# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical resident-slot planning for the fused C8 speculative-Get path.

This module is deliberately a small CPU/NPU-neutral oracle for the device
operator.  It models the *only* persistent effect that the fused operator is
allowed to expose: the exact C8 Top-K sources map to canonical resident slots
before QSFA begins reading them.

The score-time side is free to place likely misses into a fixed per-row
resident range.  It must not modify the persistent ``cache_slots`` mapping
until exact Top-K is known.  Once Top-K is known, a winner is resolved by this
strict order:

1. keep an already-resident persistent hit;
2. use its preassigned speculative resident slot when it was posted earlier;
3. assign one still-unused slot from the same fixed resident range and add it
   to Guaranteed Recall.

There is no private arena, cross-forward resource ownership, slot release, or
ABA protocol here.  The fused device kernel drains every Direct-HCCP post
before returning; QSFA subsequently reads only ``attention_slots``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


_UNSET_SLOT = -1


@dataclass(frozen=True)
class C8CanonicalResidentPlan:
    """Exact final placement calculated from an already-completed C8 Top-K.

    ``attention_slots`` has the same order as ``topk_source_ids``.  A non-negative
    entry in ``recall_source_ids`` / ``recall_destination_slots`` is a final
    winner that was not posted during score scanning and therefore needs a
    Guaranteed Recall post.  Padding is ``-1``.
    """

    attention_slots: torch.Tensor
    recall_source_ids: torch.Tensor
    recall_destination_slots: torch.Tensor
    speculative_winner_mask: torch.Tensor

    @property
    def recall_count(self) -> torch.Tensor:
        return self.recall_source_ids.ne(_UNSET_SLOT).sum(dim=1)


def _require_integer_tensor(name: str, value: torch.Tensor, ndim: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"{name} must be rank-{ndim}, got shape={tuple(value.shape)}")
    if value.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must use int32 or int64, got {value.dtype}")


def _validate_row_unique(name: str, values: torch.Tensor) -> None:
    """Reject duplicate non-negative values in each small fixed row."""

    for row in values.detach().cpu().tolist():
        seen: set[int] = set()
        for value in row:
            value = int(value)
            if value == _UNSET_SLOT:
                continue
            if value < _UNSET_SLOT:
                raise ValueError(f"{name} contains invalid negative value={value}")
            if value in seen:
                raise ValueError(f"{name} contains duplicate value={value}")
            seen.add(value)


def plan_c8_canonical_resident_slots(
    *,
    topk_source_ids: torch.Tensor,
    cache_slots: torch.Tensor,
    speculative_source_ids: torch.Tensor,
    speculative_resident_slots: torch.Tensor,
) -> C8CanonicalResidentPlan:
    """Resolve exact C8 winners into a fixed direct-resident range.

    Args:
        topk_source_ids: Exact C8 Top-K source ids, ``[batch, topk]``.  Each
            row is expected to contain unique, non-negative source ids.
        cache_slots: Existing persistent ``source -> resident slot`` mapping,
            ``[batch, source_capacity]``; ``-1`` means not resident.
        speculative_source_ids: Score-time candidate ids, ``[batch, topk]``.
            Candidate ``i`` was posted to ``speculative_resident_slots[..., i]``.
            The suffix may be ``-1`` when fewer than Top-K candidates were
            selected or a queue was unavailable.
        speculative_resident_slots: Fixed pre-reserved canonical slot range,
            ``[batch, topk]``.  These slots are distinct per row and must not
            already appear in ``cache_slots``.

    The fixed range deliberately has Top-K capacity.  Therefore any number of
    speculative winners plus the remaining final misses always fits: final
    winners are at most Top-K, and every non-speculative winner consumes one
    of the still-unused range entries.
    """

    _require_integer_tensor("topk_source_ids", topk_source_ids, 2)
    _require_integer_tensor("cache_slots", cache_slots, 2)
    _require_integer_tensor("speculative_source_ids", speculative_source_ids, 2)
    _require_integer_tensor(
        "speculative_resident_slots", speculative_resident_slots, 2
    )
    if topk_source_ids.shape != speculative_source_ids.shape:
        raise ValueError(
            "topk_source_ids and speculative_source_ids must have the same shape"
        )
    if topk_source_ids.shape != speculative_resident_slots.shape:
        raise ValueError(
            "topk_source_ids and speculative_resident_slots must have the same shape"
        )
    if cache_slots.shape[0] != topk_source_ids.shape[0]:
        raise ValueError("cache_slots batch dimension must match topk_source_ids")
    if not (
        topk_source_ids.device
        == cache_slots.device
        == speculative_source_ids.device
        == speculative_resident_slots.device
    ):
        raise ValueError("all planner tensors must be on one device")

    _validate_row_unique("topk_source_ids", topk_source_ids)
    _validate_row_unique("speculative_source_ids", speculative_source_ids)
    _validate_row_unique("speculative_resident_slots", speculative_resident_slots)

    batch, topk = topk_source_ids.shape
    source_capacity = cache_slots.shape[1]
    result_dtype = topk_source_ids.dtype
    attention_slots = torch.full_like(topk_source_ids, _UNSET_SLOT)
    recall_source_ids = torch.full_like(topk_source_ids, _UNSET_SLOT)
    recall_destination_slots = torch.full_like(topk_source_ids, _UNSET_SLOT)
    speculative_winner_mask = torch.zeros(
        (batch, topk), dtype=torch.bool, device=topk_source_ids.device
    )

    # This is the reference oracle, not a throughput path.  The device kernel
    # implements the same bounded maps in GM/UB and makes the equivalent
    # selection without a host round-trip.
    topk_rows = topk_source_ids.detach().cpu().tolist()
    cache_rows = cache_slots.detach().cpu().tolist()
    candidate_rows = speculative_source_ids.detach().cpu().tolist()
    resident_rows = speculative_resident_slots.detach().cpu().tolist()
    for batch_index in range(batch):
        slots = [int(value) for value in resident_rows[batch_index]]
        if any(slot < 0 for slot in slots):
            raise ValueError("speculative_resident_slots must be non-negative")
        resident_set = set(slots)
        if any(int(slot) in resident_set for slot in cache_rows[batch_index] if int(slot) >= 0):
            raise ValueError(
                "fixed speculative resident range overlaps an existing cache slot"
            )

        candidate_to_slot: dict[int, int] = {}
        for candidate_index, source in enumerate(candidate_rows[batch_index]):
            source = int(source)
            if source == _UNSET_SLOT:
                continue
            if not 0 <= source < source_capacity:
                raise ValueError(
                    f"speculative source={source} is outside source capacity={source_capacity}"
                )
            if int(cache_rows[batch_index][source]) >= 0:
                raise ValueError(
                    "a score-time candidate must be a miss; it is already resident"
                )
            candidate_to_slot[source] = slots[candidate_index]

        used_fixed_slots: set[int] = set()
        recall_cursor = 0
        for topk_index, source in enumerate(topk_rows[batch_index]):
            source = int(source)
            if not 0 <= source < source_capacity:
                raise ValueError(
                    f"exact Top-K source={source} is outside source capacity={source_capacity}"
                )
            persistent_slot = int(cache_rows[batch_index][source])
            if persistent_slot >= 0:
                attention_slots[batch_index, topk_index] = persistent_slot
                continue
            speculative_slot = candidate_to_slot.get(source)
            if speculative_slot is not None:
                attention_slots[batch_index, topk_index] = speculative_slot
                speculative_winner_mask[batch_index, topk_index] = True
                used_fixed_slots.add(speculative_slot)
                continue

            destination = next(
                (slot for slot in slots if slot not in used_fixed_slots), None
            )
            if destination is None:
                raise RuntimeError(
                    "Top-K reconciliation exhausted its fixed resident range"
                )
            used_fixed_slots.add(destination)
            attention_slots[batch_index, topk_index] = destination
            recall_source_ids[batch_index, recall_cursor] = source
            recall_destination_slots[batch_index, recall_cursor] = destination
            recall_cursor += 1

    return C8CanonicalResidentPlan(
        attention_slots=attention_slots.to(dtype=result_dtype),
        recall_source_ids=recall_source_ids.to(dtype=result_dtype),
        recall_destination_slots=recall_destination_slots.to(dtype=result_dtype),
        speculative_winner_mask=speculative_winner_mask,
    )
