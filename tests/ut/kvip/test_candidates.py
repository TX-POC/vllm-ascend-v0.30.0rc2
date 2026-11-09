# SPDX-License-Identifier: Apache-2.0
"""Execute the service prepare method without loading the NPU/vLLM runtime."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]


def prepare_method():
    path = ROOT / "vllm_ascend/distributed/kv_transfer/kvaf/cache_policy.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DecodeHotCache")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "prepare")
    scope = dict(torch=torch, np=np, TOPK=2048, MAX_QUERIES=4, BLOCK_SIZE=128, TAIL=384, NATIVE_BITMAP_WORDS=512)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope["prepare"]


def policy(fused):
    inputs = {
        name: torch.empty(3, dtype=torch.int32)
        for name in ("entries", "query_ends", "budgets", "lengths", "logical", "miss_counts")
    }
    inputs["blocks"] = torch.zeros(2, 3, 35, dtype=torch.int32)
    return SimpleNamespace(
        pool=SimpleNamespace(
            on_npu=False,
            max_requests=3,
            max_tokens=4096,
            storage={"token_to_slot": torch.empty(0), "slot_to_token": torch.empty(0)},
        ),
        turn=0,
        done=[None, None],
        transfer_host=[{}, {}],
        max_queries=12,
        spare_slots=torch.zeros(2, 3, 2048, dtype=torch.int32),
        spare_bitmap=torch.zeros(2, 3, 512, dtype=torch.int32),
        abc_descriptors=torch.zeros(2, 3, 4, dtype=torch.int32),
        fused_reserved_hot=torch.full((2, 3, 4096), -1, dtype=torch.int32) if fused else None,
        owners={"a": 1, "b": 2},
        map_groups=[0, 1],
        snapshots={(req, group): (4096, 32) for req in ("a", "b") for group in range(2)},
        history_capacity=4096,
        slot_capacity=4480,
        tail_start=4096,
        tail_tokens=384,
        inputs=inputs,
        write_slots=torch.empty(12, dtype=torch.int64),
        query_bases=torch.empty(12, 1, 1, dtype=torch.int64),
        cursors=torch.zeros(2),
        uploaded_table_owners=(1, 2),
        resident_tables=torch.zeros(3, 35, dtype=torch.int32),
    )


def manager(ids, shift=0):
    frames = []
    for group in range(2):
        rows = []
        for req in ids:
            candidates = (np.arange(2048, dtype=np.int32) + group * 777 + (req == "b") * 311 + shift) % 4096
            state = SimpleNamespace(
                addresses=list(range(32)), progress=SimpleNamespace(computed_tokens=4096, write_end=4097)
            )
            rows.append(
                SimpleNamespace(state=state, lru=SimpleNamespace(hot_snapshot=(candidates, np.zeros(512, np.int32))))
            )
        frames.append(SimpleNamespace(device=None, request_ids=ids, rows=rows, query_offsets=np.arange(len(ids) + 1)))
    return SimpleNamespace(groups=frames)


def test_group_mapping_reorder_shrink_and_native_unchanged():
    prepare = prepare_method()
    fused, native = policy(True), policy(False)
    abc = SimpleNamespace(update_abc=Mock())
    with patch.dict(sys.modules, {"kvip.cache.abc": abc}):
        for ids, shift in ((["a", "b"], 0), (["b", "a"], 109), (["b"], 509)):
            frame = manager(ids, shift)
            for instance in (fused, native):
                prepare(instance, frame, decode=True, padded_requests=3)
            for group in range(2):
                expected = torch.full((3, 4096), -1, dtype=torch.int32)
                for row in range(len(ids)):
                    slots = torch.from_numpy(frame.groups[group].rows[row].lru.hot_snapshot[0]).long()
                    expected[row, slots] = torch.arange(2048, dtype=torch.int32)
                assert torch.equal(fused.fused_reserved_hot[group], expected)
            # Native inputs and uploads are unchanged by enabling the extra map.
            for name in ("spare_slots", "spare_bitmap", "abc_descriptors", "write_slots", "query_bases"):
                assert torch.equal(getattr(fused, name), getattr(native, name))
            for name in native.inputs:
                assert torch.equal(fused.inputs[name], native.inputs[name])
            assert all("reserved_hot_batch" not in host for host in native.transfer_host)
    assert abc.update_abc.call_count == 6


def test_short_history_candidate_allocation_cannot_move_abc_or_future_index():
    from kvip.adapters.vllm.inputs import committed_candidates

    resident, topk = 16384, 2048
    for committed in (0, 128, 1024, 2048, 3968, 11392, 16384, 131072):
        # A valid LRU order can start at the newest/empty end of hot capacity.
        original = np.arange(resident - topk, resident, dtype=np.int32)
        candidates = committed_candidates(original, committed, resident)
        assert len(candidates) == topk and len(set(candidates.tolist())) == topk
        reads = min(committed, topk)
        assert np.all(candidates[:reads] < min(committed, resident))
        if committed >= resident:
            assert candidates is original
        else:
            # Producer's first reads can only name committed rows; closing the
            # update permutation therefore cannot relocate ABC/future rows.
            index = np.arange(resident + 384)
            for source, destination in zip(range(reads), candidates[:reads]):
                index[source], index[destination] = index[destination], index[source]
            np.testing.assert_array_equal(index[committed:], np.arange(len(index))[committed:])


def test_committed_candidate_priority_preserves_lru_order():
    from kvip.adapters.vllm.inputs import committed_candidates

    candidates = np.array([7, 1, 6, 3], dtype=np.int32)
    np.testing.assert_array_equal(committed_candidates(candidates, 4, 8), [1, 3, 0, 2])


def test_short_service_prepare_uses_same_committed_slots_in_every_group():
    prepare = prepare_method()
    fused = policy(True)
    frame = manager(["b", "a"], shift=1800)
    for group in range(2):
        for row, item in enumerate(frame.groups[group].rows):
            committed = (128, 3968)[row]
            item.state.addresses = list(range(committed // 128))
            item.state.progress.computed_tokens = committed + 1
            item.state.progress.write_end = committed + 2
            fused.snapshots[(frame.groups[group].request_ids[row], group)] = (committed, committed // 128)
    with patch.dict(sys.modules, {"kvip.cache.abc": SimpleNamespace(update_abc=Mock())}):
        prepare(fused, frame, decode=True, padded_requests=3)
    for group in range(2):
        for row, committed in enumerate((128, 3968)):
            slots = fused.spare_slots[group, row].long()
            assert (slots[: min(committed, 2048)] < committed).all()
            assert torch.equal(fused.fused_reserved_hot[group, row, slots], torch.arange(2048, dtype=torch.int32))
            assert fused.abc_descriptors[group, row, 1] == (2, 1)[row]


def test_service_workspace_uses_external_kvip():
    from kvip.operators.lifetch.workspace import LIFetchWorkspace

    path = ROOT / "vllm_ascend/distributed/kv_transfer/kvaf/cache_policy.py"
    tree = ast.parse(path.read_text())
    method = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "initialize_fused_workspace"
    )
    scope = dict(torch=torch, SimpleNamespace=SimpleNamespace)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    state = SimpleNamespace(
        fused_index_storage=[SimpleNamespace(history=4096)],
        fused_workspace=None,
        pool=SimpleNamespace(max_requests=3, device="cpu"),
        history_capacity=2048,
    )
    config = SimpleNamespace(
        build_dir="/external/KVAF/build-npu", intermediate_fetches=4, npu_core_count=4, transport="urma"
    )

    def cpu_workspace(*args, **kwargs):
        return LIFetchWorkspace(*args, **kwargs, pin_memory=False)

    with (
        patch("kvip.runtime.build.kvaf_build_paths", return_value=("/cann", "/headers")),
        patch("kvip.runtime.device.query_device_resources", return_value={"effective_aic": 32}),
        patch("kvip.operators.lifetch.workspace.LIFetchWorkspace", side_effect=cpu_workspace),
    ):
        scope["initialize_fused_workspace"](state, config, 0, "Ascend950PR_9599", index_heads=32)
    assert isinstance(state.fused_workspace, LIFetchWorkspace)
    assert state.fused_workspace.topk.shape == (2, 4, 2048)
    assert state.fused_workspace.scratch["reserved_hot"].shape == (2, 2048)
    assert state.fused_setup.batch == 2


def test_ubmem_workspace_dispatches_before_memory_profiling():
    path = ROOT / "vllm_ascend/distributed/kv_transfer/kvaf/cache_policy.py"
    tree = ast.parse(path.read_text())
    method = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "initialize_fused_workspace"
    )
    scope = dict(torch=torch)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    state = SimpleNamespace()
    config = SimpleNamespace(transport="ubmem")
    with patch("kvip.adapters.vllm.binding.prepare_workspace") as prepare:
        scope["initialize_fused_workspace"](state, config, 5, "Ascend950PR_9599", index_heads=32)
    prepare.assert_called_once_with(torch, state, config, 5, "Ascend950PR_9599", index_heads=32)
