# SPDX-License-Identifier: Apache-2.0
"""Capacity/lifecycle adapter for the original slot LRU kernels."""

import numpy as np
from numba import set_num_threads

from .lru_core import _commit_lru_rows, _reset_lru_rows, _write_oldest_victims

TOPK = 2048
BITMAP_WORD_BITS = 32
BITMAP_ALIGNMENT_WORDS = 8


class SlotLRU:
    """One request/layer chain. Only the committed slot prefix is evictable.

    Until LIDU is connected, token i owns slot i. Reserved ABC slots are
    outside ``active`` and are never linked into the eviction chain.
    """

    def __init__(self):
        self.version = 0
        self.active = 0
        self.next = np.empty((1, 0), dtype=np.int32)
        self.previous = np.empty((1, 0), dtype=np.int32)
        self.head = np.full(1, -1, dtype=np.int32)
        self.tail = np.full(1, -1, dtype=np.int32)
        self.slots = np.full((1, TOPK), -1, dtype=np.int32)
        self.bitmap = np.empty((1, 0), dtype=np.int32)
        self.unique = np.zeros(1, dtype=np.int32)
        self.rows = np.zeros(1, dtype=np.int64)

    def resize(self, capacity, committed):
        """Grow storage and append newly committed slots, preserving LRU order."""
        if not 0 <= committed <= capacity:
            raise ValueError("Committed slots must fit HBM capacity")
        set_num_threads(1)  # Same one-worker default as the original runtime.
        changed = False
        if capacity > self.next.shape[1]:
            capacity = max(capacity, 2 * self.next.shape[1])
            for name in ("next", "previous"):
                old = getattr(self, name)
                new = np.full((1, capacity), -1, dtype=np.int32)
                new[:, : old.shape[1]] = old
                setattr(self, name, new)
            words = (capacity + BITMAP_WORD_BITS - 1) // BITMAP_WORD_BITS
            words = (words + BITMAP_ALIGNMENT_WORDS - 1) // BITMAP_ALIGNMENT_WORDS * BITMAP_ALIGNMENT_WORDS
            self.bitmap = np.zeros((1, words), dtype=np.int32)
            changed = True
        count = min(TOPK, committed)
        if committed < self.active or self.active == 0 < committed:
            _reset_lru_rows(
                self.rows,
                np.array([committed], dtype=np.int32),
                self.next,
                self.previous,
                self.head,
                self.tail,
                self.slots,
                self.bitmap,
                self.unique,
                count,
                self.bitmap.shape[1],
            )
            self.slots[0, count:] = -1
            self.active = committed
            self.version += 1
            return
        if committed > self.active:
            first = self.active
            values = np.arange(first, committed, dtype=np.int32)
            self.previous[0, first:committed] = values - 1
            self.next[0, first:committed] = values + 1
            self.previous[0, first] = self.tail[0]
            self.next[0, self.tail[0]] = first
            self.next[0, committed - 1] = -1
            self.tail[0] = committed - 1
            self.active = committed
            changed = True
        if changed:
            self._select()

    def _select(self):
        self.version += 1
        self.slots.fill(-1)
        _write_oldest_victims(
            0,
            self.next,
            self.head,
            self.slots,
            self.bitmap,
            min(TOPK, self.active),
            self.bitmap.shape[1],
        )

    def commit(self, destinations):
        """Consume actual accessed slots; ABC hits do not enter the LRU chain."""
        set_num_threads(1)
        # The caller validates against all mapped HBM slots before submission.
        feedback = np.ascontiguousarray(destinations[destinations < self.active], dtype=np.int32)
        if not feedback.size:
            return
        _commit_lru_rows(
            self.rows,
            feedback,
            np.array([0, feedback.size], dtype=np.int64),
            np.array([self.active], dtype=np.int32),
            self.next,
            self.previous,
            self.head,
            self.tail,
            self.slots,
            self.bitmap,
            self.unique,
            min(TOPK, self.active),
            self.bitmap.shape[1],
        )
        self.version += 1


def warmup():
    """Compile before serving, not on the first real decode round."""
    row = SlotLRU()
    row.resize(32, 16)
    row.commit(np.array([1], dtype=np.int32))
    row.resize(32, 24)
