# SPDX-License-Identifier: Apache-2.0
"""Per-worker decode history miss statistics, consumed by the CPU LRU thread."""

import logging
import time

import numpy as np

LOG_INTERVAL_SECONDS = 5.0
logger = logging.getLogger("vllm")


class HotMissStats:
    def __init__(self, layers, context="", weights=None):
        self.ddr_fetch = False
        self.context = context
        self.weights = weights
        self.group_widths = tuple(weights) if weights is not None else (1,) * layers
        self.get_counts = {width: [] for width in sorted(set(self.group_widths))}
        layers = sum(weights) if weights is not None else layers
        self.misses = np.zeros(layers, dtype=np.int64)
        self.references = np.zeros(layers, dtype=np.int64)
        self.request_rates = []
        self.rounds = self.invalid = 0
        self.last_log = time.monotonic()

    def record(self, misses, references, get_counts=None, *, ddr_fetch=False):
        self.ddr_fetch = ddr_fetch
        # Counts and LRU unique counts already reside on CPU. The latter
        # deduplicate MTP queries and exclude ABC slots, which are not victims.
        # One sample is one group Get, summed over requests, not one physical
        # layer or query. Record before expanding grouped-layer miss percentages.
        totals = misses.sum(axis=1) if get_counts is None else get_counts
        for width, total in zip(self.group_widths, totals):
            if total >= 0:
                self.get_counts[width].append(int(total))
        if self.weights is not None:
            misses = np.repeat(misses, self.weights, axis=0)
            references = np.repeat(references, self.weights, axis=0)
        invalid = (misses < 0) | ((misses > references) & ~np.bool_(ddr_fetch))
        valid = (references > 0) & ~invalid
        self.invalid += int(np.count_nonzero(invalid))
        valid &= misses >= 0
        if np.any(valid):
            misses = np.where(valid, misses, 0)
            references = np.where(valid, references, 0)
            self.misses += misses.sum(axis=1)
            self.references += references.sum(axis=1)
            per_request = references.sum(axis=0)
            active = per_request > 0
            self.request_rates.extend((100 * misses.sum(axis=0)[active] / per_request[active]).tolist())
            self.rounds += 1
        if time.monotonic() - self.last_log >= LOG_INTERVAL_SECONDS:
            self.flush()

    @staticmethod
    def percentiles(values):
        p50, p90, p99 = np.percentile(values, [50, 90, 99])
        return f"p50={p50:.3f} p90={p90:.3f} p99={p99:.3f} avg={np.mean(values):.3f}"

    def flush(self):
        get_summary = " ".join(
            f"get_tokens_{width}layer[calls={len(values)} {self.percentiles(values)}]"
            for width, values in self.get_counts.items()
            if values
        )
        active = self.references > 0
        if np.any(active):
            misses, references = int(self.misses.sum()), int(self.references.sum())
            logger.info(
                "KVAF hot miss summary: [%s] phase=decode scope=%s "
                "rounds=%s layers=%s request_rounds=%s layer_miss_pct[%s] request_round_miss_pct[%s] "
                "overall_miss_pct=%.3f %s misses=%s references=%s invalid_samples=%s %s",
                self.context,
                "ddr_fetch_including_prefetch" if self.ddr_fetch else "history_unique_tokens",
                self.rounds,
                int(active.sum()),
                len(self.request_rates),
                self.percentiles(100 * self.misses[active] / self.references[active]),
                self.percentiles(self.request_rates),
                100 * misses / references,
                "includes_unused_prefetch=true" if self.ddr_fetch else f"hits={references - misses}",
                misses,
                references,
                self.invalid,
                get_summary,
            )
        elif self.invalid:
            logger.warning(
                "KVAF hot miss summary: [%s] invalid_samples=%s; no valid samples", self.context, self.invalid
            )
        if not np.any(active) and get_summary:
            logger.info("KVAF Get summary: [%s] phase=decode %s", self.context, get_summary)
        for values in self.get_counts.values():
            values.clear()
        self.misses.fill(0)
        self.references.fill(0)
        self.request_rates.clear()
        self.rounds = self.invalid = 0
        self.last_log = time.monotonic()
