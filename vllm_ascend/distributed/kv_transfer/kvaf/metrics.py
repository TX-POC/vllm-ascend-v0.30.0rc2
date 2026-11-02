# SPDX-License-Identifier: Apache-2.0
"""Per-worker decode history miss statistics, consumed by the CPU LRU thread."""

import logging
import time

import numpy as np

LOG_INTERVAL_SECONDS = 5.0
logger = logging.getLogger("vllm")


class HotMissStats:
    def __init__(self, layers, context="", weights=None):
        self.context = context
        self.weights = weights
        layers = sum(weights) if weights is not None else layers
        self.misses = np.zeros(layers, dtype=np.int64)
        self.references = np.zeros(layers, dtype=np.int64)
        self.request_rates = []
        self.rounds = self.invalid = 0
        self.last_log = time.monotonic()

    def record(self, misses, references):
        # Counts and LRU unique counts already reside on CPU. The latter
        # deduplicate MTP queries and exclude ABC slots, which are not victims.
        if self.weights is not None:
            misses = np.repeat(misses, self.weights, axis=0)
            references = np.repeat(references, self.weights, axis=0)
        valid = (references > 0) & (misses <= references)
        self.invalid += int(np.count_nonzero((misses > references) | (misses < 0)))
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
        active = self.references > 0
        if np.any(active):
            misses, references = int(self.misses.sum()), int(self.references.sum())
            logger.info(
                "KVAF hot miss summary: [%s] phase=decode scope=history_unique_tokens "
                "rounds=%s layers=%s request_rounds=%s layer_miss_pct[%s] request_round_miss_pct[%s] "
                "overall_miss_pct=%.3f hits=%s misses=%s references=%s invalid_samples=%s",
                self.context,
                self.rounds,
                int(active.sum()),
                len(self.request_rates),
                self.percentiles(100 * self.misses[active] / self.references[active]),
                self.percentiles(self.request_rates),
                100 * misses / references,
                references - misses,
                misses,
                references,
                self.invalid,
            )
        elif self.invalid:
            logger.warning(
                "KVAF hot miss summary: [%s] invalid_samples=%s; no valid samples", self.context, self.invalid
            )
        self.misses.fill(0)
        self.references.fill(0)
        self.request_rates.clear()
        self.rounds = self.invalid = 0
        self.last_log = time.monotonic()
