# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for actual grouped BatchGet count reporting."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "kvaf_metrics", Path(__file__).resolve().parents[3] / "vllm_ascend/distributed/kv_transfer/kvaf/metrics.py"
)
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


class TestGetMetrics(unittest.TestCase):
    def test_group_counts_are_not_repeated_per_layer(self):
        stats = METRICS.HotMissStats(2, "DP0", weights=(4, 1))
        with patch.object(METRICS.time, "monotonic", return_value=stats.last_log):
            stats.record(np.array([[3, 5], [1, 2]]), np.full((2, 2), 10), np.array([8, 3]))
            stats.record(np.zeros((2, 2), dtype=int), np.full((2, 2), 10), np.array([0, 0]))
        self.assertEqual(stats.get_counts, {1: [3, 0], 4: [8, 0]})
        self.assertEqual(stats.misses.tolist(), [8, 8, 8, 8, 3])
        with self.assertLogs("vllm", level="INFO") as captured:
            stats.flush()
        output = " ".join(captured.output)
        self.assertIn("overall_miss_pct=", output)
        self.assertIn("get_tokens_4layer[calls=2 p50=4.000 p90=7.200 p99=7.920 avg=4.000]", output)
        self.assertIn("get_tokens_1layer[calls=2 p50=1.500", output)
        self.assertEqual(stats.get_counts, {1: [], 4: []})

    def test_prefetch_ratio_above_one_is_valid_and_has_no_negative_hits(self):
        stats = METRICS.HotMissStats(1, weights=(4,))
        with patch.object(METRICS.time, "monotonic", return_value=stats.last_log):
            stats.record(np.array([[12]]), np.array([[8]]), np.array([12]), ddr_fetch=True)
        self.assertEqual(stats.invalid, 0)
        with self.assertLogs("vllm", level="INFO") as captured:
            stats.flush()
        output = " ".join(captured.output)
        self.assertIn("scope=ddr_fetch_including_prefetch", output)
        self.assertIn("overall_miss_pct=150.000", output)
        self.assertNotIn("hits=", output)
        self.assertIn("get_tokens_4layer[calls=1 p50=12.000", output)

    def test_zero_reference_gets_still_report_zero_payload(self):
        stats = METRICS.HotMissStats(1, weights=(4,))
        zeros = np.zeros((1, 2), dtype=int)
        stats.record(zeros, zeros, np.array([0]))
        with self.assertLogs("vllm", level="INFO") as captured:
            stats.flush()
        self.assertIn("get_tokens_4layer[calls=1 p50=0.000", captured.output[0])


if __name__ == "__main__":
    unittest.main()
