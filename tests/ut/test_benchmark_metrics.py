# SPDX-License-Identifier: Apache-2.0
"""Small numerical fixtures for client metrics, without simulating inference."""

import copy
import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "benchmark_metrics", Path(__file__).resolve().parents[2] / "scripts/benchmark_metrics.py"
)
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


def request(name, rank, events):
    events = [[int(seconds * 1_000_000_000), count] for seconds, count in events]
    return {
        "request_id": name,
        "dp_rank": rank,
        "status": "ok",
        "token_ids_complete": True,
        "first_ns": events[0][0],
        "last_ns": events[-1][0],
        "completion_tokens": sum(count for _, count in events),
        "token_events": events,
        "end_ns": 999_000_000_000,  # Delayed [DONE] never extends the window.
    }


def streams(dp=False):
    return [
        request("a", 0, [(0, 1), (2, 2), (3, 3), (7, 4)]),
        request("b", int(dp), [(1, 1), (2.5, 4), (6, 1)]),
    ]


class BenchmarkMetricTest(unittest.TestCase):
    def test_common_window_counts_actual_mtp_tokens_and_aligns_tpot_tps(self):
        result = METRICS.summarize_overlap(streams(), 2, 1)
        self.assertEqual(result["status"], "ready", result)
        self.assertEqual(result["window"], {"start_ns": 1_000_000_000, "end_ns": 6_000_000_000, "seconds": 5})
        self.assertEqual(result["tokens"], 9)
        self.assertAlmostEqual(result["tps"], 1.8)
        self.assertAlmostEqual(result["tpot_ms"], 10000 / 9)
        self.assertAlmostEqual(result["tps"] * result["tpot_ms"] / 1000, 2)
        self.assertEqual(result["per_request"]["a"]["events"], 2)

    def test_all_dp_use_the_same_window_and_no_tp_multiplier(self):
        rows = streams(dp=True)
        # TP is only configuration metadata, never an input to the metrics.
        for row in rows:
            row["tensor_parallel_size"] = 8
        result = METRICS.summarize_overlap(rows, 1, 2)
        self.assertEqual(result["status"], "ready", result)
        self.assertEqual(result["tokens"], 9)
        self.assertAlmostEqual(result["per_dp"]["0"]["tps"], 1)
        self.assertAlmostEqual(result["per_dp"]["1"]["tps"], 0.8)
        self.assertAlmostEqual(sum(row["tps"] for row in result["per_dp"].values()), result["tps"])
        self.assertAlmostEqual(result["tpot_ms"], 10000 / 9)

    def test_boundary_bursts_are_excluded_whole(self):
        rows = [
            request("a", 0, [(0, 1), (1, 4), (2, 3), (6, 4), (7, 1)]),
            request("b", 0, [(1, 4), (3, 2), (6, 4)]),
        ]
        result = METRICS.summarize_overlap(rows, 2, 1)
        self.assertEqual(result["tokens"], 5)
        self.assertEqual(result["window"]["seconds"], 5)

    def test_missing_failed_or_inconsistent_streams_are_unavailable(self):
        bad_rows = [streams()[:1]]
        for key, value in (
            ("request_id", "a"),
            ("dp_rank", 1),
            ("status", "error"),
            ("token_ids_complete", False),
            ("completion_tokens", 999),
            ("first_ns", 3),
            ("last_ns", 999),
            ("token_events", [[2, 1], [1, 1]]),
        ):
            rows = copy.deepcopy(streams())
            rows[1][key] = value
            bad_rows.append(rows)
        for rows in bad_rows:
            with self.subTest(rows=rows):
                result = METRICS.summarize_overlap(rows, 2, 1)
                self.assertEqual(result["status"], "unavailable")
                self.assertIsNone(result["tps"])
                self.assertIsNone(result["tpot_ms"])

    def test_empty_overlap_or_a_request_without_interior_tokens_is_unavailable(self):
        for other in (
            [(8, 1), (9, 1)],  # No shared interval.
            [(1, 6)],  # Entire output in one SSE burst.
            [(1, 1), (6, 5)],  # Other request's tokens cannot mask a silent stream.
        ):
            with self.subTest(other=other):
                rows = [streams()[0], request("b", 0, other)]
                result = METRICS.summarize_overlap(rows, 2, 1)
                self.assertEqual(result["status"], "unavailable")
                self.assertIsNone(result["tps"])
                self.assertTrue(result["reason"])


if __name__ == "__main__":
    unittest.main()
