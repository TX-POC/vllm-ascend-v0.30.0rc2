# SPDX-License-Identifier: Apache-2.0

import asyncio
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

from test_kvaf_pd_integration import SCHEDULER, load_file

decode_batch = load_file("pd_decode_batch_for_ut", "vllm_ascend/distributed/kv_transfer/kvaf/pd/decode_batch.py")


class TestDecodeBatchRegistry(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        settings = decode_batch.Settings(enabled=True, timeout_seconds=1)
        self.registry = decode_batch.Registry(settings.with_capacity({"dp_size": 2, "max_batch_size": 2}))
        self.specs = [
            dict(group="measure", dp_size=2, batch_size=2, rank=rank, index=0, phase="measure") for rank in range(2)
        ]
        self.messages = []
        for spec in self.specs:
            rank = spec["rank"]
            members = [[index, f"request-{rank}-{index}", "generation"] for index in range(2)]
            self.messages.append(dict(op="ready", spec=spec, timeout=1, members=members))
            for index, request_id, _ in members:
                await self.registry.accept(dict(spec, index=index), request_id)

    async def test_hot_kv_requires_opt_in_and_merges_rank_histograms_once(self):
        specs = [dict(spec, group="hot", hot_kv_stats=True) for spec in self.specs]
        with self.assertRaisesRegex(ValueError, "does not advertise"):
            await self.registry.accept(specs[0], "request")
        with self.assertRaisesRegex(ValueError, "measure wave"):
            decode_batch.validate_spec(dict(specs[0], phase="warmup"))
        await self.registry.abort(self.specs[0], "end previous test wave")
        self.registry.hot_kv_stats = True
        arrivals = []
        for spec, message in zip(specs, self.messages):
            for index, request_id, _ in message["members"]:
                await self.registry.accept(dict(spec, index=index), request_id)
            arrivals.append(self.registry.arrive(dict(message, spec=spec)))
        await asyncio.gather(*arrivals)
        self.assertNotIn("hot_kv", await self.registry.status("measure"))
        for rank, spec in enumerate(specs):
            histogram = [0] * decode_batch.HOT_KV_BUCKETS
            histogram[rank * 1000] = rank + 1
            report = dict(spec=spec, histogram=histogram)
            await self.registry.hot_kv(report)
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                await self.registry.hot_kv(report)
        hot = (await self.registry.status("hot"))["hot_kv"]
        self.assertEqual(hot["rank_samples"], {0: 1, 1: 2})
        self.assertEqual((hot["histogram"][0], hot["histogram"][-1], sum(hot["histogram"])), (1, 2, 3))

    async def test_waits_for_all_ranks_and_records_first_step(self):
        first = asyncio.create_task(self.registry.arrive(self.messages[0]))
        await asyncio.sleep(0)
        status = await self.registry.status("measure")
        self.assertFalse(first.done())
        self.assertFalse(status["released"])
        self.assertEqual(status["ranks"][0]["kv_ready"], 2)
        self.assertEqual(status["ranks"][1]["kv_ready"], 0)
        with self.assertRaisesRegex(RuntimeError, "unreleased"):
            await self.registry.scheduled(dict(spec=self.specs[0], cohort_requests=2, scheduled_requests=2))

        second = await self.registry.arrive(self.messages[1])
        self.assertEqual(await first, second)
        await self.registry.scheduled(dict(spec=self.specs[0], cohort_requests=1, scheduled_requests=2))
        status = await self.registry.status("measure")
        self.assertTrue(status["released"])
        self.assertEqual(status["ranks"][0]["first_step"], {"cohort_requests": 1, "scheduled_requests": 2})

    async def test_abort_wakes_waiter_without_releasing_incomplete_group(self):
        first = asyncio.create_task(self.registry.arrive(self.messages[0]))
        await asyncio.sleep(0)
        await self.registry.abort(self.specs[1], "D stream failed")
        with self.assertRaisesRegex(RuntimeError, r"D stream failed.*\[1\]"):
            await first
        with self.assertRaisesRegex(RuntimeError, "D stream failed"):
            await self.registry.arrive(self.messages[1])

    async def test_first_step_contains_only_exact_cohort_count(self):
        # Native scheduler's first step includes unrelated requests in the total.
        calls = []

        def submit(operation, *args):
            calls.append(args)

        scheduler = object.__new__(SCHEDULER.ConsumerScheduler)
        scheduler.executor = SimpleNamespace(submit=submit)
        seed = SCHEDULER.BatchReceiveRequest("generation", 128, ((1,),), self.specs[0], "http://proxy", 1)
        scheduler._first_decode_pending = {"first": seed, "second": seed}
        scheduler._observe_first_decode({"first": 1, "unrelated": 1})
        scheduler._observe_first_decode({"first": 1, "second": 1})
        self.assertEqual(calls, [(seed, 1, 2)])


class TestDecodeBatchGate(unittest.TestCase):
    def test_holds_only_tagged_completions_and_releases_entire_local_batch(self):
        spec = dict(group="measure", dp_size=2, batch_size=2, rank=0, index=0, phase="measure")
        gate = decode_batch.Gate(decode_batch.Settings(dp_size=2, max_batch_size=2), 0)
        seed = SCHEDULER.BatchReceiveRequest("generation", 128, ((1,),), spec, "http://proxy", 5)
        gate.bind("first", "external-first", seed)
        second = SCHEDULER.BatchReceiveRequest("generation", 128, ((2,),), dict(spec, index=1), "http://proxy", 5)
        gate.bind("second", "external-second", second)
        self.assertEqual(gate.poll({"first", "ordinary"}, set(), set()), {"ordinary"})
        group = gate.groups["measure"]
        future = Future()
        group["future"] = future
        self.assertEqual(gate.poll({"second"}, set(), set()), set())
        future.set_result({"status": "released"})
        self.assertEqual(gate.poll(set(), set(), set()), {"first", "second"})
        self.assertEqual(gate.poll(set(), {"first", "second"}, set()), set())
        self.assertFalse(gate.groups)


class TestHotKVWaves(unittest.TestCase):
    def test_reports_once_after_last_request_and_pending_feedback(self):
        cohorts = decode_batch.HotKVWaves()
        spec = dict(group="hot", dp_size=1, batch_size=2, rank=0, index=0, phase="measure", hot_kv_stats=True)
        seed = SCHEDULER.BatchReceiveRequest("generation", 128, ((1,),), spec, "http://proxy", 5)
        cohorts.bind("first", seed)
        cohorts.bind("second", seed)
        wave = cohorts.requests["first"]
        allow_feedback, sent = Event(), Event()

        def feedback():
            allow_feedback.wait(1)
            wave.record_miss_percent(12.34)

        def send(_wave):
            sent.set()

        with patch.object(decode_batch.HotKVWave, "send", autospec=True, side_effect=send) as report:
            with ThreadPoolExecutor(max_workers=1) as fifo:
                fifo.submit(feedback)
                cohorts.finish(["first"], fifo)
                self.assertFalse(sent.is_set())
                cohorts.finish(["first", "second"], fifo)
                self.assertFalse(sent.is_set())
                allow_feedback.set()
                self.assertTrue(sent.wait(1))
            report.assert_called_once_with(wave)
        self.assertEqual(wave.histogram[877], 1)
        self.assertFalse(cohorts.requests or cohorts.waves)
