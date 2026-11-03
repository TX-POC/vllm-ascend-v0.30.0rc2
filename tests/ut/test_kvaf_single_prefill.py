# SPDX-License-Identifier: Apache-2.0
"""Execute production scheduling loops with CPU-only resource/queue doubles."""

import ast
import time
import unittest
from collections import deque
from contextlib import nullcontext
from functools import wraps
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vllm_ascend/patch/platform/patch_balance_schedule.py"


class Queue(deque):
    def peek_request(self):
        return self[0]

    def pop_request(self):
        return self.popleft()

    def prepend_request(self, request):
        self.appendleft(request)

    def prepend_requests(self, requests):
        self.extendleft(reversed(requests))


STATUS = NS(WAITING="waiting", RUNNING="running", PREEMPTED="preempted", WAITING_FOR_REMOTE_KVS="remote")


class Request(NS):
    __hash__ = object.__hash__


def request(name, computed=0, prompt=512):
    return Request(
        request_id=name,
        num_computed_tokens=computed,
        num_prompt_tokens=prompt,
        num_tokens=max(prompt, computed + 1),
        num_tokens_with_spec=max(prompt, computed + 1),
        num_output_placeholders=0,
        max_tokens=512,
        next_decode_eligible_step=0,
        is_prefill_chunk=computed < prompt,
        has_encoder_inputs=False,
        spec_token_ids=[],
        lora_request=None,
        status=STATUS.WAITING,
        num_preemptions=0,
        priority=0,
        arrival_time=0,
        mm_features=[],
        prefill_stats=None,
    )


def scheduling_loops():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BalanceScheduler")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "schedule")
    # Keep both complete admission loops; omit output serialization, not policy.
    end = next(
        i
        for i, n in enumerate(method.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "total_num_scheduled_tokens" for t in n.targets)
    )
    method.body = method.body[:end] + [ast.Return(value=ast.Name(id="num_scheduled_tokens", ctx=ast.Load()))]
    method.returns = None
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    env = dict(
        time=time,
        PauseState=NS(PAUSED_ALL=1, UNPAUSED=0),
        RequestStatus=STATUS,
        SchedulingPolicy=NS(PRIORITY="priority"),
        create_request_queue=lambda _: Queue(),
        record_function_or_nullcontext=lambda _: nullcontext(),
        logger=MagicMock(),
        _upstream_schedule=lambda *_: "upstream",
    )
    exec(compile(module, str(SOURCE), "exec"), env)
    return env["schedule"]


class TestSinglePrefill(unittest.TestCase):
    def scheduler(self, running=(), waiting=(), enabled=True):
        s = NS(
            vllm_config=NS(additional_config={"kvaf_config": {"enabled": enabled}}),
            _balance_enabled=False,
            current_step=0,
            running=list(running),
            waiting=Queue(waiting),
            skipped_waiting=Queue(),
            max_num_scheduled_tokens=4096,
            _pause_state=0,
            max_num_encoder_input_tokens=0,
            prefill_capacity_bound=False,
            scheduler_config=NS(long_prefill_token_threshold=0, enable_chunked_prefill=True),
            max_model_len=131072,
            num_sampled_tokens_per_step=1,
            use_eagle=False,
            need_mamba_block_aligned_split=False,
            policy="fcfs",
            lora_config=None,
            max_num_running_reqs=16,
            connector=None,
            num_lookahead_tokens=3,
            has_mamba_layers=False,
            ec_connector=None,
            connector_prefix_cache_stats=None,
            scheduler_reserve_full_isl=False,
            is_encoder_decoder=False,
            log_stats=False,
            _inflight_prefills=set(),
            kv_cache_manager=MagicMock(),
        )
        s.kv_cache_manager.get_computed_blocks.return_value = (MagicMock(), 0, None)
        s._select_waiting_queue_for_scheduling = lambda: s.skipped_waiting if s.skipped_waiting else s.waiting
        s._is_blocked_waiting_status = lambda _: False
        s._inflight_prefill_reserved_blocks = lambda: 0
        s._preempt_request = lambda r, _: setattr(r, "num_computed_tokens", 0)
        return s

    def test_running_prefills_do_not_block_decode(self):
        s = self.scheduler([request("p1", 128), request("p2", 128), request("d", 512)])
        self.assertEqual(scheduling_loops()(s), {"p1": 384, "d": 1})

    def test_waiting_prefix_tails_share_one_prefill_admission(self):
        s = self.scheduler(waiting=[request("p1", 384), request("p2", 384), request("d", 512)])
        self.assertEqual(scheduling_loops()(s), {"p1": 128, "d": 1})
        self.assertEqual([r.request_id for r in s.skipped_waiting], ["p2"])
        self.assertEqual(scheduling_loops()(s).get("p2"), None)  # p1 still prefilling in this double

    def test_running_prefill_and_waiting_tail_share_limit(self):
        s = self.scheduler([request("p1", 128)], [request("p2", 384)])
        self.assertEqual(scheduling_loops()(s), {"p1": 384})

    def test_new_local_prefix_hits_are_limited(self):
        s = self.scheduler(waiting=[request("p1"), request("p2")])
        s.kv_cache_manager.get_computed_blocks.return_value = (MagicMock(), 384, None)
        s._record_prefix_cache_metrics = MagicMock()
        self.assertEqual(scheduling_loops()(s), {"p1": 128})

    def test_remote_loads_do_not_consume_compute_slot(self):
        p1, p2 = request("p1"), request("p2")
        s = self.scheduler([request("running", 128)], [p1, p2])
        s.connector = MagicMock()
        s.connector.get_num_new_matched_tokens.return_value = (384, True)
        s._record_prefix_cache_metrics = MagicMock()
        self.assertEqual(scheduling_loops()(s), {"running": 384})
        self.assertEqual([p1.status, p2.status], [STATUS.WAITING_FOR_REMOTE_KVS] * 2)
        self.assertEqual(s.connector.update_state_after_alloc.call_count, 2)

    def test_no_work_does_not_consume_slot(self):
        empty = request("empty", 128)
        empty.num_tokens_with_spec = 128
        s = self.scheduler([empty, request("p2", 128)])
        self.assertEqual(scheduling_loops()(s), {"p2": 384})

    def test_decode_batch_is_not_reduced(self):
        s = self.scheduler([request(str(i), 512) for i in range(16)])
        self.assertEqual(len(scheduling_loops()(s)), 16)

    def test_preempted_prefill_returns_its_slot(self):
        prefill = request("p1", 128)
        prefill.priority = 10
        s = self.scheduler([prefill, request("decode", 512), request("p2", 128)])
        s.policy = "priority"
        s.kv_cache_manager.allocate_slots.side_effect = [MagicMock(), None, MagicMock(), MagicMock()]
        self.assertEqual(scheduling_loops()(s), {"decode": 1, "p2": 384})

    def test_already_imported_async_base_uses_limit(self):
        class Base:
            def schedule(self, throttle_prefills=False):
                return "upstream"

        class Async(Base):
            def _update_after_schedule(self, output):
                return output

        update = Async._update_after_schedule
        tree = ast.parse(SOURCE.read_text())
        wrapper = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_schedule_with_kvaf_prefill_limit"
        )
        env = dict(wraps=wraps, _upstream_schedule=Base.schedule, BalanceScheduler=NS(schedule=scheduling_loops()))
        exec(compile(ast.Module(body=[wrapper], type_ignores=[]), str(SOURCE), "exec"), env)
        Base.schedule = env[wrapper.name]
        scheduler = Async()
        scheduler.__dict__.update(vars(self.scheduler([request("a", 128), request("b", 128)])))
        self.assertEqual(scheduler.schedule(), {"a": 384})
        self.assertIs(Async._update_after_schedule, update)
        scheduler.vllm_config.additional_config = {}
        self.assertEqual(scheduler.schedule(), "upstream")

    def test_disabled_delegates_to_upstream(self):
        self.assertEqual(scheduling_loops()(self.scheduler(enabled=False)), "upstream")


if __name__ == "__main__":
    unittest.main()
