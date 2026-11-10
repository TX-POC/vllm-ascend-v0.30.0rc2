# SPDX-License-Identifier: Apache-2.0
"""Native ConsumerScheduler methods plus actual PD destination ownership.

The tests model the Runner's release/reset on preemption; they do not execute
the full vLLM Scheduler's memory-pressure selection or NPU local recomputation.
Only SDK, device execution and the proxy transport use host substitutes.
"""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import test_kvaf_pd_flow as flow
import test_kvaf_pd_integration as integration
import test_kvaf_pd_recovery as recovery


class TestPDDecodeLifecycle(unittest.TestCase):
    def setUp(self):
        self.fixture = flow.TestPDFlow()

        def call(endpoint, message, **kwargs):
            return self.fixture.receiver.dispatch(message)

        for target, name, replacement in (
            (flow.RECEIVER, "ControlServer", recovery._DirectServer),
            (flow.SENDER, "call", call),
        ):
            patcher = patch.object(target, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        cache = NS(
            kv_cache_groups=[
                NS(kv_cache_spec=NS(block_size=128), layer_names=[name])
                for name in ("main", "indexer", "mtp", "mtp_indexer")
            ]
        )
        self.consumer = integration.SCHEDULER.ConsumerScheduler(integration.config(), True, cache)
        self.addCleanup(self.consumer.shutdown)
        self.request = integration.TestNativeSchedulers.request()
        with patch.object(integration.NATIVE.SFAPDRD2HScheduler, "_submit_metaserver_request") as submit:
            self.consumer.update_state_after_alloc(self.request, NS(get_block_ids=lambda: self.fixture.blocks), 300)
        self.rendezvous = submit.call_args.kwargs["message"]
        self.metadata = self.consumer.build_connector_meta(None)
        self.seed = self.metadata.requests["r"]
        self.fixture.receiver.seed("r", "r", self.seed)

    def receive(self):
        self.fixture.chunk(300, generation=self.seed.generation)

    def assert_destination_released(self):
        fixture = self.fixture
        self.assertNotIn("r", fixture.writer.requests)
        self.assertFalse(fixture.native.pages)
        self.assertFalse(fixture.writer.slots[656, 128].references)
        self.assertFalse(fixture.pool.policy.receive_slots)
        self.assertFalse(fixture.pool.policy.receive_blocks)
        self.assertFalse(fixture.pool.policy._received_tails)

    def test_pending_receive_defers_native_finish_until_original_sources_drain(self):
        fixture = self.fixture
        self.assertFalse(self.request.kv_transfer_params["do_remote_prefill"])
        self.assertEqual(self.rendezvous["kvaf_generation"], self.seed.generation)
        self.assertEqual(self.consumer.request_finished_all_groups(self.request, fixture.blocks), (True, None))
        fixture.writer.release_requests({"r"})
        fixture.receiver.release({"r"})
        self.consumer.update_connector_output(NS(finished_recving=set()))
        self.assertIn("r", self.consumer._pd_requests)
        self.assertIn("r", fixture.writer.receiving)
        self.assertIn("r", fixture.writer.deferred_release)
        self.assertIn("r", fixture.pool.policy.receive_slots)

        self.receive()
        completed = fixture.receiver.get_finished()
        self.assertEqual(completed, {"r"})
        self.consumer.update_connector_output(NS(finished_recving=completed))
        fixture.receiver.release(set())
        self.assertNotIn("r", self.consumer._pd_requests)
        self.assertNotIn("r", fixture.receiver.requests)
        self.assertFalse(fixture.writer.receiving)
        self.assertFalse(fixture.writer.deferred_release)
        self.assert_destination_released()
        self.assertFalse(any(source.active is not None for source in fixture.sdk.sources.values()))

    def test_published_request_reallocation_uses_native_local_recompute_without_rendezvous(self):
        fixture = self.fixture
        self.receive()
        fixture.assert_payload(300)
        completed = fixture.receiver.get_finished()
        self.assertEqual(completed, {"r"})
        self.consumer.update_connector_output(NS(finished_recving=completed))
        self.assertNotIn("r", self.consumer._dspark_pending_recv)
        self.assertIn("r", self.consumer._pd_requests)
        self.assertTrue(fixture.writer.requests["r"][0].keys)
        self.assertTrue(fixture.pool.policy._received_tails)
        sequences = tuple(source.sequence for source in fixture.sdk.sources.values())

        # Native _preempt_request resets computed tokens without invoking the
        # connector's request_finished; its next Runner frame releases IDs in
        # preempted_req_ids before native states and block tables are rebuilt.
        self.request.num_computed_tokens = 0
        fixture.writer.release_requests({"r"})
        self.assert_destination_released()
        self.assertEqual(self.consumer.get_num_new_matched_tokens(self.request, 0), (0, False))
        new_blocks = ((200, 201, 202), (0, 2, 4), (1, 3, 5), (0, 2, 4))
        with patch.object(integration.NATIVE.SFAPDRD2HScheduler, "_submit_metaserver_request") as submit:
            self.consumer.update_state_after_alloc(self.request, NS(get_block_ids=lambda: new_blocks), 0)
        submit.assert_not_called()
        self.assertFalse(self.consumer.build_connector_meta(None).requests)
        self.assertIs(self.consumer._pd_requests["r"], self.seed)
        self.assertFalse(self.request.kv_transfer_params["do_remote_prefill"])
        self.assertEqual(tuple(source.sequence for source in fixture.sdk.sources.values()), sequences)

        # Prepare actual host directory/progress for native local recompute.
        # Native model execution and the newly produced KV bytes are not run.
        fixture.writer.prepare_requests({"r": NS(num_computed_tokens=0, block_ids=new_blocks)}, {"r": 128})
        self.assertTrue(all(not state.keys and not state.addresses for state in fixture.writer.requests["r"]))
        self.assertTrue(all(state.progress.block_ids == (200,) for state in fixture.writer.requests["r"]))
        self.assertEqual(self.consumer.request_finished_all_groups(self.request, new_blocks), (False, None))
        fixture.receiver.release({"r"})
        self.assertNotIn("r", self.consumer._pd_requests)
        self.assertNotIn("r", fixture.receiver.requests)
        self.assert_destination_released()


if __name__ == "__main__":
    unittest.main()
