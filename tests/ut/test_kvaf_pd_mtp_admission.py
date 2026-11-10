# SPDX-License-Identifier: Apache-2.0
"""Host-only native MTP boundary admission and P request lifecycle checks."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from test_kvaf_pd_integration import NATIVE, SCHEDULER, config


class TestNativeMTPAdmission(unittest.TestCase):
    def setUp(self):
        self.config = config(producer=True, tp=8, dp=2)
        self.config.model_config = NS(max_model_len=1024)
        self.config.speculative_config.draft_model_config = None
        self.cache = NS(kv_cache_groups=[NS(kv_cache_spec=NS(block_size=128)) for _ in range(4)])
        self.producer = SCHEDULER.ProducerScheduler(self.config, self.cache, "P")
        self.blocks = NS(get_block_ids=lambda: ([1, 2, 3],) * 4)

    @staticmethod
    def request(req_id, tokens):
        return NS(
            request_id=req_id,
            prompt_token_ids=[1] * tokens,
            all_token_ids=[1] * tokens,
            num_computed_tokens=0,
            max_tokens=1,
            kv_transfer_params={
                "do_remote_decode": True,
                "kvaf_generation": f"generation-{req_id}",
                "remote_cached_tokens": 0,
            },
        )

    def admit(self, request):
        self.assertEqual(self.producer.get_num_new_matched_tokens(request, 0), (0, False))
        self.producer.update_state_after_alloc(request, self.blocks, 0)

    @staticmethod
    def output(request, computed, scheduled):
        return NS(
            scheduled_cached_reqs=NS(req_ids=[], new_block_ids=[], num_computed_tokens=[]),
            scheduled_new_reqs=[NS(req_id=request.request_id, num_computed_tokens=computed)],
            scheduled_spec_decode_tokens={},
            num_scheduled_tokens={request.request_id: scheduled},
        )

    @staticmethod
    def cached_output(request, computed, scheduled):
        return NS(
            scheduled_cached_reqs=NS(
                req_ids=[request.request_id], new_block_ids=[None], num_computed_tokens=[computed]
            ),
            scheduled_new_reqs=[],
            scheduled_spec_decode_tokens={},
            num_scheduled_tokens={request.request_id: scheduled},
        )

    def test_native_limit_uses_explicit_draft_limit_and_target_fallback(self):
        self.assertEqual(SCHEDULER.native_draft_limit(self.config), 1024)
        self.config.speculative_config.draft_model_config = NS(max_model_len=None)
        self.assertEqual(SCHEDULER.native_draft_limit(self.config), 1024)
        self.config.speculative_config.draft_model_config.max_model_len = 512
        self.assertEqual(SCHEDULER.native_draft_limit(self.config), 512)

    def test_lookup_has_no_admission_side_effect_after_failed_allocation(self):
        short, long = self.request("short", 64), self.request("long", 1022)
        self.assertEqual(self.producer.get_num_new_matched_tokens(short, 0), (0, False))
        # Native allocation failure does not call update_state_after_alloc.
        self.assertEqual(self.producer.get_num_new_matched_tokens(long, 0), (0, False))
        self.assertEqual(self.producer._pd_admitted_classes, {})
        self.admit(long)
        self.assertEqual(self.producer.get_num_new_matched_tokens(short, 0), (None, False))

    def test_exact_native_mtp3_boundary_and_same_class_batch(self):
        self.admit(self.request("boundary", 1021))
        self.admit(self.request("short", 128))
        self.assertEqual(self.producer.get_num_new_matched_tokens(self.request("long", 1022), 0), (None, False))
        self.assertEqual(self.config.speculative_config.num_speculative_tokens, 3)

    def test_near_limit_class_uses_full_prompt_during_chunked_prefill(self):
        long = self.request("long", 1022)
        self.admit(long)
        metadata = self.producer.build_connector_meta(self.output(long, 0, 128))
        self.assertFalse(metadata.requests["long"].chunk_finish)
        self.assertEqual(self.producer.get_num_new_matched_tokens(self.request("short", 64), 0), (None, False))
        self.admit(self.request("another-long", 1024))

    def test_local_prefix_hit_does_not_change_full_prompt_admission_class(self):
        self.admit(self.request("short", 128))
        self.assertEqual(self.producer.get_num_new_matched_tokens(self.request("long", 1022), 896), (None, False))

    def test_final_chunk_retains_class_until_native_request_finished(self):
        long = self.request("long", 1022)
        self.admit(long)
        metadata = self.producer.build_connector_meta(self.output(long, 0, 1022))
        self.assertTrue(metadata.requests["long"].chunk_finish)
        self.assertNotIn("long", self.producer._reqs_need_send_layerwise)
        short = self.request("short", 64)
        self.assertEqual(self.producer.get_num_new_matched_tokens(short, 0), (None, False))
        self.assertEqual(self.producer.request_finished_all_groups(long, ([1, 2, 3],) * 4), (True, None))
        self.assertEqual(self.producer.get_num_new_matched_tokens(short, 0), (0, False))

    def test_cancellation_releases_only_its_own_admission(self):
        a, b = self.request("a", 1022), self.request("b", 1024)
        self.admit(a)
        self.admit(b)
        self.producer.request_finished_all_groups(a, ([],) * 4)
        self.assertEqual(self.producer.get_num_new_matched_tokens(self.request("short", 64), 0), (None, False))
        self.producer.request_finished_all_groups(b, ([],) * 4)
        self.assertEqual(self.producer._pd_admitted_classes, {})

    def test_native_preemption_rechecks_same_class_without_releasing_it(self):
        long = self.request("long", 1022)
        self.admit(long)
        # Native preemption resets computed tokens and requeues the request;
        # request_finished is not called while it can resume model execution.
        long.num_computed_tokens = 0
        self.admit(long)
        self.assertEqual(self.producer._pd_admitted_classes, {"long": False})
        self.assertEqual(self.producer.get_num_new_matched_tokens(self.request("short", 64), 0), (None, False))

    def test_incompatible_allocation_is_rejected_before_native_mutation(self):
        self.admit(self.request("short", 64))
        with (
            patch.object(NATIVE.SFAPDRD2HProducerScheduler, "update_state_after_alloc") as native,
            self.assertRaisesRegex(ValueError, "mixed native MTP"),
        ):
            self.producer.update_state_after_alloc(self.request("long", 1022), self.blocks, 0)
        native.assert_not_called()
        self.assertEqual(self.producer._pd_admitted_classes, {"short": True})

    def test_direct_multitoken_p_request_is_rejected_before_admission(self):
        request = self.request("direct", 64)
        request.max_tokens = 2
        with self.assertRaisesRegex(ValueError, "max_tokens=1"):
            self.producer.get_num_new_matched_tokens(request, 0)
        self.assertEqual(self.producer._pd_admitted_classes, {})

    def test_direct_single_token_request_cannot_contaminate_pd_short_batch(self):
        request = self.request("direct", 1022)
        request.kv_transfer_params = None
        self.admit(request)
        self.assertEqual(self.producer.get_num_new_matched_tokens(self.request("short", 64), 0), (None, False))
        self.assertEqual(self.producer.request_finished_all_groups(request, ([],) * 4), (False, None))
        self.assertEqual(self.producer._pd_admitted_classes, {})

    def test_preempted_prefill_resumes_after_committed_prefix_with_new_blocks(self):
        request = self.request("r", 500)
        self.admit(request)
        first = self.producer.build_connector_meta(self.output(request, 0, 128))
        self.assertEqual(first.requests["r"].local_transed_tokens, 0)
        second = self.producer.build_connector_meta(self.cached_output(request, 128, 128))
        self.assertEqual(second.requests["r"].local_transed_tokens, 128)
        generation = second.generations["r"]

        request.num_computed_tokens = 0
        new_groups = ([9, 8, 7, 6], [6, 7, 8, 9], [5, 4, 3, 2], [2, 3, 4, 5])
        self.producer.update_state_after_alloc(request, NS(get_block_ids=lambda: new_groups), 0)
        for output in (self.output(request, 0, 64), self.cached_output(request, 64, 64)):
            suppressed = self.producer.build_connector_meta(output)
            self.assertEqual(suppressed.requests, {})
            self.assertEqual(suppressed.generations, {})

        resumed = self.producer.build_connector_meta(self.cached_output(request, 128, 256))
        self.assertEqual(resumed.requests["r"].local_transed_tokens // 128, 2)
        self.assertEqual(resumed.requests["r"].local_computed_tokens, 384)
        self.assertEqual(resumed.requests["r"].local_block_ids, list(new_groups))
        self.assertEqual(resumed.generations["r"], generation)
        final = self.producer.build_connector_meta(self.cached_output(request, 384, 116))
        self.assertTrue(final.requests["r"].chunk_finish)
        self.assertEqual(final.requests["r"].local_transed_tokens, 384)
        self.assertEqual(final.requests["r"].local_computed_tokens, 500)
        self.assertEqual(final.generations["r"], generation)
        self.producer.request_finished_all_groups(request, new_groups)
        self.assertEqual(self.producer._pd_sent_tokens, {})

    def test_partial_nonfinal_chunk_is_suppressed_but_final_tail_is_retained(self):
        request = self.request("r", 100)
        self.admit(request)
        partial = self.producer.build_connector_meta(self.output(request, 0, 64))
        self.assertEqual(partial.requests, {})
        final = self.producer.build_connector_meta(self.cached_output(request, 64, 36))
        self.assertTrue(final.requests["r"].chunk_finish)
        self.assertEqual(final.requests["r"].local_transed_tokens, 0)

    def test_reallocation_cannot_replace_generation_after_prefix_was_sent(self):
        request = self.request("r", 500)
        self.admit(request)
        self.producer.build_connector_meta(self.output(request, 0, 128))
        request.kv_transfer_params["kvaf_generation"] = "replacement-generation"
        with self.assertRaisesRegex(ValueError, "cannot change the generation"):
            self.producer.update_state_after_alloc(request, self.blocks, 0)


if __name__ == "__main__":
    unittest.main()
