# SPDX-License-Identifier: Apache-2.0
"""Focused host checks for archive admission, ACKs and scheduler lifecycle."""

import importlib
import logging
import sys
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch

package = ModuleType("kvaf_catalog_under_test")
package.__path__ = [str(Path(__file__).resolve().parents[2] / "vllm_ascend/distributed/kv_transfer/kvaf")]
sys.modules[package.__name__] = package


def load_connector():
    class Base:
        def __init__(self, config, role, cache):
            self._kv_cache_manager = None

        def bind_kv_cache_manager(self, manager):
            self._kv_cache_manager = manager

        def _get_connector_metadata(self):
            return self._connector_metadata

    base = ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")
    base.KVConnectorBase_V1 = Base
    for name in ("KVConnectorHandshakeMetadata", "KVConnectorMetadata", "KVConnectorWorkerMetadata", "SupportsHMA"):
        setattr(base, name, type(name, (), {}))
    logger = ModuleType("vllm.logger")
    logger.init_logger = logging.getLogger
    with patch.dict(sys.modules, {base.__name__: base, logger.__name__: logger}):
        return importlib.import_module(f"{package.__name__}.prefix_cache")


connector = catalog = load_connector()
GEOMETRY = (4, 128)


def capacity(slots=20):
    return catalog.Capacity(128, {GEOMETRY: slots}, {GEOMETRY: 1})


def request(name="a", tokens=512, computed=512, in_flight=0):
    return NS(
        request_id=name,
        num_prompt_tokens=tokens,
        num_tokens=tokens + 1,
        max_tokens=128,
        num_computed_tokens=computed,
        num_in_flight_tokens=in_flight,
        block_hashes=tuple(f"h{i}" for i in range(tokens // 128)),
    )


class TestArchiveCatalog(unittest.TestCase):
    def test_exact_packed_capacity_and_oversized_request(self):
        budget = catalog.Capacity(128, {GEOMETRY: 10}, {GEOMETRY: 1}, ((GEOMETRY, 128, 32), (GEOMETRY, 128, 16)))
        self.assertEqual(budget.cost(512), {GEOMETRY: 5})
        store = catalog.PrefixCatalog(budget, 2)
        with self.assertRaisesRegex(ValueError, "cannot hold request"):
            store.reserve("oversized", 2048)

    def test_save_and_eviction_ack_cross_rounds_keep_capacity_reserved(self):
        store = catalog.PrefixCatalog(capacity(6), 2)
        self.assertTrue(store.reserve("a", 768))
        job = store.save("a", ("a", "b", "c", "d"), 512, ((1, 2, 3, 4),), {})
        self.assertEqual(store.entries[job.entry_id].cost[GEOMETRY], 6)
        store.acknowledge_save(job.entry_id, {0})
        self.assertEqual(store.lookup(("a", "b", "c", "d"), 512), (None, 0))
        store.acknowledge_save(job.entry_id, {1})
        self.assertEqual(store.lookup(("a", "b", "c", "d"), 512)[1], 384)
        self.assertEqual(store.entries[job.entry_id].cost[GEOMETRY], 4)
        self.assertFalse(store.reserve("b", 768))
        self.assertEqual(store.entries[job.entry_id].state, "evicting")
        self.assertEqual(store.lookup(("a", "b", "c"), 512), (None, 0))
        store.acknowledge_eviction(job.entry_id, {1})
        self.assertFalse(store.reserve("b", 768))
        store.acknowledge_eviction(job.entry_id, {0})
        self.assertTrue(store.reserve("b", 768))

    def test_active_lease_prevents_eviction_and_generated_prefix_matches(self):
        store = catalog.PrefixCatalog(capacity(8), 1)
        store.reserve("a", 512)
        job = store.save("a", ("prompt0", "prompt1", "decode0", "decode1"), 512, ((1, 2, 3, 4),), {})
        store.acknowledge_save(job.entry_id, {0})
        store.lease(job.entry_id, "consumer")
        self.assertFalse(store.reserve("b", 768))
        self.assertFalse(store.pending_evictions)
        entry, matched = store.lookup(("prompt0", "prompt1", "decode0", "different"), 1024)
        self.assertEqual((entry.entry_id, matched), (job.entry_id, 256))
        store.release("consumer")
        self.assertFalse(store.reserve("b", 768))
        self.assertEqual(store.pending_evictions, {job.entry_id})

    def test_two_warmed_prefixes_fund_two_concurrent_borrowers(self):
        # Real accounting: Target is shared; each request still reserves its
        # own future auxiliary snapshot. Old full-copy admission evicts a warm
        # prefix here although both borrowers fit without copying Target.
        budget = catalog.Capacity(128, {GEOMETRY: 18}, {GEOMETRY: 1}, ((GEOMETRY, 128, 128),))
        store = catalog.PrefixCatalog(budget, 2)
        jobs = []
        for name in ("warm-a", "warm-b"):
            self.assertTrue(store.reserve(name, 512))
            job = store.save(name, tuple(f"{name}-{i}" for i in range(4)), 512, ((1, 2, 3, 4),), {})
            store.acknowledge_save(job.entry_id, {0, 1})
            jobs.append(job)
        for name, source in zip(("a", "b"), jobs):
            self.assertTrue(store.reserve(name, 640, source=source.entry_id, matched_tokens=384))
            # Retried HBM admission must not take another reservation/credit.
            self.assertTrue(store.reserve(name, 640, source=source.entry_id, matched_tokens=384))
        self.assertEqual(store._usage(), {GEOMETRY: 18})
        self.assertFalse(store.pending_evictions)
        self.assertFalse(store.reserve("unrelated", 640))
        self.assertFalse(store.pending_evictions, "live borrowers must protect their funding archives")
        # A finish removes the active credit: the new archive now owns a full
        # charge. Retiring its old source must not undercount shared pages.
        saved = store.save("a", ("a0", "a1", "a2", "a3", "a4"), 640, ((1, 2, 3, 4, 5),), {})
        store.acknowledge_save(saved.entry_id, {0, 1})
        store.release("a")
        store.evict(jobs[0].entry_id)
        store.acknowledge_eviction(jobs[0].entry_id, {0, 1})
        self.assertEqual(store._usage(), {GEOMETRY: 16})
        self.assertIn("b", store.entries[jobs[1].entry_id].leases)

    def test_completed_replay_retires_only_covered_unleased_archive(self):
        store = catalog.PrefixCatalog(capacity(40), 2)
        ids = []
        for name, hashes in (
            ("a", ("a0", "a1", "a2", "a3")),
            ("b", ("b0", "b1", "b2", "b3")),
            ("replay-a", ("a0", "a1", "a2", "a3", "a4")),
        ):
            store.reserve(name, len(hashes) * 128)
            job = store.save(name, hashes, len(hashes) * 128, (tuple(range(len(hashes))),), {})
            ids.append(job.entry_id)
            store.acknowledge_save(job.entry_id, {0, 1})
        store.lease(ids[0], "still-running")
        store.retire_redundant()
        self.assertFalse(store.pending_evictions)
        store.release("still-running")
        store.retire_redundant()
        self.assertEqual(store.pending_evictions, {ids[0]})
        self.assertEqual(store.lookup(("b0", "b1", "b2", "b3"), 511)[1], 384)
        self.assertEqual(store.lookup(("a0", "a1", "a2", "a3"), 511)[0].entry_id, ids[2])

    def test_mtp_replay_is_relative_to_actual_common_prefix(self):
        for replay, expected in ((1, 256), (0, 384)):
            with self.subTest(replay_blocks=replay):
                store = catalog.PrefixCatalog(capacity(), 1, replay_blocks=replay)
                store.reserve("source", 512)
                source = ("h0", "h1", "h2", "source-tail")
                query = ("h0", "h1", "h2", "different-tail")
                job = store.save("source", source, 512, ((1, 2, 3, 4),), {})
                self.assertTrue(store.has_pending_match(query, 511, expected - 1))
                self.assertFalse(store.has_pending_match(query, 511, expected))
                store.acknowledge_save(job.entry_id, {0})
                self.assertEqual(store.lookup(query, 511)[1], expected)
                self.assertEqual(store.lookup(source, 511)[1], 384)


class TestArchiveConnector(unittest.TestCase):
    def setUp(self):
        config = NS(
            parallel_config=NS(tensor_parallel_size=2),
            model_config=NS(max_model_len=1024),
            speculative_config=NS(num_speculative_tokens=3),
            cache_config=NS(enable_prefix_caching=True),
        )
        self.connector = connector.KVAFConnector(config, None, None)
        self.connector.set_xfer_handshake_metadata({rank: connector.ArchiveHandshake(capacity()) for rank in range(2)})
        self.blocks = [NS(block_hash=f"hash{i}", ref_cnt=1, block_id=i, is_null=i == 0) for i in range(20)]
        self.evicted = []

        def evict(ids):
            self.evicted.extend(ids)
            for block_id in ids:
                self.blocks[block_id].block_hash = None

        self.connector._kv_cache_manager = NS(
            block_pool=NS(blocks=self.blocks, hash_block_size=128),
            evict_blocks=evict,
            get_block_ids=lambda req_id: ((1, 2, 3, 4, 5, 6),),
        )

    def test_bind_disables_hbm_prefix_hits_and_load_ack_keeps_funded_source_lease(self):
        manager = self.connector._kv_cache_manager
        manager.enable_caching = True
        self.connector.bind_kv_cache_manager(manager)
        self.assertFalse(manager.enable_caching)
        store = self.connector.catalog
        store.reserve("source", 512)
        job = store.save("source", ("h0", "h1", "h2", "h3"), 512, ((1, 2, 3, 4),), {})
        store.acknowledge_save(job.entry_id, {0, 1})
        replay = request("replay")
        self.assertEqual(self.connector.get_num_new_matched_tokens(replay, 0), (384, True))
        self.connector.update_state_after_alloc(replay, NS(get_block_ids=lambda: ((7, 8, 9),)), 384)
        self.connector.update_connector_output(
            NS(kv_connector_worker_meta=None, finished_sending=(), finished_recving={"replay"})
        )
        self.assertEqual(store.entries[job.entry_id].leases, {"replay"})
        self.assertIn("replay", store.reservations)
        self.assertIn("replay", store.shared_targets)

    def test_lookup_precedes_reservation_and_protects_warmed_source(self):
        self.connector.catalog.capacity = capacity(9)
        store = self.connector.catalog
        store.reserve("warm", 512)
        job = store.save("warm", ("h0", "h1", "h2", "h3"), 512, ((1, 2, 3, 4),), {})
        store.acknowledge_save(job.entry_id, {0, 1})
        # Lifetime = 512 prompt + 128 output + MTP + ABC reserve = 8 slots.
        # Old admission needs 4 + 8 > 9, evicts the very prefix it will look up.
        self.assertEqual(self.connector.get_num_new_matched_tokens(request("replay"), 0), (384, True))
        self.assertEqual(store._usage(), {GEOMETRY: 9})
        self.assertFalse(store.pending_evictions)
        self.assertEqual(store.entries[job.entry_id].leases, {"replay"})

    def test_finish_corrects_inflight_retains_extra_pages_until_worker_ack(self):
        req = request(computed=640, in_flight=128)
        self.assertEqual(self.connector.get_num_new_matched_tokens(req, 0), (0, False))
        delayed, _ = self.connector.request_finished_all_groups(req, ((1, 2, 3, 4),))
        self.assertTrue(delayed)
        metadata = self.connector.build_connector_meta(None)
        job = metadata.saves[0]
        self.assertEqual(job.num_tokens, 512)
        self.assertEqual(job.drop_block_ids, (1, 2, 3, 4, 5, 6))
        self.assertEqual(job.block_ids, ((1, 2, 3, 4, 5, 6),))
        self.assertEqual(len(self.evicted), 6)
        # Partial TP completion never advertises a hit.
        self.connector.update_connector_output(
            NS(
                kv_connector_worker_meta=connector.ArchiveWorkerMetadata(saved={job.entry_id: {0}}),
                finished_sending=(),
                finished_recving=(),
            )
        )
        self.assertEqual(self.connector.catalog.lookup(req.block_hashes, 512), (None, 0))

    def test_worker_tasks_idempotent_and_finish_without_forward(self):
        calls = []
        zeroed = [False]

        def load(job):
            self.assertTrue(zeroed[0], "restore must follow runner new-block zeroing")
            calls.append(("load", job.entry_id))

        self.connector.bind_archive(
            NS(
                capacity=capacity(),
                save=lambda job: calls.append(("save", job.entry_id)),
                load=load,
                evict=lambda job: calls.append(("evict", job.entry_id)),
                drop=lambda job: calls.append(("drop", job.request_id)),
            ),
            rank=1,
        )
        metadata = connector.ArchiveMetadata(
            saves=(catalog.SaveJob(0, "a", ((1,),), 128),),
            loads=(catalog.LoadJob(2, "b", ((2,),), 128),),
            drops=(catalog.DropJob("c", (3,)),),
        )
        self.connector.handle_preemptions(metadata)
        self.assertEqual(len(calls), 2)
        zeroed[0] = True
        self.connector._connector_metadata = metadata
        self.connector.start_load_kv(None)
        self.connector.start_load_kv(None)
        self.assertEqual(len(calls), 3)
        self.assertEqual(self.connector.get_finished({"a", "c", "untracked"}), ({"a", "c"}, {"b"}))
        self.assertEqual(self.connector.build_connector_worker_meta().saved, {0: {1}})
        self.assertIsNone(self.connector.build_connector_worker_meta())
        self.assertFalse(self.connector._reported_finished)

    def test_no_archive_when_prefix_disabled_and_no_reservation_abort(self):
        self.connector.prefix_caching = False
        req = request()
        self.connector.get_num_new_matched_tokens(req, 0)
        self.assertTrue(self.connector.request_finished_all_groups(req, ((1, 2),))[0])
        metadata = self.connector.build_connector_meta(None)
        self.assertFalse(metadata.saves)
        self.assertEqual(metadata.drops[0].request_id, "a")
        self.assertFalse(self.connector.request_finished_all_groups(request("never-admitted"), ((),))[0])
        self.assertFalse(self.connector.reset_cache())

    def test_native_prefix_lease_and_stale_block_generation_eviction(self):
        store = self.connector.catalog
        store.reserve("a", 512)
        job = store.save("a", ("h0", "h1", "h2", "h3"), 512, ((1, 2, 3, 4),), {1: "hash1", 2: "hash2"})
        store.acknowledge_save(job.entry_id, {0, 1})
        consumer = request("consumer")
        store.reserve("consumer", 512)
        self.connector.update_state_after_alloc(consumer, NS(get_block_ids=lambda: ((1, 2),)), 0)
        self.assertEqual(store.entries[job.entry_id].leases, {"consumer"})
        self.assertFalse(self.connector.reset_cache())
        store.release("consumer")
        self.blocks[1].ref_cnt = 0
        self.blocks[2].ref_cnt = 0
        self.blocks[2].block_hash = "reused-for-another-prefix"
        self.assertTrue(self.connector.reset_cache())
        metadata = self.connector.build_connector_meta(None)
        self.assertEqual(metadata.evictions[0].drop_block_ids, (1,))
        self.assertEqual(self.blocks[2].block_hash, "reused-for-another-prefix")

    def test_cancel_receive_has_one_cleanup_ack_and_keeps_lease_until_then(self):
        store = self.connector.catalog
        store.reserve("source", 512)
        job = store.save("source", ("h0", "h1", "h2", "h3"), 512, ((1, 2, 3, 4),), {})
        store.acknowledge_save(job.entry_id, {0, 1})
        req = request("receiving", computed=384)
        self.assertEqual(self.connector.get_num_new_matched_tokens(req, 0), (384, True))
        self.connector.update_state_after_alloc(req, NS(get_block_ids=lambda: ((7, 8, 9),)), 384)
        self.assertTrue(self.connector.request_finished_all_groups(req, ((7, 8, 9),))[0])
        self.assertFalse(self.connector._saves)
        self.assertTrue(self.connector.has_pending_push_work())
        output = NS(kv_connector_worker_meta=None, finished_sending=set(), finished_recving={"receiving"})
        self.connector.update_connector_output(output)
        self.assertFalse(output.finished_recving, "receive ACK must not free an aborted request before cleanup")
        metadata = self.connector.build_connector_meta(None)
        self.assertEqual(metadata.drops[0].request_id, "receiving")
        # No live requests or queued jobs remain, but the cleanup still needs
        # worker rounds until its all-TP ACK arrives.
        self.assertTrue(self.connector.has_pending_push_work())
        self.assertIn("receiving", store.reservations)
        self.assertIn("receiving", store.entries[job.entry_id].leases)
        self.connector.update_connector_output(
            NS(
                kv_connector_worker_meta=None,
                finished_sending={"receiving"},
                finished_recving=(),
            )
        )
        self.assertNotIn("receiving", store.reservations)
        self.assertFalse(store.entries[job.entry_id].leases)
        self.assertFalse(self.connector.has_pending_push_work())

    def test_immediate_prefix_replay_waits_for_all_workers_save(self):
        self.connector.catalog.capacity = capacity(32)
        req = request()
        self.connector.get_num_new_matched_tokens(req, 0)
        self.connector.request_finished_all_groups(req, ((1, 2, 3, 4),))
        metadata = self.connector.build_connector_meta(None)
        entry_id = metadata.saves[0].entry_id
        replay = request("replay")
        self.assertEqual(self.connector.get_num_new_matched_tokens(replay, 0), (None, False))
        self.connector.catalog.acknowledge_save(entry_id, {0})
        self.assertEqual(self.connector.get_num_new_matched_tokens(replay, 0), (None, False))
        unrelated = request("unrelated")
        unrelated.block_hashes = ("other0", "other1", "other2", "other3")
        self.assertEqual(self.connector.get_num_new_matched_tokens(unrelated, 0), (0, False))
        self.connector.catalog.acknowledge_save(entry_id, {1})
        self.assertEqual(self.connector.get_num_new_matched_tokens(replay, 0), (384, True))

    def test_new_archive_leases_running_owner_grown_block_table(self):
        owner = request("growing")
        self.connector.catalog.reserve(owner.request_id, 512)
        self.connector.update_state_after_alloc(owner, NS(get_block_ids=lambda: ((1,),)), 0)
        # Page 3 became shared only after the first owner's admission callback.
        self.blocks[3].ref_cnt = 2
        self.connector._kv_cache_manager.get_block_ids = lambda req_id: ((1, 3),) if req_id == "growing" else ((3, 4),)
        req = request()
        self.connector.get_num_new_matched_tokens(req, 0)
        self.connector.request_finished_all_groups(req, ((3, 4),))
        entry = next(iter(self.connector.catalog.entries.values()))
        self.assertEqual(entry.leases, {"growing"})

    def test_cancel_worker_receive_preserves_finish_notice_for_later_drop(self):
        dropped = []
        self.connector.bind_archive(NS(capacity=capacity(), load=lambda job: None, drop=dropped.append))
        metadata = connector.ArchiveMetadata(loads=(catalog.LoadJob(0, "cancelled", ((1,),), 128),))
        self.connector._connector_metadata = metadata
        self.connector.start_load_kv(None)
        self.assertEqual(self.connector.get_finished({"cancelled"}), (set(), {"cancelled"}))
        metadata = connector.ArchiveMetadata(drops=(catalog.DropJob("cancelled", (1,)),))
        self.connector.handle_preemptions(metadata)
        self.assertEqual(self.connector.get_finished(set()), ({"cancelled"}, set()))
        self.assertEqual(len(dropped), 1)

    def test_preemption_cleanup_only_free_ids_and_cold_lookup_respected(self):
        req = request()
        req.skip_reading_prefix_cache = True
        self.connector.get_num_new_matched_tokens(req, 0)
        reservation = self.connector.catalog.reservations["a"].copy()
        self.blocks[2].ref_cnt = self.blocks[5].ref_cnt = 0
        metadata = self.connector.build_connector_meta(NS(preempted_req_ids={"a"}))
        self.assertEqual(metadata.cleanups[0].drop_block_ids, (2, 5))
        self.assertFalse(metadata.drops)
        self.assertEqual(self.connector.catalog.reservations["a"], reservation)
        self.assertEqual(self.blocks[3].block_hash, "hash3")


if __name__ == "__main__":
    unittest.main()
