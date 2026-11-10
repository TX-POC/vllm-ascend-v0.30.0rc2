# SPDX-License-Identifier: Apache-2.0
"""Native Scheduler reuse, external Runner allocation and feature-off contracts."""

import importlib
import importlib.util
import logging
import sys
import unittest
from enum import Enum
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import test_kvaf_adapter as helpers
import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = helpers.PACKAGE.__name__
CONFIG = importlib.import_module(f"{PACKAGE}.pd.config")


def config(*, producer=False, tp=1, dp=16, rank=0):
    return NS(
        kv_transfer_config=NS(
            kv_connector="PDKVoffloadConnector",
            kv_role="kv_producer" if producer else "kv_consumer",
            is_kv_producer=producer,
            is_kv_consumer=not producer,
            engine_id="engine",
            kv_port=31000,
            kv_connector_extra_config={"decode_endpoints": ["10.0.0.3:31000", "10.0.0.4:31008"]} if producer else {},
        ),
        parallel_config=NS(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            data_parallel_rank=rank,
            worker_cls=CONFIG.WORKER_CLASS,
        ),
        speculative_config=NS(method="deepseek_mtp", num_speculative_tokens=3, draft_model_config=None),
        model_config=NS(max_model_len=4096),
        cache_config=NS(enable_prefix_caching=False),
        scheduler_config=NS(max_num_seqs=2),
        additional_config={"kvaf_config": {"enabled": not producer, "build_dir": "/kvaf/build-pd"}},
    )


def module(name, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    return result


def load_file(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


class Role(Enum):
    SCHEDULER = 0
    WORKER = 1


class Base:
    def __init__(self, **kwargs):
        self._connector_metadata = None

    def bind_connector_metadata(self, metadata):
        self._connector_metadata = metadata

    def _get_connector_metadata(self):
        return self._connector_metadata

    def has_connector_metadata(self):
        return self._connector_metadata is not None


def load_scheduler():
    base = module(
        "vllm.distributed.kv_transfer.kv_connector.v1.base",
        KVConnectorMetadata=object,
        KVConnectorBase_V1=Base,
        KVConnectorRole=Role,
        SupportsHMA=type("HMA", (), {}),
    )
    modules = {base.__name__: base}
    definitions = {
        "vllm.config": dict(VllmConfig=object),
        "vllm.logger": dict(logger=logging.getLogger("vllm")),
        "vllm.utils.math_utils": dict(round_down=lambda value, size: value // size * size),
        "vllm.utils.network_utils": dict(get_ip=lambda: "127.0.0.1"),
        "vllm.v1.kv_cache_interface": dict(KVCacheConfig=object),
        "vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context": dict(
            DSparkContextDescriptor=NS, resident_mla_context_group_ids=lambda _: ()
        ),
        "vllm_ascend.spec_decode.dspark_utils": dict(get_dspark_aux_layer_ids=lambda _: ()),
    }
    modules.update({name: module(name, **values) for name, values in definitions.items()})
    modules["httpx"] = module("httpx")
    with patch.dict(sys.modules, modules):
        protocol = load_file(
            "pd_native_protocol_for_ut", "vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/protocol.py"
        )
        slots = load_file(
            "pd_native_slots_for_ut", "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/copy_sfa_topk_slots.py"
        )
        modules["vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol"] = protocol
        modules["vllm_ascend.distributed.kv_transfer.sparse_kv_offload.copy_sfa_topk_slots"] = slots
        with patch.dict(sys.modules, modules):
            native = load_file(
                "pd_native_scheduler_for_ut", "vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/scheduler.py"
            )
            modules["vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.scheduler"] = native
            with patch.dict(sys.modules, modules):
                scheduler = importlib.import_module(f"{PACKAGE}.pd.scheduler")
                connector = importlib.import_module(f"{PACKAGE}.pd.connector")
    return scheduler, native, connector


SCHEDULER, NATIVE, CONNECTOR = load_scheduler()


class TestPDConfig(unittest.TestCase):
    def test_cpu_and_fullmesh_eids_resolve_by_physical_device_independently(self):
        value = config()
        cpu, raw = "1" * 32, "2" * 32
        value.kv_transfer_config.kv_connector_extra_config = {
            "cpu_eids": {"4": cpu},
            "npu_eids": {"4": raw, "7": "3" * 32},
        }
        parsed = CONFIG.PDConfig.from_vllm(value)
        self.assertEqual(parsed.eids(4), (cpu, raw))
        self.assertEqual(parsed.eids(0), ("", ""))
        self.assertEqual(parsed.eids(7), ("", "3" * 32))

    def test_invalid_per_card_eid_is_rejected_before_runtime_creation(self):
        value = config()
        value.kv_transfer_config.kv_connector_extra_config = {"npu_eids": {"04": "2" * 32}}
        with self.assertRaisesRegex(ValueError, "physical NPU ID"):
            CONFIG.PDConfig.from_vllm(value)

    def test_original_production_and_temporary_topologies_use_same_entry(self):
        for producer, tp, dp in ((True, 8, 2), (False, 1, 16), (True, 4, 1), (False, 1, 4)):
            CONFIG.PDConfig.from_vllm(config(producer=producer, tp=tp, dp=dp))

    def test_explicit_connector_worker_and_backend_are_required(self):
        value = config()
        value.kv_transfer_config.kv_connector = "SfaRemoteD2HConnector"
        with self.assertRaisesRegex(ValueError, "explicitly"):
            CONFIG.PDConfig.from_vllm(value)
        value = config()
        value.parallel_config.worker_cls = "auto"
        with self.assertRaisesRegex(ValueError, "worker-cls"):
            CONFIG.PDConfig.from_vllm(value)
        value = config(producer=True)
        value.additional_config = {}
        with self.assertRaisesRegex(ValueError, "build_dir"):
            CONFIG.PDConfig.from_vllm(value)

    def test_native_mtp_and_d_prefix_rules_are_preserved(self):
        value = config()
        value.cache_config.enable_prefix_caching = True
        with self.assertRaisesRegex(ValueError, "prefix"):
            CONFIG.PDConfig.from_vllm(value)
        value = config()
        value.speculative_config.num_speculative_tokens = 0
        with self.assertRaisesRegex(ValueError, "MTP"):
            CONFIG.PDConfig.from_vllm(value)
        value = config(producer=True)
        value.cache_config.enable_prefix_caching = True
        with self.assertRaisesRegex(ValueError, "P/D prefix"):
            CONFIG.PDConfig.from_vllm(value)

    def test_unknown_flags_and_missing_full_mesh_fail_before_sdk_loading(self):
        value = config(producer=True)
        value.kv_transfer_config.kv_connector_extra_config = {}
        with self.assertRaisesRegex(ValueError, "all D"):
            CONFIG.PDConfig.from_vllm(value)
        value.kv_transfer_config.kv_connector_extra_config = {"typo": True}
        with self.assertRaisesRegex(ValueError, "unknown"):
            CONFIG.PDConfig.from_vllm(value)


class TestConnectorHooks(unittest.TestCase):
    def test_sender_owner_is_retained_when_initialization_raises(self):
        connector = CONNECTOR.PDKVoffloadConnector(config(producer=True, tp=8), Role.WORKER)
        connector.register_kv_caches({})
        sender_module = importlib.import_module(f"{PACKAGE}.pd.sender")
        worker = Mock()
        failure = RuntimeError("source initialization failed")

        def initialize():
            self.assertIs(connector.worker, worker)
            raise failure

        worker.initialize.side_effect = initialize
        modules = {
            "vllm.distributed": module("vllm.distributed", get_tensor_model_parallel_rank=lambda: 0),
            "vllm.platforms": module(
                "vllm.platforms", current_platform=NS(visible_device_id_to_physical_device_id=lambda device: device)
            ),
            "vllm.utils.network_utils": module("vllm.utils.network_utils", get_ip=lambda: "127.0.0.1"),
            "vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol": module(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol",
                get_external_request_id=lambda req: req,
            ),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(torch, "npu", NS(current_device=lambda: 0), create=True),
            patch.object(sender_module, "Sender", return_value=worker),
            self.assertRaises(RuntimeError) as raised,
        ):
            connector.bind_pd_backend(sdk=Mock(), runtime=None, memories={}, writer=None, pool=None, draft_limit=None)
        self.assertIs(raised.exception, failure)
        self.assertIs(connector.worker, worker)
        connector.shutdown()
        worker.close.assert_called_once()

    def test_registration_is_two_phase_and_profiling_does_not_require_sdk(self):
        connector = CONNECTOR.PDKVoffloadConnector(config(), Role.WORKER)
        self.assertIsNone(connector.worker)
        connector.on_kv_cache_written("m")
        connector.register_kv_caches({"m": object()})
        with self.assertRaisesRegex(RuntimeError, "repeat"):
            connector.register_kv_caches({})

    def test_consumer_seeds_native_metadata_and_returns_request_level_completion(self):
        connector = CONNECTOR.PDKVoffloadConnector(config(), Role.WORKER)
        connector.worker = Mock()
        connector.worker.get_finished.return_value = {"r"}
        connector.worker.get_errors.return_value = {5}
        connector._external_id = lambda value: value
        seed = SCHEDULER.ReceiveRequest("a1", 129, ((3, 4),))
        connector.bind_connector_metadata(SCHEDULER.ReceiveMetadata({"r": seed}))
        connector.start_load_kv(None)
        connector.worker.seed.assert_called_once_with("r", "r", seed)
        self.assertEqual(connector.get_finished({"old"}), (set(), {"r"}))
        connector.worker.release.assert_called_once_with({"old"})
        self.assertEqual(connector.get_block_ids_with_load_errors(), {5})

    def test_producer_binds_before_forward_and_reuses_native_write_and_finalize_hooks(self):
        connector = CONNECTOR.PDKVoffloadConnector(config(producer=True, tp=8, dp=2), Role.WORKER)
        connector.worker = Mock()
        connector._layer_names = ("m",)
        metadata = NS(requests={}, generations={})
        connector.bind_connector_metadata(metadata)
        connector.worker.bind.assert_called_once_with(metadata)
        connector.on_kv_cache_written("m")
        connector.worker.written.assert_called_once_with("m")
        connector.wait_for_layer_reuse(0)
        connector.worker.wait_layer.assert_called_once_with("m")
        connector.wait_for_save()
        connector.worker.finalize.assert_called_once()


class TestNativeSchedulers(unittest.TestCase):
    def setUp(self):
        self.cache = NS(
            kv_cache_groups=[
                NS(kv_cache_spec=NS(block_size=128), layer_names=[name])
                for name in ("main", "indexer", "mtp", "mtp_indexer")
            ]
        )
        self.consumer = SCHEDULER.ConsumerScheduler(config(), True, self.cache)
        self.addCleanup(self.consumer.shutdown)

    @staticmethod
    def request(req_id="r"):
        return NS(
            request_id=req_id,
            prompt_token_ids=list(range(300)),
            all_token_ids=list(range(300)),
            num_computed_tokens=0,
            max_tokens=1,
            kv_transfer_params={"do_remote_prefill": True, "metaserver": "http://proxy"},
        )

    def allocate(self, request):
        groups = ([3, 2, 5], [7, 4, 1], [1, 2, 3], [5, 4, 3])
        with patch.object(NATIVE.SFAPDRD2HScheduler, "_submit_metaserver_request") as submit:
            self.consumer.update_state_after_alloc(request, NS(get_block_ids=lambda: groups), 300)
        return groups, submit.call_args.kwargs["message"]

    def test_native_proxy_rendezvous_carries_generation_and_all_cache_groups(self):
        request = self.request()
        groups, message = self.allocate(request)
        self.assertEqual(message["remote_tp_size"], 1)
        self.assertEqual(message["remote_cached_tokens"], 0)
        metadata = self.consumer.build_connector_meta(None)
        self.assertEqual(metadata.requests["r"].blocks, tuple(tuple(group) for group in groups))
        self.assertEqual(metadata.requests["r"].generation, message["kvaf_generation"])
        self.assertFalse(self.consumer._dspark_context_groups)
        self.assertIsNone(self.consumer._copy_sfa_slot_allocator)

    def test_cancellation_defers_native_blocks_until_receive_report(self):
        request = self.request()
        groups, _ = self.allocate(request)
        self.assertEqual(self.consumer.request_finished_all_groups(request, groups), (True, None))
        self.assertIn("r", self.consumer._pd_requests)
        self.consumer.update_connector_output(NS(finished_recving={"r"}))
        self.assertNotIn("r", self.consumer._pd_requests)

    def test_pending_receivers_share_native_max_sequence_admission(self):
        for req_id in ("a", "b"):
            self.allocate(self.request(req_id))
        self.assertEqual(self.consumer.get_num_new_matched_tokens(self.request("c"), 0), (None, False))

    def test_global_dp_rank_controls_worker_port_on_both_hosts(self):
        for rank in (0, 7, 8, 15):
            consumer = SCHEDULER.ConsumerScheduler(config(rank=rank), True, self.cache)
            self.addCleanup(consumer.shutdown)
            self.assertEqual(consumer.side_channel_port, 31000 + rank)

    def test_producer_reuses_native_chunks_and_holds_blocks_until_sdk_ack(self):
        request = self.request()
        _, request.kv_transfer_params = self.allocate(request)
        producer = SCHEDULER.ProducerScheduler(config(producer=True, tp=8, dp=2), self.cache, "P")
        producer.update_state_after_alloc(request, NS(get_block_ids=lambda: ([1, 2, 3],) * 4), 0)
        output = NS(
            scheduled_cached_reqs=NS(req_ids=[], new_block_ids=[], num_computed_tokens=[]),
            scheduled_new_reqs=[NS(req_id="r", num_computed_tokens=0)],
            scheduled_spec_decode_tokens={},
            num_scheduled_tokens={"r": 300},
        )
        metadata = producer.build_connector_meta(output)
        self.assertTrue(metadata.requests["r"].chunk_finish)
        self.assertEqual(metadata.generations["r"], request.kv_transfer_params["kvaf_generation"])
        self.assertEqual(producer.request_finished_all_groups(request, ([1, 2, 3],) * 4), (True, None))

    def test_producer_cancellation_before_first_forward_is_not_lost(self):
        request = self.request()
        _, request.kv_transfer_params = self.allocate(request)
        producer = SCHEDULER.ProducerScheduler(config(producer=True, tp=8, dp=2), self.cache, "P")
        producer.update_state_after_alloc(request, NS(get_block_ids=lambda: ([1, 2, 3],) * 4), 0)
        self.assertTrue(producer.request_finished_all_groups(request, ([],) * 4)[0])
        output = NS(
            scheduled_cached_reqs=NS(req_ids=[], new_block_ids=[], num_computed_tokens=[]),
            scheduled_new_reqs=[],
            scheduled_spec_decode_tokens={},
            num_scheduled_tokens={},
        )
        self.assertIn("r", producer.build_connector_meta(output).cancellations)


class NativeRunnerBoundary:
    def _get_layer_kv_cache_specs(self, config):
        return self.specs

    def _allocate_int8_cache_tensor(self, size, alignment):
        return torch.zeros(size, dtype=torch.int8)

    def _reshape_kv_cache_tensors(self, cache, raw):
        result = {}
        for name, views in raw.items():
            if hasattr(self.specs[name], "sfa_dcp_replicated_indexer_size"):
                result[name] = (views[0].view(8, 128, 1, 128), views[1].view(torch.float32).view(8, 128, 1, 1))
            else:
                result[name] = (views[0].view(-1, 128, 1, 656),)
        return result


class TestExternalRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = {
            "vllm_ascend.worker.model_runner_v1": module("runner", NPUModelRunner=NativeRunnerBoundary),
            "vllm.distributed.kv_transfer": module(
                "transfer", get_kv_transfer_group=lambda: None, has_kv_transfer_group=lambda: False
            ),
            "vllm.platforms": module("platforms", current_platform=NS()),
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout": module(
                "reuse", get_layerwise_reuse_config=lambda _: None
            ),
        }
        with patch.dict(sys.modules, modules):
            cls.runner_module = importlib.import_module(f"{PACKAGE}.pd.runner")
            cls.runner_type = cls.runner_module.PDKVModelRunner

    def runner(self, *, decode):
        runner = self.runner_type.__new__(self.runner_type)
        runner.specs = {
            name: NS(block_size=128, page_size_bytes=128 * (132 if "idx" in name else 656))
            for name in ("m", "idx", "m2", "draft", "draft_idx")
        }
        for name in ("idx", "draft_idx"):
            runner.specs[name].sfa_dcp_replicated_indexer_size = 1
        runner.compilation_config = NS(
            static_forward_context={
                "m": NS(impl=NS(runtime_has_indexer=True, indexer=NS(k_cache=NS(prefix="idx"), num_cache_tensors=2))),
                "m2": NS(impl=NS(runtime_has_indexer=False)),
                "draft": NS(
                    impl=NS(runtime_has_indexer=True, indexer=NS(k_cache=NS(prefix="draft_idx"), num_cache_tensors=2))
                ),
            }
        )
        runner.kvaf_target_layer_names = {"m", "m2"} if decode else set()
        runner.drafter = NS(attn_layer_names=("draft", "draft_idx"))
        runner.kvaf_cache_state = (
            NS(
                indices={"m": 0, "m2": 1},
                policy=NS(hot=[torch.zeros(2, 128, 1, 656, dtype=torch.int8) for _ in range(2)]),
                storage={"pd_receive_reservation": None},
            )
            if decode
            else None
        )
        runner.device, runner.max_num_reqs, runner.vllm_config = "cpu", 2, config(producer=not decode)
        runner.pd_memories, runner._pd_indexer_names = {}, {}
        cache = NS(num_blocks=8, kv_cache_groups=[NS(layer_names=[name]) for name in runner.specs])
        return runner, cache

    def test_p_uses_one_real_layer_allocation_without_hot_or_tensor_concatenation(self):
        runner, cache = self.runner(decode=False)
        raw = runner._allocate_kv_cache_tensors(cache)
        runner._reshape_kv_cache_tensors(cache, raw)
        memory = runner.pd_memories["m"]
        self.assertEqual(
            {tensor.untyped_storage().data_ptr() for tensor in memory.planes.values()},
            {memory.tensor.untyped_storage().data_ptr()},
        )
        self.assertEqual(memory.planes["main"].shape[0], 8)
        self.assertTrue(memory.offload_main)
        self.assertFalse(runner.pd_memories["draft"].offload_main)

    def test_d_compact_ordinary_tail_and_native_hot_have_distinct_storage(self):
        runner, cache = self.runner(decode=True)
        raw = runner._allocate_kv_cache_tensors(cache)
        runner._reshape_kv_cache_tensors(cache, raw)
        memory = runner.pd_memories["m"]
        self.assertEqual(memory.planes["main"].shape[0], 2)
        self.assertNotEqual(memory.planes["main"].data_ptr(), raw["m"][0].data_ptr())
        self.assertEqual(memory.planes["indexer"].shape[0], 8)
        self.assertEqual(memory.planes["scale"].dtype, torch.float32)
        self.assertEqual(runner.pd_memories["draft"].planes["main"].shape[0], 8)
        self.assertEqual(set(runner.pd_memories["m2"].planes), {"main"})

    def test_unsupported_layout_fails_before_registering_any_memory(self):
        runner, cache = self.runner(decode=False)
        runner.specs["m"].page_size_bytes = 128 * 576
        with self.assertRaisesRegex(ValueError, "656-byte"):
            runner._allocate_kv_cache_tensors(cache)
        self.assertFalse(runner.pd_memories)

    def test_async_compute_records_fence_without_synchronizing_each_forward(self):
        runtime_module = importlib.import_module(f"{PACKAGE}.pd.runtime")
        runtime = runtime_module.SerializedRuntime(NS())
        trace = []
        runtime.compute_done = NS(record=lambda: trace.append("record"), synchronize=lambda: trace.append("wait"))
        runner = self.runner_type.__new__(self.runner_type)
        runner.kvaf_adapter = NS(runtimes={(656, 128): runtime})
        self.assertEqual(runner._run_pd_compute(lambda: trace.append("enqueue_graph") or 7), 7)
        self.assertEqual(trace, ["enqueue_graph", "record"])
        with runtime.lock:
            runtime.wait_compute()
        self.assertEqual(trace, ["enqueue_graph", "record", "wait"])

    def test_compute_exception_still_fences_enqueued_device_work(self):
        runtime_module = importlib.import_module(f"{PACKAGE}.pd.runtime")
        runtime = runtime_module.SerializedRuntime(NS())
        runtime.compute_done = Mock()
        runner = self.runner_type.__new__(self.runner_type)
        runner.kvaf_adapter = NS(runtimes={(656, 128): runtime})
        with self.assertRaisesRegex(RuntimeError, "forward"):
            runner._run_pd_compute(Mock(side_effect=RuntimeError("forward failed after enqueue")))
        runtime.compute_done.record.assert_called_once()
        runtime.compute_done.synchronize.assert_not_called()

    def test_producer_compute_has_no_runtime_fence_or_sdk_work(self):
        runner = self.runner_type.__new__(self.runner_type)
        runner.kvaf_adapter = None
        self.assertEqual(runner._run_pd_compute(lambda: 9), 9)

    def test_native_skip_marker_precedes_dummy_forward(self):
        runner = self.runner_type.__new__(self.runner_type)
        runner.is_kv_producer = True
        runner.effective_drafter_max_model_len = 4096
        trace = []
        connector = NS(skip_pd_drafting=lambda limit: trace.append(("skip", limit)))
        runner_module = self.runner_module
        with (
            patch.object(runner_module, "get_kv_transfer_group", return_value=connector),
            patch.object(
                NativeRunnerBoundary,
                "_skip_drafting",
                side_effect=lambda *args, **kwargs: trace.append("dummy") or 7,
                create=True,
            ),
        ):
            self.assertEqual(runner._skip_drafting("scheduler"), 7)
        self.assertEqual(trace, [("skip", 4096), "dummy"])

    def test_successful_native_graph_proposal_proves_mtp_write(self):
        runner = self.runner_type.__new__(self.runner_type)
        runner.is_kv_producer = True
        connector = Mock()
        runner_module = self.runner_module
        result = object()
        with (
            patch.object(runner_module, "get_kv_transfer_group", return_value=connector),
            patch.object(NativeRunnerBoundary, "propose_draft_token_ids", return_value=result, create=True),
        ):
            self.assertIs(runner.propose_draft_token_ids("sample"), result)
        connector.complete_pd_drafting.assert_called_once_with()

    def test_failed_native_proposal_cannot_prove_mtp_write(self):
        runner = self.runner_type.__new__(self.runner_type)
        runner.is_kv_producer = True
        connector = Mock()
        runner_module = self.runner_module
        with (
            patch.object(runner_module, "get_kv_transfer_group", return_value=connector),
            patch.object(
                NativeRunnerBoundary,
                "propose_draft_token_ids",
                side_effect=RuntimeError("native proposal failed"),
                create=True,
            ),
            self.assertRaisesRegex(RuntimeError, "native proposal failed"),
        ):
            runner.propose_draft_token_ids("sample")
        connector.complete_pd_drafting.assert_not_called()


if __name__ == "__main__":
    unittest.main()
