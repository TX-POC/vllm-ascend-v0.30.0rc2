# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of KVAF configuration and native-session ownership."""

import importlib
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

PACKAGE = ModuleType("kvaf_adapter_under_test")
PACKAGE.__path__ = [str(Path(__file__).resolve().parents[3] / "vllm_ascend/distributed/kv_transfer/kvaf")]
sys.modules[PACKAGE.__name__] = PACKAGE
adapter_module = importlib.import_module(f"{PACKAGE.__name__}.adapter")
layout_module = importlib.import_module(f"{PACKAGE.__name__}.layout")
KVAFConfig = adapter_module.KVAFConfig


def load_prefix_module():
    from .test_prefix_cache import load_connector

    load_connector()
    # Reuse the already-loaded protocol types for this test package namespace.
    from .test_prefix_cache import connector

    base = ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")
    for name in (
        "KVConnectorBase_V1",
        "KVConnectorHandshakeMetadata",
        "KVConnectorMetadata",
        "KVConnectorWorkerMetadata",
        "SupportsHMA",
    ):
        setattr(base, name, getattr(connector, name))
    logger = ModuleType("vllm.logger")
    import logging

    logger.init_logger = logging.getLogger
    with patch.dict(sys.modules, {base.__name__: base, logger.__name__: logger}):
        return importlib.import_module(f"{PACKAGE.__name__}.prefix_cache")


def layout(token_bytes, name):
    plane = layout_module.KVCachePlane(
        shape=(8, 128, 1, token_bytes),
        dtype="torch.int8",
        stride=(128 * token_bytes, token_bytes, token_bytes, 1),
        device="npu:0",
        data_ptr=4096,
        page_bytes=128 * token_bytes,
        page_stride_bytes=128 * token_bytes,
        token_bytes=token_bytes,
        direct_token_read=True,
    )
    return layout_module.KVCacheLayout(
        layer_names=(name,),
        group_ids=(0,),
        roles=("target",),
        cache_kind="main",
        block_size=128,
        page_bytes=128 * token_bytes,
        packed_c8=True,
        planes=(plane,),
    )


class TestKVAFAdapter(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.build_dir = Path(directory.name)
        (self.build_dir / "bin").mkdir()
        for kind in ("get", "get_urma", "put"):
            (self.build_dir / "bin" / f"kvaf_execute_batch_{kind}_kernels.o").touch()
        self.runtimes = []
        self.fail_at = None

        def make_runtime(**kwargs):
            runtime = Mock(spec=("link_to", "initialize", "shutdown"))
            self.runtimes.append(runtime)
            if len(self.runtimes) == self.fail_at:
                runtime.initialize.side_effect = OSError("allocation failed")
            return runtime

        sdk = ModuleType("kvaf")
        sdk.__file__ = str(self.build_dir / "python/kvaf/__init__.py")
        sdk.NpuRuntime = Mock(side_effect=make_runtime)
        self.sdk = sdk
        imports = patch.dict(sys.modules, {"kvaf": sdk, "torch_npu": ModuleType("torch_npu")})
        imports.start()
        self.addCleanup(imports.stop)

    def adapter(self, widths, *, reuse_device_ra=False):
        config = KVAFConfig(enabled=True, build_dir=str(self.build_dir), ddr_size_gb=0.001)
        result = adapter_module.KVAFAdapter(
            config,
            3,
            tuple(layout(width, str(i)) for i, width in enumerate(widths)),
            log_dir=str(self.build_dir),
            tp_rank=3,
            dp_rank=0,
            reuse_device_ra=reuse_device_ra,
        )
        self.addCleanup(result.close)
        return result

    def test_default_get_uses_mainline_even_with_isolated_artifact(self):
        (self.build_dir / "vllm-get-build").write_text("isolated\n")
        self.adapter((656,)).connect()
        self.assertEqual(
            self.sdk.NpuRuntime.call_args.kwargs["batch_get_kernel_object_path"],
            str((self.build_dir / "bin/kvaf_execute_batch_get_kernels.o").resolve()),
        )

    def test_runtime_borrows_existing_hccl_ra_only_when_requested(self):
        for reuse in (False, True):
            with self.subTest(reuse_device_ra=reuse):
                adapter = self.adapter((656,), reuse_device_ra=reuse)
                adapter.connect()
                self.assertIs(self.sdk.NpuRuntime.call_args.kwargs["reuse_device_ra"], reuse)
                adapter.close()

    def test_transport_selects_runtime_without_changing_urma_default(self):
        for transport in ("urma", "ubmem"):
            with self.subTest(transport=transport):
                config = KVAFConfig(enabled=True, build_dir=str(self.build_dir), ddr_size_gb=0.001, transport=transport)
                adapter = adapter_module.KVAFAdapter(
                    config, 3, (layout(656, "target"),), log_dir=self.build_dir, tp_rank=3, dp_rank=0
                )
                self.addCleanup(adapter.close)
                adapter.connect()
                kwargs = self.sdk.NpuRuntime.call_args.kwargs
                if transport == "ubmem":
                    self.assertEqual(kwargs["transport"], "ubmem")
                else:
                    self.assertNotIn("transport", kwargs)
                adapter.close()

    def test_fused_worker_budget_reaches_runtime_without_changing_config(self):
        config = KVAFConfig(enabled=True, build_dir=str(self.build_dir), ddr_size_gb=0.001, decode_backend="li_fetch")
        adapter = adapter_module.KVAFAdapter(
            config,
            3,
            (layout(656, "target"),),
            log_dir=self.build_dir,
            tp_rank=3,
            dp_rank=0,
            fused_workers=40,
        )
        self.addCleanup(adapter.close)
        adapter.connect()
        self.assertEqual(self.sdk.NpuRuntime.call_args.kwargs["npu_core_count"], 40)
        self.assertEqual(config.npu_core_count, 2)

    def test_reject_invalid_configuration(self):
        for values in (
            {"enabled": "true"},
            {"verify_put": "true"},
            {"enabled": True},
            {"cpu_address": "host:65536"},
            {"cpu_addresses": []},
            {"cpu_addresses": {4: "host:28999"}},
            {"cpu_addresses": {"-1": "host:28999"}},
            {"cpu_addresses": {"04": "host:28999"}},
            {"cpu_addresses": {"4": None}},
            {"cpu_addresses": {"4": "host:65536"}},
            {"cpu_addresses": {"4": "host"}},
            {"ddr_size_gb": float("nan")},
            {"ddr_size_gb": -1},
            {"npu_core_count": True},
            {"decode_backend": "unknown"},
            {"transport": "unknown"},
            {"intermediate_fetches": 0},
            {"intermediate_fetches": 33},
            {"intermediate_fetches": True},
            {"unknown_option": 1},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                KVAFConfig.from_dict(values)

    def test_config_import_does_not_initialize_vllm(self):
        # Platform discovery has loaded the root package, but not vLLM distributed.
        # Import through the real parent packages; the adapter tests skip them.
        root = ModuleType("kvaf_platform_import")
        root.__path__ = [str(Path(__file__).resolve().parents[3] / "vllm_ascend")]
        original_import = __import__

        def import_without_vllm(name, *args, **kwargs):
            if name == "vllm" or name.startswith("vllm."):
                raise ImportError(f"KVAF configuration must not initialize {name} during platform discovery")
            return original_import(name, *args, **kwargs)

        with (
            patch.dict(sys.modules, {root.__name__: root}),
            patch("builtins.__import__", side_effect=import_without_vllm),
        ):
            config = importlib.import_module(f"{root.__name__}.distributed.kv_transfer.kvaf.config")
            self.assertFalse(config.KVAFConfig.from_dict({}).enabled)
            # An empty map must also avoid importing vLLM during adapter setup.
            adapter = self.adapter((640,))
            self.assertEqual(adapter.cpu_address, adapter.config.cpu_address)

    def test_cpu_routes_use_physical_ids_and_runtime_keeps_logical_ids(self):
        platforms = ModuleType("vllm.platforms")
        platforms.current_platform = Mock()
        resolve = platforms.current_platform.visible_device_id_to_physical_device_id
        resolve.side_effect = [4, 5, 7, 6]
        config = replace(
            self.adapter((640,)).config,
            cpu_addresses={"0": "decoy:28999", "4": "cpu-a:28999", "5": "cpu-a:28999", "7": "cpu-b:28999"},
        )
        with patch.dict(sys.modules, {"vllm.platforms": platforms}):
            for logical, address in enumerate(("cpu-a:28999", "cpu-a:28999", "cpu-b:28999", config.cpu_address)):
                adapter = adapter_module.KVAFAdapter(
                    config,
                    logical,
                    (layout(640, "target"),),
                    log_dir=self.build_dir,
                    tp_rank=logical,
                    dp_rank=0,
                )
                self.addCleanup(adapter.close)
                adapter.connect()
                self.runtimes[-1].link_to.assert_called_once_with(address)
                self.assertEqual(self.sdk.NpuRuntime.call_args.kwargs["device_id"], logical)
        self.assertEqual([call.args[0] for call in resolve.call_args_list], [0, 1, 2, 3])

    def test_geometry_budgets_share_one_per_worker_limit(self):
        adapter = self.adapter((640, 768))
        adapter.connect()
        budgets = [runtime.initialize.call_args.args for runtime in self.runtimes]
        self.assertEqual(len(budgets), 2)
        self.assertLessEqual(sum(size for size, _ in budgets), int(adapter.config.ddr_size_gb * (1 << 30)))
        for (size, slot), call in zip(budgets, self.sdk.NpuRuntime.call_args_list):
            self.assertGreaterEqual(size, slot)
            self.assertEqual(size % slot, 0)
            self.assertEqual(slot % adapter_module.SLOT_ALIGNMENT, 0)
            self.assertEqual(call.kwargs["device_id"], 3)
        for runtime in self.runtimes:
            runtime.link_to.assert_called_once_with(adapter.config.cpu_address)

    def test_shared_geometry_uses_one_runtime_and_close_is_idempotent(self):
        adapter = self.adapter((640, 640))
        adapter.connect()
        self.assertEqual(len(self.runtimes), 1)
        self.assertEqual(len(adapter.runtimes), 1)
        with self.assertRaisesRegex(RuntimeError, "already connected"):
            adapter.connect()
        self.assertEqual(len(self.runtimes), 1)
        adapter.close()
        adapter.close()
        self.runtimes[0].shutdown.assert_called_once_with()
        self.assertFalse(adapter.runtimes)

    def test_grouped_runtimes_share_total_ddr_budget(self):
        adapter = self.adapter((656,) * 5)
        adapter.groups = (("0",), ("1", "2", "3", "4"))
        adapter.connect()
        self.assertEqual(set(adapter.runtimes), {(656, 128), (2624, 128)})
        total = sum(runtime.initialize.call_args.args[0] for runtime in self.runtimes)
        self.assertLessEqual(total, int(adapter.config.ddr_size_gb * (1 << 30)))
        self.assertEqual([layout.layer_names for layout in adapter.target_layouts], list(adapter.groups))

    def test_bind_caches_uses_allocated_slots_and_optional_readback(self):
        adapter = self.adapter((640,))
        adapter.config = replace(adapter.config, verify_put=True)
        adapter.connect()
        cache = Mock()
        cache.element_size.return_value = 1
        adapter.bind_caches({"0": cache})
        size, slot = self.runtimes[0].initialize.call_args.args
        self.assertEqual(adapter.writer.slots[(640, 128)].capacity, size // slot)
        self.assertIs(adapter.writer.verify, adapter_module.verify_put_block)
        self.assertIs(adapter.writer.groups[0].source, cache.as_strided.return_value)
        adapter.close()
        self.assertIsNone(adapter.writer)
        self.assertFalse(adapter.slot_counts)

    def test_read_layout_does_not_require_c8_but_requires_contiguous_tokens(self):
        adapter = self.adapter((640,))
        cache = replace(adapter.layouts[0], packed_c8=False)
        adapter.layouts = (cache,)
        adapter.connect()
        adapter.close()
        for planes in ((cache.planes[0], cache.planes[0]), (replace(cache.planes[0], direct_token_read=False),)):
            adapter.layouts = (replace(cache, planes=planes),)
            with self.subTest(planes=len(planes)), self.assertRaisesRegex(ValueError, "single token-contiguous"):
                adapter.connect()
        self.assertEqual(len(self.runtimes), 1)

    def test_partial_initialize_failure_releases_all_created_sessions(self):
        self.fail_at = 2
        adapter = self.adapter((640, 768))
        with self.assertRaisesRegex(RuntimeError, "allocation failed") as caught:
            adapter.connect()
        self.assertIsInstance(caught.exception.__cause__, OSError)
        self.assertEqual(len(self.runtimes), 2)
        for runtime in self.runtimes:
            runtime.shutdown.assert_called_once_with()
        self.assertFalse(adapter.runtimes)

    def test_native_logs_and_stderr_restoration(self):
        # Isolate real file-descriptor changes from the test runner's streams.
        code = textwrap.dedent("""
            import json, logging, os, sys, traceback
            from tests.ut.kvaf.test_adapter import TestKVAFAdapter, adapter_module

            case = TestKVAFAdapter()
            case.setUp()
            failed = sys.argv[1] == "failure"
            create_runtime = case.sdk.NpuRuntime.side_effect
            adapter_module.logger.addHandler(logging.StreamHandler(sys.stderr))
            adapter_module.logger.setLevel(logging.INFO)

            def initialize(*args):
                os.write(2, b"[KVAF_AIV] native initialize\\n")
                os.write(1, b"stdout during initialize\\n")
                if failed:
                    raise RuntimeError("forced metadata failure")

            def native_runtime(**kwargs):
                os.write(2, b"[KVAF_AIV] native construct\\n")
                runtime = create_runtime(**kwargs)
                runtime.link_to.side_effect = lambda address: os.write(2, b"[KVAF_AIV] native link\\n")
                runtime.initialize.side_effect = initialize
                runtime.shutdown.side_effect = lambda: os.write(2, b"[KVAF_AIV] native shutdown\\n")
                return runtime

            case.sdk.NpuRuntime.side_effect = native_runtime
            adapter = case.adapter((640,))
            other = adapter_module.KVAFAdapter(
                adapter.config, 3, adapter.layouts,
                log_dir=str(case.build_dir), tp_rank=4, dp_rank=1,
            )
            try:
                try:
                    adapter.connect()
                except RuntimeError:
                    traceback.print_exc()
                    if not failed:
                        raise
                adapter.close()
                os.write(2, b"stderr restored\\n")
                print(json.dumps({
                    "path": str(adapter.log_path), "other_path": str(other.log_path),
                    "contents": adapter.log_path.read_text(),
                }))
            finally:
                case.doCleanups()
            """)
        for outcome in ("success", "failure"):
            with self.subTest(outcome=outcome):
                result = subprocess.run(
                    [sys.executable, "-c", code, outcome],
                    cwd=Path(__file__).resolve().parents[3],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                payload = json.loads(result.stdout.splitlines()[-1])
                self.assertIn("stdout during initialize\n", result.stdout)
                self.assertIn("stderr restored\n", result.stderr)
                self.assertIn(payload["path"], result.stderr)
                self.assertIn("kvaf_DP0_TP3_device3_pid", payload["path"])
                self.assertIn("kvaf_DP1_TP4_device3_pid", payload["other_path"])
                self.assertNotEqual(payload["path"], payload["other_path"])
                for operation in ("construct", "link", "initialize", "shutdown"):
                    message = f"[KVAF_AIV] native {operation}\n"
                    self.assertIn(message, payload["contents"])
                    self.assertNotIn(message, result.stderr)
                self.assertNotIn("stderr restored", payload["contents"])
                if outcome == "failure":
                    self.assertIn("forced metadata failure", result.stderr)
                    self.assertIn("forced metadata failure", payload["contents"])
                    self.assertIn("Traceback", payload["contents"])


if __name__ == "__main__":
    unittest.main()
