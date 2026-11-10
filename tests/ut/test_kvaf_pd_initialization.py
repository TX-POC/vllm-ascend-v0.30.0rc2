# SPDX-License-Identifier: Apache-2.0
"""Host-only regression: competing TCP clients cannot initialize one K together."""

import importlib
import socket
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import test_kvaf_adapter as helpers
import test_kvaf_pd_integration as integration

INIT = importlib.import_module(f"{helpers.PACKAGE.__name__}.pd.initialization")
RUNTIME = importlib.import_module(f"{helpers.PACKAGE.__name__}.pd.runtime")


class TestMultiMRSDKContract(unittest.TestCase):
    @staticmethod
    def sdk(build):
        return NS(
            __file__=str(build / "python/kvaf/__init__.py"),
            _native=NS(reuse_device_ra_supported=True),
            NpuRuntime=NS(
                listen_hbm=lambda: None,
                cancel_raw_addr=lambda: None,
                submit_cpu_pull=lambda: None,
                batch_get_raw_addr=lambda: None,
            ),
            CpuPullHbmSource=object,
            NpuPullHbmSource=object,
            ReadyToken=NS(mr_id=None),
            CpuPullSourceDesc=NS(mr_length=None),
        )

    def test_new_api_loads_only_from_configured_build(self):
        with TemporaryDirectory() as directory:
            build = Path(directory)
            sdk = self.sdk(build)
            with patch.object(RUNTIME.importlib, "import_module", return_value=sdk):
                self.assertIs(RUNTIME.load_sdk(build), sdk)
                with self.assertRaisesRegex(RuntimeError, "outside configured"):
                    RUNTIME.load_sdk(build / "other")

    def test_old_and_intermediate_apis_fail_before_runtime_allocation(self):
        with TemporaryDirectory() as directory:
            build = Path(directory)
            for kind, field in (
                ("NpuRuntime", "listen_hbm"),
                ("NpuRuntime", "cancel_raw_addr"),
                ("ReadyToken", "mr_id"),
                ("CpuPullSourceDesc", "mr_length"),
            ):
                sdk = self.sdk(build)
                delattr(getattr(sdk, kind), field)
                with (
                    self.subTest(field=field),
                    patch.object(RUNTIME.importlib, "import_module", return_value=sdk),
                    self.assertRaisesRegex(RuntimeError, field),
                ):
                    RUNTIME.load_sdk(build)


class TestInitializationAdmission(unittest.TestCase):
    def setUp(self):
        self.server = INIT.InitializationServer(("127.0.0.1", 0))
        self.endpoint = f"127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def slot(self, endpoint="k:28997", timeout=3):
        return INIT.initialization_slot(self.endpoint, endpoint, timeout=timeout)

    def test_eight_independent_connections_serialize_sdk_work(self):
        barrier = threading.Barrier(8)
        mutex = threading.Lock()
        active = 0
        maximum = 0

        def initialize(_):
            nonlocal active, maximum
            barrier.wait(timeout=3)
            with self.slot():
                with mutex:
                    active += 1
                    maximum = max(maximum, active)
                time.sleep(0.02)
                with mutex:
                    active -= 1

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(initialize, range(8)))
        self.assertEqual(maximum, 1)
        self.assertFalse(self.server.gate("k:28997").blocked)

    def test_different_k_endpoints_do_not_block_each_other(self):
        barrier = threading.Barrier(2)

        def initialize(endpoint):
            with self.slot(endpoint):
                barrier.wait(timeout=2)

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(initialize, ("k1:28997", "k2:28997")))

    def test_waiter_timeout_does_not_release_owner_or_poison_gate(self):
        with self.slot():
            with self.assertRaisesRegex(RuntimeError, "SDK was not called"), self.slot(timeout=1):
                self.fail("waiter entered while owner is active")
            self.assertTrue(self.server.gate("k:28997").active)
        with self.slot():
            pass

    def test_native_error_poison_blocks_following_initializers(self):
        with self.assertRaisesRegex(ValueError, "native failure"), self.slot():
            raise ValueError("native failure")
        with self.assertRaisesRegex(RuntimeError, "previous initializer failed/disconnected"), self.slot():
            self.fail("poisoned endpoint admitted another initializer")

    def test_owner_disconnect_never_automatically_transfers_admission(self):
        host, port = INIT.validate_endpoint(self.endpoint)
        with socket.create_connection((host, port), timeout=3) as connection:
            INIT._send(connection, {"endpoint": "k:28997", "timeout": 3})
            with connection.makefile("rb") as reader:
                self.assertEqual(INIT._receive(reader), {"status": "granted"})
        with self.assertRaisesRegex(RuntimeError, "previous initializer failed/disconnected"), self.slot():
            self.fail("disconnected owner was silently replaced")


class TestInitializationIntegration(unittest.TestCase):
    def test_feature_off_opens_no_socket(self):
        with (
            patch.object(INIT.socket, "create_connection") as connect,
            INIT.initialization_slot("", "k:28997", timeout=120),
        ):
            pass
        connect.assert_not_called()

    def test_config_is_validated_on_d_and_rejected_on_p(self):
        config = integration.config()
        self.assertEqual(integration.CONFIG.PDConfig.from_vllm(config).initialization_coordinator, "")
        config.kv_transfer_config.kv_connector_extra_config = {"initialization_coordinator": "coordinator:28998"}
        self.assertEqual(integration.CONFIG.PDConfig.from_vllm(config).initialization_coordinator, "coordinator:28998")
        for value in (None, False, 1, "host", "host:0", "host:65536", " "):
            config.kv_transfer_config.kv_connector_extra_config = {"initialization_coordinator": value}
            with self.subTest(value=value), self.assertRaises(ValueError):
                integration.CONFIG.PDConfig.from_vllm(config)
        config = integration.config(producer=True)
        config.kv_transfer_config.kv_connector_extra_config["initialization_coordinator"] = "coordinator:28998"
        with self.assertRaisesRegex(ValueError, "D consumers only"):
            integration.CONFIG.PDConfig.from_vllm(config)

    def test_pd_adapter_wraps_all_sdk_initialization_calls(self):
        with TemporaryDirectory() as directory:
            build = Path(directory)
            (build / "bin").mkdir()
            for kind in ("get", "put", "get_raw_addr"):
                (build / "bin" / f"kvaf_execute_batch_{kind}_kernels.o").touch()
            config = helpers.KVAFConfig(enabled=True, build_dir=directory, ddr_size_gb=16)
            adapter = RUNTIME.PDAdapter(
                config,
                0,
                (helpers.layout(656, "m"),),
                groups=(("m",),),
                log_dir=directory,
                tp_rank=0,
                dp_rank=0,
                initialization_coordinator="coord:28998",
                initialization_timeout=600,
            )
            self.addCleanup(adapter.close)
            native = Mock()
            entered = False

            class Slot:
                def __enter__(self):
                    nonlocal entered
                    entered = True

                def __exit__(self, *args):
                    nonlocal entered
                    entered = False

            def check(*args):
                self.assertTrue(entered)

            native.link_to.side_effect = check
            native.initialize.side_effect = check
            native.initialize_ddr.side_effect = check
            with (
                patch.object(RUNTIME, "load_sdk", return_value=Mock(NpuRuntime=Mock(return_value=native))),
                patch.object(RUNTIME, "initialization_slot", return_value=Slot()) as slot,
                patch.object(adapter, "_native_log", return_value=nullcontext()),
            ):
                adapter.connect()
            slot.assert_called_once_with("coord:28998", adapter.cpu_address, timeout=600)
            native.initialize_ddr.assert_called_once_with(17179803648, 86016)

    def test_adapter_rejected_admission_calls_no_sdk_initialization(self):
        with TemporaryDirectory() as directory:
            build = Path(directory)
            (build / "bin").mkdir()
            for kind in ("get", "put", "get_raw_addr"):
                (build / "bin" / f"kvaf_execute_batch_{kind}_kernels.o").touch()
            adapter = RUNTIME.PDAdapter(
                helpers.KVAFConfig(enabled=True, build_dir=directory),
                0,
                (helpers.layout(656, "m"),),
                groups=(("m",),),
                log_dir=directory,
                tp_rank=0,
                dp_rank=0,
                initialization_coordinator="coord:28998",
            )
            self.addCleanup(adapter.close)
            native = Mock()
            with (
                patch.object(RUNTIME, "load_sdk", return_value=Mock(NpuRuntime=Mock(return_value=native))),
                patch.object(RUNTIME, "initialization_slot", side_effect=RuntimeError("admission denied")),
                patch.object(adapter, "_native_log", return_value=nullcontext()),
                self.assertRaisesRegex(RuntimeError, "admission denied"),
            ):
                adapter.connect()
            native.link_to.assert_not_called()
            native.initialize.assert_not_called()
            native.initialize_ddr.assert_not_called()
            native.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
