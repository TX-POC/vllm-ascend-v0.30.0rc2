#!/usr/bin/env python3
"""Static regression gate for P9's caller-thread ACLNN launch boundary.

This test intentionally stays independent of CANN/Torch: it prevents the P9
bridge from silently falling back to a deferred generic launch path, where a
host callback could outlive the forward that submitted its resident output.
"""

from __future__ import annotations

from pathlib import Path
import unittest


OPS_ROOT = Path(__file__).resolve().parents[1]
BRIDGE = (
    OPS_ROOT
    / "a5_c8_exact_speculative_resident_get"
    / "torch_extension"
    / "csrc"
    / "a5_c8_exact_speculative_resident_get.cpp"
).read_text()


class P9DirectLaunchContractTest(unittest.TestCase):
    def test_p9_uses_its_direct_workspace_launcher_not_a_deferred_host_path(
        self,
    ) -> None:
        self.assertIn("LaunchP9CallerThreadDirectAsync", BRIDGE)
        self.assertIn("RequireP9CallerThreadDirectAsyncContract", BRIDGE)
        self.assertIn("P9_DIRECT_ASYNC_STAGE", BRIDGE)
        self.assertIn("P9UseOrderedLaunchBisection", BRIDGE)
        self.assertIn("P9_ORDERED_LAUNCH_BISECT", BRIDGE)
        self.assertIn("if (P9UseOrderedLaunchBisection())", BRIDGE)
        self.assertIn("EXEC_NPU_CMD_ORDERED(", BRIDGE)
        direct_call = BRIDGE.rindex("LaunchP9CallerThreadDirectAsync(")
        self.assertLess(
            BRIDGE.index("if (P9UseOrderedLaunchBisection())"),
            direct_call,
        )
        # The ordered path is an opt-in diagnostic only.  Normal P9 launches
        # retain the direct caller-thread implementation and never implicitly
        # cross into RunOpApi.
        self.assertIn('std::getenv("P9_FORCE_ORDERED_LAUNCH")', BRIDGE)
        self.assertIn('std::strcmp(value, "1") == 0', BRIDGE)

    def test_nonzero_workspace_uses_its_creating_stream_before_cleanup(
        self,
    ) -> None:
        for token in (
            "const auto npu_stream = c10_npu::getCurrentNPUStream()",
            "npu_stream.stream(false)",
            "if (workspace_size != 0)",
            "workspace_reference.options().dtype(at::kByte)",
            "ReleaseTuple(converted)",
            "ClearTensorTuple(tensor_keepalive)",
            "workspace = at::Tensor();",
            "do not record it again on its creating stream",
        ):
            self.assertIn(token, BRIDGE)

        # The ownership contract is direct same-stream launch: the workspace
        # stays local through descriptor release, so it needs no cross-stream
        # recordStream edge before host cleanup.
        launch = BRIDGE.index("launch_fn(workspace_ptr")
        release = BRIDGE.index("ReleaseTuple(converted)")
        workspace_reset = BRIDGE.index("workspace = at::Tensor();", release)
        tls = BRIDGE.index("cleanup_host_tls();", release)
        self.assertLess(launch, release)
        self.assertLess(release, workspace_reset)
        self.assertLess(workspace_reset, tls)
        self.assertLess(release, tls)
        self.assertNotIn("RecordTensorTupleOnNpuStream", BRIDGE)

    def test_p9_direct_route_never_synchronizes_or_reintroduces_fixed_core_binding(
        self,
    ) -> None:
        self.assertNotIn("aclrtSynchronize", BRIDGE)
        self.assertNotIn("UseStreamResInCurrentThread", BRIDGE)
        self.assertIn("TASK_QUEUE_ENABLE", BRIDGE)
        self.assertIn("ASCEND_LAUNCH_BLOCKING", BRIDGE)
        self.assertIn("fixed AIV partition", BRIDGE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
