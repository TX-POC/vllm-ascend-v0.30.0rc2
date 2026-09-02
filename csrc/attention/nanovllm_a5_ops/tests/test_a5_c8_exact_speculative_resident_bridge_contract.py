#!/usr/bin/env python3
"""Cheap ABI gate for the dedicated P9 Torch/ACLNN bridge.

This deliberately does not import torch.  It catches source/JSON/bridge ABI
drift before a remote CANN build spends minutes compiling the cloned C8 OPP.
The P9 public bridge returns four user-visible values while two caller-owned
mutable tensors remain graph-visible aliases inside the ACLNN invocation.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest


OPS_ROOT = Path(__file__).resolve().parents[1]
TARGET_ROOT = OPS_ROOT / "a5_c8_exact_speculative_resident_get"
OPS = json.loads((TARGET_ROOT / "source_overlay" / "ops.json").read_text())
BRIDGE = (
    TARGET_ROOT / "torch_extension" / "csrc" / "a5_c8_exact_speculative_resident_get.cpp"
).read_text()


P9_INPUTS = (
    "runtime_enable",
    "resident_threshold",
    "resident_threshold_valid",
    "req_pool_entries",
    "kv_object_map",
    "kv_object_lens",
    "persistent_object_slots_snapshot",
    "canonical_resident_layout",
    "candidate_source_to_slot",
    "candidate_slot_cursor",
    "layer_resident_get_control",
    "rows_per_entry",
)
P9_OUTPUTS = (
    "attention_slots",
    "terminal_post_stats",
    "candidate_source_to_slot_ref",
    "candidate_slot_cursor_ref",
)


class A5C8ExactSpeculativeResidentBridgeContractTest(unittest.TestCase):
    def test_atomic_cann_abi_has_the_frozen_21_input_6_output_shape(self) -> None:
        self.assertEqual(len(OPS), 1)
        op = OPS[0]
        self.assertEqual(op["op"], "A5C8ExactSpeculativeResidentGet")
        self.assertEqual(
            [entry["name"] for entry in op["input_desc"][:9]],
            [
                "query",
                "key",
                "weights",
                "query_dequant_scale",
                "key_dequant_scale",
                "actual_seq_lengths_query",
                "actual_seq_lengths_key",
                "block_table",
                "metadata",
            ],
        )
        self.assertEqual(
            [entry["name"] for entry in op["input_desc"][9:]], list(P9_INPUTS)
        )
        self.assertEqual(
            [entry["name"] for entry in op["output_desc"][:2]],
            ["sparse_indices", "sparse_values"],
        )
        self.assertEqual(
            [entry["name"] for entry in op["output_desc"][2:]], list(P9_OUTPUTS)
        )
        self.assertEqual(len(op["input_desc"]), 21)
        self.assertEqual(len(op["output_desc"]), 6)

    def test_bridge_preserves_the_same_order_and_hides_only_internal_aliases(self) -> None:
        for token in (
            "official_tensor_inputs\"] = 9",
            "p9_tensor_inputs\"] = 12",
            "public_outputs\"] = 4",
            "mutable_output_aliases\"] = 2",
            "Tensor(a!) candidate_source_to_slot",
            "Tensor(b!) candidate_slot_cursor",
            "aclnnA5C8ExactSpeculativeResidentGet",
            "attention_slots, terminal_post_stats, candidate_source_to_slot,",
            "candidate_slot_cursor);",
        ):
            self.assertIn(token, BRIDGE)
        # Canonical resident targets are encoded once in the KVAF control
        # ABI.  A duplicated packed-output tensor would create two possible
        # destinations and violate the direct-resident contract.
        self.assertNotIn("resident_packed_output", BRIDGE)
        self.assertNotIn("a5_p8_", BRIDGE)
        self.assertNotIn("p7_", BRIDGE.lower())

    def test_control_and_alias_shapes_are_checked_before_aclnn(self) -> None:
        for token in (
            "kLayerResidentGetControlWords = 16",
            "kCanonicalResidentLayoutWords = 4",
            "layer_resident_get_control must be int64[16]",
            "canonical_resident_layout must be int32[4]",
            "candidate_source_to_slot must be int32[R,S]",
            "candidate_slot_cursor",
            "rows_per_entry",
        ):
            self.assertIn(token, BRIDGE)

    def test_bridge_and_generated_opp_share_runnable_layout_defaults(self) -> None:
        """Default dispatch must not select the invalid dense-key layout."""
        self.assertIn(
            "str layout_query='BSND', str layout_key='PA_BSND'", BRIDGE
        )
        attrs = {entry["name"]: entry["default_value"] for entry in OPS[0]["attr"]}
        self.assertEqual(attrs["layout_query"], "BSND")
        self.assertEqual(attrs["layout_key"], "PA_BSND")


if __name__ == "__main__":
    unittest.main(verbosity=2)
