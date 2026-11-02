# SPDX-License-Identifier: Apache-2.0
"""Host-only layout checks; run directly without vLLM/NPU dependencies."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

MODULE_PATH = Path(__file__).resolve().parents[2] / "vllm_ascend/distributed/kv_transfer/kvaf/layout.py"
SPEC = importlib.util.spec_from_file_location("kvaf_layout_under_test", MODULE_PATH)
layout = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = layout
SPEC.loader.exec_module(layout)


def tensor(width, pointer, *, item_bytes=1, strides=None):
    return NS(
        shape=(8, 128, 1, width),
        dtype=f"int{item_bytes * 8}",
        device="npu:0",
        stride=lambda: strides or (128 * width, width, width, 1),
        element_size=lambda: item_bytes,
        data_ptr=lambda: pointer,
    )


def config(specs):
    return NS(kv_cache_groups=[NS(layer_names=list(specs), kv_cache_spec=NS(kv_cache_specs=specs))])


class TestKVAFLayout(unittest.TestCase):
    def test_packed_target_shared_with_draft_is_recorded_once(self):
        spec = NS(block_size=128, page_size_bytes=128 * 640, cache_sparse_sfa_c8=True)
        cache = tensor(640, 4096)
        result = layout.describe_cache_layouts(
            {"target": (cache,), "draft_alias": (cache,)},
            config({"target": spec, "draft_alias": spec}),
            ("draft_alias",),
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].roles, ("draft", "target"))
        self.assertEqual(result[0].cache_kind, "main")
        self.assertEqual(result[0].layer_names, ("target", "draft_alias"))
        self.assertTrue(result[0].packed_c8)
        self.assertEqual(result[0].planes[0].token_bytes, 640)

    def test_indexer_keeps_key_and_scale_separate(self):
        spec = NS(block_size=128, page_size_bytes=128 * 132, sfa_dcp_replicated_indexer_size=1)
        result = layout.describe_cache_layouts(
            {"index": (tensor(128, 4096), tensor(1, 8192, item_bytes=4))}, config({"index": spec})
        )[0]
        self.assertEqual(result.roles, ("indexer",))
        self.assertEqual(result.cache_kind, "indexer")
        self.assertEqual([plane.token_bytes for plane in result.planes], [128, 4])
        self.assertEqual(sum(plane.page_bytes for plane in result.planes), result.page_bytes)

    def test_draft_indexer_retains_its_cache_kind(self):
        spec = NS(block_size=128, page_size_bytes=128 * 128, sfa_dcp_replicated_indexer_size=1)
        result = layout.describe_cache_layouts(
            {"draft_index": tensor(128, 4096)}, config({"draft_index": spec}), ("draft_index",)
        )[0]
        self.assertEqual(result.roles, ("draft",))
        self.assertEqual(result.cache_kind, "indexer")

    def test_noncontiguous_token_is_not_direct_readable(self):
        spec = NS(block_size=128, page_size_bytes=128 * 16)
        result = layout.describe_cache_layouts(
            {"target": tensor(16, 4096, strides=(4096, 32, 32, 2))}, config({"target": spec})
        )[0]
        self.assertFalse(result.planes[0].direct_token_read)
        self.assertIsNone(result.planes[0].token_bytes)
        self.assertEqual(result.planes[0].page_stride_bytes, 4096)

    def test_live_layer_spec_overrides_group(self):
        group_spec = NS(block_size=128, page_size_bytes=128 * 16)
        index_spec = NS(block_size=128, page_size_bytes=128 * 16, sfa_dcp_replicated_indexer_size=1)
        result = layout.describe_cache_layouts(
            {"index": tensor(16, 4096)}, config({"index": group_spec}), layer_specs={"index": index_spec}
        )[0]
        self.assertEqual(result.roles, ("indexer",))

    def test_shared_views_cannot_disagree_about_page_size(self):
        cache = tensor(16, 4096)
        specs = {name: NS(block_size=128, page_size_bytes=size) for name, size in (("a", 2048), ("b", 4096))}
        with self.assertRaisesRegex(ValueError, "incompatible specs"):
            layout.describe_cache_layouts({"a": cache, "b": cache}, config(specs))


if __name__ == "__main__":
    unittest.main()
