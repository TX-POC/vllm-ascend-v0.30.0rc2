# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for pruning checkpoint streams and synchronizing MTP config."""

import __future__

import ast
import logging
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_mtp.py"
TREE = ast.parse(SOURCE.read_text())
FUNCTIONS = ("_validate_expert_pruning", "_sync_pruned_mtp_config", "_filter_pruned_expert_weights")
NAMESPACE = {"re": re, "logger": logging.getLogger(__name__)}
NODES = [
    node
    for node in TREE.body
    if isinstance(node, ast.FunctionDef)
    and node.name in FUNCTIONS
    or isinstance(node, ast.Assign)
    and any(getattr(t, "id", "") in ("_EXPERT_ID", "_ROUTER_WEIGHTS") for t in node.targets)
]
exec(
    compile(ast.Module(body=NODES, type_ignores=[]), str(SOURCE), "exec", __future__.annotations.compiler_flag),
    NAMESPACE,
)
validate, sync_mtp, filter_weights = (NAMESPACE[name] for name in FUNCTIONS)


class Tensor:
    def __init__(self, shape):
        self.shape, self.ndim = shape, len(shape)
        self.rows = tuple(range(shape[0])) if shape else ()

    def __getitem__(self, key):
        result = Tensor((len(self.rows[key]), *self.shape[1:]))
        result.rows = self.rows[key]
        return result


def config():
    return SimpleNamespace(
        original_n_routed_experts=8, n_routed_experts=4, n_group=2, topk_group=1, num_experts_per_tok=2
    )


class ExpertPruningTests(unittest.TestCase):
    def test_disabled_preserves_original_weights(self):
        weights = [
            ("model.layers.2.mlp.experts.7.down_proj.weight", Tensor((3, 2))),
            ("model.layers.2.mlp.gate.weight", Tensor((8, 2))),
        ]
        self.assertEqual(list(filter_weights(SimpleNamespace(), iter(weights))), weights)
        validate(SimpleNamespace())

    def test_filters_experts_and_scales_and_slices_only_router_rows(self):
        prefix = "model.layers.45.mlp."
        retained = [(prefix + f"experts.3.down_proj.{suffix}", Tensor((2, 3))) for suffix in ("weight", "weight_scale")]
        removed = [(name.replace("experts.3.", "experts.4."), value) for name, value in retained]
        shared = (prefix + "shared_experts.down_proj.weight", Tensor((8, 3)))
        routers = [
            (prefix + suffix, Tensor(shape))
            for suffix, shape in (("gate.weight", (8, 3)), ("gate.bias", (8,)), ("gate.e_score_correction_bias", (8,)))
        ]
        result = dict(filter_weights(config(), iter(retained + removed + [shared] + routers)))
        self.assertEqual(set(result), {name for name, _ in retained + [shared] + routers})
        for name, weight in retained + [shared]:
            self.assertIs(result[name], weight)
        for name, weight in routers:
            self.assertEqual(result[name].shape, (4, *weight.shape[1:]))
            self.assertEqual(result[name].rows, weight.rows[:4])

    def test_mtp_sync_updates_independent_config_and_derived_expert_count(self):
        target = config()
        hf = SimpleNamespace(model_type="deepseek_mtp", n_routed_experts=8)
        hf.get_text_config = lambda: hf
        draft = SimpleNamespace(hf_config=hf, hf_text_config=hf, model_arch_config=SimpleNamespace(num_experts=8))
        draft.get_model_arch_config = lambda: SimpleNamespace(num_experts=hf.n_routed_experts)
        spec = SimpleNamespace(target_model_config=SimpleNamespace(hf_config=target), draft_model_config=draft)
        sync_mtp(SimpleNamespace(speculative_config=spec, model_config=draft))
        self.assertEqual(
            (hf.n_routed_experts, hf.original_n_routed_experts, draft.model_arch_config.num_experts), (4, 8, 4)
        )
        self.assertIs(draft.hf_text_config, hf)
        self.assertEqual(hf.model_type, "deepseek_mtp")
        self.assertIsNot(hf, target)

    def test_rejects_invalid_counts_groups_and_router_shapes(self):
        for changes in (
            {"n_routed_experts": 0},
            {"n_routed_experts": 9},
            {"n_routed_experts": 3},
            {"num_experts_per_tok": 3},
            {"n_routed_experts": 2, "num_experts_per_tok": 1, "topk_method": "noaux_tc"},
        ):
            invalid = config()
            vars(invalid).update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate(invalid)
        for shape in ((), (7, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                list(filter_weights(config(), [("model.layers.2.mlp.gate.weight", Tensor(shape))]))

    def test_both_loaders_filter_and_mtp_sync_precedes_allocation(self):
        classes = {node.name: node for node in TREE.body if isinstance(node, ast.ClassDef)}
        for name in ("AscendDeepSeekMTP", "AscendGlmMoeDsaForCausalLM"):
            methods = {node.name: node for node in classes[name].body if isinstance(node, ast.FunctionDef)}
            self.assertIn("_filter_pruned_expert_weights", ast.unparse(methods["load_weights"].body[0]))
        init = next(node for node in classes["AscendDeepSeekMTP"].body if getattr(node, "name", None) == "__init__")
        self.assertIn("_sync_pruned_mtp_config", ast.unparse(init.body[0]))
        self.assertIn("super().__init__", ast.unparse(init.body[1]))


if __name__ == "__main__":
    unittest.main()
