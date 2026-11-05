# SPDX-License-Identifier: Apache-2.0
"""Run the actual startup branch without a generated SoC build attribute."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


def startup_branch():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text())
    method = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_initialize_kvaf_cache_state"
    )
    branch = next(
        n
        for n in method.body
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "fused_history"
    )
    return compile(ast.Module(body=[branch], type_ignores=[]), str(path), "exec")


@pytest.mark.parametrize("soc", ["Ascend950PR_9599", "Ascend950DT_9591"])
def test_runtime_soc_reaches_fused_build(soc):
    initialize = Mock()
    runner = SimpleNamespace(
        kvaf_cache_state=SimpleNamespace(policy=SimpleNamespace(initialize_fused_workspace=initialize)),
        ascend_config=SimpleNamespace(kvaf_config=object()),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_n_heads=32)),
    )
    query = Mock(return_value=soc)
    with patch.dict(sys.modules, {"acl": SimpleNamespace(get_soc_name=query)}):
        exec(
            startup_branch(),
            dict(
                fused_history=131328, self=runner, torch=SimpleNamespace(npu=SimpleNamespace(current_device=lambda: 6))
            ),
        )
    initialize.assert_called_once_with(runner.ascend_config.kvaf_config, 6, soc, index_heads=32)


def test_native_path_does_not_query_acl():
    with patch.dict(sys.modules, {"acl": None}):
        exec(startup_branch(), dict(fused_history=0))


def test_failed_runtime_query_does_not_guess_a_chip():
    with (
        patch.dict(sys.modules, {"acl": SimpleNamespace(get_soc_name=lambda: None)}),
        pytest.raises(RuntimeError, match="ACL could not query"),
    ):
        exec(startup_branch(), dict(fused_history=131328))
