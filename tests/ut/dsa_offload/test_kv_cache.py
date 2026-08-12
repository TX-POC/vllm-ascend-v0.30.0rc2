# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.dsa_offload import kv_cache as dsa_kv_cache
from vllm_ascend.dsa_offload.kv_cache import (
    DSAIndexerKVSpec,
    DSAResidentMLAAttentionSpec,
    build_dsa_kv_cache_config,
    build_dsa_kv_cache_groups,
    dsa_max_memory_usage_bytes,
    dsa_pool_bytes_per_base_block,
    get_dsa_group_num_blocks,
    get_dsa_kv_cache_binding_order,
    get_dsa_kv_cache_group_ids,
    get_dsa_request_limit_base_blocks,
    get_dsa_request_limit_blocks,
    validate_dsa_kv_cache_config,
)
from vllm_ascend.dsa_offload.kv_cache_coordinator import (
    DSABlockPoolView,
    DSAKVCacheCoordinator,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _ascend_get_kv_cache_groups,
)


def _make_specs(
    num_layers: int = 2,
) -> dict[str, DSAIndexerKVSpec | DSAResidentMLAAttentionSpec]:
    specs: dict[
        str,
        DSAIndexerKVSpec | DSAResidentMLAAttentionSpec,
    ] = {}
    for layer_idx in range(num_layers):
        specs[f"model.layers.{layer_idx}.self_attn.indexer.k_cache"] = DSAIndexerKVSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
        )
        specs[f"model.layers.{layer_idx}.self_attn.attn"] = DSAResidentMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            sparse_head_dim=(512, 64, 0),
            dtype=torch.bfloat16,
            cache_dtype_str="auto",
        )
    return specs


def _make_vllm_config(
    *,
    max_model_len: int = 32768,
    max_num_seqs: int = 1,
    override: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=override),
        model_config=SimpleNamespace(max_model_len=max_model_len),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )


@pytest.fixture(autouse=True)
def _mock_dsa_offload_config(monkeypatch) -> SimpleNamespace:
    """Keep capacity-planning unit tests independent of global config state."""

    dsa_config = SimpleNamespace(
        capacity_mode="auto",
        capacity_guard_blocks=0,
        indexer_mla_block_ratio=3,
        keep_decode_kv_in_hbm=False,
    )
    monkeypatch.setattr(
        "vllm_ascend.ascend_config.get_ascend_config",
        lambda: SimpleNamespace(dsa_offload_config=dsa_config),
    )
    return dsa_config


def test_indexer_spec_accounts_for_one_vector_per_token() -> None:
    spec = next(spec for spec in _make_specs(num_layers=1).values() if isinstance(spec, DSAIndexerKVSpec))

    assert spec.page_size_bytes == 128 * 1 * 128 * 2


def test_split_groups_keep_stable_plane_order() -> None:
    groups = build_dsa_kv_cache_groups(_make_specs())

    assert len(groups) == 2
    assert isinstance(groups[0].kv_cache_spec, DSAIndexerKVSpec)
    assert isinstance(
        groups[1].kv_cache_spec,
        DSAResidentMLAAttentionSpec,
    )
    assert len(groups[0].layer_names) == len(groups[1].layer_names) == 2


def test_engine_core_grouping_initializes_process_local_ascend_config(
    monkeypatch,
) -> None:
    initialized_with: list[object] = []
    vllm_config = object()
    monkeypatch.setattr(
        "vllm_ascend.ascend_config.init_ascend_config",
        initialized_with.append,
    )

    groups = _ascend_get_kv_cache_groups(  # type: ignore[arg-type]
        vllm_config,
        _make_specs(),
    )

    assert initialized_with == [vllm_config]
    assert len(groups) == 2


def test_split_group_ids_follow_spec_identity_not_position() -> None:
    groups = list(reversed(build_dsa_kv_cache_groups(_make_specs())))
    config = SimpleNamespace(kv_cache_groups=groups)

    group_ids = get_dsa_kv_cache_group_ids(config)  # type: ignore[arg-type]

    assert group_ids.indexer == 1
    assert group_ids.resident_mla == 0


def test_split_cache_binding_orders_each_layer_resident_then_indexer() -> None:
    groups = build_dsa_kv_cache_groups(_make_specs())
    config = SimpleNamespace(kv_cache_groups=groups)

    assert get_dsa_kv_cache_binding_order(config) == [
        "model.layers.0.self_attn.attn",
        "model.layers.0.self_attn.indexer.k_cache",
        "model.layers.1.self_attn.attn",
        "model.layers.1.self_attn.indexer.k_cache",
    ]


def test_split_cache_binding_rejects_duplicate_layer_name() -> None:
    groups = build_dsa_kv_cache_groups(_make_specs(num_layers=1))
    groups[1].layer_names.append(groups[0].layer_names[0])
    config = SimpleNamespace(kv_cache_groups=groups)

    with pytest.raises(RuntimeError, match="more than one group"):
        get_dsa_kv_cache_binding_order(config)


def test_ratio_is_expressed_by_final_tensor_sizes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    groups = build_dsa_kv_cache_groups(_make_specs())
    bytes_per_base_block = dsa_pool_bytes_per_base_block(groups)

    config = build_dsa_kv_cache_config(
        _make_vllm_config(),
        groups,
        available_memory=bytes_per_base_block * 256,
    )

    assert config.num_blocks == 256
    assert get_dsa_group_num_blocks(config, groups[0]) == 768
    assert get_dsa_group_num_blocks(config, groups[1]) == 256
    assert not hasattr(groups[0], "dsa_num_blocks")
    assert not hasattr(groups[1], "dsa_num_blocks")


def test_num_gpu_blocks_override_keeps_base_block_semantics(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    groups = build_dsa_kv_cache_groups(_make_specs())

    config = build_dsa_kv_cache_config(
        _make_vllm_config(override=320),
        groups,
        available_memory=dsa_pool_bytes_per_base_block(groups) * 256,
    )

    assert config.num_blocks == 320
    assert get_dsa_group_num_blocks(config, groups[0]) == 960
    assert get_dsa_group_num_blocks(config, groups[1]) == 320


@pytest.mark.parametrize(
    ("max_num_seqs", "max_model_len", "block_size", "guard", "expected"),
    [
        (1, 131072, 128, 0, 1024),
        (1, 131073, 128, 0, 1025),
        (2, 129, 128, 7, 11),
        (4, 1, 128, 0, 4),
    ],
)
def test_request_limit_blocks_follow_scheduler_limits(
    _mock_dsa_offload_config,
    max_num_seqs: int,
    max_model_len: int,
    block_size: int,
    guard: int,
    expected: int,
) -> None:
    _mock_dsa_offload_config.capacity_mode = "request_limit"
    _mock_dsa_offload_config.capacity_guard_blocks = guard

    assert get_dsa_request_limit_blocks(
        _make_vllm_config(
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
        ),
        block_size=block_size,
    ) == expected

    # BlockPool block 0 is a null mapping and is not allocatable to a request.
    assert get_dsa_request_limit_base_blocks(
        _make_vllm_config(
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
        ),
        block_size=block_size,
    ) == expected + 1


def test_auto_capacity_mode_does_not_impose_a_request_limit(
    _mock_dsa_offload_config,
) -> None:
    assert get_dsa_request_limit_blocks(
        _make_vllm_config(max_num_seqs=1, max_model_len=131072),
        block_size=128,
    ) is None


def test_request_limit_capacity_sets_base_and_indexer_planes(
    monkeypatch,
    _mock_dsa_offload_config,
) -> None:
    """HBM needs one extra physical block for BlockPool's null mapping."""

    _mock_dsa_offload_config.capacity_mode = "request_limit"
    _mock_dsa_offload_config.capacity_guard_blocks = 128
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    groups = build_dsa_kv_cache_groups(_make_specs())

    config = build_dsa_kv_cache_config(
        _make_vllm_config(max_num_seqs=1, max_model_len=131072),
        groups,
        available_memory=dsa_pool_bytes_per_base_block(groups) * 4096,
    )

    # 1024 request blocks + 128 guard + BlockPool null block.
    assert config.num_blocks == 1153
    assert get_dsa_group_num_blocks(config, groups[0]) == 3459
    assert get_dsa_group_num_blocks(config, groups[1]) == 1153


def test_request_limit_capacity_rejects_insufficient_hbm(
    monkeypatch,
    _mock_dsa_offload_config,
) -> None:
    _mock_dsa_offload_config.capacity_mode = "request_limit"
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    groups = build_dsa_kv_cache_groups(_make_specs())

    with pytest.raises(ValueError, match="Insufficient HBM"):
        build_dsa_kv_cache_config(
            _make_vllm_config(max_num_seqs=1, max_model_len=131072),
            groups,
            available_memory=dsa_pool_bytes_per_base_block(groups) * 1024,
        )


def test_request_limit_base_includes_block_pool_null_block(
    monkeypatch,
    _mock_dsa_offload_config,
) -> None:
    """1024 physical blocks expose only 1023 request-allocatable blocks."""

    _mock_dsa_offload_config.capacity_mode = "request_limit"
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 1)
    groups = build_dsa_kv_cache_groups(_make_specs())

    config = build_dsa_kv_cache_config(
        _make_vllm_config(max_num_seqs=1, max_model_len=131072),
        groups,
        available_memory=dsa_pool_bytes_per_base_block(groups) * 1025,
    )

    assert config.num_blocks == 1025
    assert get_dsa_group_num_blocks(config, groups[0]) == 1025
    assert get_dsa_group_num_blocks(config, groups[1]) == 1025


def test_request_limit_rejects_mixed_group_block_sizes(
    monkeypatch,
    _mock_dsa_offload_config,
) -> None:
    _mock_dsa_offload_config.capacity_mode = "request_limit"
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 1)
    groups = build_dsa_kv_cache_groups(_make_specs())
    groups[1].kv_cache_spec = DSAResidentMLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=576,
        sparse_head_dim=(512, 64, 0),
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
    )

    with pytest.raises(ValueError, match="one shared block_size"):
        build_dsa_kv_cache_config(
            _make_vllm_config(max_num_seqs=1, max_model_len=131072),
            groups,
            available_memory=dsa_pool_bytes_per_base_block(groups) * 4096,
        )


def test_request_limit_accepts_exact_non_aligned_hbm_capacity(
    monkeypatch,
    _mock_dsa_offload_config,
) -> None:
    """Request-limit planning must not discard usable tail blocks."""

    _mock_dsa_offload_config.capacity_mode = "request_limit"
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    groups = build_dsa_kv_cache_groups(_make_specs())

    config = build_dsa_kv_cache_config(
        _make_vllm_config(max_num_seqs=1, max_model_len=131073),
        groups,
        available_memory=dsa_pool_bytes_per_base_block(groups) * 1026,
    )

    assert config.num_blocks == 1026
    assert get_dsa_group_num_blocks(config, groups[0]) == 3078
    assert get_dsa_group_num_blocks(config, groups[1]) == 1026


def test_request_limit_override_must_cover_target_capacity(
    monkeypatch,
    _mock_dsa_offload_config,
) -> None:
    _mock_dsa_offload_config.capacity_mode = "request_limit"
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    groups = build_dsa_kv_cache_groups(_make_specs())
    vllm_config = _make_vllm_config(
        max_num_seqs=1,
        max_model_len=131072,
        override=1024,
    )

    with pytest.raises(ValueError, match="below DSA request-limit"):
        build_dsa_kv_cache_config(
            vllm_config,
            groups,
            available_memory=dsa_pool_bytes_per_base_block(groups) * 4096,
        )


def test_request_limit_override_above_target_remains_base_semantic(
    monkeypatch,
    _mock_dsa_offload_config,
) -> None:
    _mock_dsa_offload_config.capacity_mode = "request_limit"
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    groups = build_dsa_kv_cache_groups(_make_specs())

    config = build_dsa_kv_cache_config(
        _make_vllm_config(
            max_num_seqs=1,
            max_model_len=131072,
            override=1280,
        ),
        groups,
        available_memory=dsa_pool_bytes_per_base_block(groups) * 4096,
    )

    assert config.num_blocks == 1280
    assert get_dsa_group_num_blocks(config, groups[0]) == 3840
    assert get_dsa_group_num_blocks(config, groups[1]) == 1280


def test_cross_rank_base_capacity_shrink_preserves_plane_ratio(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    groups = build_dsa_kv_cache_groups(_make_specs())
    config = build_dsa_kv_cache_config(
        _make_vllm_config(),
        groups,
        available_memory=dsa_pool_bytes_per_base_block(groups) * 256,
    )

    old_base_blocks = config.num_blocks
    config.num_blocks = 128
    for tensor in config.kv_cache_tensors:
        tensor.size = tensor.size // old_base_blocks * config.num_blocks

    validate_dsa_kv_cache_config(config)
    assert get_dsa_group_num_blocks(config, groups[0]) == 384
    assert get_dsa_group_num_blocks(config, groups[1]) == 128


def test_prefill_admission_uses_weighted_base_block_cost(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    groups = build_dsa_kv_cache_groups(_make_specs())
    vllm_config = _make_vllm_config(max_model_len=2048)

    assert dsa_max_memory_usage_bytes(
        vllm_config,
        groups,
    ) == 16 * dsa_pool_bytes_per_base_block(groups)


def test_request_limit_raises_startup_hbm_precheck(
    monkeypatch,
    _mock_dsa_offload_config,
) -> None:
    """Profiling must reserve the same request-limit base target as build."""

    _mock_dsa_offload_config.capacity_mode = "request_limit"
    _mock_dsa_offload_config.capacity_guard_blocks = 8
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    groups = build_dsa_kv_cache_groups(_make_specs())

    assert dsa_max_memory_usage_bytes(
        _make_vllm_config(max_num_seqs=2, max_model_len=131072),
        groups,
    ) == 2057 * dsa_pool_bytes_per_base_block(groups)


def test_split_groups_reject_missing_per_layer_indexer() -> None:
    specs = _make_specs(num_layers=2)
    del specs["model.layers.1.self_attn.indexer.k_cache"]

    with pytest.raises(RuntimeError, match="one Indexer cache per"):
        build_dsa_kv_cache_groups(specs)


def test_block_pool_view_resets_every_physical_pool() -> None:
    class _Pool:
        num_gpu_blocks = 4

        def __init__(self, result: bool) -> None:
            self.result = result
            self.reset_calls = 0

        def reset_prefix_cache(self) -> bool:
            self.reset_calls += 1
            return self.result

    first = _Pool(False)
    second = _Pool(True)
    view = DSABlockPoolView([first, second])  # type: ignore[list-item]

    assert view.reset_prefix_cache() is False
    assert first.reset_calls == 1
    assert second.reset_calls == 1


def test_component_admission_checks_each_physical_pool() -> None:
    class _Pool:
        def __init__(self, free_blocks: int) -> None:
            self.free_blocks = free_blocks

        def get_num_free_blocks(self) -> int:
            return self.free_blocks

    coordinator = object.__new__(DSAKVCacheCoordinator)
    coordinator.physical_block_pools = (  # type: ignore[attr-defined]
        _Pool(6),
        _Pool(2),
    )

    assert coordinator.can_allocate((6, 2))
    assert not coordinator.can_allocate((7, 1))
    assert not coordinator.can_allocate((1, 3))
    with pytest.raises(RuntimeError, match="reservations"):
        coordinator.can_allocate((1, 1), reserved_blocks=1)
