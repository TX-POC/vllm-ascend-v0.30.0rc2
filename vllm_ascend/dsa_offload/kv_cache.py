# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA Indexer/MLA 解耦后的 KV-cache 规格与物理容量规划。

DSA target model 固定由一个 Indexer group 和一个 resident MLA group
组成。打开 ``keep_decode_kv_in_hbm`` 且配置 MTP 时，draft layer 保持为
独立 ``is_eagle_group``，不参与 target DSA dump/selection 上下文。
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
)

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec

DSA_KV_BLOCK_COUNT_ALIGNMENT = 128


def _may_override_num_blocks(
    vllm_config: VllmConfig,
    num_blocks: int,
) -> int:
    override = vllm_config.cache_config.num_gpu_blocks_override
    return int(override) if override is not None else num_blocks


@dataclass(frozen=True, kw_only=True)
class DSAIndexerKVSpec(FullAttentionSpec):
    """完整驻留 HBM 的 target Indexer dense plane。"""

    @property
    def real_page_size_bytes(self) -> int:
        return self.block_size * self.num_kv_heads * self.head_size * get_dtype_size(self.dtype)


@dataclass(frozen=True, kw_only=True)
class DSAResidentMLAAttentionSpec(AscendMLAAttentionSpec):
    """target attention resident working set 的 MLA plane。"""


@dataclass(frozen=True)
class DSAKVCacheGroupIds:
    """finalized KVCacheConfig 中 target 与 MTP draft 的稳定 group id。"""

    indexer: int
    resident_mla: int
    draft: tuple[int, ...] = ()


def is_dsa_indexer_spec(spec: KVCacheSpec) -> bool:
    return isinstance(spec, DSAIndexerKVSpec)


def is_dsa_resident_mla_spec(spec: KVCacheSpec) -> bool:
    return isinstance(spec, DSAResidentMLAAttentionSpec)


def is_dsa_target_group(group: KVCacheGroupSpec) -> bool:
    return is_dsa_indexer_spec(group.kv_cache_spec) or is_dsa_resident_mla_spec(group.kv_cache_spec)


def is_dsa_draft_group(group: KVCacheGroupSpec) -> bool:
    return bool(group.is_eagle_group) and not is_dsa_target_group(group)


def has_dsa_split_kv_cache_specs(
    kv_cache_specs: dict[str, KVCacheSpec],
) -> bool:
    return any(is_dsa_indexer_spec(spec) or is_dsa_resident_mla_spec(spec) for spec in kv_cache_specs.values())


def has_dsa_split_kv_cache_groups(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> bool:
    return any(is_dsa_target_group(group) for group in kv_cache_groups)


def get_dsa_kv_cache_group_ids(
    kv_cache_config: KVCacheConfig,
) -> DSAKVCacheGroupIds:
    """解析并校验唯一 target pair 与可选 MTP draft groups。"""

    indexer_group_ids = [
        group_id
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        if is_dsa_indexer_spec(group.kv_cache_spec)
    ]
    resident_group_ids = [
        group_id
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        if is_dsa_resident_mla_spec(group.kv_cache_spec)
    ]
    draft_group_ids = [
        group_id
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        if is_dsa_draft_group(group)
    ]
    foreign_group_ids = [
        group_id
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        if not is_dsa_target_group(group) and not is_dsa_draft_group(group)
    ]
    if (
        len(indexer_group_ids) != 1
        or len(resident_group_ids) != 1
        or foreign_group_ids
    ):
        raise RuntimeError(
            "DSA KV-cache config must contain exactly one target Indexer "
            "group, one target resident MLA group, and only typed MTP draft "
            "groups: "
            f"indexer_group_ids={indexer_group_ids}, "
            f"resident_group_ids={resident_group_ids}, "
            f"draft_group_ids={draft_group_ids}, "
            f"foreign_group_ids={foreign_group_ids}"
        )
    return DSAKVCacheGroupIds(
        indexer=indexer_group_ids[0],
        resident_mla=resident_group_ids[0],
        draft=tuple(draft_group_ids),
    )


def _merge_group_specs(
    layer_specs: dict[str, KVCacheSpec],
) -> KVCacheSpec:
    specs = list(layer_specs.values())
    assert specs
    merged = type(specs[0]).merge(specs)
    assert type(merged) is type(specs[0])
    return merged


def _keep_decode_kv_in_hbm() -> bool:
    from vllm_ascend.ascend_config import get_ascend_config

    return bool(
        get_ascend_config().dsa_offload_config.keep_decode_kv_in_hbm
    )


def build_dsa_kv_cache_groups(
    kv_cache_specs: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """构造 target 两组，并在 MTP keep 模式保留独立 draft groups。"""

    indexer_specs = {name: spec for name, spec in kv_cache_specs.items() if is_dsa_indexer_spec(spec)}
    resident_specs = {name: spec for name, spec in kv_cache_specs.items() if is_dsa_resident_mla_spec(spec)}
    foreign_specs = {
        name: spec
        for name, spec in kv_cache_specs.items()
        if not is_dsa_indexer_spec(spec) and not is_dsa_resident_mla_spec(spec)
    }

    if not indexer_specs or not resident_specs:
        raise RuntimeError(
            "DSA split KV-cache requires exactly one Indexer plane and one "
            "resident MLA plane: "
            f"indexer_layers={len(indexer_specs)}, "
            f"resident_layers={len(resident_specs)}"
        )
    if len(indexer_specs) != len(resident_specs):
        raise RuntimeError(
            "DSA split KV-cache requires one Indexer cache per resident MLA "
            f"layer: indexer_layers={len(indexer_specs)}, "
            f"resident_layers={len(resident_specs)}"
        )
    if foreign_specs and not _keep_decode_kv_in_hbm():
        raise RuntimeError(
            "DSA split KV-cache received foreign specs while "
            "keep_decode_kv_in_hbm=False: "
            f"foreign_specs={tuple(sorted(type(spec).__name__ for spec in foreign_specs.values()))}"
        )

    groups = [
        KVCacheGroupSpec(
            layer_names=list(indexer_specs),
            kv_cache_spec=_merge_group_specs(indexer_specs),
        ),
        KVCacheGroupSpec(
            layer_names=list(resident_specs),
            kv_cache_spec=_merge_group_specs(resident_specs),
        ),
    ]
    groups.extend(
        KVCacheGroupSpec(
            layer_names=[layer_name],
            kv_cache_spec=spec,
            is_eagle_group=True,
        )
        for layer_name, spec in foreign_specs.items()
    )
    return groups


def _get_dsa_ratio() -> int:
    from vllm_ascend.ascend_config import get_ascend_config

    ratio = int(get_ascend_config().dsa_offload_config.indexer_mla_block_ratio)
    if ratio <= 0:
        raise ValueError(f"DSA indexer_mla_block_ratio must be positive, got {ratio}")
    return ratio


def dsa_pool_bytes_per_base_block(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """返回一个 resident base block 对应的 target+draft 总物理字节数。"""

    ratio = _get_dsa_ratio()
    total = 0
    for group in kv_cache_groups:
        if is_dsa_indexer_spec(group.kv_cache_spec):
            weight = ratio
        elif is_dsa_resident_mla_spec(group.kv_cache_spec) or is_dsa_draft_group(group):
            weight = 1
        else:
            raise TypeError(
                "Unexpected KV-cache spec in DSA split layout: "
                f"{type(group.kv_cache_spec).__name__}"
            )
        total += group.kv_cache_spec.page_size_bytes * len(group.layer_names) * weight
    if total <= 0:
        raise ValueError("DSA KV-cache groups contain no physical layers")
    return total


def build_dsa_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """为 worker 构造 ratio 解耦、包含 draft groups 的最终 tensor 布局。"""

    ratio = _get_dsa_ratio()
    bytes_per_base_block = dsa_pool_bytes_per_base_block(kv_cache_groups)
    num_base_blocks = available_memory // bytes_per_base_block
    num_base_blocks = num_base_blocks // DSA_KV_BLOCK_COUNT_ALIGNMENT * DSA_KV_BLOCK_COUNT_ALIGNMENT
    num_base_blocks = _may_override_num_blocks(
        vllm_config,
        num_base_blocks,
    )
    if num_base_blocks <= 0:
        raise ValueError(
            "No KV-cache blocks fit in the DSA split layout: "
            f"available_memory={available_memory}, "
            f"bytes_per_base_block={bytes_per_base_block}"
        )

    tensors: list[KVCacheTensor] = []
    for group in kv_cache_groups:
        group_num_blocks = (
            num_base_blocks * ratio
            if is_dsa_indexer_spec(group.kv_cache_spec)
            else num_base_blocks
        )
        for layer_name in group.layer_names:
            tensors.append(
                KVCacheTensor(
                    size=(group.kv_cache_spec.page_size_bytes * group_num_blocks),
                    shared_by=[layer_name],
                )
            )

    config = KVCacheConfig(
        num_blocks=num_base_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=kv_cache_groups,
    )
    validate_dsa_kv_cache_config(config)
    return config


def dsa_max_memory_usage_bytes(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """计算完整 prefill 加最大 resident budget 所需的物理字节数。"""

    resident_groups = [group for group in kv_cache_groups if is_dsa_resident_mla_spec(group.kv_cache_spec)]
    if len(resident_groups) != 1:
        raise RuntimeError("DSA memory admission requires exactly one resident MLA group")
    resident_group = resident_groups[0]
    resident_page_size = resident_group.kv_cache_spec.page_size_bytes
    resident_bytes_per_layer = resident_group.kv_cache_spec.max_memory_usage_bytes(vllm_config)
    if resident_bytes_per_layer % resident_page_size != 0:
        raise RuntimeError("DSA resident max-memory requirement is not page aligned")

    required_base_blocks = resident_bytes_per_layer // resident_page_size
    if _keep_decode_kv_in_hbm():
        from vllm_ascend.ascend_config import get_ascend_config

        max_budget_tokens = int(
            get_ascend_config().dsa_offload_config.max_resident_budget_tokens
        )
        required_base_blocks += cdiv(
            max_budget_tokens,
            int(resident_group.kv_cache_spec.block_size),
        )
    return required_base_blocks * dsa_pool_bytes_per_base_block(kv_cache_groups)


def _layer_tensor_sizes(
    kv_cache_config: KVCacheConfig,
) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for tensor in kv_cache_config.kv_cache_tensors:
        for layer_name in tensor.shared_by:
            if layer_name in sizes:
                raise RuntimeError(f"DSA layer is backed by more than one KVCacheTensor: {layer_name}")
            sizes[layer_name] = int(tensor.size)
    return sizes


def get_dsa_group_num_blocks(
    kv_cache_config: KVCacheConfig,
    group: KVCacheGroupSpec,
) -> int:
    """从最终 tensor 大小反推 group 的真实物理 block 数。"""

    if not group.layer_names:
        return 0
    tensor_sizes = _layer_tensor_sizes(kv_cache_config)
    page_size = int(group.kv_cache_spec.page_size_bytes)
    block_counts: set[int] = set()
    for layer_name in group.layer_names:
        try:
            tensor_size = tensor_sizes[layer_name]
        except KeyError as exc:
            raise RuntimeError(f"DSA KV-cache tensor is missing for layer {layer_name}") from exc
        if tensor_size % page_size != 0:
            raise RuntimeError(
                "DSA KV-cache tensor is not page aligned: "
                f"layer={layer_name}, tensor_size={tensor_size}, "
                f"page_size={page_size}"
            )
        block_counts.add(tensor_size // page_size)
    if len(block_counts) != 1:
        raise RuntimeError(
            "Layers in one DSA KV-cache group have different capacities: "
            f"group_spec={type(group.kv_cache_spec).__name__}, "
            f"block_counts={sorted(block_counts)}"
        )
    return block_counts.pop()


def validate_dsa_kv_cache_config(
    kv_cache_config: KVCacheConfig,
) -> None:
    """校验 target ratio 及 draft group 类型/容量。"""

    group_ids = get_dsa_kv_cache_group_ids(kv_cache_config)
    indexer_group = kv_cache_config.kv_cache_groups[group_ids.indexer]
    resident_group = kv_cache_config.kv_cache_groups[group_ids.resident_mla]
    indexer_blocks = get_dsa_group_num_blocks(kv_cache_config, indexer_group)
    resident_blocks = get_dsa_group_num_blocks(kv_cache_config, resident_group)
    ratio = _get_dsa_ratio()
    if resident_blocks != int(kv_cache_config.num_blocks):
        raise RuntimeError(
            "DSA resident MLA capacity must equal KVCacheConfig.num_blocks: "
            f"resident={resident_blocks}, num_blocks={kv_cache_config.num_blocks}"
        )
    if indexer_blocks != resident_blocks * ratio:
        raise RuntimeError(
            "DSA Indexer/MLA capacity ratio mismatch: "
            f"indexer={indexer_blocks}, resident={resident_blocks}, ratio={ratio}"
        )
    for draft_group_id in group_ids.draft:
        draft_blocks = get_dsa_group_num_blocks(
            kv_cache_config,
            kv_cache_config.kv_cache_groups[draft_group_id],
        )
        if draft_blocks != resident_blocks:
            raise RuntimeError(
                "DSA MTP draft capacity must equal resident base capacity: "
                f"group_id={draft_group_id}, draft={draft_blocks}, "
                f"resident={resident_blocks}"
            )


def get_dsa_kv_cache_binding_order(
    kv_cache_config: KVCacheConfig,
) -> list[str]:
    """返回 target 同层双 plane 后接独立 draft layer 的稳定绑定顺序。"""

    from vllm.model_executor.models.utils import extract_layer_index

    target_layers: list[tuple[int, int, str]] = []
    draft_layers: list[str] = []
    seen_layers: set[str] = set()
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if is_dsa_resident_mla_spec(spec):
            plane_order = 0
        elif is_dsa_indexer_spec(spec):
            plane_order = 1
        elif is_dsa_draft_group(group):
            for layer_name in group.layer_names:
                if layer_name in seen_layers:
                    raise RuntimeError(f"DSA KV-cache layer appears in more than one group: {layer_name}")
                seen_layers.add(layer_name)
                draft_layers.append(layer_name)
            continue
        else:
            raise RuntimeError(
                "Unexpected KV-cache group in DSA binding: "
                f"{type(spec).__name__}"
            )
        for layer_name in group.layer_names:
            if layer_name in seen_layers:
                raise RuntimeError(f"DSA KV-cache layer appears in more than one group: {layer_name}")
            seen_layers.add(layer_name)
            target_layers.append(
                (
                    extract_layer_index(layer_name),
                    plane_order,
                    layer_name,
                )
            )

    if not target_layers:
        raise RuntimeError("DSA KV-cache binding received no target cache layers")
    return [layer_name for _, _, layer_name in sorted(target_layers)] + draft_layers


def report_dsa_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> None:
    """按最终 tensor 大小输出一次确定性的 DSA HBM 容量报告。"""

    validate_dsa_kv_cache_config(kv_cache_config)
    group_ids = get_dsa_kv_cache_group_ids(kv_cache_config)
    indexer_group = kv_cache_config.kv_cache_groups[group_ids.indexer]
    resident_group = kv_cache_config.kv_cache_groups[group_ids.resident_mla]
    indexer_blocks = get_dsa_group_num_blocks(kv_cache_config, indexer_group)
    resident_blocks = get_dsa_group_num_blocks(kv_cache_config, resident_group)
    indexer_tokens = indexer_blocks * indexer_group.kv_cache_spec.block_size
    resident_tokens = resident_blocks * resident_group.kv_cache_spec.block_size
    max_model_len = int(vllm_config.model_config.max_model_len)
    max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
    block_size = int(resident_group.kv_cache_spec.block_size)

    from vllm_ascend.ascend_config import get_ascend_config

    dsa_config = get_ascend_config().dsa_offload_config
    resident_slots_per_request = cdiv(dsa_config.max_resident_budget_tokens, block_size) * block_size + block_size
    decode_by_mla = resident_tokens // resident_slots_per_request
    indexer_blocks_per_request = cdiv(max_model_len, indexer_group.kv_cache_spec.block_size)
    decode_by_indexer = indexer_blocks // indexer_blocks_per_request
    configured_decode_limit = min(
        decode_by_mla,
        decode_by_indexer,
        max_num_seqs,
    )
    total_bytes = sum(int(tensor.size) for tensor in kv_cache_config.kv_cache_tensors)
    draft_capacities = {
        group_id: get_dsa_group_num_blocks(
            kv_cache_config,
            kv_cache_config.kv_cache_groups[group_id],
        )
        for group_id in group_ids.draft
    }

    logger.info_once(
        "\n"
        "================ DSA HBM CACHE CAPACITY REPORT ================\n"
        "  Split ratio             : indexer:mla = %d:1; base blocks = %s\n"
        "  Allocated HBM KV bytes  : %s bytes (%s GiB)\n"
        "  MLA resident plane      : %s tokens (%s blocks x %s tokens)\n"
        "  Indexer dense plane     : %s tokens (%s blocks x %s tokens)\n"
        "  MTP draft groups        : %s\n"
        "  Batched prefill limit   : %s tokens (dense cache required in both planes)\n"
        "  Sparse decode MLA limit : %s requests (%s resident slots/request)\n"
        "  Dense Indexer limit     : %s requests at max_model_len=%s\n"
        "  Configured decode limit : %s requests (max_num_seqs=%s)\n"
        "=================================================================",
        _get_dsa_ratio(),
        f"{resident_blocks:,}",
        f"{total_bytes:,}",
        format_gib(total_bytes),
        f"{resident_tokens:,}",
        f"{resident_blocks:,}",
        f"{block_size:,}",
        f"{indexer_tokens:,}",
        f"{indexer_blocks:,}",
        f"{indexer_group.kv_cache_spec.block_size:,}",
        str(draft_capacities),
        f"{min(resident_tokens, indexer_tokens):,}",
        f"{decode_by_mla:,}",
        f"{resident_slots_per_request:,}",
        f"{decode_by_indexer:,}",
        f"{max_model_len:,}",
        f"{configured_decode_limit:,}",
        f"{max_num_seqs:,}",
    )
