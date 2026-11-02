import re
from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP
from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.sequence import IntermediateTensors

from vllm_ascend.utils import is_rot_weight_used

logger = init_logger(__name__)

_EXPERT_ID = re.compile(r"\.mlp\.experts\.(\d+)\.")
_ROUTER_WEIGHTS = (".mlp.gate.weight", ".mlp.gate.bias", ".mlp.gate.e_score_correction_bias")


def _validate_expert_pruning(config) -> None:
    original = getattr(config, "original_n_routed_experts", None)
    if original is None:
        return
    retained = config.n_routed_experts
    groups = getattr(config, "n_group", 1)
    selected_groups = getattr(config, "topk_group", 1)
    if not 0 < retained <= original:
        raise ValueError(f"n_routed_experts must be in [1, {original}], got {retained}")
    # Grouped routing still uses the checkpoint's top-k and group settings.
    if retained % groups or retained // groups * selected_groups < config.num_experts_per_tok:
        raise ValueError("Retained experts must divide evenly into n_group and support num_experts_per_tok")
    if getattr(config, "topk_method", None) == "noaux_tc" and retained // groups < 2:
        raise ValueError("noaux_tc routing requires at least two retained experts per group")


def _sync_pruned_mtp_config(vllm_config: VllmConfig) -> None:
    spec = vllm_config.speculative_config
    target = spec.target_model_config.hf_config
    original = getattr(target, "original_n_routed_experts", None)
    if original is None:
        return
    _validate_expert_pruning(target)
    draft = spec.draft_model_config
    draft.hf_config.n_routed_experts = target.n_routed_experts
    draft.hf_config.original_n_routed_experts = original
    draft.hf_text_config = draft.hf_config.get_text_config()
    # The loader's EP expert filter reads this derived config, not hf_config.
    draft.model_arch_config = draft.get_model_arch_config()
    logger.info("MTP expert pruning: retaining %d/%d routed experts per layer", target.n_routed_experts, original)


def _filter_pruned_expert_weights(config, weights: Iterable[tuple[str, torch.Tensor]]):
    original = getattr(config, "original_n_routed_experts", None)
    if original is None:
        yield from weights
        return
    retained = config.n_routed_experts
    for name, weight in weights:
        expert = _EXPERT_ID.search(name)
        if expert is not None and int(expert[1]) >= retained:
            continue  # Includes the removed experts' quantization weights/scales.
        if name.endswith(_ROUTER_WEIGHTS):
            if weight.ndim == 0 or weight.shape[0] not in (original, retained):
                raise ValueError(f"Unexpected router shape for {name}: {tuple(weight.shape)}")
            weight = weight[:retained]
        yield name, weight


@support_torch_compile
class AscendDeepSeekMTP(DeepSeekMTP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        _sync_pruned_mtp_config(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.is_rot_weight_used = is_rot_weight_used(vllm_config)
        if self.is_rot_weight_used:
            self.rot = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if self.is_rot_weight_used:
            hidden_states = self.rot(hidden_states)
        return super().forward(input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds, spec_step_idx)

    def _maybe_set_own_lm_head(self, loaded_weights: set[str]) -> None:
        """Expose a checkpoint-provided MTP head to the runner.

        DeepSeekMTP always constructs ``shared_head``, so module existence does
        not prove head ownership. Treat it as independent only when its weight
        was loaded from the checkpoint.
        """
        mtp_layer_idx = self.model.mtp_start_layer_idx
        own_head_weight = f"model.layers.{mtp_layer_idx}.shared_head.head.weight"
        if own_head_weight not in loaded_weights:
            # GLM-5.3 and friends ship no MTP head, leaving shared_head.head at
            # its allocation-time contents. Record that so the proposer shares
            # the target head instead of inspecting those values.
            self.has_own_lm_head = False
            return

        self.has_own_lm_head = True
        mtp_layer = self.model.layers[str(mtp_layer_idx)]
        self.lm_head = mtp_layer.shared_head.head

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = _filter_pruned_expert_weights(self.config, weights)
        if self.quant_config is not None and (cache_scale_mapper := self.quant_config.get_cache_scale_mapper()):
            weights = cache_scale_mapper.apply(weights)

        weights_mapper = WeightsMapper(
            orig_to_new_prefix={"rot.": f"model.layers.{self.config.num_hidden_layers}.rot."},
        )
        loaded_weights = super().load_weights(weights_mapper.apply(weights))
        self._maybe_set_own_lm_head(loaded_weights)
        return loaded_weights

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        if "rot" in name:
            name = name.replace(f"model.layers.{spec_layer}.rot.", "rot.")
            return name
        return super()._rewrite_spec_layer_name(spec_layer, name)


class AscendGlmMoeDsaForCausalLM(GlmMoeDsaForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        _validate_expert_pruning(config)
        if (original := getattr(config, "original_n_routed_experts", None)) is not None:
            logger.info(
                "GLM expert pruning: retaining %d/%d routed experts per layer", config.n_routed_experts, original
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if vllm_config.use_v2_model_runner and vllm_config.parallel_config.pipeline_parallel_size > 1:
            # EPLB maps and expert weights must describe the same local layers.
            self.num_moe_layers = len(self.moe_layers)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = _filter_pruned_expert_weights(self.config, weights)
        mapper = WeightsMapper(orig_to_new_prefix={"rot.": None})
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=mapper)
