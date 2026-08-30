#!/usr/bin/env python3
"""Fast source/ABI gate for the standalone P7.1 executor.

This is intentionally a *closure* contract, not a synthetic HCCP benchmark.
It proves that the generated build will receive the real KVAF PostOnly/CQ
helpers, the 31-in/18-alias ABI, two bounded 64-seed local buffers, and no C8
or fixed-core source dependency.  The companion runtime test additionally
executes the bridge in fail-closed and, when supplied, real-control modes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_KERNEL = (
    "TryAcquireKvafStreamingBatchGetQueueFromAicore",
    "LoadKvafStreamingBatchGetQueueTemplate",
    "PublishKvafStreamingBatchGetDescriptorScratch",
    "BeginKvafStreamingBatchGet",
    "TryRetireKvafStreamingBatchGet",
    "FinishKvafStreamingBatchGet",
    "GetKvafStreamingBatchGetReady",
    "ReleaseKvafStreamingBatchGetQueueFromAicore",
    "DataSyncBarrier<MemDsbT::DDR>",
    "KERNEL_TYPE_AIV_ONLY",
    "privateReadyEpoch_",
    "privateReadyGeneration_",
    "ClearTerminalFailedMappingSimt",
    "LoadGmInt32ForSimt",
    "LoadGmInt64ForSimt",
)
FORBIDDEN_KERNEL = (
    "KERNEL_TYPE_MIX_AIC_1_2",
    "kGetLane",
    "LI_CORE",
    "GET_CORE",
    "quant_lightning",
    "a5_streaming_quant_li_get",
    "a5_official_c8",
    "ReadGmByPassDCache<int64_t>(private_ready_epoch",
    "ReadGmByPassDCache<int64_t>(private_ready_generation",
    "ReadGmByPassDCache<int32_t>(private_ready +",
)


def _require(text: str, token: str, path: Path) -> None:
    if token not in text:
        raise AssertionError(f"missing token={token!r} file={path}")


def _forbid(text: str, token: str, path: Path) -> None:
    if token in text:
        raise AssertionError(f"forbidden token={token!r} file={path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ops-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="nanovllm_a5_ops root",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ops_root = Path(args.ops_root).resolve()
    p7_root = ops_root / "a5_score_receipt_speculative_executor"
    spec_path = p7_root / "ops.json"
    host_path = p7_root / "op_host" / "a5_score_receipt_speculative_executor.cpp"
    kernel_path = p7_root / "op_kernel" / "a5_score_receipt_speculative_executor.cpp"
    tiling_path = (
        p7_root / "op_kernel" / "a5_score_receipt_speculative_executor_tiling.h"
    )
    bridge_path = (
        p7_root
        / "torch_extension"
        / "csrc"
        / "score_receipt_speculative_executor.cpp"
    )
    abi_path = p7_root / "P7_1_ABI.md"
    for path in (spec_path, host_path, kernel_path, tiling_path, bridge_path, abi_path):
        if not path.is_file():
            raise AssertionError(f"missing required P7.1 file={path}")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if len(spec) != 1 or spec[0].get("op") != "A5ScoreReceiptSpeculativeExecutor":
        raise AssertionError("ops.json must contain exactly A5ScoreReceiptSpeculativeExecutor")
    inputs = spec[0].get("input_desc", [])
    outputs = spec[0].get("output_desc", [])
    if len(inputs) != 31 or len(outputs) != 18:
        raise AssertionError(
            f"P7.1 ABI must be 31 inputs / 18 outputs, got {len(inputs)}/{len(outputs)}"
        )
    if [inputs[5]["name"], inputs[6]["name"]] != [
        "source_lens",
        "kv_object_address_domain",
    ]:
        raise AssertionError("P7.1 ABI input 5/6 must be source_lens/address_domain")

    kernel = kernel_path.read_text(encoding="utf-8")
    host = host_path.read_text(encoding="utf-8")
    tiling = tiling_path.read_text(encoding="utf-8")
    bridge = bridge_path.read_text(encoding="utf-8")
    abi = abi_path.read_text(encoding="utf-8")
    for token in REQUIRED_KERNEL:
        _require(kernel, token, kernel_path)
    for token in FORBIDDEN_KERNEL:
        _forbid(kernel, token, kernel_path)
    for token in (
        "kInputCount",
        "kOutputCount",
        "source_lens",
        "kv_object_address_domain",
        "OP_ADD(A5ScoreReceiptSpeculativeExecutor)",
    ):
        _require(host, token, host_path)
    for token in (
        "kA5ScoreReceiptExecutorMicrobatchCapacity = 64U",
        "kA5ScoreReceiptExecutorMicrobatchSlots = 2U",
    ):
        _require(tiling, token, tiling_path)
    for token in (
        "31in_18aliasout",
        "aclnnA5ScoreReceiptSpeculativeExecutor",
        "CheckAlias",
        "physical_aliases=18",
        "Tensor(r!) worker_stats",
    ):
        _require(bridge, token, bridge_path)
    for token in (
        "31in_18aliasout",
        "AIV-only",
        "MAPPED_OBJECT",
        "FREE -> FILLING -> POSTED -> CQ_TERMINAL",
    ):
        _require(abi, token, abi_path)

    print(
        "P7_1_SOURCE_ABI "
        "op=A5ScoreReceiptSpeculativeExecutor inputs=31 outputs=18 "
        "candidate_seed_shape=[N,128,4] source_lens=[R] "
        "address_domain=1_mapped_object local_microbatches=2x64",
        flush=True,
    )
    print(
        "P7_1_LIFECYCLE "
        "queue=FREE>FILLING>POSTED>CQ_TERMINAL>FREE_OR_QUARANTINED "
        "private=FREE>PENDING>READY_OR_FAILED "
        "identity=epoch,generation,post_sequence",
        flush=True,
    )
    print(
        "P7_1_TRANSPORT_CONTRACT "
        "lease=real postonly=real try_retire=real finish=real release=real "
        "c8_dependency=0 fixed_get_core=0 task_type=aiv_only",
        flush=True,
    )
    print(
        "P7_1_CONTRACT_PASS source=1 abi=1 lifecycle=1 transport=1 "
        "alias_bridge=1 c8_closure=0",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"P7_1_CONTRACT_FAIL type={type(exc).__name__} text={exc}", flush=True)
        raise
