#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# GLM-5.1 W4A4C8 / Ascend 950 DSA two-card online service.
#
# This profile uses request-limit capacity planning.  It deliberately does not
# pass --num-gpu-blocks-override: base HBM blocks and the swapped-DRAM arena
# are derived from MAX_NUM_SEQS, MAX_MODEL_LEN and block_size at startup.

set -euo pipefail

# =========================
# User configuration
# =========================

MODEL_PATH="${MODEL_PATH:-/home/s30013950/glm5-mxfp4-4l}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.1-dsa}"
RUN_MODE="${RUN_MODE:-eager}" # eager / graph

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-EMPTY}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
BLOCK_SIZE="${BLOCK_SIZE:-128}" # DSA's current operator contract requires 128.
ENABLE_CHUNKED_PREFILL="true"

# Capacity policy:
# request/DRAM usable = MAX_NUM_SEQS * ceil(MAX_MODEL_LEN / block_size) + guard
# resident HBM base = request/DRAM usable + 1  (BlockPool block 0 is null)
# indexer HBM = resident HBM base * DSA_INDEXER_MLA_BLOCK_RATIO
# ratio=1 is the minimum safe topology for this one-request profile.
DSA_INDEXER_MLA_BLOCK_RATIO="${DSA_INDEXER_MLA_BLOCK_RATIO:-1}"
DSA_CAPACITY_MODE="request_limit"
DSA_CAPACITY_GUARD_BLOCKS="${DSA_CAPACITY_GUARD_BLOCKS:-0}"
DSA_HOT_CPU_BLOCK_MULTIPLE="1" # ignored by capacity_mode=request_limit

DSA_SPARSE_ACTIVATION_TOKENS="6144"
DSA_MAX_ACTIVE_REQS="256"

# Logical NPU 0/1 map to physical NPU 6/7.  The Docker container must already
# expose /dev/davinci6 and /dev/davinci7.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-6,7}"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE="200"
export OMP_NUM_THREADS="10"
export OMP_PROC_BIND="false"
export PYTHONHASHSEED="114514"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_ASCEND_ENABLE_MLAPO="1"
export VLLM_LOGGING_LEVEL="INFO"

case "${RUN_MODE}" in
    eager)
        ENABLE_DSA_GRAPH="false"
        EXECUTION_ARGS=(--enforce-eager)
        ;;
    graph)
        ENABLE_DSA_GRAPH="true"
        EXECUTION_ARGS=(
            --no-enforce-eager
            --compilation-config
            '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}'
        )
        ;;
    *)
        echo "RUN_MODE must be eager or graph, got: ${RUN_MODE}" >&2
        exit 2
        ;;
esac

ADDITIONAL_CONFIG="$(
    cat <<JSON
{"enable_sparse_sfa_c8":true,"enable_sparse_li_c8":true,"dsa_sparse_config":{"enabled":true,"split_indexer_cache":true,"op_backend":"nanovllm_a5","indexer_mla_block_ratio":${DSA_INDEXER_MLA_BLOCK_RATIO},"capacity_mode":"${DSA_CAPACITY_MODE}","capacity_guard_blocks":${DSA_CAPACITY_GUARD_BLOCKS},"sparse_activation_tokens":${DSA_SPARSE_ACTIVATION_TOKENS},"prompt_budget_thresholds":[32768,65536],"resident_budget_tokens":[6144,10240,12288],"max_active_reqs":${DSA_MAX_ACTIVE_REQS},"hot_cpu_block_multiple":${DSA_HOT_CPU_BLOCK_MULTIPLE},"enable_prefill_full_kv_offload":true,"keep_decode_kv_in_hbm":true,"enable_row_mode_decode_graph":${ENABLE_DSA_GRAPH},"trace_points":{"enabled":false,"points":["first_sample"],"ranks":[0]}}}
JSON
)"

echo "[dsa-online] mode=${RUN_MODE} model=${MODEL_PATH}"
echo "[dsa-online] endpoint=http://${HOST}:${PORT}/v1"
echo "[dsa-online] visible_devices=${ASCEND_RT_VISIBLE_DEVICES} tp=${TENSOR_PARALLEL_SIZE}"

exec vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --api-key "${API_KEY}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --pipeline-parallel-size 1 \
    --data-parallel-size 1 \
    --quantization ascend \
    --seed 1024 \
    --enable-expert-parallel \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --block-size "${BLOCK_SIZE}" \
    --no-async-scheduling \
    --stream-interval 1 \
    --generation-config vllm \
    --reasoning-parser glm45 \
    --additional-config "${ADDITIONAL_CONFIG}" \
    --enable-chunked-prefill \
    --scheduler-reserve-full-isl \
    "${EXECUTION_ARGS[@]}"
