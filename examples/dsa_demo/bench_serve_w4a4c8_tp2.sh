#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run an SSE serving benchmark against an already-running W4A4C8 DSA service.
#
# This is a client-only script.  It never starts, stops, or resets vLLM/NPU
# processes.  The default profile matches serve_dsa_w4a4c8_tp2.sh:
# physical NPU 6,7, TP=2, MAX_NUM_SEQS=1, and max_model_len=131072.

set -euo pipefail

# =========================
# User-overridable settings
# =========================

API_BASE="${API_BASE:-http://127.0.0.1:8000}"
API_KEY="${API_KEY:-EMPTY}"
MODEL_PATH="${MODEL_PATH:-/home/s30013950/glm5-mxfp4-4l}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.1-dsa}"
VLLM_BIN="${VLLM_BIN:-vllm}"

# INPUT_LEN=8192 is intentionally above sparse_activation_tokens=6144, so the
# workload exercises the sparse DSA path rather than only the dense short-prompt
# path.  All lengths are tokenizer token counts, not character counts.
INPUT_LEN="${INPUT_LEN:-8192}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-5}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
NUM_WARMUPS="${NUM_WARMUPS:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
SEED="${SEED:-1024}"

# Disable GLM thinking so the requested output length measures answer-token
# streaming rather than an unbounded reasoning preamble.
EXTRA_BODY="${EXTRA_BODY:-{\"chat_template_kwargs\":{\"enable_thinking\":false}}}"
RESULT_DIR="${RESULT_DIR:-benchmarks/results/dsa}"
RESULT_FILENAME="${RESULT_FILENAME:-c8_tp2_sse_in${INPUT_LEN}_out${OUTPUT_LEN}_c1_$(date +%Y%m%d-%H%M%S).json}"

die() {
    echo "[dsa-bench] error: $*" >&2
    exit 2
}

[[ "${MAX_CONCURRENCY}" == "1" ]] || die \
    "this service was started with MAX_NUM_SEQS=1; set MAX_CONCURRENCY=1"

[[ "${INPUT_LEN}" =~ ^[1-9][0-9]*$ ]] || die "INPUT_LEN must be a positive integer"
[[ "${OUTPUT_LEN}" =~ ^[1-9][0-9]*$ ]] || die "OUTPUT_LEN must be a positive integer"
[[ "${NUM_PROMPTS}" =~ ^[1-9][0-9]*$ ]] || die "NUM_PROMPTS must be a positive integer"
[[ "${NUM_WARMUPS}" =~ ^[0-9]+$ ]] || die "NUM_WARMUPS must be a non-negative integer"

# MAX_MODEL_LEN is the total context limit.  Reserve a small margin for the
# chat template and special tokens; use a restarted service with a larger limit
# for a true 131K-prompt benchmark.
if (( INPUT_LEN + OUTPUT_LEN >= 131072 )); then
    die "INPUT_LEN + OUTPUT_LEN must be below the service max_model_len (131072)"
fi

command -v "${VLLM_BIN}" >/dev/null 2>&1 || die \
    "cannot find ${VLLM_BIN}; run this inside the vLLM Docker environment"

curl -fsS \
    -H "Authorization: Bearer ${API_KEY}" \
    "${API_BASE}/v1/models" >/dev/null || die \
    "service is not reachable at ${API_BASE}/v1/models"

mkdir -p "${RESULT_DIR}"

echo "[dsa-bench] endpoint=${API_BASE}/v1/chat/completions"
echo "[dsa-bench] model=${SERVED_MODEL_NAME} tokenizer=${MODEL_PATH}"
echo "[dsa-bench] stream=true input=${INPUT_LEN} output=${OUTPUT_LEN} prompts=${NUM_PROMPTS} concurrency=${MAX_CONCURRENCY}"
echo "[dsa-bench] result=${RESULT_DIR}/${RESULT_FILENAME}"

# The openai-chat backend sends stream=true and stream_options.include_usage=true
# for every request.  It reports TTFT, TPOT, ITL, and throughput from the SSE
# response; --no-stream is deliberately not used because it is a dataset-loader
# option, not an HTTP streaming switch.
OPENAI_API_KEY="${API_KEY}" "${VLLM_BIN}" bench serve \
    --backend openai-chat \
    --base-url "${API_BASE}" \
    --endpoint /v1/chat/completions \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tokenizer "${MODEL_PATH}" \
    --trust-remote-code \
    --seed "${SEED}" \
    --dataset-name random \
    --random-input-len "${INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --num-prompts "${NUM_PROMPTS}" \
    --request-rate "${REQUEST_RATE}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --num-warmups "${NUM_WARMUPS}" \
    --temperature "${TEMPERATURE}" \
    --ignore-eos \
    --extra-body "${EXTRA_BODY}" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --save-result \
    --save-detailed \
    --result-dir "${RESULT_DIR}" \
    --result-filename "${RESULT_FILENAME}" \
    --disable-tqdm
