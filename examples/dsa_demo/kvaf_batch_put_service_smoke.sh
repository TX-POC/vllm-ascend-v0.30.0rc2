#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# KVAF Batch Put online-service smoke test.

set -euo pipefail

print_usage() {
    cat <<'EOF'
Usage:
  bash examples/dsa_demo/kvaf_batch_put_service_smoke.sh \
    MODEL_PATH [TENSOR_PARALLEL_SIZE] [PORT] [STARTUP_TIMEOUT_SECONDS]

Example:
  bash examples/dsa_demo/kvaf_batch_put_service_smoke.sh \
    /mnt/weight/GLM-5.1-w4a8 16 8000 900
EOF
}

if [[ $# -lt 1 || $# -gt 4 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_usage
    [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && exit 0
    exit 2
fi

MODEL_PATH="$1"
TENSOR_PARALLEL_SIZE="${2:-16}"
PORT="${3:-8000}"
STARTUP_TIMEOUT_SECONDS="${4:-900}"
SERVED_MODEL_NAME="kvaf-batch-put-smoke"
HOST="127.0.0.1"
API_KEY="EMPTY"
MAX_MODEL_LEN="16384"
MAX_NUM_BATCHED_TOKENS="8192"
PROMPT_REPEAT_COUNT="320"

for value_name in TENSOR_PARALLEL_SIZE PORT STARTUP_TIMEOUT_SECONDS; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer, got: ${value}" >&2
        exit 2
    fi
done

for command_name in curl python3 vllm; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Required command is unavailable: ${command_name}" >&2
        exit 2
    fi
done

if ! python3 -c "import kvaf" >/dev/null 2>&1; then
    echo "The Python environment used by vLLM cannot import kvaf." >&2
    exit 2
fi

BASE_URL="http://${HOST}:${PORT}"
if curl --silent --max-time 2 "${BASE_URL}/v1/models" >/dev/null 2>&1; then
    echo "Port ${PORT} already has a responding service; refusing to reuse it." >&2
    exit 2
fi

ARTIFACT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kvaf-batch-put-smoke.XXXXXX")"
SERVER_LOG="${ARTIFACT_DIR}/server.log"
REQUEST_JSON="${ARTIFACT_DIR}/request.json"
RESPONSE_JSON="${ARTIFACT_DIR}/response.json"
SERVER_PID=""

cleanup() {
    exit_code=$?
    trap - EXIT
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -INT "${SERVER_PID}" 2>/dev/null || true
        for _ in {1..30}; do
            kill -0 "${SERVER_PID}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${SERVER_PID}" 2>/dev/null; then
            kill -TERM "${SERVER_PID}" 2>/dev/null || true
        fi
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    if [[ ${exit_code} -ne 0 ]]; then
        echo "[kvaf-smoke] failed; last server log lines:" >&2
        tail -n 80 "${SERVER_LOG}" >&2 || true
    fi
    echo "[kvaf-smoke] artifacts: ${ARTIFACT_DIR}"
    exit "${exit_code}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

ADDITIONAL_CONFIG='{"enable_sparse_sfa_c8":true,"enable_sparse_li_c8":true,"dsa_sparse_config":{"enabled":true,"split_indexer_cache":true,"indexer_mla_block_ratio":3,"sparse_activation_tokens":6144,"prompt_budget_thresholds":[8192,12288],"resident_budget_tokens":[6144,10240,12288],"max_active_reqs":1,"hot_cpu_block_multiple":1.0,"enable_row_mode_decode_graph":false,"enable_prefill_full_kv_offload":true,"keep_decode_kv_in_hbm":false,"op_backend":"nanovllm_a5","trace_points":{"enabled":false}}}'

echo "[kvaf-smoke] starting model=${MODEL_PATH} tp=${TENSOR_PARALLEL_SIZE}"
echo "[kvaf-smoke] server log: ${SERVER_LOG}"

HCCL_OP_EXPANSION_MODE=AIV \
HCCL_BUFFSIZE=200 \
OMP_NUM_THREADS=10 \
OMP_PROC_BIND=false \
PYTHONHASHSEED=114514 \
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
VLLM_ASCEND_ENABLE_MLAPO=1 \
VLLM_LOGGING_LEVEL=INFO \
vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --api-key "${API_KEY}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --pipeline-parallel-size 1 \
    --data-parallel-size 1 \
    --quantization ascend \
    --enable-expert-parallel \
    --max-num-seqs 1 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --gpu-memory-utilization 0.90 \
    --block-size 128 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --no-async-scheduling \
    --enforce-eager \
    --generation-config vllm \
    --additional-config "${ADDITIONAL_CONFIG}" \
    >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

ready="false"
deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
while ((SECONDS < deadline)); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        wait "${SERVER_PID}" || true
        echo "vLLM exited before becoming ready." >&2
        exit 1
    fi
    if curl --fail --silent --max-time 5 \
        -H "Authorization: Bearer ${API_KEY}" \
        "${BASE_URL}/v1/models" >/dev/null 2>&1; then
        ready="true"
        break
    fi
    sleep 2
done

if [[ "${ready}" != "true" ]]; then
    echo "vLLM did not become ready within ${STARTUP_TIMEOUT_SECONDS}s." >&2
    exit 1
fi

if ! grep -q '\[DSA_C8_RESIDENT_LAYOUT\]' "${SERVER_LOG}"; then
    echo "The service started without the expected DSA C8 resident layout." >&2
    exit 1
fi

python3 - "${REQUEST_JSON}" "${SERVED_MODEL_NAME}" "${PROMPT_REPEAT_COUNT}" <<'PY'
import json
import sys

output_path, model_name, repeat_count = sys.argv[1:]
context = " ".join(
    f"KV cache validation block {index} contains stable context."
    for index in range(int(repeat_count))
)
payload = {
    "model": model_name,
    "prompt": context + "\nReply with exactly: batch put passed",
    "temperature": 0,
    "max_tokens": 8,
    "stream": False,
}
with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(payload, output_file, ensure_ascii=False)
PY

echo "[kvaf-smoke] service ready; sending multi-block prefill request"
http_code="$(curl --silent --show-error --max-time 300 \
    -o "${RESPONSE_JSON}" \
    -w '%{http_code}' \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "Content-Type: application/json" \
    --data-binary "@${REQUEST_JSON}" \
    "${BASE_URL}/v1/completions")"

if [[ "${http_code}" != "200" ]]; then
    echo "Completion request failed with HTTP ${http_code}." >&2
    cat "${RESPONSE_JSON}" >&2 || true
    exit 1
fi

python3 - "${RESPONSE_JSON}" "${SERVED_MODEL_NAME}" <<'PY'
import json
import sys

response_path, expected_model = sys.argv[1:]
with open(response_path, encoding="utf-8") as response_file:
    response = json.load(response_file)
choices = response.get("choices")
if not isinstance(choices, list) or not choices:
    raise SystemExit(f"response has no completion choices: {response}")
text = choices[0].get("text")
if not isinstance(text, str):
    raise SystemExit(f"completion text is invalid: {response}")
actual_model = response.get("model", expected_model)
print(f"[kvaf-smoke] passed model={actual_model} output={text!r}")
PY
