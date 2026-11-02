#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

# Migrated from mtp_xzw_0903:tests/e2e/dsa_request_test.sh.
# Usage: bash scripts/test_model_curl.sh [both|short|long]
# DSA_BASE_URL=http://127.0.0.1:8080 DSA_MODEL=glm DSA_MAX_TOKENS=512
set -euo pipefail
case_mode="${1:-both}"
case "$case_mode" in both|short|long) ;; *) echo "Usage: $0 [both|short|long]" >&2; exit 2 ;; esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROMPTS_FILE="${DSA_PROMPTS_FILE:-${SCRIPT_DIR}/prompts/dsa_request_prompts.json}"
BASE_URL="${DSA_BASE_URL:-http://127.0.0.1:8080}"
MODEL="${DSA_MODEL:-glm}"
API_KEY="${DSA_API_KEY:-EMPTY}"
MAX_TOKENS="${DSA_MAX_TOKENS:-512}"
OUTPUT_DIR="${DSA_OUTPUT_DIR:-${PWD}/scripts/result/curl_$(date +%Y%m%d_%H%M%S)_$$}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ENDPOINT="${BASE_URL%/}/v1/chat/completions"

command -v curl >/dev/null 2>&1 || {
    echo "curl is required" >&2
    exit 1
}
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
    echo "${PYTHON_BIN} is required" >&2
    exit 1
}
[[ -f "${PROMPTS_FILE}" ]] || {
    echo "Prompt file not found: ${PROMPTS_FILE}" >&2
    exit 1
}

mkdir -p "${OUTPUT_DIR}"
REQUEST_DIR="${OUTPUT_DIR}"

run_case() {
    local case_name="$1"
    local prompt_key="${case_name}_prompt"
    local request_file="${REQUEST_DIR}/${case_name}_request.json"
    local response_file="${OUTPUT_DIR}/${case_name}_response.json"
    local http_code

    "${PYTHON_BIN}" - \
        "${PROMPTS_FILE}" \
        "${prompt_key}" \
        "${request_file}" \
        "${MODEL}" \
        "${MAX_TOKENS}" <<'PY'
import json
import sys
from pathlib import Path

prompts_path, prompt_key, request_path, model, max_tokens = sys.argv[1:]
with Path(prompts_path).open(encoding="utf-8") as file:
    prompts = json.load(file)

prompt = prompts.get(prompt_key)
if not isinstance(prompt, str) or not prompt.strip():
    raise SystemExit(f"{prompt_key} must be a non-empty string")
if prompt_key == "short_prompt":
    long_prompt = prompts.get("long_prompt")
    if not isinstance(long_prompt, str) or prompt not in long_prompt:
        raise SystemExit("short_prompt must be extracted from long_prompt")

request = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "max_tokens": int(max_tokens),
    "stream": False,
}
with Path(request_path).open("w", encoding="utf-8") as file:
    json.dump(request, file, ensure_ascii=False)
PY

    echo "[dsa-e2e] sending ${case_name} prompt to ${ENDPOINT}"
    http_code="$(curl \
        --noproxy '*' \
        --connect-timeout 10 \
        --silent \
        --show-error \
        --output "${response_file}" \
        --write-out '%{http_code}' \
        --header "Authorization: Bearer ${API_KEY}" \
        --header "Content-Type: application/json" \
        --data-binary "@${request_file}" \
        "${ENDPOINT}")"

    if [[ ! "${http_code}" =~ ^2[0-9][0-9]$ ]]; then
        echo "${case_name} request failed with HTTP ${http_code}" >&2
        "${PYTHON_BIN}" -m json.tool "${response_file}" 2>/dev/null || \
            sed -n '1,120p' "${response_file}" >&2
        return 1
    fi

    "${PYTHON_BIN}" - "${case_name}" "${response_file}" <<'PY'
import json
import sys
from pathlib import Path

case_name, response_path = sys.argv[1:]
with Path(response_path).open(encoding="utf-8") as file:
    response = json.load(file)

if "error" in response:
    raise SystemExit(f"{case_name} response contains an error: {response['error']}")
choices = response.get("choices")
if not isinstance(choices, list) or not choices:
    raise SystemExit(f"{case_name} response has no choices")
message = choices[0].get("message")
if not isinstance(message, dict):
    raise SystemExit(f"{case_name} response has no message")

texts = [
    value
    for key in ("reasoning", "reasoning_content", "content")
    if isinstance((value := message.get(key)), str) and value.strip()
]
if not texts:
    raise SystemExit(f"{case_name} response contains no generated text")
if any("\ufffd" in value for value in texts):
    raise SystemExit(f"{case_name} response contains Unicode replacement characters")

print(f"[dsa-e2e] {case_name} response:")
print("\n".join(texts))
print("usage:", json.dumps(response.get("usage", {}), ensure_ascii=False))
print("finish_reason:", choices[0].get("finish_reason"))
if choices[0].get("finish_reason") == "length":
    print("NOTE: output reached max_tokens; increase DSA_MAX_TOKENS if the answer is unfinished.")
PY
    echo "[dsa-e2e] saved ${response_file}"
}

[[ "$case_mode" == long ]] || run_case short
[[ "$case_mode" == short ]] || run_case long
echo "[dsa-e2e] HTTP/text checks passed; inspect answers and compare against KVAF-disabled baseline."
