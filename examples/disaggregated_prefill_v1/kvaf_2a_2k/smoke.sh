#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

[[ $# -le 1 ]] || { echo "Usage: $0 [PROMPT_FILE]" >&2; exit 2; }
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
python_bin=${KVAF_PD_PYTHON:-/usr/local/python3.12.13/bin/python3}
if [[ $# == 1 ]]; then
    "$python_bin" "$root/scripts/kvaf_pd_smoke.py" --base-url http://141.62.38.12:8000 \
        --prompt-file "$1" --max-tokens 32 --rounds 1 --timeout 600
else
    curl --fail --silent --show-error --noproxy '*' --connect-timeout 3 --max-time 180 \
        http://141.62.38.12:8000/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"glm","messages":[{"role":"user","content":"请回复测试成功"}],"temperature":0,"max_tokens":32,"stream":false}'
    printf '\n'
fi
