#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Native vllm serve arguments follow --, including MODEL and both JSON configs.
set -euo pipefail
dry_run=false
if [[ ${1:-} == --dry-run ]]; then dry_run=true; shift; fi
if [[ $# -lt 5 || $4 != -- || ( $1 != p && $1 != d ) ]]; then
    echo "Usage: $0 [--dry-run] p|d LOCAL_IP NIC -- MODEL [native vllm serve options...]" >&2
    exit 2
fi
role=$1
local_ip=$2
nic=$3
shift 4
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${KVAF_PD_PYTHON:-python3}
# Validate deployment-critical native fields without loading vLLM/NPU modules.
build_dir=$("$python_bin" - "$role" "$@" <<'PY'
import json
import sys

role, *args = sys.argv[1:]
def option(name):
    values = []
    for i, arg in enumerate(args):
        if arg == name:
            values.append(args[i + 1])
        elif arg.startswith(name + "="):
            values.append(arg.split("=", 1)[1])
    if len(values) != 1:
        raise SystemExit(f"Supply exactly one native {name}")
    return json.loads(values[0])
transfer = option("--kv-transfer-config")
additional = option("--additional-config")
expected = "kv_producer" if role == "p" else "kv_consumer"
if (transfer.get("kv_connector"), transfer.get("kv_role")) != ("PDKVoffloadConnector", expected):
    raise SystemExit("Connector/role does not match the selected PD role")
if transfer.get("kv_connector_module_path") != "vllm_ascend.distributed.kv_transfer.kvaf.pd.connector":
    raise SystemExit("Supply the native PD Connector module path")
if role == "p" and not transfer.get("kv_connector_extra_config", {}).get("decode_endpoints"):
    raise SystemExit("P requires all D worker decode_endpoints")
config = additional.get("kvaf_config", {})
if config.get("enabled") is not (role == "d") or not config.get("build_dir"):
    raise SystemExit("Set kvaf_config.build_dir and enabled=false on P / true on D")
if any(arg.split("=", 1)[0] == "--enable-prefix-caching" for arg in args):
    raise SystemExit("PD requires --no-enable-prefix-caching on P and D")
print(config["build_dir"])
PY
)
args=("$@")
has_option() {
    local arg
    for arg in "${args[@]}"; do
        if [[ $arg == "$1" || $arg == "$1="* ]]; then return 0; fi
    done
    return 1
}
default_option() {
    if ! has_option "$1"; then args+=("$@"); fi
}
default_option --worker-cls vllm_ascend.distributed.kv_transfer.kvaf.pd.worker.PDKVWorker
default_option --host 0.0.0.0
default_option --distributed-executor-backend mp
default_option --seed 1024
default_option --served-model-name glm
default_option --enable-expert-parallel
default_option --trust-remote-code
default_option --no-enable-prefix-caching
default_option --quantization ascend
default_option --kv-cache-dtype fp8
default_option --attention_config.indexer_kv_dtype fp8
default_option --block-size 128
default_option --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp","enforce_eager":true}'
if ! has_option --no-enforce-eager; then default_option --enforce-eager; fi
if [[ $dry_run == true ]]; then
    printf '%q -m vllm.entrypoints.cli.main serve ' "$python_bin"
    printf '%q ' "${args[@]}"
    printf '\n'
    exit 0
fi
test -d "$build_dir/python/kvaf"
source "$root/third_party/KVOP/env.sh"
export PYTHONPATH="$build_dir/python${PYTHONPATH:+:$PYTHONPATH}"
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
export VLLM_HOST_IP="$local_ip"
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic" TP_SOCKET_IFNAME="$nic" HCCL_SOCKET_IFNAME="$nic"
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_PROCESS_LOG_PATH="${ASCEND_PROCESS_LOG_PATH:-$PWD/pd-logs/$role/plog}"
export PROMETHEUS_MULTIPROC_DIR="${PROMETHEUS_MULTIPROC_DIR:-/dev/shm/vllm_metrics_pd_$role}"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$PROMETHEUS_MULTIPROC_DIR"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
export TASK_QUEUE_ENABLE=1 HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=1800
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-400}" HCCL_OP_EXPANSION_MODE=AIV
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0 VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export HCCL_INTRA_PCIE_ENABLE=1 HCCL_INTRA_ROCE_ENABLE=0 HCCL_ASYNC_ERROR_HANDLING=0
export OMP_PROC_BIND=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-100}"
# Use the same interpreter that built/loaded the native KVAF extension.
exec "$python_bin" -m vllm.entrypoints.cli.main serve "${args[@]}"
