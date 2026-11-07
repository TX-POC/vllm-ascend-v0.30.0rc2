#!/usr/bin/env bash
# Run inside the container after install_vllm_in_container.sh.
# Usage: bash scripts/start_vllm_server.sh

# ---- Editable configuration (GLM-5.3 TP1 / DP8 with KVAF) ----
model_path="/home/models/GLM-5.3-w4a4c8-mxfp4"
served_model_name="glm"
host="0.0.0.0"
port=8080
nic_name="enp34s0f1"
local_ip="141.62.38.12"
device_ids="0,1,2,3,4,5,6,7"
tensor_parallel_size=1
routed_experts=0 # 0: full model; positive: retain the first N routed experts per MoE layer (GLM only).
data_parallel_size=8
data_parallel_size_local=8
data_parallel_address="$local_ip"
data_parallel_rpc_port=13389
max_num_seqs=16
max_model_len=131072
max_num_batched_tokens=4096
gpu_memory_utilization=0.95
hccl_buffsize=400
enable_dsa_cp=false # true or false: DSA context parallelism.
kvaf_enabled=true # Target hot decode + transient prefill + DDR prefix cache.
kvaf_cpu_address="127.0.0.1:28999"
kvaf_cpu_addresses='{
  "0":"127.0.0.1:28880",
  "1":"127.0.0.1:28881",
  "2":"127.0.0.1:28882",
  "3":"127.0.0.1:28883",
  "4":"127.0.0.1:28884",
  "5":"127.0.0.1:28885",
  "6":"127.0.0.1:28886",
  "7":"127.0.0.1:28887"
}' # JSON: physical NPU ID -> host:port; unlisted cards use kvaf_cpu_address.
kvaf_build_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/third_party/KVAF/build-npu"
kvaf_ddr_size_gb=128 # Per NPU worker; TP8 allocates up to 8 times this amount.
kvaf_decode_backend=batch_get # batch_get (native LI/LIDU/Get) or li_fetch.
kvaf_prefetch_tokens=512 # Total prefetch tokens/request/round; final repair is not capped.
kvaf_intermediate_fetches=8 # Includes the initial hot window; li_fetch only.
kvaf_hot_tokens=16384 # History tokens per request/layer; excludes ABC.
kvaf_put_interval_steps=32 # One shared decode Put window across requests.
kvaf_put_overlap_steps=8 # Extra ABC space for forward/Put overlap; wait only before reuse.
kvaf_verify_put=false # Validation: read back the first prefill/decode block per worker and compare bytes.
profiler_enabled=true # Configure on-demand /start_profile and /stop_profile; does not start capture.
profiler_dir="$PWD/profiling/$(date +%Y%m%d_%H%M%S)"
# ---- End configuration ----

# Use the container shell's CANN/ATB environment, as in the verified launch.
set -euo pipefail
if [[ "$kvaf_enabled" == true ]]; then
    source "$(dirname "${BASH_SOURCE[0]}")/../third_party/KVIP/scripts/env.sh"
fi

# Only this service process drops proxies; container installs can still use them.
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY

if [[ ! -d "$model_path" ]]; then
    printf 'Model directory not found: %s\nCheck model_path and the container volume mounts.\n' "$model_path" >&2
    exit 1
fi

export ASCEND_PROCESS_LOG_PATH="$PWD/$(date +%Y%m%d_%H%M%S)/plog"
export ASCEND_GLOBAL_LOG_LEVEL=3
export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$PROMETHEUS_MULTIPROC_DIR"

export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
export ASCEND_RT_VISIBLE_DEVICES="$device_ids"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export HCCL_IF_IP="$local_ip"
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export HCCL_BUFFSIZE="$hccl_buffsize"
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
# Set ASCEND_LAUNCH_BLOCKING=1 when launching to locate asynchronous operator errors.
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"
export HCCL_ASYNC_ERROR_HANDLING=0

printf 'Starting %s on %s:%s\nModel: %s\nAscend logs: %s\n' \
    "$served_model_name" "$host" "$port" "$model_path" "$ASCEND_PROCESS_LOG_PATH"

additional_config="$(python3 - "$enable_dsa_cp" "$kvaf_enabled" "$kvaf_cpu_address" "$kvaf_build_dir" "$kvaf_ddr_size_gb" "$kvaf_cpu_addresses" "$kvaf_verify_put" "$kvaf_hot_tokens" "$kvaf_decode_backend" "$kvaf_intermediate_fetches" "$kvaf_put_interval_steps" "$kvaf_put_overlap_steps" "$kvaf_prefetch_tokens" <<'PY'
import json
import sys

dsa_cp, enabled, address, build_dir, ddr_gb, addresses, verify, hot_tokens, backend, windows, put_interval, put_overlap, prefetch_tokens = sys.argv[1:]
if any(value not in ("true", "false") for value in (dsa_cp, enabled, verify)):
    raise SystemExit("enable_dsa_cp, kvaf_enabled and kvaf_verify_put must be true or false")
config = {"enable_dsa_cp": dsa_cp == "true", "enable_cpu_binding": True,
          "multistream_overlap_shared_expert": True}
if enabled == "true":
    config["kvaf_config"] = {
        "enabled": True, "cpu_address": address, "cpu_addresses": json.loads(addresses),
        "build_dir": build_dir, "ddr_size_gb": float(ddr_gb), "verify_put": verify == "true",
        "hot_tokens": int(hot_tokens), "decode_backend": backend, "intermediate_fetches": int(windows),
        "prefetch_tokens": int(prefetch_tokens), "put_interval_steps": int(put_interval), "put_overlap_steps": int(put_overlap)
    }
print(json.dumps(config))
PY
)"

hf_overrides="{}"
if [[ "$routed_experts" != "0" ]]; then
    hf_overrides="$(python3 - "$model_path/config.json" "$routed_experts" <<'PY'
import json
import sys

with open(sys.argv[1]) as source:
    config = json.load(source)
if config["model_type"] != "glm_moe_dsa":
    raise SystemExit("Expert pruning currently supports glm_moe_dsa only")
print(json.dumps({"n_routed_experts": int(sys.argv[2]), "original_n_routed_experts": config["n_routed_experts"]}))
PY
)"
fi

profiler_args=()
if [[ "$profiler_enabled" == true ]]; then
    export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
    mkdir -p "$profiler_dir"
    profiler_dir="$(cd "$profiler_dir" && pwd)"
    profiler_config="$(python3 - "$profiler_dir" <<'PYCONFIG'
import json
import sys
print(json.dumps({"profiler": "torch", "torch_profiler_dir": sys.argv[1],
                  "torch_profiler_with_stack": False, "torch_profiler_with_memory": False}))
PYCONFIG
)"
    profiler_args=(--profiler-config "$profiler_config")
    printf 'On-demand CPU/NPU profiler ready; traces: %s\n' "$profiler_dir"
fi

# Fixed defaults; keep frequently changed settings in the configuration above.
exec vllm serve "$model_path" \
    ${profiler_args[@]+"${profiler_args[@]}"} \
    --hf-overrides "$hf_overrides" \
    --host "$host" \
    --port "$port" \
    --distributed-executor-backend mp \
    --data-parallel-size "$data_parallel_size" \
    --data-parallel-size-local "$data_parallel_size_local" \
    --data-parallel-address "$data_parallel_address" \
    --data-parallel-rpc-port "$data_parallel_rpc_port" \
    --tensor-parallel-size "$tensor_parallel_size" \
    --seed 1024 \
    --served-model-name "$served_model_name" \
    --enable-expert-parallel \
    --max-num-seqs "$max_num_seqs" \
    --max-model-len "$max_model_len" \
    --max-num-batched-tokens "$max_num_batched_tokens" \
    --trust-remote-code \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --gpu-memory-utilization "$gpu_memory_utilization" \
    --quantization ascend \
    --async-scheduling \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --kv-cache-dtype fp8 \
    --attention_config.indexer_kv_dtype fp8 \
    --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp","enforce_eager":true}' \
    --additional-config "$additional_config"
