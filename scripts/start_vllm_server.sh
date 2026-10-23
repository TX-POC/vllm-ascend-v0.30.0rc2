#!/usr/bin/env bash
# Run inside the container after install_vllm_in_container.sh.
# Usage: bash scripts/start_vllm_server.sh

# ---- Editable configuration (user-tested GLM-5.3 TP8 baseline) ----
model_path="/home/models/GLM-5.3-w4a4c8-mxfp4"
served_model_name="glm"
host="0.0.0.0"
port=8080
nic_name="enp34s0f1"
local_ip="141.62.38.12"
device_ids="0,1,2,3,4,5,6,7"
tensor_parallel_size=8
data_parallel_size=1
data_parallel_size_local=1
data_parallel_address="$local_ip"
data_parallel_rpc_port=13389
max_num_seqs=16
max_model_len=131072
max_num_batched_tokens=4096
gpu_memory_utilization=0.95
hccl_buffsize=400
enable_dsa_cp=false # true or false: DSA context parallelism.
# ---- End configuration ----

# Use the container shell's CANN/ATB environment, as in the verified launch.
set -euo pipefail

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
export ASCEND_LAUNCH_BLOCKING=0
export HCCL_ASYNC_ERROR_HANDLING=0

printf 'Starting %s on %s:%s\nModel: %s\nAscend logs: %s\n' \
    "$served_model_name" "$host" "$port" "$model_path" "$ASCEND_PROCESS_LOG_PATH"

# Fixed defaults; keep frequently changed settings in the configuration above.
exec vllm serve "$model_path" \
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
    --no-enable-prefix-caching \
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
    --additional-config '{"enable_dsa_cp":'"$enable_dsa_cp"'}'
