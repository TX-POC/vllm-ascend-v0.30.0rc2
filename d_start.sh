#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 手动配置：按现场环境修改本节，无需启动参数。
set -euo pipefail
if [[ $# -ne 0 ]]; then
    echo "Usage: ./d_start.sh; edit configuration inside the script" >&2
    exit 2
fi
python_bin=/usr/local/python3.12.13/bin/python3
model_path=/home/models/glm5-mtp-4l
kvaf_build=/home/c00612283/KVAF-Devlop/build-pd-py312
nic_name=enp34s0f1
local_ip=141.62.38.12
device_ids=4,5
port=8200
tensor_parallel_size=1
data_parallel_size=2
data_parallel_size_local=2
data_parallel_rpc_port=13489
kv_port=31000
max_model_len=8192
max_num_seqs=2
max_num_batched_tokens=4096
gpu_memory_utilization=0.85
diagnostics_requests=0   # 日志：改为2；默认关闭。
verify_kv=false          # 字节校验：P/D均改为true，同时开启上面的日志。
env_file=""              # 可填现场CANN/ATB环境脚本；空值沿用当前环境。
k_address=141.62.38.12:28997
ddr_size_gb=1             # 每个D worker的DDR预算。

# 原生Connector配置：地址、拓扑变化时手动修改上面的字段。
kv_transfer_config=$(cat <<JSON
{
  "kv_connector": "PDKVoffloadConnector",
  "kv_connector_module_path": "vllm_ascend.distributed.kv_transfer.kvaf.pd.connector",
  "kv_role": "kv_consumer",
  "kv_port": $kv_port,
  "kv_connector_extra_config": {
    "diagnostics_requests": $diagnostics_requests,
    "verify_kv": $verify_kv
  }
}
JSON
)
additional_config=$(cat <<JSON
{
  "enable_cpu_binding": true,
  "multistream_overlap_shared_expert": true,
  "c8_enable_reshape_optim": false,
  "kvaf_config": {
    "enabled": true,
    "build_dir": "$kvaf_build",
    "cpu_address": "$k_address",
    "ddr_size_gb": $ddr_size_gb
  }
}
JSON
)

# 加载现场环境，清除代理。
if [[ -n $env_file ]]; then
    set +u
    # shellcheck disable=SC1090
    source "$env_file"
    set -u
fi
unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY all_proxy ALL_PROXY
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ -d "$model_path" ]] || { echo "Model directory missing: $model_path" >&2; exit 1; }
[[ -d "$kvaf_build/python/kvaf" ]] || { echo "KVAF binding missing: $kvaf_build/python/kvaf" >&2; exit 1; }
# shellcheck disable=SC1091
source "$script_dir/third_party/KVOP/env.sh"
export PYTHONPATH="$kvaf_build/python${PYTHONPATH:+:$PYTHONPATH}"

# 通信环境：IP与网卡均使用顶部手动配置。
export VLLM_HOST_IP="$local_ip"
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export HCCL_BUFFSIZE=400
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
export HCCL_ASYNC_ERROR_HANDLING=0
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES="$device_ids"
export ASCEND_GLOBAL_LOG_LEVEL=3

# 每次启动独立保存日志；P/D的metrics目录分开。
stamp=$(date +%Y%m%d_%H%M%S)_$$
log_dir="$script_dir/pd-logs/d/$stamp"
export ASCEND_PROCESS_LOG_PATH="$log_dir/plog"
export PROMETHEUS_MULTIPROC_DIR="/dev/shm/vllm_metrics_d_$stamp"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$PROMETHEUS_MULTIPROC_DIR"
printf 'Starting D: IP=%s NIC=%s devices=%s TP=%s DP=%s HTTP=%s\nLog: %s/server.log\n' \
  "$local_ip" "$nic_name" "$device_ids" "$tensor_parallel_size" "$data_parallel_size" "$port" "$log_dir"
exec > >(tee "$log_dir/server.log") 2>&1

# 原生vLLM启动参数：后续可在这里直接增删参数。
# 固定使用同一个Python解释器调用vllm CLI，避免SDK ABI混用。
exec "$python_bin" -m vllm.entrypoints.cli.main serve "$model_path" \
  --host 0.0.0.0 \
  --port "$port" \
  --tensor-parallel-size "$tensor_parallel_size" \
  --data-parallel-size "$data_parallel_size" \
  --data-parallel-size-local "$data_parallel_size_local" \
  --data-parallel-address "$local_ip" \
  --data-parallel-rpc-port "$data_parallel_rpc_port" \
  --pipeline-parallel-size 1 \
  --distributed-executor-backend mp \
  --enable-expert-parallel \
  --served-model-name glm \
  --seed 1024 \
  --max-model-len "$max_model_len" \
  --max-num-batched-tokens "$max_num_batched_tokens" \
  --max-num-seqs "$max_num_seqs" \
  --gpu-memory-utilization "$gpu_memory_utilization" \
  --trust-remote-code \
  --enforce-eager \
  --quantization ascend \
  --no-enable-prefix-caching \
  --worker-cls vllm_ascend.distributed.kv_transfer.kvaf.pd.worker.PDKVWorker \
  --kv-cache-dtype fp8 \
  --attention_config.indexer_kv_dtype fp8 \
  --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp","enforce_eager":true}' \
  --kv-transfer-config "$kv_transfer_config" \
  --additional-config "$additional_config"
