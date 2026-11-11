#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 编辑本文件中的 IP、网卡、路径和原生 JSON，然后不带参数执行。
set -euo pipefail
[[ $# == 0 ]] || { echo "Edit this script, then run it without arguments" >&2; exit 2; }

cd /home/c00612283/vllm-ascend-v0.30.0rc2
unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY all_proxy ALL_PROXY
source third_party/KVOP/env.sh
export PYTHONPATH="/home/c00612283/KVAF-Devlop/build-pd-py312/python${PYTHONPATH:+:$PYTHONPATH}"

local_ip=141.62.38.12
nic_name=enp34s0f1
export VLLM_HOST_IP="$local_ip" HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name" TP_SOCKET_IFNAME="$nic_name" HCCL_SOCKET_IFNAME="$nic_name"
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
export HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=1800 HCCL_BUFFSIZE=400
export HCCL_OP_EXPANSION_MODE=AIV HCCL_INTRA_PCIE_ENABLE=1 HCCL_INTRA_ROCE_ENABLE=0
export TORCH_HCCL_ASYNC_ERROR_HANDLING=0
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0 VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export OMP_PROC_BIND=false OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1 ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

dir="$PWD/pd-logs/p0/$(date +%Y%m%d_%H%M%S)_$$"
export ASCEND_PROCESS_LOG_PATH="$dir/plog"
export PROMETHEUS_MULTIPROC_DIR="/dev/shm/vllm_metrics_p0_$$"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$PROMETHEUS_MULTIPROC_DIR"
printf 'p0 server log: %s/server.log\n' "$dir"
exec > >(tee "$dir/server.log") 2>&1

exec /usr/local/python3.12.13/bin/python3 -m vllm.entrypoints.cli.main serve \
  /home/models/GLM-5.3-w4a4c8-mxfp4 \
  --host 0.0.0.0 \
  --port 8100 \
  --tensor-parallel-size 8 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 0 \
  --data-parallel-address 141.62.38.12 \
  --data-parallel-rpc-port 13389 \
  --pipeline-parallel-size 1 \
  --distributed-executor-backend mp \
  --enable-expert-parallel \
  --served-model-name glm \
  --seed 1024 \
  --max-model-len 131072 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --quantization ascend \
  --no-enable-prefix-caching \
  --worker-cls vllm_ascend.distributed.kv_transfer.kvaf.pd.worker.PDKVWorker \
  --block-size 128 \
  --kv-cache-dtype fp8 \
  --attention_config.indexer_kv_dtype fp8 \
  --enforce-eager \
  --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
  --kv-transfer-config '{
  "kv_connector": "PDKVoffloadConnector",
  "kv_connector_module_path": "vllm_ascend.distributed.kv_transfer.kvaf.pd.connector",
  "kv_role": "kv_producer",
  "kv_port": 31000,
  "kv_connector_extra_config": {
    "source_tp0_only": false,
    "timeout": 600,
    "diagnostics_requests": 1,
    "verify_kv": true,
    "decode_endpoints": [
      "141.62.38.24:31000",
      "141.62.38.24:31001",
      "141.62.38.24:31002",
      "141.62.38.24:31003",
      "141.62.38.24:31004",
      "141.62.38.24:31005",
      "141.62.38.24:31006",
      "141.62.38.24:31007",
      "141.62.38.25:31008",
      "141.62.38.25:31009",
      "141.62.38.25:31010",
      "141.62.38.25:31011",
      "141.62.38.25:31012",
      "141.62.38.25:31013",
      "141.62.38.25:31014",
      "141.62.38.25:31015"
    ]
  }
}' \
  --additional-config '{
  "enable_cpu_binding": true,
  "enable_dsa_cp": false,
  "multistream_overlap_shared_expert": true,
  "c8_enable_reshape_optim": false,
  "kvaf_config": {
    "enabled": false,
    "build_dir": "/home/c00612283/KVAF-Devlop/build-pd-py312"
  }
}'
