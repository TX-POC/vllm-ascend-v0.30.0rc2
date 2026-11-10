#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/config.sh"
[[ $# == 2 || ( $# == 3 && $3 == --dry-run ) ]] || { echo "Usage: $0 p|d 0|1 [--dry-run]" >&2; exit 2; }
role=$1
node=$2
[[ ( $role == p || $role == d ) && ( $node == 0 || $node == 1 ) ]] || exit 2
if [[ $role == p ]]; then
    local_ip=${P_HOSTS[node]}; master=${P_HOSTS[0]}
    tp=8; dp=2; local_dp=1; start_rank=$node; port=8100; rpc=13389
else
    local_ip=${D_HOSTS[node]}; master=${D_HOSTS[0]}
    tp=1; dp=16; local_dp=8; start_rank=$((node * 8)); port=8200; rpc=13489
fi
# Build only native JSON; this does not import vLLM, CANN or KVAF.
transfer=$("$PYTHON" - "$role" "$DIAGNOSTICS_REQUESTS" "$VERIFY_KV" "$COORDINATOR" "$CONTROL_BACKLOG" "${D_HOSTS[@]}" <<'JSON'
import json, sys
role, budget, verify, coordinator, backlog, *hosts = sys.argv[1:]
extra = dict(source_tp0_only=True, timeout=600, diagnostics_requests=int(budget), verify_kv=json.loads(verify))
if role == 'p':
    extra['decode_endpoints'] = [f'{host}:{31000 + node * 8 + local}' for node, host in enumerate(hosts) for local in range(8)]
else:
    extra['initialization_coordinator'] = coordinator
    extra['control_backlog'] = int(backlog)
print(json.dumps(dict(kv_connector='PDKVoffloadConnector', kv_connector_module_path='vllm_ascend.distributed.kv_transfer.kvaf.pd.connector', kv_role='kv_producer' if role == 'p' else 'kv_consumer', kv_port=31000, kv_connector_extra_config=extra)))
JSON
)
additional=$("$PYTHON" - "$role" "$node" "$BUILD" "${K_HOSTS[@]}" "${K_PORTS[@]}" <<'JSON'
import json, sys
role, node, build, *values = sys.argv[1:]
hosts, ports = values[:8], values[8:]
kvaf = dict(enabled=role == 'd', build_dir=build)
config = dict(enable_cpu_binding=True, enable_dsa_cp=False, multistream_overlap_shared_expert=True, c8_enable_reshape_optim=False, kvaf_config=kvaf)
if role == 'd':
    routes = {str(i): f'{hosts[int(node)*4+i//2]}:{ports[i%2]}' for i in range(8)}
    kvaf.update(cpu_address=routes['0'], cpu_addresses=routes, ddr_size_gb=16, npu_core_count=1, jettys_per_core=1)
    config.update(ascend_compilation_config=dict(enable_npugraph_ex=True, enable_static_kernel=False, enable_super_kernel=False), recompute_scheduler_enable='True', finegrained_tp_config=dict(lmhead_tensor_parallel_size=8), enable_mlapo=True)
print(json.dumps(config))
JSON
)
args=("$PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL"
  --host 0.0.0.0 --port "$port" --tensor-parallel-size "$tp"
  --data-parallel-size "$dp" --data-parallel-size-local "$local_dp"
  --data-parallel-start-rank "$start_rank" --data-parallel-address "$master" --data-parallel-rpc-port "$rpc"
  --pipeline-parallel-size 1 --distributed-executor-backend mp --enable-expert-parallel
  --served-model-name glm --seed 1024 --max-model-len 131072 --max-num-batched-tokens 4096
  --max-num-seqs 1 --gpu-memory-utilization 0.85 --trust-remote-code --quantization ascend
  --no-enable-prefix-caching --worker-cls vllm_ascend.distributed.kv_transfer.kvaf.pd.worker.PDKVWorker
  --block-size 128 --kv-cache-dtype fp8 --attention_config.indexer_kv_dtype fp8
  --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp","enforce_eager":true}'
  --kv-transfer-config "$transfer" --additional-config "$additional")
if [[ $node == 1 ]]; then args+=(--headless); fi
if [[ $role == p ]]; then
    args+=(--enforce-eager)
else
    args+=(--async-scheduling --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
      --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILER_DIR\",\"torch_profiler_with_stack\":false}")
fi
if [[ ${3:-} == --dry-run ]]; then printf '%q ' "${args[@]}"; printf '\n'; exit 0; fi
cd "$REPO"
unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY all_proxy ALL_PROXY
source third_party/KVOP/env.sh
export PYTHONPATH="$BUILD/python${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_HOST_IP="$local_ip" HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$NIC" TP_SOCKET_IFNAME="$NIC" HCCL_SOCKET_IFNAME="$NIC"
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
export HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=1800 HCCL_BUFFSIZE=400
export HCCL_OP_EXPANSION_MODE=AIV HCCL_INTRA_PCIE_ENABLE=1 HCCL_INTRA_ROCE_ENABLE=0
export TORCH_HCCL_ASYNC_ERROR_HANDLING=0
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0 VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1 ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
dir="$LOG_ROOT/$role$node/$(date +%Y%m%d_%H%M%S)_$$"
export ASCEND_PROCESS_LOG_PATH="$dir/plog" PROMETHEUS_MULTIPROC_DIR="/dev/shm/vllm_pool_${role}${node}_$$"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$PROMETHEUS_MULTIPROC_DIR"
if [[ $role == d ]]; then mkdir -p "$PROFILER_DIR"; fi
printf 'Server log: %s/server.log\n' "$dir"
exec > >(tee "$dir/server.log") 2>&1
exec "${args[@]}"
