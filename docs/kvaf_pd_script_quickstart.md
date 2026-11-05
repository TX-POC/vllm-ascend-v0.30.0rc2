# PD 源码编译与单机联调脚本

当前现场（已有 vLLM、141.62.38.12、P TP2 / D DP2、K 28997）请优先使用[更新后的完整教程](kvaf_pd_existing_vllm_tutorial.md)，包含当天问题汇总、编译检查点和诊断流程。

适用：已下载本仓 v0.30.0rc2-develop、KVAF cpu-pull-hbm-reference，以及匹配的 vLLM v0.30.0 源码。下面用单机 8 卡、已能运行的四层 GLM5.3 W4A4C8 模型联调；生产代码仍支持原始多机拓扑。命令在已验证的 torch/torch_npu/CANN 容器中运行，先加载现场 CANN、ATB、UMDK 环境；安装脚本不会替换匹配的 torch 包，也不下载模型或修改模型层数。

## 1. 源码安装和编译

在框架仓根目录，将以下路径替换成现场路径；所有终端使用同一个 Python：

```bash
export KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3
export KVAF_PD_SRC=/home/c00612283/kvaf-hostcpu-test/sdk
export KVAF_PD_BUILD="$KVAF_PD_SRC/build-pd-py312"
export VLLM_SRC=/path/to/vllm
export MODEL=/path/to/GLM-5.3-w4a4c8-4layer
export LOCAL_IP=192.168.1.10
export NIC=enp34s0f1

bash scripts/install_kvaf_pd_source.sh "$VLLM_SRC" "$KVAF_PD_SRC" "$KVAF_PD_BUILD"
bash scripts/kvaf_pd_k.sh build "$KVAF_PD_SRC" "$KVAF_PD_SRC/build-hostcpu-k"
```

install_kvaf_pd_source.sh 依次检查 Python/torch/torch_npu、安装构建依赖、editable 安装 vLLM 和 vLLM-Ascend、编译 KVOP、复用 install_kvaf_pd.sh 应用 SDK RA 借用补丁并编译完整 binding。最后应看到 PD_SOURCE_INSTALL_OK；CMake/NPU 编译异常立即退出。K 构建可追加原生 CMake 参数，例如 -DKVAF_URMA_ROOT=/path/to/umdk；两类 build 目录独立。

如果仅下载了框架和 KVAF 两个仓库，且镜像内的 vLLM 已匹配当前框架，将第一个参数改为 -，即 bash scripts/install_kvaf_pd_source.sh - "$KVAF_PD_SRC" "$KVAF_PD_BUILD"，保留已安装的 vLLM；脚本会打印其版本供核对。不要使用旧四层分组的 install_kvaf.sh 替代 PD 构建。

## 2. K：机头 CPU 服务

独立终端运行，保留前台日志：

```bash
ss -lntp 'sport = :28999'
bash scripts/kvaf_pd_k.sh start "$KVAF_PD_SRC/build-hostcpu-k" "$LOCAL_IP" --port 28999
```

已有可用 K 服务时直接复用。端口占用先用 ss 确认 PID，不自动终止进程。候选 UB 设备按服务原生参数选择，可在 start 后追加原生选项；TCP 监听成功不代表 UB 数据面通过。CPU 服务启动前不要给它设置 KVAF_REUSE_DEVICE_RA=1。

## 3. D：TP1/DP4

新终端重新导出第 1 节变量。显式选择新 Connector 才启用 PD；两侧关闭本地 prefix caching，MTP3 保留。ddr_size_gb 是每个 D worker 的预算，下例最多四份 1 GiB，实际按机头空闲内存调整。

```bash
D_TRANSFER='{"kv_connector":"PDKVoffloadConnector","kv_connector_module_path":"vllm_ascend.distributed.kv_transfer.kvaf.pd.connector","kv_role":"kv_consumer","kv_port":31000}'
D_ADDITIONAL=$("$KVAF_PD_PYTHON" - "$KVAF_PD_BUILD" "$LOCAL_IP" <<'PY'
import json
import sys
print(json.dumps({"enable_cpu_binding": True, "multistream_overlap_shared_expert": True,
    "kvaf_config": {"enabled": True, "build_dir": sys.argv[1],
                    "cpu_address": sys.argv[2] + ":28999", "ddr_size_gb": 1}}))
PY
)
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 bash scripts/start_vllm_pd.sh d "$LOCAL_IP" "$NIC" -- \
  "$MODEL" --port 8200 --tensor-parallel-size 1 \
  --data-parallel-size 4 --data-parallel-size-local 4 \
  --data-parallel-address "$LOCAL_IP" --data-parallel-rpc-port 13489 \
  --max-num-seqs 2 --max-model-len 8192 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 \
  --kv-transfer-config "$D_TRANSFER" --additional-config "$D_ADDITIONAL"
```

该脚本继承原启动脚本的通信环境、量化、fp8 主 KV/indexer、MTP3 和 EP 参数；初次联调默认 eager，不启用原脚本的按需 profiler。通过原生 CLI 参数可调整模型、并行和内存预算。模型目标层数取已减层模型本身，不传猜测的层数覆盖参数。

## 4. P：TP4/DP1

另一终端导出同一组路径/IP，等待 D 完成 warmup 并开放控制端口：

```bash
P_TRANSFER=$("$KVAF_PD_PYTHON" - "$LOCAL_IP" <<'PY'
import json
import sys
print(json.dumps({"kv_connector": "PDKVoffloadConnector",
    "kv_connector_module_path": "vllm_ascend.distributed.kv_transfer.kvaf.pd.connector",
    "kv_role": "kv_producer", "kv_port": 31000,
    "kv_connector_extra_config": {"decode_endpoints":
        [f"{sys.argv[1]}:{31000 + i}" for i in range(4)]}}))
PY
)
P_ADDITIONAL=$("$KVAF_PD_PYTHON" - "$KVAF_PD_BUILD" <<'PY'
import json
import sys
print(json.dumps({"enable_cpu_binding": True, "multistream_overlap_shared_expert": True,
    "kvaf_config": {"enabled": False, "build_dir": sys.argv[1]}}))
PY
)
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 bash scripts/start_vllm_pd.sh p "$LOCAL_IP" "$NIC" -- \
  "$MODEL" --port 8100 --tensor-parallel-size 4 \
  --data-parallel-size 1 --data-parallel-size-local 1 \
  --data-parallel-address "$LOCAL_IP" --data-parallel-rpc-port 13389 \
  --max-num-seqs 2 --max-model-len 8192 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 \
  --kv-transfer-config "$P_TRANSFER" --additional-config "$P_ADDITIONAL"
```

在 role 前增加 --dry-run 可先打印最终命令，不导入 NPU、不建立连接；例如 bash scripts/start_vllm_pd.sh --dry-run p ...。真实启动自动将原生 additional-config 中的 build_dir/python 加入 PYTHONPATH，使用 KVAF_PD_PYTHON 对应解释器，避免 Python 3.11/3.12 binding 混用。启动变量 ASCEND_RT_VISIBLE_DEVICES 由调用者提供；脚本不改物理卡映射。

## 5. Proxy 与 HTTP 联调

```bash
curl --fail --noproxy '*' "http://$LOCAL_IP:8200/health"
curl --fail --noproxy '*' "http://$LOCAL_IP:8100/health"
bash scripts/kvaf_pd_proxy.sh --host "$LOCAL_IP" --port 8000 \
  --prefiller-hosts "$LOCAL_IP" --prefiller-ports 8100 \
  --decoder-hosts "$LOCAL_IP" --decoder-ports 8200
```

在另一终端发送请求：

```bash
printf '%s\n' '请简短介绍中国的首都。' > /tmp/pd-prompt.txt
"$KVAF_PD_PYTHON" scripts/kvaf_pd_smoke.py --base-url "http://$LOCAL_IP:8000" \
  --prompt-file /tmp/pd-prompt.txt --rounds 3 --max-tokens 16
```

应看到三次 PD_HTTP_PASS，任意 HTTP/响应错误返回非零。该标记只说明经过 Proxy 的接口请求成功；还需保存 P/D/K 日志核对双 Pull、DDR 目录、tail、MTP 和资源回收，并与相同模型的原生结果比较精度。prompt-file 是文本，其真实 token 数取 usage 或同模型 tokenizer；不要用重复字符数冒充 127/128/129 等 token 边界。

## 6. 图模式、生产与退出

eager 通过后，两侧追加原生 --no-enforce-eager --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'，再验证图模式；需要时显式增加 --async-scheduling。P/D 使用一致的 drafter 有效上限，P 由 Proxy 保持 max_tokens=1。生产沿同一脚本传入原始 TP8/DP2/EP16、TP1/DP16/EP16 的现场已验证多机 DP 参数；P decode_endpoints 必须列出全部 16 个 D worker，端口按全局 DP rank 计算。八台 K 的映射使用原生 kvaf_config.cpu_addresses，不自动将单机地址广播到生产。

停止顺序：先停止 Proxy 新请求，等待在途请求结束，再正常退出 P、D，最后退出 K；未知 DMA/Commit 时保活资源，不使用脚本自动 kill/强制回收。详细硬件验收仍按 [Connector 说明](kvaf_pd_connector.md) 和 [机头测试](kvaf_pd_ascend_host_cpu_test.md) 执行。本次云环境仅验证脚本和模拟命令，没有执行现场编译或 NPU 模型运行。
