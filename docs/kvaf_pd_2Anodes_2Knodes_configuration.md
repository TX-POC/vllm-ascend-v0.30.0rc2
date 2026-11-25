# KVAF PD 分离 2A+2K 安装部署说明

模型为 GLM5.3 W4A4C8。两台 950DT 负责 P/D，两台鲲鹏提供 DDR。启动脚本沿用每个 D worker 独占一个 K 服务端口的直连配置。

## 1. 拓扑与版本

| 角色 | 主机和环境 | 并行或服务参数 |
| --- | --- | --- |
| P | `141.62.38.12` 的昇腾容器 | TP8 / DP1 / EP8，eager，HTTP `8100` |
| D | `141.62.38.24` 的昇腾容器 | TP1 / DP8 / EP8，异步调度，Target `FULL_DECODE_ONLY`，HTTP `8200` |
| K1 | `141.62.38.71` 宿主机 | CPU 服务 `28997`、`28999`、`28995`、`28996` |
| K2 | `141.62.38.73` 宿主机 | CPU 服务 `28997`、`28999`、`28995`、`28996` |
| Proxy | P 容器 | `141.62.38.12:8000` |

| D 物理 NPU | K IP | 按卡号依次对应的端口 |
| --- | --- | --- |
| 0–3 | `141.62.38.71` | `28997`、`28999`、`28995`、`28996` |
| 4–7 | `141.62.38.73` | `28997`、`28999`、`28995`、`28996` |

`cpu_addresses` 按物理卡号查找。D 每 worker 配置 `128 GiB` DDR，单台 K 四个端点合计约 `512 GiB`，另需预留系统和 SDK 开销。启动前按实际 NUMA 绑定核对每个节点的可用容量。

| 依赖 | 正式来源 | 本轮固定提交 |
| --- | --- | --- |
| KVAF | `TX-POC/KVAF-Develop/main` | `95a14f1cfaa6d7c76666f5d692df4c4f182360ec` |
| KVIP | `TX-POC/KVAF-KVIP/main` | `d6cfd392f722dd3f96ef6e05175f503aac511dbb` |

## 2. 获取框架源码

P、D 和两台 K 使用同一份框架分支。以下以 `/home/c00612283/kvaf_vllm_ascend` 为部署根目录，框架与外部依赖放在同一级。

```bash
mkdir -p /home/c00612283/kvaf_vllm_ascend
cd /home/c00612283/kvaf_vllm_ascend
git clone --branch v0.30.0rc2-develop https://github.com/TX-POC/vllm-ascend-v0.30.0rc2.git
cd vllm-ascend-v0.30.0rc2
git log -1 --oneline
```

已有 checkout 时在框架根目录执行 `git pull --ff-only origin v0.30.0rc2-develop`，先确认 P、D 和 Proxy 都使用这个目录。容器内需要核对实际挂载路径和启动进程的脚本路径。

## 3. 编译安装

### P/D 容器

使用当前镜像配套的 vLLM、torch、torch_npu 和 ATB。两侧从框架根目录执行：

```bash
export KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.0
bash scripts/build_kvaf_pd_a.sh --full
```

`--full` 安装框架并构建配套 SDK，成功时输出 `PD_SDK_IMPORT_OK` 和 `PD_SOURCE_INSTALL_OK`。只重建 SDK 和 KVIP 时使用 `--sdk-only`，成功时输出 `PD_SDK_IMPORT_OK`。
给定匹配的本地 vLLM 源码时，`--full` 也可加 `--vllm-source /path/to/vllm`。脚本默认在框架同级查找 `KVAF` 和 `KVIP`，可通过 `KVAF_PD_SDK_ROOT` 和 `KVAF_PD_KVIP_ROOT` 指定其他路径。缺失时会检出固定提交；已有目录版本不符时停止。

**构建源码一致性**。锁定提交是构建补丁的基线。PD 构建的源码准备阶段会应用 KVIP 兼容和 checked raw READ 补丁。独立运行 `install_kvip.sh --source-only` 也会应用 KVIP 所需补丁。如果编译输入必须与远端 `main` 完全一致，需等兼容改动进入上游并更新锁定版本。

Boost 使用 `BOOST_INCLUDEDIR`、`BOOST_ROOT` 或系统开发头文件。离线时可在框架同级目录放置官方 `boost_1_83_0.tar.gz`，也可用 `KVAF_PD_BOOST_ARCHIVE` 指定归档。
如果历史补丁留下已知的未跟踪头文件，核对来源后可加 `--remove-stale-headers`。已经完整应用的补丁会保留。
首次编译框架还需要 CATLASS、makeself、Abseil 和 Protobuf。离线环境应提前准备相同版本的本地依赖。

### K1/K2 宿主机

两台 K 使用同一 KVAF SHA 构建 CPU 服务。首次构建前检出 KVAF 到框架同级。按宿主机的 UMDK 安装位置填写头文件和 `liburma` 路径：

```bash
cd /home/c00612283/kvaf_vllm_ascend
git clone --branch main https://github.com/TX-POC/KVAF-Develop.git KVAF
git -C KVAF switch --detach 95a14f1cfaa6d7c76666f5d692df4c4f182360ec
cd vllm-ascend-v0.30.0rc2
bash examples/disaggregated_prefill_v1/kvaf_2a_2k/build_k.sh \
  -DKVAF_URMA_INCLUDE_DIR=/usr/include/ub/umdk/urma \
  -DKVAF_URMA_LIBRARY=/usr/lib64/liburma.so
ldd ../KVAF/build-hostcpu-k/runtime/kvaf_cpu_service
```

用 `ldd` 确认库都能解析。P/D 和 K1/K2 统一切换到 `95a14f1` 的配套产物。新版 P/D 使用 KVHR v9，K 侧需要支持扩展的 UB 候选查询。切换前先排空旧请求，停止旧 P/D，再重启 K 服务。K 重启后旧 DDR ticket 会失效。

## 4. 启动顺序

从框架检出目录执行。每个 P/D 容器终端先加载 CANN／ATB 环境。启动脚本默认从框架同级查找 `KVAF` 和 `KVIP`。`KVAF_PD_BUILD_DIR` 默认从 `KVAF_PD_SDK_ROOT/build-npu` 推导，若之前已设置，确认它指向同一轮产物。现有部署若使用其他目录，可设置相同的 `KVAF_PD_SDK_ROOT` 和 `KVAF_PD_KVIP_ROOT`。

| 顺序 | 执行位置 | 命令 |
| --- | --- | --- |
| 1 | K1 和 K2 宿主机 | 按下节启动每台 K 的四个端点 |
| 2 | D 容器 | `bash scripts/d_start.sh` |
| 3 | P 容器 | `bash scripts/p_start.sh` |
| 4 | P 容器，另一终端 | `bash examples/disaggregated_prefill_v1/kvaf_2a_2k/proxy_start.sh` |
| 5 | 能访问 Proxy 的终端 | 按第 5 节选择普通冒烟或同步 Decode 压测 |

- 两台 K 各确认四个服务均在监听后启动 D。D 的 `initialization_coordinator` 默认留空，直接向映射的 K 端点初始化。
- D 就绪后，P 需要访问 D `31000`～`31007` 和 SDK ticket 返回的动态 HBM 端口。P/D 还需能访问对应 K 的业务端口及 UB 数据面。
- 跨机 NPU Pull 默认从 HCCL rootinfo 选择 CLOS EID。启动时核对 P/D 容器里的 rootinfo 和宿主机 boot ID，以及 K 日志中的候选设备是否符合实际 CPU 绑定。
- `decode_endpoints` 由 D 的实际监听端口决定。P 改为 DP2 时，只要 D 仍为 DP8，就继续列出 `31000`～`31007`。P/D 两端的 `source_tp0_only` 必须一致。
- P、D、Proxy 前台运行。D／P 各自使用 `curl --fail --noproxy '*' --max-time 5 http://本机IP:端口/health` 确认就绪。Proxy 脚本复查两端再监听 `8000`。
- A 侧路径可用 `KVAF_PD_PYTHON`、`KVAF_PD_SDK_ROOT`、`KVAF_PD_BUILD_DIR`、`KVAF_PD_KVIP_ROOT`、`KVAF_PD_MODEL` 覆盖。K 侧用 `KVAF_PD_K_BUILD_DIR`、`KVAF_PD_K_PYTHON`、`KVAF_PD_K_LOG_DIR`。

### K 宿主机指定 UB 设备与 NUMA 绑定

如果现场使用 `start_k.sh` 手动指定 UB 设备，K1 的实机脚本提供了以下端口和设备组合。K2 以本机实际设备和 EID 为准。

| K1 端口 | `--ub-device` | `--eid-index` | Gather |
| --- | --- | --- | --- |
| `28997` | `udmac0d1e3` | `0` | `8` 线程预算，`active` |
| `28999` | `udmac1d1e3` | `0` | `8` 线程预算，`active` |
| `28995` | `udmac0d1e2` | `0` | `8` 线程预算，`active` |
| `28996` | `udmac1d1e2` | `0` | `8` 线程预算，`active` |

直连配置使用下列四个端点。K2 按现场的 UB 设备、EID 和 NUMA 拓扑填写同样的四个端口。旧的两端点模式可使用既有 K 启动脚本和协调器，同时将 D 的 `cpu_addresses` 改为共享端口映射，并设置 `initialization_coordinator`。

在 K1 的 KVAF 源码目录执行以下命令。`cpu_service` 指向本说明固定 KVAF 版本的构建产物。示例沿用现场的 `build-cpu` 路径，使用第 3 节的构建产物时改为 `./build-hostcpu-k/runtime/kvaf_cpu_service`。

```bash
lscpu -e=CPU,SOCKET,NODE
numactl --hardware

cpu_service=./build-cpu/runtime/kvaf_cpu_service
numa0=0  # 按第一块 UB 设备所在的 NUMA 节点修改
numa1=1  # 按第二块 UB 设备所在的 NUMA 节点修改
numa2=0  # 按第三块 UB 设备所在的 NUMA 节点修改
numa3=1  # 按第四块 UB 设备所在的 NUMA 节点修改
log_dir="/home/c00612283/pd-pool-logs/k1/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$log_dir"

start_k_service() {
    local node=$1 port=$2 device=$3
    nohup numactl --cpunodebind="$node" --membind="$node" \
        "$cpu_service" --listen 0.0.0.0 --port "$port" \
        --ub-device "$device" --eid-index 0 \
        --gather-threads 8 --gather-mode active --memory-page-mode default \
        > "$log_dir/kserver_${port}.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$log_dir/kserver_${port}.pid"
}

start_k_service "$numa0" 28997 udmac0d1e3
start_k_service "$numa1" 28999 udmac1d1e3
start_k_service "$numa2" 28995 udmac0d1e2
start_k_service "$numa3" 28996 udmac1d1e2

ss -lntp | grep -E ':(28995|28996|28997|28999)[[:space:]]'
```

`--ub-device` 和 `--eid-index` 选择 UB 端点。`numactl` 限制进程及其 Gather 线程的 CPU 调度范围，以及 DDR 分配的 NUMA 节点。`--gather-threads 8` 是线程预算，`active` 表示主动轮询。手工启动后确认每个端口的监听 PID 与对应的 `.pid` 一致，并检查各自日志中的 `LISTEN`。D 初始化后可用 `numastat -p <PID>` 核对 DDR 的实际分布。

## 5. 请求、验收和回滚

1. 普通 Proxy 模式先用 `smoke.sh` 发短请求，再用 `smoke.sh /path/to/prompt.txt` 发确定性长请求。同步模式用 `benchmark.py --levels 1` 验证每个 D Rank 的首步 batch。建议长输入跨越 128 token block 和 4096 token 调度分段，输入和输出总长控制在 `131072` 以内。
2. D 核对八个 `KVAF CPU route`，P 核对多 MR 注册，K 核对对应端口的探测与 Pull 日志。字节验收时可在两端启动脚本同时改为 `diagnostics_requests=1`、`verify_kv=true`，再运行 `python3 scripts/check_kvaf_pd_diagnostics.py --require-ddr --require-compute P日志 D日志`。测性能前恢复默认值。
3. 默认的 `--max-num-seqs` 为 P 侧 20、D 侧 16。压测前核对 K 的 NUMA 可用 DDR 和 D 的 Indexer/MTP HBM。用现有 benchmark 逐级验证实际 D batch、TTFT、TPOT 和 TPS。需要整组 KV 就绪后再 Decode 时，在 Proxy 侧设置 `enabled: true`，使用能够读取运行中 D 容量的同版 Proxy 和 benchmark。 [同步 Decode 使用说明](kvaf_pd_performance_tool/DECODE_BATCH.md)
4. 日志位于框架检出的 `pd-logs/{p,d,proxy}/时间_PID/server.log` 以及 K 宿主机的 `pd-pool-logs/{k1,k2,coordinator}/时间_PID/`。停止时先停新流量并排空，按 Proxy → P → D 停止，再处理本轮 K 服务和协调器。回滚恢复旧框架、SDK、K 二进制和启动配置，然后按本说明的启动顺序恢复。

脚本保留 P/D 的 MTP3 和 block size 128，P 使用 TP0-only 传输，D 使用 URMA 和 `batch_get`。更新构建路径或依赖版本后，按以上步骤重新验收。

## 脚本入口

- [A 容器安装](../scripts/build_kvaf_pd_a.sh)
- [K 宿主机编译](../examples/disaggregated_prefill_v1/kvaf_2a_2k/build_k.sh)
- [K1/K2 启动](../examples/disaggregated_prefill_v1/kvaf_2a_2k/k_start.sh) · [初始化协调器](../examples/disaggregated_prefill_v1/kvaf_2a_2k/coordinator_start.sh)
- [P 启动](../scripts/p_start.sh) · [D 启动](../scripts/d_start.sh)
- [Proxy 启动](../examples/disaggregated_prefill_v1/kvaf_2a_2k/proxy_start.sh) · [请求冒烟](../examples/disaggregated_prefill_v1/kvaf_2a_2k/smoke.sh)
