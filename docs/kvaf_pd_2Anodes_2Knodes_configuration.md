# KVAF PD 分离 2A+2K 安装部署说明

模型为 GLM5.3 W4A4C8。两台 950DT 负责 P/D，两台鲲鹏提供 DDR。启动脚本默认使用已验证的直连配置，每个 D worker 独占一个 K 服务端口。

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
| KVAF | `TX-POC/KVAF-Develop/main` | `813f1750a50b7e6c92d5f5397f2fbc05befef6da` |
| KVIP | `TX-POC/KVAF-KVIP/main` | `d6cfd392f722dd3f96ef6e05175f503aac511dbb` |

## 2. 准备候选源码

从 `TX-POC/vllm-ascend-v0.30.0rc2` 的 `v0.30.0rc2-develop` 分支获取候选源码。在各主机保留旧框架和旧 SDK，候选版本用独立目录。

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2
git fetch origin refs/heads/v0.30.0rc2-develop
git worktree add --detach /home/c00612283/pd-main-framework FETCH_HEAD
git -C /home/c00612283/pd-main-framework log -2 --oneline
```

无法访问远端的主机可以通过部署包离线导入。将其中的 `kvaf-pd-2a2k.bundle` 放到 `/home/c00612283/kvaf-pd-2a2k.bundle`，使用已有基线提交 `f9bf84aab` 的框架仓库执行：

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2
git bundle verify /home/c00612283/kvaf-pd-2a2k.bundle
git fetch /home/c00612283/kvaf-pd-2a2k.bundle HEAD
git worktree add --detach /home/c00612283/pd-main-framework FETCH_HEAD
git -C /home/c00612283/pd-main-framework log -2 --oneline
```

四台主机各准备 KVAF。仅 P/D 准备 KVIP。以下目录用于首次检出，如果已有同版本独立目录，可通过脚本的路径变量指定。

```bash
mkdir -p /home/c00612283/pd-main-deps
cd /home/c00612283/pd-main-deps

git clone --branch main https://github.com/TX-POC/KVAF-Develop.git KVAF
git -C KVAF switch --detach 813f1750a50b7e6c92d5f5397f2fbc05befef6da

# 仅在 P/D 需要 KVIP。
git clone --branch main https://github.com/TX-POC/KVAF-KVIP.git KVIP
git -C KVIP switch --detach d6cfd392f722dd3f96ef6e05175f503aac511dbb
```

## 3. 编译安装

### P/D 容器

两侧各执行一次。使用当前镜像的 Python、torch、torch_npu 和配套 CANN／ATB 环境。候选框架采用 editable 安装，使用独立的测试容器或独立 Python 环境保留旧实例的安装。

```bash
cd /home/c00612283/pd-main-framework
export KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.0
# 按当前镜像路径加载 CANN。已加载时沿用现有环境。
if [[ -f "$ASCEND_CANN_PACKAGE_PATH/set_env.sh" ]]; then
    source "$ASCEND_CANN_PACKAGE_PATH/set_env.sh"
else
    source "$ASCEND_CANN_PACKAGE_PATH/bin/setenv.bash"
fi
bash examples/disaggregated_prefill_v1/kvaf_2a_2k/install_a.sh
```

安装脚本复用镜像中的 vLLM，安装候选 vLLM-Ascend，并用锁定的 KVAF/KVIP 构建 PD SDK。成功标记为 `PD_SDK_IMPORT_OK` 和 `PD_SOURCE_INSTALL_OK`。P/D 的 `build_dir` 指向 `pd-main-deps/KVAF/build-npu`。SDK 安装会打上 checked raw READ 补丁，D 在 NPU Pull 设备完成状态确认后才报告 HBM 就绪。

如果原生算子构建卡在 catlass 下载，先按候选仓库的 `.gitmodules` 在测试容器准备固定版本的 `csrc/third_party/catlass`，确认头文件目录可读后再运行安装脚本。

### K1/K2 宿主机

两台 K 使用同一 KVAF SHA 构建 CPU 服务。按宿主机的 UMDK 安装位置填写头文件和 `liburma` 路径：

```bash
cd /home/c00612283/pd-main-framework
bash examples/disaggregated_prefill_v1/kvaf_2a_2k/build_k.sh \
  -DKVAF_URMA_INCLUDE_DIR=/usr/include/ub/umdk/urma \
  -DKVAF_URMA_LIBRARY=/usr/lib64/liburma.so
ldd /home/c00612283/pd-main-deps/KVAF/build-hostcpu-k/runtime/kvaf_cpu_service
```

`ldd` 确认库都能解析。K 侧只需要 CPU 服务产物以及运行初始化协调器所需的宿主机 Python。P/D 的新绑定和 K 二进制配套切换，涉及 K 服务重启时旧 DDR ticket 会失效。

## 4. 启动顺序

从框架检出目录执行。每个 P/D 容器终端先加载 CANN／ATB 环境。启动脚本默认从 `/home/c00612283/kvaf_vllm_ascend` 查找 `KVAF` 和 `KVIP`。如果按第 2 节检出依赖，启动前执行：

```bash
export KVAF_PD_SDK_ROOT=/home/c00612283/pd-main-deps/KVAF
export KVAF_PD_KVIP_ROOT=/home/c00612283/pd-main-deps/KVIP
```

| 顺序 | 执行位置 | 命令 |
| --- | --- | --- |
| 1 | K1 和 K2 宿主机 | 按下节启动每台 K 的四个端点 |
| 2 | D 容器 | `bash scripts/d_start.sh` |
| 3 | P 容器 | `bash scripts/p_start.sh` |
| 4 | P 容器，另一终端 | `bash examples/disaggregated_prefill_v1/kvaf_2a_2k/proxy_start.sh` |
| 5 | 能访问 Proxy 的终端 | `bash examples/disaggregated_prefill_v1/kvaf_2a_2k/smoke.sh` |

- 两台 K 各确认四个服务均在监听后启动 D。D 的 `initialization_coordinator` 默认留空，直接向映射的 K 端点初始化。
- D 就绪后，P 需要访问 D `31000`～`31007` 和 SDK ticket 返回的动态 HBM 端口。P/D 还需能访问对应 K 的业务端口及 UB 数据面。
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

1. 先用 `smoke.sh` 发短请求，再用 `smoke.sh /path/to/prompt.txt` 发确定性长请求。建议长输入跨越 128 token block 和 4096 token 调度分段，输入和输出总长控制在 `131072` 以内。
2. D 核对八个 `KVAF CPU route`，P 核对多 MR 注册，K 核对对应端口的探测与 Pull 日志。字节验收时可在两端启动脚本同时改为 `diagnostics_requests=1`、`verify_kv=true`，再运行 `python3 scripts/check_kvaf_pd_diagnostics.py --require-ddr --require-compute P日志 D日志`。测性能前恢复默认值。
3. 默认的 `--max-num-seqs` 为 P 侧 20、D 侧 16。压测前核对 K 的 NUMA 可用 DDR 和 D 的 Indexer/MTP HBM。用现有 benchmark 逐级验证实际 D batch、TTFT、TPOT 和 TPS。需要整组 KV 就绪后再 Decode 时，按同步 Decode 说明设置 Proxy、D 和客户端。 [同步 Decode 使用说明](kvaf_pd_performance_tool/DECODE_BATCH.md)
4. 日志位于框架检出的 `pd-logs/{p,d,proxy}/时间_PID/server.log` 以及 K 宿主机的 `pd-pool-logs/{k1,k2,coordinator}/时间_PID/`。停止时先停新流量并排空，按 Proxy → P → D 停止，再处理本轮 K 服务和协调器。回滚恢复旧框架、SDK、K 二进制和启动配置，然后按本说明的启动顺序恢复。

脚本保留 P/D 的 MTP3 和 block size 128，P 使用 TP0-only 传输，D 使用 URMA 和 `batch_get`。本次部署脚本在主机侧核对，昇腾和鲲鹏实机验收仍需按以上步骤进行。

## 脚本入口

- [A 容器安装](../examples/disaggregated_prefill_v1/kvaf_2a_2k/install_a.sh)
- [K 宿主机编译](../examples/disaggregated_prefill_v1/kvaf_2a_2k/build_k.sh)
- [K1/K2 启动](../examples/disaggregated_prefill_v1/kvaf_2a_2k/k_start.sh) · [初始化协调器](../examples/disaggregated_prefill_v1/kvaf_2a_2k/coordinator_start.sh)
- [P 启动](../scripts/p_start.sh) · [D 启动](../scripts/d_start.sh)
- [Proxy 启动](../examples/disaggregated_prefill_v1/kvaf_2a_2k/proxy_start.sh) · [请求冒烟](../examples/disaggregated_prefill_v1/kvaf_2a_2k/smoke.sh)
