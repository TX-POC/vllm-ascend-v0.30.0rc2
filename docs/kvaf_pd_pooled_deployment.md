# 四智算节点与八鲲鹏节点的 PD 部署指南

本方案使用两个跨机 DP/EP 集群和八台 K 通算服务器，不修改框架计算、KVAF SDK 或 KV 内存布局。
这是扩容部署模板，尚未完成该规模的硬件联调；单 P/单 D 的成功不能替代本方案验收。
基础安装排障见 [PD 部署指南](kvaf_pd_deployment.md)，字段含义见 [配置说明](kvaf_pd_configuration.md)。

## 1. 拓扑与独立脚本

每台智算服务器有 8 张 950DT，每台 K 服务器假定两个 CPU socket。
以下 IP 按已有地址尾号递增，使用前确认本机地址、网卡和 NUMA 拓扑。

| 角色 | 示例地址 | 全局 TP / DP / EP | 本地 DP；全局起始 rank | HTTP | 独立脚本 |
| --- | --- | --- | --- | --- | --- |
| P0 | 141.62.38.12 | TP8 / DP2 / EP16 | 1；0 | 8100 | [p0_start.sh](../examples/disaggregated_prefill_v1/kvaf_pooled/p0_start.sh) |
| P1 | 141.62.38.13 | TP8 / DP2 / EP16 | 1；1 | headless | [p1_start.sh](../examples/disaggregated_prefill_v1/kvaf_pooled/p1_start.sh) |
| D0 | 141.62.38.24 | TP1 / DP16 / EP16 | 8；0 | 8200 | [d0_start.sh](../examples/disaggregated_prefill_v1/kvaf_pooled/d0_start.sh) |
| D1 | 141.62.38.25 | TP1 / DP16 / EP16 | 8；8 | headless | [d1_start.sh](../examples/disaggregated_prefill_v1/kvaf_pooled/d1_start.sh) |
| K0 | 141.62.38.26 | 两个 CPU/socket 服务 | 对应 D0 NPU 0、1 | 28997 / 28999 | [k0_start.sh](../examples/disaggregated_prefill_v1/kvaf_pooled/k0_start.sh) |
| K1 | 141.62.38.27 | 两个 CPU/socket 服务 | 对应 D0 NPU 2、3 | 28997 / 28999 | [k1_start.sh](../examples/disaggregated_prefill_v1/kvaf_pooled/k1_start.sh) |
| K2 | 141.62.38.28 | 两个 CPU/socket 服务 | 对应 D0 NPU 4、5 | 28997 / 28999 | [k2_start.sh](../examples/disaggregated_prefill_v1/kvaf_pooled/k2_start.sh) |
| K3 | 141.62.38.29 | 两个 CPU/socket 服务 | 对应 D0 NPU 6、7 | 28997 / 28999 | [k3_start.sh](../examples/disaggregated_prefill_v1/kvaf_pooled/k3_start.sh) |
| K4 | 141.62.38.30 | 两个 CPU/socket 服务 | 对应 D1 NPU 0、1 | 28997 / 28999 | [k4_start.sh](../examples/disaggregated_prefill_v1/kvaf_pooled/k4_start.sh) |
| K5 | 141.62.38.31 | 两个 CPU/socket 服务 | 对应 D1 NPU 2、3 | 28997 / 28999 | [k5_start.sh](../examples/disaggregated_prefill_v1/kvaf_pooled/k5_start.sh) |
| K6 | 141.62.38.32 | 两个 CPU/socket 服务 | 对应 D1 NPU 4、5 | 28997 / 28999 | [k6_start.sh](../examples/disaggregated_prefill_v1/kvaf_pooled/k6_start.sh) |
| K7 | 141.62.38.33 | 两个 CPU/socket 服务 | 对应 D1 NPU 6、7 | 28997 / 28999 | [k7_start.sh](../examples/disaggregated_prefill_v1/kvaf_pooled/k7_start.sh) |
| 协调器 | K0：141.62.38.26 | 单实例 | 初始化门控 | 28998 | [coordinator_start.sh](../examples/disaggregated_prefill_v1/kvaf_pooled/coordinator_start.sh) |
| Proxy | P0：141.62.38.12 | 单实例 | P0:8100 → D0:8200 | 8000 | [proxy_start.sh](../examples/disaggregated_prefill_v1/kvaf_pooled/proxy_start.sh) |

共 14 份完整脚本，每份直接声明本机环境、路径、启动命令和配置。
P/D 使用原生 CLI 和内联 JSON；K、协调器、Proxy 包含各自后台启动与就绪检查。
可将单份脚本复制到对应服务器后独立执行，不需要同目录的其他模板文件。
运行仍需要本机已安装的框架、KVOP、SDK 和模型。

本版本移除了旧的 config.sh、start_npu.sh、k_start.sh；不再通过共享配置或编号参数转发。
已有用户先备份本地脚本，再将配置填写到对应新脚本。所有入口均不接收参数，也不支持 --dry-run。
直接编辑并用 `bash -n` 检查语法，JSON 可复制给 `python3 -m json.tool` 单独校验。

P 的 TP8 × DP2 和 D 的 TP1 × DP16 分别形成各自的 EP16；没有额外的 --expert-parallel-size 参数。
P1/D1 必须通过全局 DP 大小、start-rank 和主地址加入对应集群。
将两机分别运行成独立 DP1/DP8 实例不会形成所需跨机 EP16。

## 2. K 映射、全建链与容量

“一个 NPU 对应一个 CPU”在本方案中特指每个 D NPU 独占一个 K 服务，对应一个 CPU socket 的内存分区。
P 不另分配 K Runtime；P 将源注册到目标 D 拥有的 K region。
若要求全部 32 个 P/D NPU 各独占一个 CPU socket，需要 32 个 socket，八台双路 K 不足。

每台 D 的 additional-config.kvaf_config.cpu_addresses 显式列出本机物理 NPU ID 0～7 的完整映射。
D1 全局 DP rank 从 8 开始，但物理 NPU 键仍是本机 0～7。
每服务 16 GiB、每 K 主机约 32 GiB，总计约 256 GiB，另预留 SDK 开销。
这是固定分区接入；当前没有跨 K 的全局分配、容量均衡、故障接管或自动数据迁移。
一个 D 的 region 用满时不会自动借用其他 K 的空闲 region。

八份 K 脚本各自有 `numa_nodes=(0 1)`。先在对应宿主机检查：

```bash
lscpu -e=CPU,SOCKET,NODE
numactl --hardware
```

NUMA 节点号不等于 socket 号。将本机脚本改为两个完整 socket 的节点集合，例如
`numa_nodes=("0-3" "4-7")`，不能盲用默认值；不同 K 可以填写不同集合。
numactl 绑定不等同于固定 SDK 的 URMA/EID 候选，仍需验证实际内存归属和 CPU Pull 的 UB 路径。

四份 P/D 脚本显式使用 source_tp0_only=false，采用原始全建链模式；省略字段也默认为 false。
全部 P TP rank 按原规则连接全部 16 个 D endpoint，主 KV、Indexer 和 Scale 继续按原传输规则分工。
这与旧扩容模板中的 true 不同。临时 TP0 单源模式仍可使用，但须在四份脚本同时改为 true 后重启 P/D。
切换不会改变原生 KV 内存布局；全建链的队列资源、初始化时间和稳定性须按新规模重新验收。

按 79 层估算，全建链中每个 P rank 每路径 16 个源组、1264 个逐层 MR；不是 Jetty 实测数量。
多 MR 通道复用依赖匹配 SDK，不能把逐层 MR 数当作实际队列数。

## 3. 安装与启动前修改

四个智算容器保持相同模型、tokenizer、CANN/torch_npu、框架提交及 SDK 构建版本。
八个 K 宿主机使用同一 SDK 提交。基准 SDK 为 codex/cpu-pull-hbm-reference 的
`74c3fc491a257bae73356da54eafb08e1e9e4e45`，不可混用旧 Pull 协议。

四个 NPU 容器加载匹配 CANN 环境后，各执行：

```bash
(
set -euo pipefail
cd /home/c00612283/vllm-ascend-v0.30.0rc2
export KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3
export SOC_VERSION=ascend950dt_9582
export COMPILE_CUSTOM_KERNELS=1 MAX_JOBS=8
bash scripts/install_kvaf_pd_source.sh - \
  /home/c00612283/KVAF-Devlop \
  /home/c00612283/KVAF-Devlop/build-pd-py312
)
```

同版本 Ascend 原生扩展和 KVOP 已成功构建时，可用基础指南的 SDK-only 构建步骤。
检查实际导入路径及 listen_hbm、cancel_raw_addr、ReadyToken.mr_id 等接口。
安装入口保留现有 HCCL device RA 借用补丁，本次模板调整不修改 SDK。

八个 K 宿主机需有匹配 UMDK、liburma、编译工具、numactl、Python3，以及框架脚本目录。各执行：

```bash
(
set -euo pipefail
cd /home/c00612283/vllm-ascend-v0.30.0rc2
bash scripts/kvaf_pd_k.sh build \
  /home/c00612283/KVAF-Devlop \
  /home/c00612283/KVAF-Devlop/build-hostcpu-k \
  -DKVAF_URMA_INCLUDE_DIR=/usr/include/ub/umdk/urma \
  -DKVAF_URMA_LIBRARY=/usr/lib64/liburma.so \
  '-DCMAKE_CXX_FLAGS=-DURMA_ACCESS_REMOTE_READ=URMA_ACCESS_READ -DURMA_ACCESS_REMOTE_WRITE=URMA_ACCESS_WRITE -DURMA_ACCESS_REMOTE_ATOMIC=URMA_ACCESS_ATOMIC'
)
```

K0 的协调器及后台就绪检查只依赖 Python 标准库，使用宿主机实际 python3。
证书、离线依赖、原生扩展问题按基础指南处理；RunAicpuCommInit 缺符号属于 HCCL/AICPU 配套问题。

逐份编辑以下内容，不会从其他脚本自动继承：

| 脚本 | 需核对的配置 |
| --- | --- |
| p0/p1 | 本机 IP/网卡；P0 DP 主地址；16 个 decode_endpoints；Python、模型、KVOP、SDK 路径 |
| d0/d1 | 本机 IP/网卡；D0 DP 主地址；协调器地址；本机完整 cpu_addresses；相同模型和 SDK 路径 |
| k0～k7 | 本机 local_ip、numa_nodes、两个端口、K_BIN、日志路径 |
| coordinator | K0 local_ip、port、COORDINATOR 路径；使用宿主机 command -v python3 |
| proxy | 本机地址、Python/框架路径、P0:8100、D0:8200 |

D 保留已验证单机配置中的 FULL_DECODE_ONLY、异步调度、MLAPO、LMHead TP8、npugraph_ex、MTP eager 和 profiler。
profiler 输出 `/home/d00843668/profiling_no_pooling` 须可创建和写入；扩容到 EP16 后仍需重新验证。
P 使用 eager。两侧均关闭 enable_dsa_cp 和 c8_enable_reshape_optim，禁用 prefix caching；MTP tokens=3。
四份脚本 diagnostics_requests=1、verify_kv=true，用于验收；性能测试前应同时改为 0/false。

D 的 control_backlog=128 是每个框架控制监听器的 TCP 等待队列，允许整数 1～4096；P 不配置。
它不代表并行度或 Jetty 数，内核 somaxconn 仍可能限制实际容量；修改后须重启 D listener。
本次仅调整脚本无需重编译框架或 SDK；健康 K/协调器可保留。改变 K 映射或服务时必须排空旧会话。

## 4. 网络及进程检查

- P 两机之间、D 两机之间均需支持跨机 HCCL/EP 通信，并具备所需 NPU/UB 权限和配套驱动。
- P DP 主地址 P0:13389，D DP 主地址 D0:13489；HCCL 及原生 DP 通信还会使用其他端口。
- 所有 P worker 可访问 D0:31000～31007、D1:31008～31015，以及 SDK ticket 返回的动态 HBM 端口。
- P/D 可访问对应 K 的 28997/28999；全部 D 可访问 K0:28998。
- Proxy 可访问 P0:8100 和 D0:8200；P1/D1 为 headless，不填写独立 HTTP endpoint。
- 在实际启动 NPU 的 shell 检查 soft/hard nofile，建议至少 65536；修改其他 shell 不会影响本终端。
- 先排空并停止旧 Proxy/P/D。替换 K 前确认全部旧 DMA 已停止，且对应端口已释放。

K/协调器在宿主机运行，P/D/Proxy 在智算容器运行。下文命令均从框架源码目录执行。
业务 IP 必须属于当前网络命名空间；Cannot assign requested address 需检查宿主机/容器及地址配置。

## 5. 按节点启动

### 5.1 八台 K 宿主机

在表中对应的服务器执行一条命令；八条命令分别属于八台主机：

| 主机 | 启动命令 |
| --- | --- |
| K0 .26 | `bash examples/disaggregated_prefill_v1/kvaf_pooled/k0_start.sh` |
| K1 .27 | `bash examples/disaggregated_prefill_v1/kvaf_pooled/k1_start.sh` |
| K2 .28 | `bash examples/disaggregated_prefill_v1/kvaf_pooled/k2_start.sh` |
| K3 .29 | `bash examples/disaggregated_prefill_v1/kvaf_pooled/k3_start.sh` |
| K4 .30 | `bash examples/disaggregated_prefill_v1/kvaf_pooled/k4_start.sh` |
| K5 .31 | `bash examples/disaggregated_prefill_v1/kvaf_pooled/k5_start.sh` |
| K6 .32 | `bash examples/disaggregated_prefill_v1/kvaf_pooled/k6_start.sh` |
| K7 .33 | `bash examples/disaggregated_prefill_v1/kvaf_pooled/k7_start.sh` |

每份脚本启动本机两个独立服务，均打印 SERVICE_READY 后返回 0。检查实际绑定与内存归属。
仅在 K0 再运行一个协调器：

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2
bash examples/disaggregated_prefill_v1/kvaf_pooled/coordinator_start.sh
netstat -lntp 2>/dev/null | grep -E ':28997[[:space:]]|:28998[[:space:]]|:28999[[:space:]]'
```

### 5.2 D0 与 D1 容器

两个终端先后启动，使两机均加入集群，然后等待就绪。不要等 D0 独自 ready 才启动 D1。

```bash
# D0 容器
cd /home/c00612283/vllm-ascend-v0.30.0rc2
bash examples/disaggregated_prefill_v1/kvaf_pooled/d0_start.sh
```

```bash
# D1 容器；headless，全局 DP start-rank=8
cd /home/c00612283/vllm-ascend-v0.30.0rc2
bash examples/disaggregated_prefill_v1/kvaf_pooled/d1_start.sh
```

另开终端等全部 16 个 worker 完成初始化和图 warmup：

```bash
curl --fail --noproxy '*' --connect-timeout 3 --max-time 10 -i http://141.62.38.24:8200/health
# D0 宿主机
netstat -lntp 2>/dev/null | grep -E ':3100[0-7][[:space:]]'
# D1 宿主机
netstat -lntp 2>/dev/null | grep -E ':310(0[89]|1[0-5])[[:space:]]'
```

D1 不需要独立 HTTP API；控制端口按全局 DP rank 计算，两机不会都从 31000 开始。

### 5.3 P0 与 P1 容器

```bash
# P0 容器
cd /home/c00612283/vllm-ascend-v0.30.0rc2
bash examples/disaggregated_prefill_v1/kvaf_pooled/p0_start.sh
```

```bash
# P1 容器；headless，全局 DP start-rank=1
cd /home/c00612283/vllm-ascend-v0.30.0rc2
bash examples/disaggregated_prefill_v1/kvaf_pooled/p1_start.sh
```

两机均加入后，另开终端检查 P0：

```bash
curl --fail --noproxy '*' --connect-timeout 3 --max-time 10 -i http://141.62.38.12:8100/health
```

### 5.4 Proxy：P0 容器

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2
bash examples/disaggregated_prefill_v1/kvaf_pooled/proxy_start.sh
curl --fail --noproxy '*' --max-time 5 http://141.62.38.12:8000/healthcheck
curl --fail --noproxy '*' --connect-timeout 3 --max-time 180 \
  http://141.62.38.12:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm","messages":[{"role":"user","content":"请回复测试成功"}],"temperature":0,"max_tokens":32,"stream":false}'
```

Proxy 只连接两个集群的主 HTTP 入口，跨机 DP 路由由原生 vLLM 处理。
/healthcheck 仅证明 Proxy 已启动并完成自身客户端初始化；不替代 P/D 的 /health 或实际推理请求。

## 6. 退出码、日志和恢复

P/D 脚本前台 exec 原生 vLLM，失败退出码直接传递；输出与 plog 位于仓库 pd-logs/p0、p1、d0、d1。
保持运行终端和 tee 正常，不使用 Ctrl+Z 暂停。metrics 目录为每次启动单独创建。

K、协调器、Proxy 用 nohup 直接写日志文件，标准输入来自 /dev/null，不通过后台 tee。
K/协调器日志位于 `/home/c00612283/pd-pool-logs/`，Proxy 位于仓库 pd-logs/proxy。
每次独立目录保存 server.pid 或 socket-0.pid/socket-1.pid，并打印完整日志路径。

后台启动后的 PID 只表示创建了进程，不能作为成功依据。脚本检查：

1. 新进程仍存活且未暂停，启动期间 PID 没有被复用。
2. 就绪日志已出现，且目标 IPv4 监听 socket 确实属于这个新进程。
3. Proxy 额外要求自己的 /healthcheck 返回 HTTP 200 和 status=ok。

K 每服务等待最多 60 秒，协调器 20 秒，Proxy 180 秒；可在对应独立脚本修改 startup_timeout。
K 两个服务依次检查，因此最坏等待约 120 秒。
进程退出、暂停、端口冲突或超时均返回非零、输出失败日志，不会因为旧进程占端口就误报成功。
K/协调器就绪检查只观察日志与本地 /proc，不发额外控制协议请求。

K 仅一个 socket 服务成功时，脚本也返回非零。已启动服务及 PID/日志保留，不自动重试、替换或杀进程。
超时的服务可能稍后就绪，不要立即重复执行；先核对打印的 PID、监听者和日志。
需要清理时先停止请求、Proxy、P 两机、D 两机，确认 DMA 静止，再停止明确属于本次部署的 K。
协调器失联/failed 时按基础指南恢复，不能在旧 initializer 仍活着时重启协调器放行。

## 7. 验收

1. 从原生 rank 日志确认 P DP2/TP8/EP16、D DP16/TP1/EP16，且两组跨机 EP 实际建成。
2. 检查 16 个 D worker 的 CPU route 与映射表一一对应；检查 16 个 K region 驻留及 socket 内存归属。
3. 发送足够多的新请求覆盖两个 P DP 和全部 D DP；一个短请求无法证明所有路径可用。
4. 固定长请求覆盖 DDR 满 block、NPU Indexer/Scale/tail、MTP；按 worker/request/generation 关联日志。
5. 汇总两 P/两 D 日志运行 check_kvaf_pd_diagnostics.py，检查 DDR/compute 和字节诊断证据。
6. 与非 PD 同模型、同采样基线比较精度，再验证取消、重复请求、并发、持续运行和资源释放。
7. 性能测试前，四份 P/D 脚本同时改为 diagnostics_requests=0、verify_kv=false 并重启；
   profiler 是否保留按测量目标决定，带字节校验的时延不能当作稳态性能。
