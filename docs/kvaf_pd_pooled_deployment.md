# 四智算节点与八鲲鹏节点的 PD 部署指南

本方案在当前多 MR PD Connector 上配置两个跨机 DP/EP 集群，不修改框架、KVAF 或内存布局。
它是新的扩容部署模板，尚未做该规模的硬件联调；单 P/单 D 的成功不能替代本方案验收。
基础安装排障见 [PD 部署指南](kvaf_pd_deployment.md)，字段见 [配置说明](kvaf_pd_configuration.md)。
启动脚本位于 [kvaf_pooled](../examples/disaggregated_prefill_v1/kvaf_pooled)。

## 1. 拓扑及资源假设

以下 IP 按已有地址尾号递增，均为可编辑示例；使用前确认属于对应服务器。
每台智算节点 8 张 950DT，每台 K 服务器假定两个 CPU socket。
“一个 NPU 对应一个 CPU”在本方案中特指每个 D NPU 独占一个 K 服务、对应一个 CPU socket 的内存分区。
P 的 NPU 不另分配 K Runtime；P 将源注册到请求目标 D 所拥有的 K region。
若希望全部 32 个 P/D NPU 各独占一个 CPU socket，则需要 32 个 socket，八台双路 K 不足。

| 角色 | 地址 | 本地 DP / TP | 全局 DP rank | HTTP |
| --- | --- | --- | --- | --- |
| P0 | 141.62.38.12 | DP1 / TP8 | 0 | 8100，P 集群入口 |
| P1 | 141.62.38.13 | DP1 / TP8 | 1 | headless，不对 Proxy 注册 |
| D0 | 141.62.38.24 | DP8 / TP1 | 0～7 | 8200，D 集群入口 |
| D1 | 141.62.38.25 | DP8 / TP1 | 8～15 | headless，不对 Proxy 注册 |
| K0～K7 | 141.62.38.26～33 | 每机两个服务 | 每服务一个 D NPU | 28997 / 28999 |
| 协调器 | K0:141.62.38.26 | 一个实例 | 以实际 K endpoint 为门控键 | 28998 |
| Proxy | P0:141.62.38.12 | 一个实例 | 路由到两个集群主入口 | 8000 |

P 的 TP8 × DP2 = 16 个 rank，开启 EP 后预期 EP16；D 的 TP1 × DP16 = 16 个 rank，预期 EP16。
没有额外的 `--expert-parallel-size 16` 参数。两个集群各自形成 EP 组，P 与 D 不组成同一个 EP32。
不能把两台机器分别运行 DP1 或 DP8 独立实例后，声称已经组成跨机 EP16。

本次继续使用 source_tp0_only=true：P0/P1 各自逻辑 TP0 向全部 16 个 D 建源组，其他 P TP 仍计算。
每个活跃 P TP0 每路径 16 个源组，79 层时 1264 个逐层 MR；不是 Jetty 实测数量。
多机 EP 通信、初始化时间、队列资源和长请求稳定性需要重新测量。

上述 true 是临时 TP0 单源模式，不是原始全建链的必要设置。原始全建链可在 P/D 同时省略
source_tp0_only 或设置 false，默认就是 false；全部 P TP rank 将对全部 16 个 D 注册多 MR 源组，
主 KV/Indexer/Scale 按原传输规则分工，P 的 16 个 decode_endpoints 不减少。

本目录 start_npu.sh 当前在生成 extra 时显式写入 source_tp0_only=True，
仅从其他 JSON 示例删除字段不会改变该脚本。要用这些模板运行全建链，须将该赋值改为
source_tp0_only=False，或从 extra 构造中删除此参数，并同步到四个 P/D 容器。
不要在 config.sh 中添加未被脚本读取的同名变量后假定已经生效。
启动前用四个入口的 --dry-run 核对最终命令，确保两侧一致，再按本文顺序启动；
已有实例切换须先排空并停止 P/D。全建链新增的源组及硬件资源需求需要重新验收。

## 2. K 固定映射与池化边界

| D 主机 / 物理 NPU | K 主机 | CPU/socket 服务 |
| --- | --- | --- |
| D0 / 0、1 | K0 .26 | 0:28997、1:28999 |
| D0 / 2、3 | K1 .27 | 0:28997、1:28999 |
| D0 / 4、5 | K2 .28 | 0:28997、1:28999 |
| D0 / 6、7 | K3 .29 | 0:28997、1:28999 |
| D1 / 0、1 | K4 .30 | 0:28997、1:28999 |
| D1 / 2、3 | K5 .31 | 0:28997、1:28999 |
| D1 / 4、5 | K6 .32 | 0:28997、1:28999 |
| D1 / 6、7 | K7 .33 | 0:28997、1:28999 |

通过每台 D 的 kvaf_config.cpu_addresses，按本机物理 NPU ID 0～7 精确映射。
D1 的全局 DP rank=8 不表示物理 NPU ID=8；仍使用本机键 0～7。
每个 D worker 分配 16 GiB，合计约 256 GiB；每 K 服务约 16 GiB、每双路服务器约 32 GiB，加 SDK 开销。
预算不是自动共享的统一容量。当前没有跨 K 的全局 key 分配、容量均衡、故障接管或自动数据迁移。
某 D 的 region 用满时，不能自动借用其他 K 的空闲 region；这里是池化资源的固定分区接入。

K 启动脚本为每服务使用 numactl 绑定 CPU 和内存 NUMA 节点；NUMA 节点号不等于 socket 号。
先执行 `lscpu -e=CPU,SOCKET,NODE`、`numactl --hardware`，将 config.sh 的 K_NUMA_NODES
改成每个完整 socket 对应的节点集合，例如 `("0-3" "4-7")`，不能盲用默认 `(0 1)`。
此绑定不等同于固定 SDK 的 URMA/EID 候选，仍需逐 endpoint 验证实际 CPU Pull、内存归属及 UB 路径。
八台 K 若拓扑不同，分别调整本机配置中的 K_NUMA_NODES。

## 3. 安装与版本

四台智算节点容器使用相同模型、tokenizer、CANN/torch_npu、框架提交和 SDK 构建版本。
八台 K 宿主机也使用同一 SDK 提交。基准 SDK 为 codex/cpu-pull-hbm-reference 的
`74c3fc491a257bae73356da54eafb08e1e9e4e45`，不可混用旧 Pull 协议。

配置集中在 `examples/disaggregated_prefill_v1/kvaf_pooled/config.sh`：编辑地址、网卡、
Python、模型、源码/build 路径及 NUMA 集合，再分发到所有节点。
脚本无需修改框架源码。脚本修改仅对下一次启动生效，不支持热更新路由。
容器需具备 NPU/HCCL/UB 所需设备权限和网络；以下按业务地址在容器内可绑定的部署环境编写。

在四个 NPU 容器，加载匹配 CANN 环境后各执行：

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

若同版本 Ascend 原生扩展和 KVOP 已成功构建，只需沿用基础指南的 SDK-only 构建命令。
检查实际导入目录，以及 listen_hbm、cancel_raw_addr、ReadyToken.mr_id 等新接口。
安装入口仍沿用已有 HCCL device RA 借用补丁。该示例不新增 SDK 修改。

在八个 K 宿主机各执行，需已安装匹配 UMDK、liburma、编译工具、numactl：

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

K0 另需宿主机 Python3 和框架目录中的协调器脚本，仅依赖标准库。
证书、离线依赖和原生扩展问题按基础指南处理。RunAicpuCommInit 缺符号属于 HCCL/AICPU
配套问题，先解决再扩容，不能靠修改 PD endpoint 修复。

## 4. 网络和启动前检查

- P 两机之间、D 两机之间均需可进行跨机 HCCL/EP 通信；检查驱动、固件、CANN 和网络配套。
- P 集群主地址为 P0:13389，D 集群为 D0:13489，远端启动进程必须可达各自主地址。
  HCCL、原生 DP 数据通信还会使用其他端口，不能只放行这两个控制端口。
- 所有 P worker 必须访问 D0:31000～31007、D1:31008～31015，及 SDK ticket 返回的动态 HBM 监听端口。
- P/D 访问所有对应 K endpoint；所有 D 访问 K0:28998。Proxy 访问 P0:8100 和 D0:8200。
- 所有 NPU 进程在实际启动 shell 检查 soft/hard nofile，建议至少 65536；不要只提升其他 shell。
- 启动前排空旧 Proxy/P/D；替换 K 或旧状态不明时，等旧 DMA 停止再清理/重启。
- K 启动脚本每次启动两个后台实例，不自动替换旧服务。先确认 28997/28999 未被占用。

所有脚本和配置以相同相对目录分发。下面命令均以源码目录为当前目录。

## 5. 启动顺序与命令

### K0～K7：八个通算宿主机

分别在 K0 到 K7 执行对应编号，例如 K0：

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2
bash examples/disaggregated_prefill_v1/kvaf_pooled/k_start.sh 0
netstat -lntp 2>/dev/null | grep -E ':28997[[:space:]]|:28999[[:space:]]'
```

其余七台将末尾编号换为 1～7。检查日志实际 LISTEN、进程状态和 numactl 绑定；后台 PID 不代表成功。
只在 K0 启动一个协调器：

```bash
bash examples/disaggregated_prefill_v1/kvaf_pooled/coordinator_start.sh
netstat -lntp 2>/dev/null | grep ':28998[[:space:]]'
```

### D0、D1：智算容器

两个终端先后启动，不等待单个主节点独自 ready；跨机集群需要两侧加入：

```bash
# D0 容器
bash examples/disaggregated_prefill_v1/kvaf_pooled/d0_start.sh
```

```bash
# D1 容器（headless，全局 start-rank=8）
bash examples/disaggregated_prefill_v1/kvaf_pooled/d1_start.sh
```

D0 对外监听 8200；D1 不要求出现独立 HTTP API。D0/D1 控制端口采用全局 DP rank，
不是两机都从 31000 开始。等待全部 16 个 worker 完成初始化、图 warmup：

```bash
curl --fail --noproxy '*' --max-time 10 -i http://141.62.38.24:8200/health
# 在 D0 宿主机
netstat -lntp 2>/dev/null | grep -E ':3100[0-7][[:space:]]'
# 在 D1 宿主机
netstat -lntp 2>/dev/null | grep -E ':310(0[89]|1[0-5])[[:space:]]'
```

D 配置沿用用户已确认的单机 D 参数：FULL_DECODE_ONLY、async-scheduling、MLAPO、
LMHead TP8、npugraph_ex、MTP eager、profiling；扩大到 DP16 后仍须重新验证这些组合。

### P0、P1：智算容器

```bash
# P0 容器
bash examples/disaggregated_prefill_v1/kvaf_pooled/p0_start.sh
```

```bash
# P1 容器（headless，全局 start-rank=1）
bash examples/disaggregated_prefill_v1/kvaf_pooled/p1_start.sh
```

P 两端同样必须加入后再等 P0 的 8100 健康。P 使用 eager，不能自动沿用 D 的图编译参数。
P/D 默认均采集 1 个请求的 KV 字节诊断。全部 4 个容器中的 config.sh 保持相同设置。

### Proxy：P0 容器

```bash
bash examples/disaggregated_prefill_v1/kvaf_pooled/proxy_start.sh
curl --noproxy '*' --connect-timeout 3 --max-time 180 \
  http://141.62.38.12:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm","messages":[{"role":"user","content":"请回复测试成功"}],"temperature":0,"max_tokens":32,"stream":false}'
```

Proxy 只配置 P0 和 D0 的集群 HTTP 入口，内部多机 DP 路由由原生 vLLM 处理。
不要向 Proxy 填写 headless 的 P1/D1 HTTP endpoint。

## 6. 脚本预览、日志和验收

`p0_start.sh`、`p1_start.sh`、`d0_start.sh`、`d1_start.sh` 支持 `--dry-run`：
只需要 config.sh 中的 Python 可执行，不加载 CANN/KVAF，不创建服务，输出完整原生命令。

```bash
bash examples/disaggregated_prefill_v1/kvaf_pooled/p1_start.sh --dry-run
bash examples/disaggregated_prefill_v1/kvaf_pooled/d1_start.sh --dry-run
```

日志统一保存在 config.sh 的 LOG_ROOT：p0/p1/d0/d1 为每次启动独立目录，K 按主机/socket 分开。
NPU 服务前台 tee 需持续运行，不使用 Ctrl+Z；K/协调器/Proxy 直接后台写文件。
保留各脚本打印的 PID、日志路径；只停止明确属于此部署的进程。

验收至少包含：

1. 从原生 rank 日志确认 P DP2/TP8/EP16、D DP16/TP1/EP16，且跨机 EP 组实际建成。
2. 检查 16 个 D 的 KVAF CPU route，与映射表一一对应，不能因缺少键回退到同一默认 K。
3. 检查全部 16 个 K region 驻留，内存实际落在指定 socket；每个 K 服务仅对应预期 D owner。
4. 发送足够多的新请求覆盖两 P DP 和全部 D DP，不能由单个短请求证明全部路径可用。
5. 固定长请求覆盖 DDR 满 block、NPU indexer/scale/tail、MTP，按 worker/request/generation 关联日志；
   汇总两 P/两 D 日志运行 check_kvaf_pd_diagnostics.py，要求 DDR/compute 证据。
6. 做非 PD 同模型、同采样基线精度对比；记录拓扑差异，再测取消、重复请求、并发及持续运行资源。
7. 性能测试前在四容器同时将 DIAGNOSTICS_REQUESTS=0、VERIFY_KV=false 并重启 P/D；
   profiling 是否继续保留按测量目标决定，不能把带字节校验的时延当作稳态性能。

恢复顺序：停止新请求和 Proxy，排空并停止 P 两机，再 D 两机；需要重启 K 时确认全部旧会话静止。
协调器 failed/disconnected 时仍按基础指南处理，不能在旧 initializer 还活着时重启放行。
本方案没有 K 自动故障迁移；K 服务或主入口故障需按该集群的恢复步骤重新建链。
