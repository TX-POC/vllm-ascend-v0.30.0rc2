# KVAF PD 配置说明

部署步骤见 [PD 分离部署操作指南](kvaf_pd_deployment.md)。
本文按当前代码字段说明，不适用于 develop2 的跨层 arena 或旧 SDK 单 MR 协议。

## 1. 功能入口和支持边界

PD 是显式选择的外部 Worker / Runner / Connector，不在导入包时注册，也不替换原生全局 Runner。
原生 CLI 配置如下；P 使用 kv_producer，D 使用 kv_consumer：

```json
{
  "kv_connector": "PDKVoffloadConnector",
  "kv_connector_module_path": "vllm_ascend.distributed.kv_transfer.kvaf.pd.connector",
  "kv_role": "kv_producer",
  "kv_port": 31000,
  "kv_connector_extra_config": {}
}
```

必须同时指定：

```bash
--worker-cls vllm_ascend.distributed.kv_transfer.kvaf.pd.worker.PDKVWorker
```

当前固定约束：PP=PCP=DCP=1；原生 MTP 三个 speculative tokens，method 为 mtp 或 deepseek_mtp；
两端关闭原生 prefix caching。Target 与 MTP 的有效 drafter 上限、缓存布局和诊断校验模式必须匹配。
不自动兼容其他模型、量化布局、拆分策略或 SDK 版本。

`source_tp0_only=true` 另要求 glm_moe_dsa、D TP=1、主 KV 与 indexer 均 fp8，
`enable_dsa_cp=false` 且 `c8_enable_reshape_optim=false`。P 示例使用 eager、TP8/DP1；用户最新确认可跑通的 D 使用 TP1/DP8 和 FULL_DECODE_ONLY，
主模型不设置 enforce-eager，MTP 子模型保持 enforce_eager=true。
关闭 TP0-only 不代表任意 TP/布局都已通过硬件验收。

## 2. Connector 扩展字段

下列字段放入 `--kv-transfer-config` 的 `kv_connector_extra_config`；未知字段会报错。

| 字段 | 默认值 | 配置端 | 含义及限制 |
| --- | --- | --- | --- |
| decode_endpoints | [] | P | 所有 D worker 的框架控制 host:port，必须非空、无重复；不是 HTTP 8200 或 K 端口 |
| source_tp0_only | false | P/D 一致 | true 时每个 P DP 实例仅自己的逻辑 TP0 注册和发送全部所需 KV |
| initialization_coordinator | 空字符串 | D | 统一协调器 host:port；空字符串关闭初始化准入；P 禁止配置非空值 |
| timeout | 120 秒 | P/D | 1～3600 秒，PD 控制操作及初始化等待预算；不修改 SDK 内部 socket 超时 |
| control_backlog | 128 | 仅 D | 框架控制端口的 TCP 监听队列；整数 1～4096，拒绝 bool/字符串；不是并发线程或 Jetty 数 |
| diagnostics_requests | 0 | P/D | 整数 0～16；D 每 worker 限定请求代数，P 配合对应请求；0 不安装诊断对象 |
| verify_kv | false | P/D 一致 | 对已选诊断请求做有效字节 SHA256 比较；true 必须配 diagnostics_requests>0 |
| cpu_eids | {} | P/D | 物理 NPU ID 字符串到 32 位十六进制 EID 的覆盖，选择 CPU 路径的本地 NPU EID |
| npu_eids | {} | P/D | 同上，用于 NPU HBM READ 路径；不是远端 CPU EID |

EID 映射键使用规范十进制字符串，如 "0"、"7"，不接受 "07"；缺省由 SDK 发现候选。
不同设备或迁移服务器后不能复制旧 EID 当作通用值。
超时、源 ID、epoch 和 ReadyToken 属于不同层级，不能用延长控制 timeout 代替完成确认。

### 本次 P 配置

```json
{
  "decode_endpoints": [
    "141.62.38.24:31000", "141.62.38.24:31001",
    "141.62.38.24:31002", "141.62.38.24:31003",
    "141.62.38.24:31004", "141.62.38.24:31005",
    "141.62.38.24:31006", "141.62.38.24:31007"
  ],
  "source_tp0_only": true,
  "timeout": 600,
  "diagnostics_requests": 1,
  "verify_kv": true
}
```

### 本次 D 配置

```json
{
  "source_tp0_only": true,
  "initialization_coordinator": "141.62.38.24:28998",
  "control_backlog": 128,
  "timeout": 600,
  "diagnostics_requests": 1,
  "verify_kv": true
}
```

### D 控制连接突发与监听队列

control_backlog 只放在 D 的 kv_connector_extra_config，不放在 additional-config 或 kvaf_config，
P 不配置此字段；P 显式传入会报错。省略时 D 默认 128，无需两侧同步这个值。
这是现有 PD 控制服务的队列参数，不是额外功能开关，不改变 source_tp0_only 或全建链选择。

Python 3.12 TCPServer 原默认监听队列为 5。全建链时多个 P TP rank 会短时集中连接同一 D 控制端口；
默认 128 用来容纳尚未被 accept 的突发连接。不是“第六个请求必失败”，实际排队取决于接收速度与内核行为。
该值不是活动 RPC、推理请求、线程、SDK source 或 Jetty 数量上限，也不会提高原生计算并行度。
ThreadingTCPServer 的处理方式、Runtime 锁、RPC timeout、Ready/Commit/取消/ACK 规则均不改变。

配置在 D 启动监听前生效，不热更新；每个 D worker 的框架控制端口独立应用。
不修改 K:28997、初始化协调器:28998、HTTP:8200 或 SDK 动态 HBM listener 的队列。
它也不能修复 RunAicpuCommInit 缺符号等 HCCL/AICPU 配套问题。

Linux 实际队列可能受 net.core.somaxconn 等系统设置限制；1～4096 是本配置允许范围，不是硬件容量声明。
先在 D 所在网络命名空间读取 `/proc/sys/net/core/somaxconn`，再决定是否需要单独调整系统配置，
框架不会自动修改 sysctl。要观察效果，比较受控突发测试前后的 ListenOverflows/ListenDrops 增量，
不能把系统累计值或其他监听端口的丢包认定为本次 D 的溢出；同时记录连接失败率和 RPC 延迟。
CPU 回环测试验证 32 个排队连接及控制响应，不能代替真实多机网络和 NPU 数据面验收。

## 3. additional-config 和 KVAF Runtime

以下字段放入 `--additional-config`；KVAF 专有字段必须位于其 `kvaf_config` 子对象中。

| 字段 | 当前部署值 | 作用 |
| --- | --- | --- |
| enable_cpu_binding | true | 使用框架原有 CPU 绑核能力 |
| enable_dsa_cp | false | DSA 上下文并行计算路由；不是 DSA 总开关或 KVAF 开关；TP0-only 要求关闭 |
| multistream_overlap_shared_expert | true | 原生共享专家多流重叠配置 |
| ascend_compilation_config | 见下文 D 配置 | D 的图扩展和 kernel 编译配置 |
| recompute_scheduler_enable | D: "True" | 保留成功配置的兼容字段及字符串取值；当前日志可能提示迁移到 scheduler_config，P 不沿用 |
| finegrained_tp_config | D: lmhead_tensor_parallel_size=8 | D 的细粒度 lmhead 配置，不改变命令中的主 TP=1 / DP=8 |
| enable_mlapo | D: true | 保留 D 已验证的 MLAPO 配置，P 示例不新增此字段 |
| c8_enable_reshape_optim | false | 选择相应 C8 缓存布局路径；TP0-only 要求显式关闭 |

`enable_dsa_cp=true` 会启用 TP 组内的 DSA CP 计算/通信路径，并自动启用所需 FlashComm。
它不控制 PD 建链，不能为解决 Jetty 问题而开启；false 不关闭 Indexer、稀疏注意力或 EP。

| kvaf_config 字段 | 代码默认值 | PD 说明 |
| --- | --- | --- |
| enabled | false | P 必须 false，D 必须 true；PD 功能由 Connector 显式启用 |
| build_dir | 空字符串 | P/D 均须指定；实际 import 必须来自此目录的 python/kvaf |
| cpu_address | 127.0.0.1:28999 | D 实际 K host:port；本次显式 141.62.38.24:28997 |
| cpu_addresses | {} | 物理 NPU ID 到 K endpoint 的覆盖，优先于 cpu_address；未配置时共用默认 K |
| ddr_size_gb | 1.0 | 每个 D worker 的 DDR 预算，正有限数；本次 16，DP8 总计约 128 GiB，不是八卡合计 16 |
| hot_tokens | 8192 | 每请求/层热区，不含固定 384-token ABC tail；整数至少 8192 且为 128 的倍数 |
| npu_core_count | 2 | SDK AIV 核数，整数 1～48；本次 1，SDK 仍有自己的组合约束 |
| jettys_per_core | 2 | 每核 Jetty 参数，整数 1～64；本次 1，不能据此推断整机总连接数 |
| verify_put | false | 原混部 writer 的 Put 抽样回读校验；不等同于 PD verify_kv 的双路径校验 |

多 K 配置时，使用相同 K 的所有 D 必须填写同一个协调器，并使用相同 endpoint 字符串作为门控键。
不同 IP/主机名别名不会自动合并为同一个 K；多 K 拓扑需要独立验证，不能由单 K 验收推定通过。

## 4. 原生参数与启动工具

`scripts/start_vllm_pd.sh p|d LOCAL_IP NIC -- MODEL [原生参数...]` 不发明 JSON 字段。
它校验 Connector/角色及 build_dir，并在未显式提供时填入以下值：

| 原生参数 | helper 默认值 |
| --- | --- |
| worker-cls | 外部 PDKVWorker |
| host / distributed-executor-backend | 0.0.0.0 / mp |
| seed / served-model-name | 1024 / glm |
| enable-expert-parallel / trust-remote-code | 开启 |
| no-enable-prefix-caching | 关闭 prefix caching |
| quantization | ascend |
| kv-cache-dtype / attention_config.indexer_kv_dtype | fp8 / fp8 |
| block-size | 128 |
| speculative-config | num_speculative_tokens=3，method=deepseek_mtp，enforce_eager=true |
| enforce-eager | 开启，除非显式传入 no-enforce-eager |

TP、DP、最大长度和内存预算由部署命令显式填写，不由 helper 猜测。
本次 `max-model-len=131072`、`max-num-batched-tokens=4096`、`max-num-seqs=1`、
`gpu-memory-utilization=0.85`。最后一项是 HBM 预算，不是 DDR 比例。
`max-model-len` 包括输入与输出总 token，HTTP usage 数量不能替代实际字节校验。

编译环境与入口按用户 2026-11-11 验证命令记录，完整步骤见
[部署指南第 3、4 节](kvaf_pd_deployment.md#3-pd-容器编译与检查)。

| 项目 | P/D 容器 | K 宿主机 |
| --- | --- | --- |
| Python | /usr/local/python3.12.13/bin/python3 | K 编译不启用 Python；协调器使用 command -v python3 |
| CANN / SOC | cann-9.2.0 / ascend950dt_9582 | 不使用容器 CANN/Python 编译 NPU kernel |
| 主入口 | install_kvaf_pd_source.sh -，保留现有 vLLM | cmake -S SDK -B build-hostcpu-k |
| SDK 构建 | CPU_OFFLOAD、AIV Get/Put、Python ON；REMOTE_CPU_SERVICE OFF | REMOTE_CPU_SERVICE ON；CPU_OFFLOAD、AIV Get/Put、Python OFF |
| 单独重建 | 原生 pip editable 安装 Ascend；KVOP/build.sh 8 Python | cmake --build --parallel 8，检查二进制和 ldd |
| 成功标记 | ASCEND_KVOP_KVAF_BUILD_OK，并检查 ASCEND_EXTENSION_OK / PD_SDK_IMPORT_OK | K_SERVICE_BUILD_OK；运行时另查 LISTEN |

完整容器安装已包含 Ascend、KVOP、SDK，不要求随后再各重建一遍。
编译 shell 中的 source 不会跨括号保留；启动脚本仍须加载自己的 KVOP 环境。
当前确认 K 和协调器分别是 141.62.38.24:28997、141.62.38.24:28998，D 的 cpu_address 与 K 一致。

主要脚本职责如下，均保留现有调用接口：

| 工具 | 用途 |
| --- | --- |
| install_kvaf_pd_source.sh | 保留已有 vLLM 或指定 vLLM 源码，构建 Ascend、KVOP、PD SDK |
| install_kvaf_pd.sh | 只构建匹配的 PD SDK/binding，保留已有 RA 借用补丁 |
| kvaf_pd_k.sh | 构建或前台启动宿主机 K |
| kvaf_pd_init_coordinator.py | 启动仅使用标准库的初始化协调器 |
| start_vllm_pd.sh | 包装环境及默认值，转发原生 serve 参数，支持 --dry-run |
| kvaf_pd_proxy.sh | 调用已有 layerwise Proxy，清除代理环境后转发参数 |
| kvaf_pd_ddr_residency.py | 无模型 DDR 初始化与共驻留回归 |
| check_kvaf_pd_diagnostics.py | 审计 PD 诊断日志，不是模型精度认证 |
| kvaf_pd_smoke.py | completions 请求连通测试，不修改服务端配置 |

部署脚本使用 `KVAF_PD_PYTHON` 选择解释器，默认 python3；`KVAF_PD_BUILD_JOBS` 控制构建并发，默认 8。
helper 设置网卡、HCCL、KVOP、SDK PYTHONPATH、独立 metrics/plog 路径等运行环境。
`KVAF_REUSE_DEVICE_RA` 是 PD loader 显式借用 HCCL device RA 的 SDK 内部接入开关，不是用户新增的建链模式。

### 当前 D 的直接启动配置

部署指南中的 D 直接运行 Python CLI，不使用 helper 的 eager 默认值；P 仍使用 helper。
以下配置按用户确认可跑通的命令记录，不扩展为任意硬件/模型均支持的承诺：

| D 配置 | 实际取值 |
| --- | --- |
| async-scheduling | 显式启用 |
| compilation-config.cudagraph_mode | FULL_DECODE_ONLY |
| 主模型 enforce-eager | 不传入 |
| speculative-config.enforce_eager | true，仅 draft |
| ascend_compilation_config.enable_npugraph_ex | true |
| ascend_compilation_config.enable_static_kernel | false |
| ascend_compilation_config.enable_super_kernel | false |
| recompute_scheduler_enable | 字符串 "True"，保留兼容入口 |
| finegrained_tp_config.lmhead_tensor_parallel_size | 8 |
| enable_mlapo | true |
| profiler-config | torch，输出 /home/d00843668/profiling_no_pooling，with_stack=false |
| TORCH_HCCL_ASYNC_ERROR_HANDLING | 0，按 D 命令设置 |
| diagnostics_requests / verify_kv | 1 / true，P 同步配置；性能测量时两侧一起恢复 0 / false |

D 没有显式传 enable_dsa_cp，当前默认 false；c8_enable_reshape_optim 显式 false。
保持这些区别，不能让启动包装脚本自动添加主模型 enforce-eager 后声称等价。
前台日志通过 tee 写入 server.log；保持日志消费者运行，避免 Ctrl+Z 造成阻塞。

## 5. TP0、多 MR 与实际数据流

### 原始全建链与临时 TP0 单源模式

原始需求是全建链：`source_tp0_only` 默认 false，P、D 两侧均可省略此字段，
或均显式设置为以下值；无需设置 true：

```json
{"source_tp0_only": false}
```

`true` 是此前为减少通信参与 rank、缓解连接资源压力引入的临时妥协方案，不是启用 PD 的必要条件。
本文和部署指南中保留 true 的启动示例，用于记录已有验收配置，不表示原始全建链必须开启它。
P/D 握手校验该模式，两侧必须一致；省略与显式 false 等价，不能只在 P 或 D 单侧更改。

恢复 false 后，所有 P TP rank 都对全部目标 D 注册 CPU/NPU 多 MR 源组；主 KV 仍按原规则
取 member 0，Indexer KV/Scale 恢复原 TP 区间分工，不是每个 rank 都重复发送完整 KV。
P 的 decode_endpoints 仍必须列出全部 D worker。D 的 K endpoint、初始化协调器、
原生 TP/DP/EP 及其他配置继续保留；不因切换此字段而自动改变其他能力开关。
关闭 TP0-only 不代表 DSA CP、其他布局或并行组合已获得兼容性和精度验证。

切换时先停止新请求和 Proxy，排空并停止两侧 P/D；同步删除字段或设为 false，
再按 D → P → Proxy 启动。健康且无遗留会话的 K/协调器可保留，不能在线混用两种模式。
全建链会增加参与 rank 及源组数量，使用当前多 MR SDK 后仍需重新验证注册资源、初始化、
长请求、MTP、取消及基线精度；源组/MR 数不等于硬件 Jetty 的实测数量。

| source_tp0_only | P 计算 | P 参与通信的 rank | 数据来源 |
| --- | --- | --- | --- |
| false | 原生 TP/DP/EP 不变 | 原有各 TP | 主 KV 按 member 0，indexer/scale 按原 TP 区间分工 |
| true | 原生 TP/DP/EP 不变 | 每个 P DP 的逻辑 TP0 | 所需主 KV、indexer、scale 都来自该 TP0 |

逻辑 TP0 不一定是物理卡 0。非 TP0 仍计算和完成记账，不加载 PD SDK，不增加额外 KV gather。
该模式依赖受支持布局下 TP0 含有完整有效 KV，不能推广到唯一数据分片仅存于其他 rank 的布局。

每个参与通信的 P rank 对每个 D 创建一个 CPU 源组和一个 NPU 源组，各组一次注册所有逐层 Tensor。
每层仍独立 MR、Tensor 视图、量化布局及 BlockTable；没有跨层 arena 或额外 KV 拼接复制。
79 层、8 个 D 时，每个活跃 P rank 每条路径为 8 个源组、632 个 MR；这不是 Jetty/CTP 的实测数量。
P 启动预注册所有目标 D，不是按请求动态建链，也不是请求广播；单请求只使用所路由到的 D。

组内一次只允许一个未释放 Ready，同一 P→D 的层任务串行持有源组，不同 D 独立门控。
`ready(layer_tensor)` 的 ReadyToken 含 source、sequence、mr_id，框架保留实际层引用，
并在生产事件完成后才发布；不能用其他层的新 Ready 覆盖未决批次。

数据路径：

- Target 满 block 主 KV：P HBM → K DDR，PreparePut → submit_cpu_pull → Wait → Commit 后发布目录。
- indexer KV、scale、普通 tail 主 KV：P HBM → D HBM，使用 batch_get_raw_addr。
- MTP 主 KV：沿现有业务规则进入 D HBM，不作为 Target 满块主 KV 卸载。
- 后续解码：由 D 原生后端按 Hot/Get 和 ABC/tail 规则消费，模型算子不是被 KVAF 替代。

CPU 提交的 dst_offsets 相对 runtime.remote_region.base，原目录仍使用 Put 返回的绝对地址。
SDK 管理 PA/UBA、MR、访问凭据和物理队列；框架只维护逐层业务映射及不透明 ReadyToken。
D 每 Runtime 启动一个 SDK HBM listener，端口通过 receiver ticket 发现，不是额外的逐层监听。

## 6. 完成、取消与初始化协调

每次层发送任务使用独立的生产事件，首次确认完成后，CPU/NPU 各批次共享本任务的完成确认，
不重复调用原生事件同步。完成状态不保存在层注册对象或源组中，不跨层、请求或 chunk 复用。
直接调用 LayerRegistration.ready 并传入原始事件时，仍须先同步生产事件再发布 Ready。
此优化不改变 SDK Ready/release、源组锁及目录发布规则，也不增加配置开关；
只减少重复原生同步调用，实际精度和性能仍须在目标 NPU 上做回归验证。

只有全部必需层/贡献方完成、DDR Commit 与目录发布、HBM 就绪，才能放行 D 首次 Decode。
原生实际跳过 MTP 时使用原有跳过贡献，不读取 dummy KV；普通 tail 在末 prompt token 重放前导入。
P 物理源复用必须等 SDK release 与原有完成确认，不以 RPC 已发送或 HTTP 200 代替 DMA 完成。

NPU 失败取消先由 D cancel_raw_addr；回复不明只 resolve 原批次，不重放 READ。
部分注册或 Ready 状态不明时，保留整个源组、MR、Tensor 和 bootstrap；停止该组后续批次。
正常退出先排空、撤销 D 业务映射，再由 SDK source.shutdown 协调连接退役。
这些失败保留和清理代码属于业务正确性，不能作为临时排障冗余删除。

初始化协调器只串行化同 K 的 link_to → initialize → initialize_ddr，成功后释放准入，
不转发请求或 KV，不参与运行期 Get/Pull，也不是 vLLM 原生 DP Coordinator。
关闭它会回到原并行初始化；新 SDK 是否可去掉该保护需单独做八进程驻留验证。
持有者失败/断连后，endpoint 会 blocked；必须等所有相关 D 静止后重启协调器恢复。
不能通过重试或租约自动接管可能仍在运行的初始化。

## 7. 诊断字段和证据边界

日志固定前缀为 KVAF_PD_DIAG，后面是单行 JSON；通过 worker/request/generation/job 关联，
不要只按日志总数或相同层名判断同一批次。

| 字段或事件 | 可说明什么 |
| --- | --- |
| SOURCE_READY、mr_id | 对应逐层 MR 的源就绪发布 |
| PULL_PLAN / PULL_DONE，path=cpu/npu | 业务传输计划和相应完成路径 |
| KV_CHECK，status=PASS/FAIL | 选定请求有效字节的比较结果 |
| MAIN_REPLICA_CHECK | 参与者之间的主 KV 副本证据；TP0-only 仅一个贡献者，不证明其他计算 TP 副本 |
| DECODE_READY，expected/contributions | 必需贡献完成并通过接收门控 |
| MODEL_FORWARD_DONE / SFA_DONE 等 | 原生计算路径已执行的诊断证据 |
| source_groups_per_path / mrs_per_path | 框架注册预算，不是硬件队列用量 |
| K REMOTE_PULL_DONE | 对应 CPU Pull 完成；不能代替 NPU READ 或精度证据 |

verify_kv 会同步设备、拷贝数据到 CPU，并对 DDR 做额外回读，影响延迟。
默认 0/false 不创建 PD 诊断对象或装饰器，也不进行额外哈希/回读；SDK 自有日志不受这两个字段控制。
仅日志模式使用 diagnostics_requests>0、verify_kv=false；不能通过要求字节证据的审计。
完整日志中任意诊断 FAIL 不能由 HTTP 成功覆盖；缺少证据也不能解释为校验成功。

四智算节点与八鲲鹏节点的扩容模板及脚本见 [池化部署指南](kvaf_pd_pooled_deployment.md)。
其中 P0/P1、D0/D1、K0～K7、协调器和 Proxy 均提供独立启动脚本，直接编辑本机参数及原生 JSON。
后台脚本检查新进程持有的监听和就绪状态，失败返回非零；K 部分成功时保留服务，需人工核对后恢复。
