# KVAF 1P1D 双机部署配置说明（P 八卡 / D 八卡）

本文逐项解释 [双机部署指南](kvaf_pd_2nodes_deployment.md) 中用户实际验证过的 P/D 启动参数。
完整编译、启动脚本和请求命令以该指南为准；本文补充作用、单位、默认值和修改时的关联关系，不改动启动基线。
字段说明依据 `v0.30.0rc2-develop` 的 PD 实现，基准框架提交为 `3927ee424`，对应启动脚本见同分支的双机部署指南。
表中的“未显式配置”不等于关闭；只在源码可确认时列出默认值，实际执行能力还受模型、量化和硬件条件影响。

## 1. 当前 P/D 配置总览

| 项目 | P 节点 | D 节点 |
| --- | --- | --- |
| 业务 IP | `141.62.38.12` | `141.62.38.24` |
| 物理 NPU | `0,1,2,3,4,5,6,7` | `0,1,2,3,4,5,6,7` |
| TP / DP / 本地 DP / PP | `8 / 1 / 1 / 1` | `1 / 8 / 8 / 1` |
| EP | 开启，本机 EP8 | 开启，本机 EP8 |
| HTTP / DP RPC 基础端口 | `8100 / 13389` | `8200 / 13489` |
| PD 角色 | `kv_producer`，产生并提供 Prefill KV | `kv_consumer`，接收 KV 并执行 Decode |
| Target 主模型 | `--enforce-eager` | `FULL_DECODE_ONLY`，不传 `--enforce-eager` |
| MTP Draft | `deepseek_mtp`，3 个推测 token，eager | 同 P |
| KVAF Runtime 开关 | `enabled=false` | `enabled=true` |
| 通信源模式 | `source_tp0_only=true` | 必须与 P 一致 |
| 字节诊断 | `diagnostics_requests=0`、`verify_kv=false` | 同 P |
| 原生 prefix caching | 关闭 | 关闭 |

K 数据服务和初始化协调器位于 **24 号宿主机**，分别监听 `28997` 和 `28998`；Proxy 在 12 号容器监听 `8000`。
本方案只有两台八卡智算服务器。P/D 的 EP 各在本机，不合并成跨 P/D 的 EP16。

## 2. 原生模型、并行和资源参数

以下字段直接跟在 `serve` 后，不放进 `--additional-config` 或 `--kv-transfer-config`。

| 参数 | 本次取值（P / D） | 作用与单位 | 修改要点 |
| --- | --- | --- | --- |
| 模型位置参数 | 两端 `/home/models/GLM-5.3-w4a4c8-mxfp4` | 本地权重、模型和 tokenizer 配置来源 | 两端使用相同权重、tokenizer 和量化配置 |
| `--served-model-name` | `glm / glm` | HTTP API 使用的模型名 | 请求 JSON 中的 `model` 与之匹配；不负责选择磁盘模型 |
| `--host` | `0.0.0.0 / 0.0.0.0` | HTTP 绑定地址 | `0.0.0.0` 是监听地址；Proxy 和其他节点使用实际业务 IP |
| `--port` | `8100 / 8200` | HTTP TCP 端口 | 与 Proxy 的 prefiller/decoder 端口同步修改 |
| `--tensor-parallel-size` | `8 / 1` | 主模型张量并行度 TP | P 使用八卡 TP；D 每个 DP 引擎为 TP1；TP0-only 当前要求 D TP1 |
| `--data-parallel-size` | `1 / 8` | 全局数据并行引擎数 DP | D 有 8 个 DP 引擎；不能把 DP8 直接当作单卡 batch=8 |
| `--data-parallel-size-local` | `1 / 8` | 本机 DP 引擎数 | 本次均为单机 DP 组，本地值等于各自全局 DP 值 |
| `--data-parallel-address` | P `.12`，D `.24` | 各自 DP 组的协调地址 | 不是 K 地址，也不是 P/D KV 传输目标 |
| `--data-parallel-rpc-port` | `13389 / 13489` | 原生 DP RPC 基础端口 | 不替代 HTTP、PD 控制或 HCCL 的其他端口 |
| `--pipeline-parallel-size` | `1 / 1` | 流水线并行度 PP | 当前 PD 实现校验 PP=PCP=DCP=1；不要仅改 PP 来扩展部署 |
| `--distributed-executor-backend` | `mp / mp` | 原生多进程执行后端 | 沿用当前本机八卡进程组织 |
| `--enable-expert-parallel` | 两端开启 | 在各自并行组内分配 MoE 专家计算 | 改为关闭涉及专家布局与通信变化，需要重新验证 |
| `--seed` | `1024 / 1024` | 随机种子 | 有助复现；不能保证不同并行拓扑输出逐 token 相同 |
| `--max-model-len` | `131072 / 131072` | 单序列输入与输出合计的 token 上限 | 长输入须为输出预留额度；不是本轮立即处理的 token 数 |
| `--max-num-batched-tokens` | `4096 / 4096` | 每个引擎单次调度的 token 预算 | Prefill 可以分块；它不把总上下文长度限制为 4096 |
| `--max-num-seqs` | `1 / 1` | 每个引擎的并发序列上限 | D DP8 每引擎上限为 1；实际请求分布由路由和调度决定 |
| `--gpu-memory-utilization` | `0.85 / 0.85` | 每实例设备内存预算比例，范围按 vLLM 校验 | 作用于 NPU HBM 预算，不是 K DDR 比例，也不是只给 KV 的固定比例 |
| `--trust-remote-code` | 两端开启 | 允许加载模型目录中的自定义代码 | 保留已验证模型的加载方式 |
| `--quantization` | `ascend / ascend` | 选择 Ascend 量化实现 | 权重实际位宽由模型量化配置决定；不等价于将全部权重设为 fp8 |
| `--no-enable-prefix-caching` | 两端显式关闭 | 关闭原生 prefix KV 复用 | PD 校验要求关闭，避免 MTP 跳过场景复用陈旧 KV |
| `--block-size` | `128 / 128` | 每个 KV block 的 token 数 | 当前 PD Target DDR 路径使用 block-128、656 bytes/token 的 C8 布局 |
| `--kv-cache-dtype` | `fp8 / fp8` | 主 KV cache 数据类型 | 与 indexer dtype、模型布局及 TP0-only 约束一起核对 |
| `--attention_config.indexer_kv_dtype` | `fp8 / fp8` | Indexer KV cache 数据类型 | 与主 KV dtype 是两个字段；本次两端均为 fp8 |
| `--worker-cls` | 两端 `vllm_ascend.distributed.kv_transfer.kvaf.pd.worker.PDKVWorker` | 显式选择 PD Worker / Runner 接入 | 必须与 `PDKVoffloadConnector` 配套，不能只填 Connector 而使用普通 Worker |

`gpu-memory-utilization`、`max-model-len`、`max-num-seqs`、`max-num-batched-tokens` 和图 workspace 共同影响 HBM 使用。
单独提高其中某个值不代表当前硬件能承载对应负载；扩大并发后同时核算 HBM 热区、staging、图捕获和 K DDR。

## 3. 图模式、异步调度与 MTP

| 位置 / 参数 | P 值 | D 值 | 作用 |
| --- | --- | --- | --- |
| 原生 `--enforce-eager` | 显式传入 | 不传入 | 控制主模型是否强制 eager；D 不能被启动 helper 自动补入该参数 |
| `--compilation-config.cudagraph_mode` | 不显式配置，主模型由 eager 约束 | `FULL_DECODE_ONLY` | D 的 Decode 使用整图模式，Prefill 不在该模式的整图范围 |
| `--async-scheduling` | 未显式传入 | 显式传入 | 使用原生异步调度，减少 CPU 调度与设备执行的串行等待；不替代 KV 传输完成确认 |
| `--speculative-config.method` | `deepseek_mtp` | `deepseek_mtp` | 选择原生 MTP；当前 PD 也接受方法名 `mtp` |
| `--speculative-config.num_speculative_tokens` | `3` | `3` | 当前 PD 校验固定为三个推测 token；不是保证每轮接受三个 token |
| `--speculative-config.enforce_eager` | `true` | `true` | 仅控制 Draft 执行模式；不会关闭 D Target 的 Decode 图模式 |

D 的图相关配置分属两个入口，需要区分层级：

```json
{
  "cudagraph_mode": "FULL_DECODE_ONLY"
}
```

上面是 `--compilation-config` 的完整值。下面是 `--additional-config` 中的一个子对象：

```json
{
  "ascend_compilation_config": {
    "enable_npugraph_ex": true,
    "enable_static_kernel": false,
    "enable_super_kernel": false
  }
}
```

| `ascend_compilation_config` 字段 | 本次 D 值 | 当前字段默认值 | 作用与依赖 |
| --- | --- | --- | --- |
| `enable_npugraph_ex` | `true` | `true` | 选择 Ascend npugraph_ex 图优化路径；不支持该能力的硬件 profile 会将其关闭 |
| `enable_static_kernel` | `false` | `false` | 是否生成用于图执行的静态 shape kernel；开启时要求 npugraph_ex 开启 |
| `enable_super_kernel` | `false` | `false` | super kernel 图优化配置；开启时要求 static kernel 开启 |

`enable_static_kernel=false` 不等于关闭图模式。本文将两个 kernel 开关都显式置为 false，保持实测组合。
源码在只传 static kernel 开关且省略 super kernel 时有默认联动，不应将两者视为始终互不相关。
图回放要求相关 buffer 地址和容量稳定；调整并发、MTP、热区或布局后重启并重新捕获，不能在线修改后继续复用旧图。
配置启用不等于证明每个 batch 都实际命中图回放，需结合启动捕获和运行日志判断。

## 4. PD Connector：角色、地址与传输控制

本节所有字段放在 `--kv-transfer-config` 内，KVAF Runtime 字段另见第 6 节。

| 字段 | P / D 基线 | 作用与修改约束 |
| --- | --- | --- |
| `kv_connector` | 两端 `PDKVoffloadConnector` | PD 功能的显式入口 |
| `kv_connector_module_path` | 两端 `vllm_ascend.distributed.kv_transfer.kvaf.pd.connector` | Connector 类所在模块，不是磁盘 build 路径 |
| `kv_role` | P `kv_producer`；D `kv_consumer` | P 产生并注册源，D 接收并消费 KV；当前实现要求独立角色 |
| `kv_port` | 两端填写 `31000` | D 控制监听基础端口；P 仍按 `decode_endpoints` 寻址，不能只改 P 的 kv_port 来改变目标 |
| `kv_connector_extra_config` | 以下扩展字段 | PD 实现拒绝未知字段，不放入 `additional-config.kvaf_config` |

### 4.1 Connector 扩展字段

| 字段 | 基线值 / 源码默认值 | 配置端 | 作用、单位与约束 |
| --- | --- | --- | --- |
| `decode_endpoints` | `.24:31000`～`.24:31007` / 空列表 | P | 所有 D worker 的控制地址，非空、无重复；不是 D HTTP 或 K endpoint |
| `source_tp0_only` | `true` / `false` | P/D 一致 | 每个 P DP 实例仅逻辑 TP0 参与源注册和传输；其他 TP rank 仍参与模型计算 |
| `initialization_coordinator` | `.24:28998` / 空字符串 | 仅 D | 同一 K endpoint 初始化的准入门控；空字符串关闭门控，不能理解为初始化完成 |
| `timeout` | `600` / `120` | P/D | 秒，范围 1～3600，控制 PD 控制操作及初始化等待预算；不改 SDK 内部 socket 超时 |
| `diagnostics_requests` | `0` / `0` | P/D | 整数 0～16；D 每 worker 选择有限请求代数，P 配合诊断；0 关闭诊断对象 |
| `verify_kv` | `false` / `false` | P/D 一致 | 对诊断请求做有效 KV 字节校验；设 true 时要求 `diagnostics_requests>0` |
| `control_backlog` | 未显式配置 / `128` | 仅 D | 控制监听的 TCP 等待队列，整数 1～4096；不是推理并发、线程或 Jetty 数；P 显式填写会报错 |
| `cpu_eids` | 未配置 / `{}` | P/D | 可选：物理 NPU ID 字符串到本地 NPU EID 的覆盖，用于 CPU Pull 路径；不是 K CPU EID |
| `npu_eids` | 未配置 / `{}` | P/D | 可选：同类映射，用于 NPU HBM READ 路径 |

EID 映射使用规范十进制键，如 `"0"`，值为 32 位十六进制字符串。缺省由 SDK 发现，不把旧设备 EID 当作通用常量。

### 4.2 D 控制端口计算

当前代码按以下公式建立 D listener：

```text
D 控制端口 = kv_port + 全局 DP rank × TP size + TP rank
```

本次 D TP=1、DP rank=0～7，所以端口为 `31000`～`31007`，P 必须完整列出这八个地址。
控制连接用于注册和协调；实际 HBM Pull 还使用 SDK ticket 返回的动态监听端口。
`control_backlog` 的实际效果受 D 网络命名空间中的 `net.core.somaxconn` 限制；增加它不能修复 UB 数据面或 MR 探测错误。

### 4.3 TP0-only 的适用范围

本次 `source_tp0_only=true` 要求模型类型 `glm_moe_dsa`、D TP=1、主 KV 与 Indexer KV 均 fp8，
并显式设置 `enable_dsa_cp=false`、`c8_enable_reshape_optim=false`。
逻辑 TP0 不一定是物理 NPU 0；`ASCEND_RT_VISIBLE_DEVICES` 重排后仍按逻辑 rank 判断。

改为原始全建链时，P/D 同时设为 false 或同时删除该字段，停止并排空旧 P/D 后重启。
全建链增加参与注册的 P rank；MR 数、源组数不等于实测 Jetty 数，必须重新验证初始化、资源占用和精度。
P 预注册所有 D，不表示一个推理请求会被广播给全部 D；单请求使用实际路由到的 D。

数据路径也不同：Target 满块主 KV 从 P HBM 进入 K DDR；Indexer、scale、tail 等按实现进入 D HBM，
D Decode 再按 Hot/Get 和 ABC/tail 规则消费。不能将所有 PD 流量都归为 K DDR 或将它们都视为图内算子。

## 5. Ascend `additional-config` 参数

| 字段路径（相对 `additional-config`） | P / D 基线 | 源码默认值 | 作用及注意点 |
| --- | --- | --- | --- |
| `enable_cpu_binding` | `true / true` | `true` | 启用框架 CPU 绑核能力；不等价于 K 内存 NUMA 绑定 |
| `enable_dsa_cp` | `false / false` | `false` | DSA 上下文并行计算开关；false 不关闭 Indexer 或稀疏注意力；TP0-only 要求关闭 |
| `multistream_overlap_shared_expert` | `true / true` | `false` | 共享专家计算与路由专家等工作的多流重叠；不控制 PD 或图模式；基线显式开启，删除后默认关闭 |
| `c8_enable_reshape_optim` | `false / false` | `true` | C8 缓存 reshape 优化路径；TP0-only 要求显式 false，不能随意删除此项 |
| `recompute_scheduler_enable` | P 未配置；D `"True"` | `false` | D 重计算调度兼容入口：抢占时尝试保留 KV，必要时让请求回 P 重做 Prefill，而非让 D 本地跑 Prefill |
| `finegrained_tp_config.lmhead_tensor_parallel_size` | P 未配置；D `8` | `0` | 为输出 logits 的 LMHead 建立独立并行组，不把整个 D 主模型改为 TP8 |
| `enable_mlapo` | P 未显式配置；D `true` | `true` | 允许受支持的融合注意力预处理路径；实际选择仍检查量化类型、模型层与硬件，true 不保证每层都走 MLAPO |

共享专家重叠在某些其他配置下会被框架关闭，例如 `enable_fused_mc2=1`；本基线不据此新增其他开关。
遇到共享专家函数栈仍需查看最终异常，不能仅按开关取值认定故障原因。

### 5.1 D 的 LMHead TP8 为什么可与主 TP1 共存

`--tensor-parallel-size=1` 定义主模型 TP；`lmhead_tensor_parallel_size=8` 选择细粒度 LMHead 分组。
当前实现沿 DP rank 维度为相同 TP rank 组建该分组，因此本次 D 的 8 个 DP rank 可以组成 LMHead TP8。
配置要求细粒度分组大小能整除 DP 大小，改变 D DP 时须一起核对；不能因主 TP1 就将 LMHead TP8 判为无效。

### 5.2 `"True"` 字符串与 JSON 布尔值

D 实测脚本保留顶层兼容写法：

```json
{
  "recompute_scheduler_enable": "True"
}
```

当前解析允许旧顶层入口，再通过类型化配置处理；新配置入口是
`additional_config.scheduler_config.recompute_scheduler_enable`，嵌套值优先于旧顶层值，并有迁移提示。
本文不更改已验证脚本，也不建议同时写两个入口。

其他字段应按各自类型填写，例如 `source_tp0_only`、`verify_kv`、`kvaf_config.enabled` 都严格要求布尔值，
不能照搬 `"True"` 字符串。JSON 使用小写 `true/false`；同一对象不要重复定义键。

## 6. `additional-config.kvaf_config`：SDK、DDR 和执行资源

| 字段 | P / D 基线 | 源码默认值 | 作用、单位与限制 |
| --- | --- | --- | --- |
| `enabled` | `false / true` | `false` | D 启用 KVAF Runtime；P 必须为 false，但其 PD producer 仍由 Connector 启用 |
| `build_dir` | 两端 `/home/c00612283/KVAF-Devlop/build-pd-py312` | 空字符串 | SDK 编译产物根目录；PD 两端都必须有值，不是 SDK 源码目录 |
| `cpu_address` | P 未配置；D `141.62.38.24:28997` | `127.0.0.1:28999` | K 数据服务 endpoint；本基线必须显式指向 `.24:28997` |
| `cpu_addresses` | 两端未配置 | `{}` | 可选物理 NPU ID → K endpoint 路由，命中时优先于 cpu_address；本次 D 全部共用一个 K endpoint |
| `ddr_size_gb` | P 未配置；D `16` | `1.0` | 每个 D worker 的 DDR 预算；代码按 GiB，即 `2^30` bytes 换算；须为正有限数 |
| `npu_core_count` | P 未配置；D `1` | `2` | 传给 SDK 的 AIV 执行核数；框架允许整数 1～48，不是 TP、DP 或 CPU 核数 |
| `jettys_per_core` | P 未配置；D `1` | `2` | 每核 Jetty 配置；框架允许整数 1～64，不代表整机队列总数 |
| `hot_tokens` | 未显式配置 | `8192` | 每请求/层的历史热区容量，单位 token，不含固定 384-token ABC tail；至少 8192 且为 128 的倍数 |
| `verify_put` | 未显式配置 | `false` | Put 抽样回读校验入口；与 Connector 的 verify_kv 不同，本次不新增开启 |

表中未配置的 P 侧默认值不表示 P 自动建立对应 DDR Runtime；PD 校验要求 P enabled=false。
`build_dir/python` 被加入 PYTHONPATH，实际 `kvaf` import 还会校验位于该目录下，防止导入旧 binding。
PD loader 检查多 MR Pull API、`ReadyToken.mr_id` 等接口；P/D binding 与 K 服务必须来自同一 SDK 提交。

### 6.1 DDR 容量计算

```text
每 worker 预算 = 16 × 2^30 bytes
8 个 D worker 的合计预算 ≈ 128 GiB
实际 slot 容量 = floor(预算 / 对齐后的 slot_size)
```

当前 PD 用 128-token × 656-byte payload，并按 slot 对齐向下取整；初始化的实际字节数可略低于配置预算。
同一 worker 的预算覆盖该 Runtime 的 Target DDR 层存储，不是每层各分配 16 GiB。
宿主机还要预留 SDK、系统和其他进程内存；增大 DDR 不会自动扩大 NPU hot cache。

### 6.2 地址优先级与初始化门控

`cpu_addresses` 按实际物理设备 ID 查找，未命中回退到 `cpu_address`。
同一 K endpoint 的 D 应使用同一个协调器及同一种 endpoint 字符串；不同 IP/主机名别名不会自动归为同一门控键。
协调器只控制初始化准入，不承载 KV 数据，不是用于修复 MR READ 失败的重试服务。
初始化失败可封锁对应 endpoint；不能在旧初始化仍执行时重启协调器放行新一轮。

## 7. 启动环境变量和日志

以下是实测脚本中的设置值，不作为其他硬件/软件版本的默认值声明。
HCCL、torch_npu 和驱动相关变量的具体实现以部署的配套版本为准。

| 参数 / 变量 | 本次值 | 作用与范围 |
| --- | --- | --- |
| `local_ip` | P `.12`，D `.24` | shell 变量，供本机业务 IP、DP 地址及通信变量使用；不是自动修改 JSON 地址的模板 |
| `nic_name` | `enp34s0f1` | shell 网卡变量，须与当前网络命名空间一致 |
| `VLLM_HOST_IP` | 本机业务 IP | 框架使用的本机地址；不等价于 HTTP 的 `--host 0.0.0.0` |
| `HCCL_IF_IP` | 本机业务 IP | HCCL 网络地址选择 |
| `GLOO_SOCKET_IFNAME` | `enp34s0f1` | Gloo socket 网卡选择 |
| `TP_SOCKET_IFNAME` | `enp34s0f1` | TP socket 通信网卡选择 |
| `HCCL_SOCKET_IFNAME` | `enp34s0f1` | HCCL socket 网卡选择，不是 KVAF UB EID 覆盖 |
| `ASCEND_RT_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | 限定可见物理 NPU 及逻辑顺序；调整后核对物理 ID 路由 |
| `PYTHONPATH` | SDK `build-pd-py312/python` 置于已有路径前 | 优先导入本次构建的 KVAF binding，保留已有路径 |
| `source third_party/KVOP/env.sh` | P/D 均执行 | 加载已构建 KVOP 的运行环境；不能替代编译 |
| `HCCL_DFS_CONFIG` | `task_exception:off,inconsistent_check:off` | 当前 HCCL 任务异常与一致性检查策略配置 |
| `HCCL_CONNECT_TIMEOUT` / `HCCL_EXEC_TIMEOUT` | `1800 / 1800` | HCCL 建链/执行超时，单位秒；不同于 PD timeout=600 |
| `HCCL_BUFFSIZE` | `400` | HCCL 通信缓冲区配置，文档单位 MB；实际内存开销由配套库决定，不是 DDR region 大小 |
| `HCCL_OP_EXPANSION_MODE` | `AIV` | HCCL 算子展开方式；不表示所有 KVAF PD 搬运都使用同一 AIV 路径 |
| `HCCL_INTRA_PCIE_ENABLE` / `HCCL_INTRA_ROCE_ENABLE` | `1 / 0` | HCCL 机内通信路径设置，不等同于关闭全部跨机 RDMA |
| `TORCH_HCCL_ASYNC_ERROR_HANDLING` | `0` | torch_npu/HCCL 异步错误处理设置；不关闭 D 的异步调度或 MTP |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | `0` | 保留当前 FlashComm1 环境设置；配置解析还受其他通信条件影响，不能理解为关闭全部通信 |
| `VLLM_ASCEND_ENABLE_PREFETCH_MLP` | `1` | 保留用户实测环境变量；当前仓库未定位到该变量读取点，是否生效需核对配套上游，不能据设置值宣称已执行预取 |
| `OMP_PROC_BIND` / `OMP_NUM_THREADS` | `false / 10` | OpenMP 绑核及线程数设置；线程数不是整机 CPU 核预算，与 enable_cpu_binding 是不同入口 |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | torch_npu 内存分配器配置；不是模型图开关 |
| `TASK_QUEUE_ENABLE` | `1` | torch_npu 任务队列配置；不替代 stream/event 依赖 |
| `ASCEND_GLOBAL_LOG_LEVEL` | `3` | Ascend 运行日志级别，不控制 vLLM Python logger 的全部输出 |
| `ASCEND_PROCESS_LOG_PATH` | `pd-logs/p或d/时间_PID/plog` | Ascend 进程日志目录 |
| `PROMETHEUS_MULTIPROC_DIR` | `/dev/shm/vllm_metrics_p或d_PID` | 每轮独立的多进程指标目录，不复用旧服务目录 |
| `unset ...proxy` | 清除当前服务 shell 的代理变量 | P/D 服务连接不走这些代理；不会修改其他终端的安装代理 |

`exec > >(tee "$dir/server.log") 2>&1` 同时输出终端和日志，后续 `exec python3 ...` 保持前台运行。
不要用 `Ctrl+Z` 暂停服务或 tee；独立 metrics/log 目录也不表示旧模型进程和 HBM 已自动释放。
若要定位 SDK 原生错误，除 P/D server.log 外，也检查本轮打印的 KVAF 原生日志和 K 服务日志。

## 8. 编译参数与 K/协调器/Proxy 参数

### 8.1 P/D 编译环境

| 参数 | 本次值 | 作用 |
| --- | --- | --- |
| `KVAF_PD_PYTHON` | `/usr/local/python3.12.13/bin/python3` | 选择安装、KVOP 和 SDK binding 的解释器 |
| `ASCEND_REPO` | `/home/c00612283/vllm-ascend-v0.30.0rc2` | 框架源码目录 |
| `KVAF_PD_SRC` | `/home/c00612283/KVAF-Devlop` | 已准备好的匹配 SDK 源码，不是输出目录 |
| `KVAF_PD_BUILD` | SDK `build-pd-py312` | SDK 输出目录，与 P/D build_dir、PYTHONPATH 一致 |
| `MODEL` | GLM 模型目录 | 部署记录中的模型路径；serve 仍显式传入该路径，修改变量不会自动重写启动脚本 |
| `SOC_VERSION` | `ascend950dt_9582` | 选择对应 SoC 的原生构建目标 |
| `CANN_ROOT` | `/usr/local/Ascend/cann-9.2.0` | 供脚本加载 set_env.sh 或 bin/setenv.bash；变量本身不完成加载 |
| `install_kvaf_pd_source.sh` 第一个参数 | `-` | 保留已安装的 vLLM，仍编译 vLLM-Ascend、KVOP 和 KVAF SDK |

K 的 CPU-only CMake 使用 `KVAF_BUILD_REMOTE_CPU_SERVICE=ON`，并关闭 Python、AIV Get/Put 和 NPU offload 构建。
`KVAF_ENABLE_CPU_OFFLOAD=OFF` 不等于关闭 K 数据服务。
协调器只需 Python 标准库和保持相对目录的两个框架 Python 文件，不需在 K 安装/编译完整 vLLM、Ascend 或 KVOP。

### 8.2 运行地址与端口对应关系

| 组件 / 参数 | 本次值 | 必须对应的位置 |
| --- | --- | --- |
| K `--listen / --port` | `141.62.38.24 / 28997` | D `kvaf_config.cpu_address` |
| K `--memory-page-mode` | `default` | CPU 服务内存页模式，保留实测配置，不表示已经启用大页 |
| 协调器 `--host / --port` | `141.62.38.24 / 28998` | D `initialization_coordinator` |
| 协调器 Python | `command -v python3` | 使用宿主机解释器，不使用容器 Python 的绝对路径 |
| Proxy `--host / --port` | `141.62.38.12 / 8000` | 客户端请求入口 |
| Proxy `--prefiller-hosts / --prefiller-ports` | `141.62.38.12 / 8100` | P HTTP 服务，不是 P 的 DP RPC 端口 |
| Proxy `--decoder-hosts / --decoder-ports` | `141.62.38.24 / 8200` | D HTTP 服务，不是 D 控制端口列表 |

改变 shell 的 local_ip 不会替换内联 JSON 或其他机器的地址，需逐项对齐上表。
K 的 `LISTEN`、协调器的 `KVAF_INIT_COORDINATOR_LISTEN`、P/D `/health`、Proxy `/healthcheck` 分别证明不同组件状态。
后台 PID 存在或 HTTP 健康都不代替源 MR 探测、实际 KV Pull 和模型精度验收。

## 9. 常见修改及关联检查

| 修改目标 | 需要一起核对 |
| --- | --- |
| 更换 P IP 或 HTTP 端口 | P 本机环境、DP 地址、Proxy prefiller 地址和客户端 Proxy 入口 |
| 更换 D IP 或 kv_port | D 本机环境、DP 地址、P 的全部 decode_endpoints；Proxy decoder HTTP 地址单独同步 |
| 将 K/协调器迁移到其他主机 | 新主机监听 IP、D cpu_address/cpu_addresses、initialization_coordinator；两项服务可能独立迁移，不能只改一项 |
| 增大 D 并发 | max-num-seqs、HBM hot/staging、MTP query 数、图捕获容量和 DDR 总预算；不只增加 max-num-batched-tokens |
| 增大 DDR | 按每 worker × DP 数核算 K 总内存，重建 Runtime；不自动增加 HBM 热区 |
| 调整 D DP | 本地 DP、可见设备、控制端口清单、LMHead 分组整除条件及 K 总容量 |
| 开启字节诊断 | P/D 同时 diagnostics_requests>0、verify_kv=true；恢复性能测量前同时改回 0/false 并重启 |
| 恢复原始全建链 | P/D 同时 source_tp0_only=false，排空重启并重新验证源注册资源、MTP、长请求及精度 |
| 遇到 Probe rejected | 核对实际 K endpoint、同轮 CPU probe 日志、SDK 提交和 MR 数据面；延长 PD timeout 不会直接修复 |

上述运行配置不作为热更新接口。修改前停止新请求、排空并停止相关 P/D，按
`K → 协调器 → D → P → Proxy` 顺序恢复；健康且未更换的 K/协调器不要求机械重启。
K 一旦重启，旧 P/D ticket 失效，不能复用原客户端继续运行。

## 10. 源码依据

| 源码 | 本文对应内容 |
| --- | --- |
| [PD 配置校验](../vllm_ascend/distributed/kv_transfer/kvaf/pd/config.py) | 角色、端点、TP0-only、MTP、prefix cache、PP/PCP/DCP 和字段类型约束 |
| [KVAFConfig](../vllm_ascend/distributed/kv_transfer/kvaf/config.py) | Runtime 默认值、范围、DDR 与热区字段 |
| [PD Connector](../vllm_ascend/distributed/kv_transfer/kvaf/pd/connector.py) | 源 rank 选择、D 控制端口计算 |
| [PD Runtime](../vllm_ascend/distributed/kv_transfer/kvaf/pd/runtime.py) | SDK 导入检查、对齐分配、初始化门控与 dual Pull |
| [KVAF Adapter](../vllm_ascend/distributed/kv_transfer/kvaf/adapter.py) | 物理设备 cpu_addresses 优先级和原生日志路径 |
| [AscendConfig](../vllm_ascend/ascend_config.py) | 图扩展、共享专家、MLAPO、调度兼容字段与细粒度 TP 配置 |
| [并行组初始化](../vllm_ascend/distributed/parallel_state.py) | 沿 DP rank 维度建立 LMHead 等细粒度组 |
| [初始化协调器](../vllm_ascend/distributed/kv_transfer/kvaf/pd/initialization.py) | 初始化准入与失败封锁语义 |

部署脚本来自用户实测记录；本文对参数含义与源码的核对不代表新增组合已经完成硬件性能或精度验收。
