# 临时 TP0 单源传输模式

分支：`v0.30.0rc2-develop`。新增开关 `source_tp0_only` 默认关闭。
这是现有 KVAF SDK 能力上的临时接入策略，不是 Jetty 生命周期优化。
不包含 develop2 的跨层 arena 或该分支的其他修改。

## 行为与边界

| 项目 | false 或省略 | true |
| --- | --- | --- |
| P KVAF 参与者 | 原有所有 TP worker | 每个 P DP 实例自己的 TP0 |
| KVAF Runtime、逐层注册、连接和发送线程 | 原行为 | 仅 P TP0 创建，非 TP0 不加载 SDK |
| 主 KV 来源 | 原有 member 0 | 所选 P DP 的 TP0 |
| indexer KV 和 scale 来源 | 原有 TP 传输区间分工 | 所选 P DP 的 TP0 提供完整有效区间 |
| D 每层预期传输贡献方 | 原 TP 比例 | 1 |
| 原生计算 TP/EP 和缓存写入 | 原行为 | 原行为，非 TP0 仍参与计算及完成记账 |

TP0 是逻辑 TP rank，不等于物理 NPU 0；例如可见设备为 2、3，TP0 对应物理卡 2。
P DP2 时有两个 TP0，分别为各自请求提供 KV，不跨 DP 混用缓存。
所有 D worker 仍预先注册可接入的 TP0 源；单条请求只使用路由到的 D worker。

当前仅支持 `model_type=glm_moe_dsa` 的 FP8 主 KV/FP8 indexer 缓存路径，
D TP=1，PP=PCP=DCP=1，关闭 DSA-CP 和 C8 reshape 优化。
继续沿用已有 MTP3、关闭原生 prefix caching 等 PD 校验，不扩展其他模型/缓存布局。
普通 indexer 的缓存写入路径保留全量有效 token；原来的 TP 区间划分是传输工作量分摊，
不能据此认定其他 rank 才拥有某段唯一 KV。该模式不新增 KV gather、重算或搬运。
硬件验收仍须确认实际模型的 TP0 缓存覆盖以及同模型基线精度。

## 内存、接口与所有权

不改变现有逐层 Tensor 分配、视图、BlockTable、量化布局、KVAF 注册内存范围及 PA/UBA 映射。
不合并跨层内存，不从非连续 Tensor 伪造 arena，不将其他 TP 的数据复制到 TP0。

- 满 Block 主 KV：K CPU 调用 `submit_cpu_pull`，Wait/Commit 后发布逐层 DDR 目录。
- indexer KV、scale、普通 tail 主 KV：D 调用 `batch_get_raw_addr`，有效字节进入现有 D HBM。
- MTP 主 KV：仍按现有 MTP 业务规则进入 HBM，不作为 Target 主 KV 卸载到 DDR。
- D 的 Hot/Get、tail 到 ABC 交接、Scheduler 分配及模型算子沿原路径执行。

地址公式保持原逻辑，TP0 使用自己的物理 block table：

```text
主 KV offset = (主KV_PA - Layer_PA) + P物理BlockID * 128 * 656
indexer offset = (indexer_PA - Layer_PA) + P物理BlockID * 128 * 128
scale offset = (scale_PA - Layer_PA) + P物理BlockID * 128 * 4
```

传输 cohort 的 member=0、ratio=1，不修改真实模型 tensor_parallel_size。
D 只等待 TP0 的全部层，P 非 TP0 在原生 forward 完成后报告本地完成；原生 worker
完成汇总仍等待 TP0 的传输完成，不能提前释放 TP0 的源。
取消也只由 TP0 向 D 投票，原有 DMA/Ready/Commit 和源复用规则不变。
图模式省略 Python 写钩子时继续使用成功原生 forward 的补发路径；
MTP 仅在实际 proposal 完成后发送，原生 skip 时不读取 dummy KV。

## 启动配置

P、D 均须运行支持该字段的 develop 代码，且开关值一致。
P 在 hello 时检查模式；D 同时拒绝错误模式、非 TP0 来源和多贡献方请求。
旧 D 没有能力字段时被视为 false，不能与开启模式的 P 混用。
此模式由已有原生 `--kv-transfer-config` 设置，不新增环境变量或启动参数格式。

P 的 `kv_connector_extra_config` 示例（保留实际部署的 D endpoint 清单）：

```json
{
  "decode_endpoints": [
    "141.62.38.20:31000", "141.62.38.20:31001",
    "141.62.38.20:31002", "141.62.38.20:31003",
    "141.62.38.20:31004", "141.62.38.20:31005",
    "141.62.38.20:31006", "141.62.38.20:31007"
  ],
  "source_tp0_only": true,
  "timeout": 600,
  "diagnostics_requests": 0,
  "verify_kv": false
}
```

D 的 `kv_connector_extra_config` 示例：

```json
{
  "source_tp0_only": true,
  "initialization_coordinator": "141.62.38.20:28998",
  "timeout": 600,
  "diagnostics_requests": 0,
  "verify_kv": false
}
```

两侧仍使用原生参数：

```bash
--kv-cache-dtype fp8 \
--attention_config.indexer_kv_dtype fp8
```

两侧 `--additional-config` 中保留原有项目，并明确设置：

```json
{
  "enable_dsa_cp": false,
  "c8_enable_reshape_optim": false
}
```

不要用以上 JSON 替换完整 additional-config；kvaf_config.build_dir、D enabled=true、
P enabled=false、D cpu_address 等仍按现有部署配置。
develop 不接受 develop2 的 `source_arena_layers`，切换分支时从启动配置中移除该字段。

## 切换与回退

先保存本轮日志，停止请求，正常结束 P/D 并确认旧传输结束、注册清理完成。
异常退出后若远端旧源状态不明，不要直接重复启动堆叠注册；确认无在途 DMA 后再清理。
重启 K 会使所有旧 ticket 失效，必须重新启动 D/P，不能继续使用旧实例。
K 和协调器健康且无遗留占用时可保留；按 K/协调器、D、P、Proxy 的顺序启动。

使用 editable 安装时，当前新增代码仅为 Python 修改，无需重编译 vLLM、Ascend 算子、
KVOP、KVAF binding 或 K。两端都更新同一 develop 版本。
关闭方式：P/D 同时省略 source_tp0_only 或设为 false，正常重启后恢复原全建链路径。

服务建议直接写文件，不通过可能被暂停的 tee 管道。已有 K 被 Ctrl+Z 暂停，或 tee
暂停导致 pipe_write 的问题，与该开关无关；先恢复日志消费者和服务再测试。

## 验证

开启模式后，每个 P DP 只有 TP0 打印一次完成注册的信息：

```text
KVAF PD source mode: mode=tp0_only dp_rank=0 tp_rank=0 compute_tp_size=8 D_peers=8 layers=79 registrations_per_path=632
```

示例数量取决于实际模型。79 层、P TP8/DP1、D DP8 时，每条 CPU/NPU source 路径
注册数从 8×79×8=5056 减至 79×8=632，不含 SDK 其他内部资源。
K 端剩余总资源仍可能触及上限，不能承诺该模式彻底解决 Jetty 资源问题。
全部读取集中于 TP0，也可能增加其带宽压力；不宣称有吞吐收益或 HBM 节省。

联调时可在两端设置 diagnostics_requests=1、verify_kv=true，发送同一长请求。
检查 P SOURCE_READY 的 worker 均为各 DP 的 TP0，D PULL_PLAN 的 member=0、ratio=1，
两种 PULL_DONE/KV_CHECK 路径齐全，DECODE_READY 的 expected 等于实际层数。
这些诊断读取 KV 并同步设备，只用于测试，完成后恢复 0/false。

MAIN_REPLICA_CHECK 在此模式仅含一个传输贡献方，不证明 TP0～TP7 的副本字节一致。
必须继续进行同模型、同请求、固定采样参数下的模式关闭/开启输出及 token 对比，
覆盖短 prompt、128 token 边界、分块长请求、MTP 正常/跳过、取消和多 DP 路由。
CPU UT 的 SDK/NPU 边界使用替代实现，不能代替真实全模型精度或硬件资源验收。
