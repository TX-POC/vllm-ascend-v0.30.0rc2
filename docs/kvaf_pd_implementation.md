# GLM5.3 W4A4C8 PD：首批实现与验证记录

日期：2026-11-04。开发分支：`v0.30.0rc2-develop`。

用户已授权按已确认设计开始实现，并明确要求提交开发分支前新增 UT。本记录描述**第一批组件实现**，不代表完整 `PDKVoffloadConnector` 已可部署。

框架基线：`af69306b27f703c4227d2ab312208c5df2c1a986`。SDK 契约依据：KVAF `codex/cpu-pull-hbm-reference`，`a2f212b2811a40ea2df4351a451381897351dd49` 的 `docs/connector_pull_api_design.md`、Python 绑定及 Runtime 源码。

## 1. 本批实现

所有运行时代码均新增于 `vllm_ascend/distributed/kv_transfer/kvaf/pd/`，没有修改原生核心、注册表、CLI、算子或已有 KVAF 逻辑。包导入不注册 Connector、不创建 Runtime/MR、连接、线程或 Tensor，不加载 KVAF/torch_npu。现有启动方式继续走现有路径。

| 文件 | 已实现内容 |
| --- | --- |
| `layout.py` | 从实际 Tensor 检查逐层 PA 布局；逻辑 block 查各自 block table；生成源 offset、HBM output offset；满块/tail 分流；原生 TP 贡献区间；Put entries 地址绑定及范围检查 |
| `registration.py` | 每层、每 D peer 独立 CPU/NPU source；返回 PA layout 和不解包的 SDK descriptor/ticket；生产事件完成后 Ready；按 source/sequence 匹配释放；失败保活 |
| `transfer.py` | 原 Put 的 Submit/Wait/Commit/Abort 与恢复；raw READ output/三数组保活和 `resolve_raw_addr`；复用 `KVTransfer._publish_addresses` 做 Commit 后目录发布 |
| `state.py` | 请求 epoch、实际 P 贡献者、层/chunk 的完成汇总；区分 P 源复用与 D 首轮 Decode 门控；覆盖空贡献者和显式列入的 MTP 层 |
| `backend.py` | PD 专用逐层 DDR 几何；复用原生目录/Get；完成接收的 tail 在首次 Decode prepare 才导入 ABC；复用原生续写、稳定满块发布及释放 |
| `tests/ut/test_kvaf_pd.py` | CPU UT，使用真实 Tensor/字节数据、原生目录发布方法和 SDK 边界替身 |
| `tests/ut/test_kvaf_pd_backend.py` | 逐层目录/Get、首次 Decode tail 导入、4-token 写入节奏及稳定满块续写的 CPU 回归 |

本批没有增加可启动的 `PDKVoffloadConnector` 注册入口，也没有提供可执行部署配置。Scheduler/Worker 接线、ZMQ 消息服务、Runtime 初始化、全网 MR attach、真实接收区分配尚未完成。Backend 的接收引用/续写已在 CPU 层实现，但未接入 Runner。组件不能单独用于生产 PD 推理。

## 2. 保持的需求与地址契约

- P：TP8/DP2/EP16，D：TP1/DP16/EP16；D worker 单 K endpoint、独立 Runtime/region。EP 不再乘出额外 worker。
- P→D 静态 `KVAF_Meta` 布局字段是 PA。SDK descriptor/ticket 原样传递，内部 UBA 信息由 KVAF 注册、导入和同步。
- 每个 plane 先由其 block table 取得 P 物理 block，再计算 `(plane_PA - layer_PA) + physical_block * 128 * token_bytes`，宽度分别为 656、128、4 字节。
- HBM 目的偏移相对本次实际 `output.data_ptr()`；不拿不同分配的最小/最大指针伪造连续源或 output。
- 只把主 KV 满块送 DDR；tail 只传有效字节到 HBM；Indexer/Scale 保留完整有效历史。N=300 时 DDR 主 KV 167,936 字节、HBM tail 28,864 字节，84 个 padding token 不搬运。
- Connector 不计算 Hot/LRU/ABC 地址。只有 D Backend 在计算准备阶段导入有效 tail 并按原生逻辑装载 Hot。
- MTP3、请求级双路径门控、D 本地 prefix 关闭，以及原事务恢复后走原生错误通道的 C01～C06 决策不变。

`plan_layer` 的 `first_block` 表示首个尚未传输的逻辑块。非最终 chunk 只发送新满块；最终 chunk 才带 tail。调用方在完成后推进该值，不能直接取上一 chunk 长度的向上取整值。TP 贡献规则复用原生 SFA 的块区间算法；C8 模型真实复制性、draft ownership 和 alias-storage 生命周期仍须在 Worker 接线时核实。

## 3. 调用及所有权约束

1. 初始化时，每个真实层存储、每个 D peer 建立独立 `LayerRegistration`。SDK 当前 NPU source 仅允许一个 reader，不能复用一个对象服务全部 D。CPU source 绑定该 D region 的 ticket。生产端和消费端布局都必须对应实际连续 Tensor。
2. 原生分配器仍拥有 D HBM block。DDR key 必须来自现有 `writer.slots[geometry]`，请求目录来自 `writer.requests`，本模块不建立第二个 key 池。
3. 源写事件完成后发布 Ready。收到完整源信息后，在 Runtime 可独占的短事务窗口内执行 PreparePut；当前 SDK 从 Prepare 起阻止普通 Get，不能长期预留后等待 Prefill。
4. `LayerPlan.ddr_segments` 从本次 `put.entries` 取绝对 DDR 地址并核对 key/长度；空满块集合不调用 CPU Pull。当前逐层计划会拒绝四层交错 payload。
5. `CpuPullTransfer` 使用调用方已有的 Put；Submit 只提交，Wait 只确认搬运，Commit 成功后才允许 publish。丢失响应只恢复原 Put，Commit 决定不得改 Abort。DMA 不确定时保留源和目的空间。
6. `NpuPullTransfer` 在给定 NPU stream 创建连续 int64 三数组，数据进入 output，返回值只代表完成。异常后只有原 Ready 的 `resolve_raw_addr` 成功才能记完成。失败清理/取消不能当作数据就绪。
7. 每批最多 4096 段。`Segments.batches()` 只分描述符；每个子批必须申请独立 Ready。CPU 批次须同时划分 Put/key，不能拆同一 Put 后用多个单源 Submit 填孔。
8. Worker 的 Runtime 访问，包括既有 Decode Get，必须统一串行协调。组件不启动后台线程，也未提供跨线程并发保证。
9. 只有已完成的传输才能产生完成事件；业务 ACK 不能代替 DMA 证明。P 物理 storage 复用还必须汇总同一存储的所有注册源/reader。取消、抢占或超时本身不解除保活。
10. 退出先停止新任务并恢复/完成旧任务，再 release batch、D detach、P source shutdown，最后释放 Runtime/DDR。源码组件保留对象引用，但完整 Worker 仍须承担全生命周期协调。

## 4. M1 已证实的接入缺口

以下是源码接口事实，不是重新询问已确认的业务定义。

| 技术项 | 源码依据 | 对完整接入的影响与下一步 |
| --- | --- | --- |
| 四层交错 DDR 与逐层源 | 框架 `kvaf/layout.py:grouped_layouts`、`cache_policy.py`；SDK `runtime/src/connector_runtime.cc:SubmitCpuPull`、API 第 12 节 | 用户已选择 PD 外部 Backend 改用逐层 DDR。已实现 singleton group、目录/Get 复用及 CPU UT；不修改原生全局分组 |
| 单层注册的真实存储 | `model_runner_v1.py:_allocate_kv_cache_tensors`、`_allocate_sparse_c8_indexer_tensors`；SDK source register 绑定 | 当前 main/indexer 并非天然一个共同层 allocation。本批只接受经实际存储验证的布局。下一步在外部 Runner/Backend 的分配阶段组织逐层 arena；不在传输时伪造范围或增加整层拼接复制 |
| D 普通 tail 接收区 | `model_runner_v1.py:_initialize_kvaf_cache_state`、`_allocate_kv_cache_tensors`、`_reshape_kv_cache_tensors` | 当前 Target 绑定 Hot。PD Backend 已提供完成接收的引用绑定、首次 Decode 导入和续写；Runner 普通接收 buffer 分配与原生 block 关联仍需接线 |
| RA/HCCL 兼容 | `kvaf/adapter.py:connect`、`scripts/patches/kvaf-reuse-device-ra.patch`；SDK `bindings/src/npu_python_runtime.cc` | 旧 Adapter 传 `reuse_device_ra`，目标分支构造签名未包含它。必须适配依赖的 RA 生命周期，不能通过删参数忽略 TP/EP 已有 HCCL 会话 |
| 初始化顺序/归档绑定 | `model_runner_v1.py` 中 register_kv_caches 早于 KVAFAdapter.connect，后续调用 connector.bind_archive | Worker 注册阶段不能假定 DDR 已初始化。新 Connector 需要明确两阶段握手并兼容原生 bind_archive 边界，不能把旧 prefix Connector 当作 PD 实现 |
| Runtime 串行事务 | SDK PreparePut 的 pending 保护以及普通 Get 的 Busy 检查 | 必须协调 PD 接收与既有 Decode Get；本批提供事务组件，未接入调度排他窗口 |
| 错误/MTP/首 token 交接 | 原生 `sfa_pd_rd2h` Scheduler/Worker 与匹配的 vLLM engine | 继续复用原生错误、worker 完成汇总和请求元数据生命周期；本批 UT 不代表 engine/模型接线已验证 |

### 4.1 DDR 适配路线（已确认并实施组件）

用户于本轮明确选择：仅在新 PD 外部 Backend 中改用逐层 DDR 布局，保留现有 Pull 接口，同时调整 PD 的目录与 Get 布局。

`create_decode_cache` 使用实际 Target 层名建立 `((layer0,), (layer1,), ...)`，每层几何为 `(656, 128)`，一个 D worker 可共用同一个该几何的 Runtime/region。`PDCachePolicy` 继承原生 KVCachePolicy，显式使用全 1 的 group_sizes，保留原生目录下标、Get 描述符和 key 池；拒绝四层组绑定。没有修改 `GROUP_WIDTH=4`，非 PD 分组保持原实现。

`bind_received_tail` 只保存已完成接收的 Tensor 和有效长度；首次 Decode prepare 调用 Backend 的 `_seed` 时才复制有效字节进入 ABC。短上下文的完整 KV 从 DDR 加载也发生在计算准备阶段。后续 4-token 写入、MTP 对齐和稳定满块发布复用原生策略。UT 已验证跨层 Get 数据、tail 不提前进入 Hot、续写后前 384 个 token 的 DDR 字节内容、同一 key 池的释放。

每层 Get 次数与原四层合并路径不同，当前不承诺性能；内存预算和算子实际输入仍需随 Runner 接线及硬件验证。用户三条 PA→offset 公式、满块/tail 规则及单 K endpoint 策略不变，不新增 SDK Pull 签名。

## 5. 验证

CPU 环境：Python 3.12、torch 2.10.0+cpu。

```bash
PYTHONPATH=third_party/KVOP python -m unittest discover -s tests/ut -p 'test_kvaf_pd*.py' -v
PYTHONPATH=third_party/KVOP python -m unittest discover -s tests/ut -p 'test_kvaf_*.py' -v
ruff check vllm_ascend/distributed/kv_transfer/kvaf/pd tests/ut/test_kvaf_pd*.py
ruff format --check vllm_ascend/distributed/kv_transfer/kvaf/pd tests/ut/test_kvaf_pd*.py
```

- 新增 43 项 UT：通过。
- 已有 78 项 KVAF UT：通过；与本批一起运行共 121 项通过。
- 新增文件的 Ruff check/format、codespell、typos、markdownlint、Gitleaks：通过；Python compileall：通过。
- 全仓 `bash format.sh ci`：已在隔离文件副本执行，未整体通过。失败来自基线文件的格式、`prefix_cache.py` 拼写/日志入口、既有禁止导入和 GitHub workflow 配置；自动格式化触及的 11 个原有文件均未带回开发检出。本次新增文件没有格式修正差异。
- 首次全仓检查还遇到 ShellCheck 工具并行解压冲突；固定工具路径后，单独重跑全仓 shellcheck 已通过。

云工作区保留验证记录：`/workspace/scratch/pd-kvaf-ut.log`、`pd-component-ut.log`、`pd-format-ci.log`、`pd-scoped-lint.log`、`pd-shellcheck.log`。全仓检查失败不能写成全仓 CI 通过；本批没有为修复基线问题改动原生文件。

覆盖范围包括真实字节数据落点、padding、分块边界、不同 P/D block table、TP 贡献和空贡献、非连续 allocation 拒绝、段数/越界/重叠、精确覆盖、源对象隔离、stale sequence、Submit/Commit 丢包恢复、raw READ 恢复、复用原生目录发布与错误保活。

没有运行模型精度、NPU 算子、UB/RDMA、全网建链、性能或端到端 PD 测试。原因包括当前云环境无硬件，以及完整 Connector/Backend 尚未接线；CPU/mock 通过不替代这些验证。

## 6. 后续开发顺序

1. 将已实现的 PD 逐层 Backend 接入外部 Runner，并完成 RA 依赖兼容；保留现有 Pull 签名，同步更新相应 UT。
2. 用外部 Worker/Runner 扩展提供真实逐层 arena、D 普通接收区和可复用 Runtime；落实初始化顺序和内存预算。
3. 新增原生 `KVConnectorBase_V1` 的 `PDKVoffloadConnector`，明确 Scheduler/Worker 分工，通过原生模块路径/插件注册加载，默认不选择。复用现有 ZMQ/调度元数据边界及 P/D 握手。
4. 接入逐层事件、全建链、请求 block 映射、按层双 Pull 与请求级门控；由 Backend 导入 tail/目录并继续 Decode/MTP3。
5. 完成取消/抢占/错误上报和物理 storage 复用闭环，再给出可执行的原生格式部署配置。每批代码新增对应 UT，通过后提交开发分支。

安装、当前 CPU UT、独立 SDK 链路验证及完整 PD 的后续验收步骤见 [安装、部署与测试方法](kvaf_pd_install_deploy_test.md)。
