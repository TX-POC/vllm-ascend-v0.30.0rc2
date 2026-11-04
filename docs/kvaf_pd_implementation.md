# GLM5.3 W4A4C8 PD：实现与验证记录

日期：2026-11-04。开发分支：v0.30.0rc2-develop。

本轮已在首批组件基础上接入独立 PDKVoffloadConnector、外部 Worker/Runner、真实内存注册、布局交换与双 Pull。原生 Runner、Worker、注意力、Connector factory 均未修改。功能通过原生 Connector/module path 与 --worker-cls 显式启用，默认路径不加载这些扩展。

原始目标保持两台 P 的 TP8/DP2/EP16、两台 D 的 TP1/DP16/EP16、八台 K、原生 Proxy。单机 8 卡四层是临时验证配置，不限制最终代码的 worker 数、实际层数或 K 地址数量。

框架原始基线：af69306b27f703c4227d2ab312208c5df2c1a986；首批组件：cef996c1b0c6a7c0f6c85c51ceb91e2a53d9f937。SDK 契约核对：KVAF codex/cpu-pull-hbm-reference 的 a2f212b2811a40ea2df4351a451381897351dd49。

## 1. 开发计划落实

| 阶段 | 本轮实现 | 验证边界 |
| --- | --- | --- |
| M0 需求与地址契约 | 保持 C01～C06、三条 PA offset 公式、主 KV 满块/有效 tail 分流 | 延续用户确认结果 |
| M1 外部扩展与依赖 | 原生 Connector 接口、继承的 Worker/Runner；SDK device RA 借用补丁与源码构建脚本 | 对照 vLLM v0.30.0 抽象接口；补丁应用检查、C++ 语法检查；HCCL 共存待硬件 |
| M2 初始化与注册 | D ticket、按层真实 arena、每 D peer 独立 CPU/NPU source、PA metadata、SDK attach、detach | 真 TCP、真实 CPU Tensor、SDK 边界替身；16×16 注册矩阵 |
| M3 资源与地址 | 全部 cache group、generation、原生 tail block 到普通接收槽的绑定、单层 DDR key/Put entries | block table、typed scale、PA/UBA 区分、TP8→TP1、空贡献 |
| M4 按层双 Pull | 写事件后 Ready、HBM raw READ、主 KV CPU Pull、4096 分批、原 Put 恢复、完成汇总 | 字节、padding、chunk、Commit 失联、失败/取消、源复用 |
| M5 Decode Backend | singleton DDR、原生目录/Get、普通 tail 在计算准备期导入 ABC、后续稳定满块发布 | 实际原生 Backend 的 CPU Get/续写；保持全局 GROUP_WIDTH=4 |
| M6 轻量验证与交付 | 新增 UT、已有 KVAF 回归、原生格式配置与部署说明 | CPU/mock 与语法检查；NPU 精度、性能、多机长稳待现场 |

详细配置和生命周期说明见 [Connector 部署说明](kvaf_pd_connector.md)。

## 2. 接入与所有权

- Scheduler 继承原生 SFA PD 的请求路由、metaserver rendezvous 和 block 分配；新增 generation 与全部 cache group seed。不把 DSpark 的上下文协议带入普通 MTP。
- P 普通主 KV、indexer、scale 在真实逐层 arena 中生成最终原生 Tensor views。实际没有 indexer 的层只注册 main；MTP 层以实际 drafter ownership 判断。
- D 的 Hot/ABC 保持原生计算 Backend 所有。独立普通 tail 接收容量在 profile 前计入；最终 arena 的 indexer 使用原生 blocks，tail 查询 Backend 对本请求原生 block 的绑定。
- 一个 D worker 使用一个 K endpoint 的独立 Runtime/region。KVAF 内部完成 PA/UBA 映射与 K 同步；业务 TCP 只传 PA 布局和不透明 SDK 对象。
- P 到所有 D region 建链不等于请求广播。复制的主 KV 由一个贡献方发送，indexer/scale 沿原生 TP 区间分配；空贡献仍确认。
- PreparePut 仅在源已就绪、Runtime 可独占时发起，避免 SDK pending Put 长期阻断普通 Get。PD 接收和原生 Decode Get 使用同一串行锁，并在 PD 使用 Runtime 前等待 Runner 记录的计算事件，覆盖图重放异步 Get。D 控制监听延后到原生 warmup 完成。
- 满块主 KV 使用现有 writer.slots 与 writer.requests，Commit 后通过原生目录方法发布，不引入第二套 DDR 分配器或手算绝对 DDR 地址。
- P 图重放通过 frame 入口复用保护和成功 forward/sample 后的 Target 通知补发接入；CPU/fullmesh EID 按本机物理卡分别配置，支持多 TP/DP worker。
- MTP snapshot 在原生 deferred finalization 后发送。所有实际必需层和贡献方完成、DDR 发布与 HBM 数据就绪后，才通过原生 finished_recving 放行 D。
- 在途请求的原生释放会暂缓；安全完成后释放 DDR key 和 tail 槽。源复用须经过 SDK release。未知 DMA/Commit 保留原句柄和存储、停止后续 Get；终态失败接入原生 load-error 和请求完成回报，不额外增加业务 rePrefill。
- 正常退出先排空，再 D detach、P source shutdown、Runtime/DDR 关闭。完成后不保留每批 NPU descriptor arrays 到请求结束。

## 3. 依赖兼容与关闭方式

PDAdapter 适配新分支现有的 CPU Pull/raw READ 方法签名。scripts/install_kvaf_pd.sh 从用户既有源码构建，应用 scripts/patches/kvaf-pd-borrow-device-ra.patch；它在显式启用时跳过已有 HCCL device RA 的重复初始化，不修改其默认独立运行行为。新 binding 以能力标记防止误加载未补丁版本。SDK 不透明类型只提供 pickle 构造，工具导入检查仅对当前受限元数据通道增加定点例外。

重新使用既有原生 Worker 和既有 Connector 配置，新 PD 路径保持关闭；不需要修改原生文件或运行一次全局替换补丁。旧非 PD KVAF 安装脚本和全局四层 DDR 分组保持原实现。回退前仍须先排空在途传输。

## 4. UT 与静态验证

```bash
PYTHONPATH=third_party/KVOP python -m unittest discover -s tests/ut -p 'test_kvaf_*.py' -v
ruff check vllm_ascend/distributed/kv_transfer/kvaf/pd tests/ut/test_kvaf_pd*.py
ruff format --check vllm_ascend/distributed/kv_transfer/kvaf/pd tests/ut/test_kvaf_pd*.py
bash -n scripts/install_kvaf_pd.sh
```

新增 test_kvaf_pd_flow.py 使用真实回环 TCP，替换 SDK/NPU 边界但运行实际注册、地址、事务和 Backend 代码：四层加 MTP 字节、TP4/TP8→TP1、chunk、空贡献、尾块、不提前放行、Commit 原事务恢复、失败/取消及释放。生产注册测试覆盖 TP8/DP2 的 16 个 P worker、TP1/DP16 的 16 个 D region 和 8 个 K endpoint。

新增 test_kvaf_pd_integration.py 直接执行复用的原生 Scheduler 实现，检查原生 Proxy 参数、全局 DP 端口、MTP cache groups、容量与取消；Runner 的 NPU 边界替身检查最终 Tensor 存储和 typed scale；Connector 生命周期和显式开关独立校验。首批地址、事务、门控及 Backend UT 继续保留。

最终全量 KVAF UT：160 项通过（原有 121 项与本轮新增 39 项）。修改范围的 Ruff check/format、Python 编译、Shell 语法、禁止导入检查通过；文档 Markdown、拼写和 typos 检查通过，Gitleaks 使用缓存二进制重跑通过。SDK Host binding 构建与实际 CPU/NPU Ready 类型往返通过；SDK CTest 的 13 项 Host 用例通过，1 项 NPU binding 参数测试因本次 Host 构建没有 NpuRuntime 而失败，未将其计为硬件验证通过。全仓 bash format.sh ci 在隔离文件副本执行；基线存在原有格式、拼写、日志和 workflow 检查问题，不通过修改原生核心来修复无关项。新增/修改文件单独检查；全仓未通过不能写成全仓 CI 通过。

没有执行真实 NPU binding 全量构建、模型精度、UB 数据面、生产全建链、性能或长稳测试。云环境无 NPU/CANN/UB；CPU/mock 通过不替代现场验证，也不证明原生 engine 在所有硬件错误下的最终行为。

## 5. 现场后续验收

先用已能运行的单机四层模型验证实际 cache views、RA/HCCL、双 Pull 和 MTP，再按原始生产拓扑检查全层、多请求、长上下文、并发、图模式、取消/抢占与资源回收。采用 SDK 独立测试、临时单机模型、原始生产部署三个验收层次，分别记录版本和 PASS，不互相替代。

入口：[通用安装测试](kvaf_pd_install_deploy_test.md)、[机头 CPU 测试](kvaf_pd_ascend_host_cpu_test.md)、[临时单机配置](kvaf_pd_single_server_8npu_4layer.md)。
