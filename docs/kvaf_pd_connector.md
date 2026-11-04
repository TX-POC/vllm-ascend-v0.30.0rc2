# PDKVoffloadConnector：接入、配置与验收

本次实现面向原始 GLM5.3 W4A4C8、DSA、远端 DDR 的 PD 方案。生产拓扑仍为两台 P 上 TP8/DP2/EP16、两台 D 上 TP1/DP16/EP16、八台 K；单机 8 卡与四层模型只是临时验证配置。层数取实际 cache specs，worker 数和端口取原生并行配置，K 地址沿用原生 kvaf_config，不在代码中固定四层、八卡或八个 endpoint。

## 1. 代码入口与数据流

只在显式选择 PDKVoffloadConnector 和外部 PDKVWorker 时启用。未修改原生 Runner、Worker、注意力实现或 Connector factory。

| 入口 | 职责 |
| --- | --- |
| pd/connector.py | 原生 KVConnectorBase_V1 / SupportsHMA 的 Scheduler、Worker 接口及完成回报 |
| pd/scheduler.py | 继承原生 SFA PD Scheduler，复用 Proxy rendezvous、block 分配、请求生命周期；保留全部 MTP cache groups |
| pd/worker.py、runner.py、memory.py | 外部 Worker/Runner；真正的逐层连续 arena、原生 cache views、D 的普通 tail 接收池 |
| pd/sender.py、registration.py | P 按层、按 D peer 注册与 Ready；保护源缓存复用 |
| pd/receiver.py、layout.py、transfer.py | PA offset、双 Pull、原 Put 恢复、DDR 目录发布、请求级汇总 |
| pd/runtime.py、backend.py | 每 D worker 独立 Runtime/region、逐层 DDR、Get 串行协调、计算阶段 tail 导入 |

初始化：先启动 K 和 D。D link_to、初始化 DDR，在原生 warmup/图捕获结束后开启控制监听；P 独立初始化实际 KV，向每个 D 获取 ticket，按层按 peer 分别注册 CPU/NPU source，再把 PA 布局和不透明 SDK descriptor/ticket 交给 D attach。每个 NPU source 只供一个 D reader；不同 D region 不共享 Put 或 source 对象。

业务层的静态 KVAF_Meta 是 **PA**。Register 映射 PA/UBA 并把 UBA 信息同步给 K；Connector 不交换或拼接 UBA 绝对地址。三类源偏移继续使用：

```text
main:    (main_PA - layer_PA) + P_physical_block * 128 * 656
indexer: (indexer_PA - layer_PA) + P_physical_block * 128 * 128
scale:   (scale_PA - layer_PA) + P_physical_block * 128 * 4
```

物理 block 来自对应 cache group 的 block table。D 的原生主 KV block 是虚拟调度容量；Backend 将该请求的原生 tail block 绑定到普通接收池的物理槽，Connector 只查询该绑定计算目的地址，Hot 地址仍由计算 Backend 自己管理。PA 必须属于同一个真实 layer Tensor，禁止用多个独立分配的地址拼出虚构注册范围。最终 reshape 后再次检查原生 Tensor 的存储、stride 和类型。

请求阶段沿用原生 Proxy 先到 D 分配，再通过 metaserver 调用 P。D generation 随原生参数到达 P；每层写事件完成后，P 发送实际源 block metadata。D 规划目的地址，按需建立短 PreparePut 事务，再执行：

- indexer、scale 和未满块的有效主 KV tail：batch_get_raw_addr，进入 D 普通 HBM；padding 不传。
- 仅完整主 KV blocks：submit_cpu_pull，直接进入对应 D region 的远端 DDR；Wait、Commit 和目录发布完成后计入就绪。
- 实际 MTP cache 按原生所有权保留在 HBM；在原生 deferred connector finalization、MTP 写入完成后发送其有效 prompt snapshot，不把 MTP3 当作三个独立模型层，不启用 DSpark 协议。

主 KV 使用复制域的 member 0；indexer/scale 复用原生 TP 块区间分摊和旋转规则，空贡献也完成汇总。全建链只决定可访问范围，请求只使用实际选中的 P DP/TP 贡献组和 D worker。P 在新 frame 写缓存前排空旧读取；图重放省略 Python 写通知时，在原生 forward/sample 成功后补发 Target 层通知，MTP 仍以实际写入通知为依据，不伪造未生成的 KV。

每次最多 4096 段，超限分成独立 Ready/Put 批次。DDR 地址以本次 Put entries 为准。PreparePut 会阻止该 Runtime 的普通 Get，因此不长期持有 Prepare 等待 P 计算；接收与原生 Decode Get 共用串行锁。

## 2. SDK 源码构建

继续使用已经 git 下载的 KVAF cpu-pull-hbm-reference 源码。接口核对版本为 a2f212b2811a40ea2df4351a451381897351dd49，不使用旧的四层分组 SDK 安装脚本来替代这个分支。

TP/EP 的 HCCL 已持有 device RA。新脚本只给 KVAF 依赖应用显式借用 device RA 的小补丁，并完整构建 Get、Put、raw READ 和 CPU Pull；不修改 vLLM-Ascend 核心，也不改变 Pull 方法签名。环境变量 KVAF_REUSE_DEVICE_RA 默认不启用，只有外部 PD Worker 的 SDK 适配器将其设为 1；能力标记检查防止误用未补丁的 binding。

```bash
export KVAF_PD_SRC=/home/c00612283/kvaf-hostcpu-test/sdk
export KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3
export KVAF_PD_BUILD="$KVAF_PD_SRC/build-pd-py312"

# 在框架源码仓根目录执行；先沿用现场已验证的 CANN/编译器环境。
bash scripts/install_kvaf_pd.sh "$KVAF_PD_SRC" "$KVAF_PD_BUILD"
export PYTHONPATH="$KVAF_PD_BUILD/python${PYTHONPATH:+:$PYTHONPATH}"
"$KVAF_PD_PYTHON" -c 'import kvaf; assert kvaf._native.reuse_device_ra_supported; print(kvaf.__file__)'
```

构建目录、Python ABI 必须与推理进程一致。保留现场已通过测试的 Python 3.12、torch 2.10.0+cpu 和 torch_npu 配对。完整 NPU 编译仍需要现场的 CANN、A5 工具链与配套头文件。K 服务继续使用原 cpu-pull-hbm-reference CPU service；不存在统一池容量的新增启动参数。

## 3. 原生配置入口

P、D 均增加原生参数：

```bash
--worker-cls vllm_ascend.distributed.kv_transfer.kvaf.pd.worker.PDKVWorker
```

D 的 kv-transfer-config：

```json
{
  "kv_connector": "PDKVoffloadConnector",
  "kv_connector_module_path": "vllm_ascend.distributed.kv_transfer.kvaf.pd.connector",
  "kv_role": "kv_consumer",
  "kv_port": 31000
}
```

D 的 additional-config 在已有模型配置上合并以下原生字段：

```json
{
  "kvaf_config": {
    "enabled": true,
    "build_dir": "/path/to/KVAF/build-pd-py312",
    "cpu_address": "10.1.0.1:28999",
    "ddr_size_gb": 1
  }
}
```

ddr_size_gb 是每 worker 预算，生产按容量另行设置。多 K 使用已有 cpu_addresses，键为本机物理 NPU ID，值为 K endpoint；两个 D 主机可以分别配置映射。一个 K 可承载多个独立 D region，不将八台 K 虚构为一个全局连续地址空间。

P 使用相同 Connector/module path，kv_role 改成 kv_producer；additional-config.kvaf_config 设置 enabled=false 和同样正确的 build_dir。P 不分配 DDR，其 CPU source 注册使用 D ticket。P 的 kv_connector_extra_config 必须包含完整 decode_endpoints 列表。

D 控制地址的原生规则为：

```text
host = 该主机 VLLM_HOST_IP / 原生 get_ip()
port = kv_port + 全局 data_parallel_rank * tensor_parallel_size + TP_rank
```

生产示例：D 主机 D0 承担全局 DP0～7，D1 承担 DP8～15，TP1，kv_port=31000。P 两台主机上的配置均列出 D0:31000～31007 和 D1:31008～31015 全部 16 个 endpoint。仅为示意，替换实际 IP，并沿用已经验证的多机 DP/HCCL 启动参数；不能用每台主机重新从 0 起算的 local DP rank 生成端口。

```json
{
  "decode_endpoints": [
    "10.0.0.3:31000", "10.0.0.3:31001", "10.0.0.3:31002", "10.0.0.3:31003",
    "10.0.0.3:31004", "10.0.0.3:31005", "10.0.0.3:31006", "10.0.0.3:31007",
    "10.0.0.4:31008", "10.0.0.4:31009", "10.0.0.4:31010", "10.0.0.4:31011",
    "10.0.0.4:31012", "10.0.0.4:31013", "10.0.0.4:31014", "10.0.0.4:31015"
  ]
}
```

该对象放进 P 的 kv_connector_extra_config；不另建中间配置协议。可选 cpu_eids、npu_eids 是“本机物理 NPU ID 字符串 → 32 位十六进制 EID”的字典，分别固定 CPU↔NPU 路径和 NPU↔NPU fullmesh 路径。每台 P/D 主机填本机真实 EID；例如 D 只暴露物理 4～7 时，键仍填 4～7。缺项使用 SDK 自动发现，但必须现场确认能发现正确的 fullmesh EID，不能直接套用 CPU 路由 EID。timeout 默认 120 秒。

保持原生 W4A4C8、block_size=128、MTP3；支持 PP=PCP=DCP=1，P TP 必须为 D TP 的整数倍。D 关闭本地 vLLM prefix caching；P 沿用已验证的 prefix 配置。新 PD 模式使用逐层 (656,128) DDR，不修改非 PD 的 GROUP_WIDTH=4，不混用旧 KVAF prefix Connector、local-CPU sparse offload、KVPP 或 layerwise storage reuse。

完整源码编译/安装仍沿用原模型环境。新 Worker 继承原 NPUWorker，原模型、量化和算子选择保持原生入口。

## 4. 启停与控制网络

1. 启动八台 K（临时测试可替换为机头 CPU），确认各 D 所选 endpoint 可达。
2. 启动 D 原生进程组。每 worker 初始化自己的 DDR region，完成原生 warmup 后开启控制监听。
3. 启动 P 原生进程组。每个 P worker 完成到所有 D region 的逐层注册和 attach 后，才结束 cache 初始化。
4. P/D 健康检查均通过后，启动原生 examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py，再发请求。
5. 停止接入新请求并排空已有请求；关闭 P 时先请求 D detach，再释放 source；D 关闭时也显式 detach，随后由 Runner 关闭 Runtime。

TCP 控制消息携带 SDK 可序列化 descriptor/Ready。该 SDK 只暴露这些不透明类型的 pickle 构造；通道反序列化限制为明确的布局、元数据和 SDK 类型，仅在部署的可信 P/D 网络开放。导入检查只为此文件增加定点例外。NPU source listen 使用动态端口；生产网络需允许 P/D 间 SDK source 控制连接，不能只开放 kv_port 的固定范围。KV 数据不经此 TCP 业务通道搬运。

源对象数量随实际注册层数和 D worker 数增长。当前 SDK 每个 NPU source 单独监听；生产按实际层数核对进程的文件描述符、线程/PID、Host 内存及设备 MR/上下文容量，不能仅用四层通过结果外推全层资源上限。

取消/抢占保留原生 HBM block 生命周期。PDTransfer 对在途请求暂缓原生 DDR/tail 释放，等安全完成后再回收。终态、已安全取消的 DMA 通过原生 load-error block 和 finished_recving 回报；未知 DMA/Commit 状态保留原 Put、Ready、源/目的内存并显式报错，停止 Runtime 后续 Get，绝不以新 Put 或重复 READ 掩盖错误，也不额外实现业务自动 rePrefill。

## 5. 已执行与现场验收

```bash
PYTHONPATH=third_party/KVOP python -m unittest discover -s tests/ut -p 'test_kvaf_*.py' -v
ruff check vllm_ascend/distributed/kv_transfer/kvaf/pd tests/ut/test_kvaf_pd*.py
ruff format --check vllm_ascend/distributed/kv_transfer/kvaf/pd tests/ut/test_kvaf_pd*.py
```

当前全量 160 项 KVAF UT 通过。新增 UT 使用真实 CPU Tensor、真实回环 TCP，替换 SDK/NPU 边界，覆盖：逐层 PA 注册、typed scale、普通 tail 与 Hot 隔离、双 Pull 字节落点、原生目录/Get/ABC 续算、TP4/TP8→TP1、空贡献、chunk、MTP、Commit 丢失恢复、失败/取消和释放。另以 16 个 P worker、16 个 D region、8 个 K endpoint 检查完整注册矩阵，并直接执行复用的原生 Scheduler 代码。

云环境没有 NPU/CANN/UB，没有执行真实 NPU binding 全量编译、模型精度、生产多机全建链、长稳或性能测试。CPU/mock 通过只代表对应关键逻辑与接口检查通过。现场仍按原设计进行短/长上下文、127/128/129/255/256/300 边界、MTP、并发、取消和生产拓扑验收；单机四层结果不替代生产验收。

单机验证的卡分配和字节标准见 [临时单机教程](kvaf_pd_single_server_8npu_4layer.md)；通用实施状态见 [实施记录](kvaf_pd_implementation.md)。
