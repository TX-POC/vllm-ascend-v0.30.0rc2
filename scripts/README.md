# KVAF 服务接入

当前支持单机 TP、Model Runner V1、GLM 的 block-128 packed C8 Target KV
（656 字节/token）、最多四个 decode query/request（含 MTP）。
当前验收基线为 TP8；DP、流水线并行、上下文并行不在本次验收范围内。

## 安装与启动

在服务容器中加载 CANN 环境，然后执行：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
bash scripts/install_kvaf.sh
bash third_party/KVOP/build.sh
```

安装脚本把 [KVAF](https://gitcode.com/CacheOS/KVAF) 固定提交
`c4375e13d7a9519d907ab6a5d11227795f207286` 放入 `third_party/KVAF`，应用
`scripts/patches/kvaf-reuse-device-ra.patch`，编译 CPU 服务、NPU Get/Put 和 Python 绑定。
补丁让 KVAF 借用 HCCL 已初始化的 device RA 会话，避免重复初始化造成通信回包错位；
这是运行依赖，不是测试脚本。已有源码版本不匹配或补丁冲突时保留目录并停止。
CPU 构建需要 UMDK；已有环境可指定 `KVAF_UMDK_ROOT`、`KVAF_URMA_LIBRARY`。
KVOP 的来源和构建限制见 [KVOP 说明](../third_party/KVOP/README.md)。

启动 CPU 服务，例如：

```bash
./third_party/KVAF/build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28999
```

根据硬件拓扑选择 `--ub-device` 和 `--eid-index`。TCP 地址可达不代表 UB 链路可用。
然后编辑 `scripts/start_vllm_server.sh` 顶部配置：

```bash
kvaf_enabled=true
kvaf_cpu_address="127.0.0.1:28999"
kvaf_cpu_addresses='{}'
kvaf_ddr_size_gb=16
kvaf_hot_tokens=8192
kvaf_verify_put=false
```

`cpu_addresses` 的键为宿主机物理 NPU ID，例如
`'{"4":"10.0.0.20:28999","5":"10.0.0.20:28999"}'`；未列出的卡使用默认地址，
多个卡可共用一个服务。`KVAF CPU route` 打印实际逻辑号、物理号和服务地址。
16 GiB 只是配置示例，是否可容纳请求取决于模型、长度、并发和辅助缓存。

运行 `bash scripts/start_vllm_server.sh`。修改容量或路由后须重启服务；
`kvaf_enabled=false` 使用原生 KV 路径，不需要 KVAF CPU 服务或 KVOP。
绑核与共享专家流水重叠在开关两种情况下均默认开启，比较性能时保持配置一致。

## 数据与容量约定

- Prefill 使用分层 chunk 暂存区，稳定完整块 Put 到 DDR；历史按请求/层组/chunk
  获取一次，四层共享 LI 的组一起读取。混合 prefill/decode 批次按请求分别处理。
- Decode 直接写入独立 hot 区的 ABC 尾部，不向旧 prefill 页镜像数据。
  历史 hot 容量为每请求每层 `kvaf_hot_tokens`，默认 8192，另加固定 384 个 ABC 槽位。
  配置必须至少 8192 且按 128 对齐，并受原 LIDU 映射容量上限约束。
  当前是固定配额，不随实际并发动态均分。
- 长请求进入 decode 时历史 hot 区为空，按 LI/LIDU miss 从 DDR 填充；prefill 不预热历史 hot。
  短于 2048 token 的上下文走专用密集映射路径。
- 单层 Runtime 使用 656 字节/token，四层组使用 2624 字节/token，布局为
  `[token, local_layer, 656]`；组内共享 LIDU、双表、LRU 和 Get，SFA 使用各层索引。
- `kvaf_ddr_size_gb` 是每 worker 全部层的总预算，不是每层或每 Runtime 的预算。
  两种 Runtime 按 Target 和辅助快照需求分配预算，计入对齐及旧、新辅助快照共存。
- Target 原始全长 HBM 页已移除；Indexer/MTP draft 仍使用调度器管理的 HBM 页。
  请求结束保留 Target DDR 引用，并保存辅助缓存；只有全部 TP worker 确认后才发布 prefix hit。
  命中后恢复辅助缓存并共享 Target 引用；MTP 末尾匹配块及不完整块会重算。
- 未提交的 ABC 数据不能被 LRU 驱逐。稳定块按已纠正的长度同步 Put，成功后才发布 DDR 地址。
  LRU 反馈沿用异步 D2H/FIFO 路径；这里的同步 Put 不等于异步传输。
- DDR 不足时淘汰未被使用的已完成缓存，等待所有 TP 的释放确认后复用空间。
  活跃请求占满预算时新请求等待；单请求自身无法容纳时明确报错。

## 验证与日志

先用同一个确定性短请求比较 KVAF 开关两种输出，再运行：

```bash
python scripts/benchmark.py --levels 1,2 --prefix-warmup
python scripts/benchmark.py --levels 2 --profile-decode --profile-decode-batch 2 --profile-decode-tokens 32
```

服务默认配置按需 profiler；HTTP `/start_profile` 才开始采集。
检查两请求均有 cached tokens，并存在共同 decode 窗口；仅 HTTP 成功不能证明并发或字节正确。
还应验证混合批次、请求结束后的重复命中、DDR 淘汰及相同请求的输出一致性。

- `KVAF connected`：建链完成；不代表模型 KV 已经传输。
- `KVAF first BatchPut`：区分 prefill/decode 首次写入。
- `kvaf_verify_put=true`：每 worker 抽查两阶段各一个块，读回与 HBM 比较；会增加等待，测性能时关闭。
- `KVAF archive READY/HIT/RESTORED`：归档确认、命中、恢复；`KVAF DDR admission WAIT`：容量等待。
- `KVAF request released`：释放请求引用；`freed_pages=0` 可能表示仍被 prefix archive 引用。
- `KVAF hot miss summary`：每卡约五秒汇总一次，历史 token 按请求/轮去重，排除 ABC、padding 和短上下文旁路。
  `layer_miss_pct` 是各层区间 miss 率的分位数；`request_round_miss_pct` 是各请求/轮跨层 miss 率的分位数。
  `overall_miss_pct` 按访问数量加权。四层组按物理层数计入统计，不等于 Get 调用次数。

原生日志写入连接时打印的 `kvaf_DP*_TP*_device*_pid*.log`；控制台保留 vLLM 风格摘要。

## 独立实验与减专家

带宽、路由/EID、1/2/4/8 卡并发、多 Runtime、不同 DDR/包大小等独立测试脚本及其单测
已保留到 `origin/codex/kvaf-bandwidth-tools`（基于 `cbcebefba` 的完整快照）。
当前服务分支不包含这些脚本；可在单独 checkout 中运行，避免切换正在运行的 editable 安装目录。
HTTP benchmark、集成回归测试和 KVOP 硬件正确性检查仍保留。

`routed_experts=0` 为完整模型；正数仅加载前 N 个路由专家，支持 `glm_moe_dsa` 主模型和 MTP。
该调试功能改变输出，不可用于完整模型的精度/性能结论，也不是 KVAF 的必要依赖。
