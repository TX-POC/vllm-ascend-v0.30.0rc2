# PD DDR 初始化驻留对照测试

本脚本不加载模型，不修改正式 Connector、Backend、SDK 或原生算子。
它显式运行时才创建资源：使用 spawn 为每张卡启动独立进程，依次完成
`link_to → initialize → initialize_ddr`；先前成功的进程保持 Runtime 存活。
全部就绪后校验 region 身份、地址范围不重叠，再统一通知释放。

这与此前“逐卡初始化后立即释放”不同，可检查八份 DDR/通道同时驻留。
它只验证初始化与驻留，不验证完整 KV 传输、模型精度或推理性能。

## 执行

正常停止本次 P/D，确保八张卡空闲；保留 24 号宿主机上的 K 服务，
监听 `141.62.38.24:28997`。在 **24 号 D 容器内**执行：

```bash
(
set -euo pipefail

REPO=/home/c00612283/vllm-ascend-v0.30.0rc2
BUILD=/home/c00612283/KVAF-Devlop/build-pd-py312
export PYTHONPATH="$BUILD/python${PYTHONPATH:+:$PYTHONPATH}"

test -f "$REPO/scripts/kvaf_pd_ddr_residency.py"
LOG="/home/c00612283/ddr-residency-$(date +%Y%m%d_%H%M%S).log"

/usr/local/python3.12.13/bin/python3 -u \
  "$REPO/scripts/kvaf_pd_ddr_residency.py" \
  --cpu 141.62.38.24:28997 \
  --build-dir "$BUILD" \
  --devices 0 1 2 3 4 5 6 7 \
  --ddr-size-gb 16 \
  2>&1 | tee "$LOG"
)
```

路径和 endpoint 可按部署替换。设备编号应在同一容器内对 torch 与 KVAF 均有效，
不要在本次测试中新增或更改设备可见性映射。
默认 `npu_core_count=2`、`jettys_per_core=2`，对应当前 PD 默认配置；
若正式配置修改过这两项，用同名连字符 CLI 参数显式传入实际值。

每个进程使用三种 AIV kernel（get、put、get_raw_addr），C8 参数为
656 bytes/token、128 tokens/block，slot 向上按 4096 bytes 对齐，得到 86016 bytes。
每份 DDR 为预算向下取整后的 17179803648 bytes，八份约占 128 GiB；
此外还有 SDK 通道和 NPU 工作空间开销。

## 判定

每行摘要以 `KVAF_DDR_DIAG` 开头，随后为 JSON。

| 事件 | 含义 |
| --- | --- |
| `START` | 启动当前卡进程，记录 PID |
| `READY` | 该进程初始化完成，返回 DDR ticket 并等待释放 |
| `ALL_RESIDENT_PASS` | 所有进程存活，region 身份不同且地址范围不重叠；尚未发送释放命令 |
| `SHUTDOWN_PASS` | 单进程清理返回；需结合 READY 和最终结果判断 |
| `SERIAL_RESIDENCY_PASS` | 全部驻留检查及所有进程正常退出通过 |
| `FAIL` / `RESIDENCY_FAIL` | 初始化或协调失败，停止启动更多进程并请求清理 |
| `RETAINED` / `CLEANUP_PENDING` | 清理失败或等待超时，不能作为测试通过 |

预期收到八条 READY、ALL_RESIDENT_PASS、八条 SHUTDOWN_PASS，
最后是 SERIAL_RESIDENCY_PASS，退出码为 0。

八卡并发曾失败而本测试通过，会进一步支持“并发初始化相关”这一判断，
但仍不能单凭本测试确认某个 Describe 请求发生 5 秒超时。
若本测试也失败，需结合失败卡阶段和 K 同期日志检查累计资源及清理问题。
当前脚本不覆盖模型加载后的 HBM 压力、HCCL 上下文和全请求生命周期。

## 失败与退出

脚本不自动重试、不会强制终止 worker，也不重启 K。
父进程初始化等待默认 180 秒，清理等待默认 180 秒；这些参数不改变 SDK socket 超时。
达到清理期限时打印未退出进程 PID，并继续等待，不把超时当作释放完成。
若 worker 的 `shutdown()` 抛异常，该进程保持存活，避免通过退出释放可能仍被 DMA 使用的内存。

需要中止时，在协调进程终端按一次 Ctrl+C，请求有序释放；不要按 Ctrl+Z。
若出现 RETAINED/CLEANUP_PENDING，保留终端、K 和日志，先确认传输静止再手动处理，
不要连续重试或直接 kill -9。尚未就绪的 worker 会在初始化返回后处理释放命令。

提取本次摘要：

```bash
grep 'KVAF_DDR_DIAG ' /home/c00612283/ddr-residency-实际时间.log
```

请同时保留 K 在本次运行期间的 REGION_ALLOCATED、CHANNEL_CREATED、REGION_RELEASE 日志。
