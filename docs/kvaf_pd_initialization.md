# 多 D worker 初始化协调

## 适用问题与边界

在相同 24 号 K endpoint 上，实测八卡逐卡初始化通过、两卡并发通过、八卡并发部分失败；
逐个初始化并保留八份 16 GiB DDR 同时驻留通过。错误为
`CPU control response unavailable; retain source and task`。
SDK 对应版本的 K 连接处理为串行，普通 remote-pull 控制请求的 socket 超时为 5 秒；
失败在 AIV 通道日志之前，符合描述符查询排队超时的方向，但尚无逐请求耗时证据。

本修复提供默认关闭的初始化准入：所有 D 主机通过**同一个协调服务**，按 K endpoint
串行执行 `link_to → initialize → initialize_ddr`。成功后立刻释放准入，Runtime/DDR 继续存活。
它不改变 KVAF 协议、SDK 超时、原生框架核心和算子；Get/Pull、注册、请求元数据和推理不经过协调服务。
它针对已复现的初始化并发条件，不证明所有运行期控制请求均不会超时。

这是显式部署的单协调点，不是自动选主或高可用服务。双 D、8 K 的正式部署同样可用：
两台 D 的 worker 均填写同一协调地址，各自实际选中的 `cpu_address` 作为门控键，
不同 K endpoint 可并行。必须使用统一的 `host:port` 写法；不同 IP/主机名别名不会自动识别为同一 K。
所有参与初始化的 D 必须启用；未接入的 SDK 客户端不会受此服务约束。

## 1. 更新源码

在 P/D 源码容器更新开发分支。代码为 Python，不需要重新构建 vLLM、Ascend 原生扩展、KVOP 或 KVAF。
确认容器采用当前 editable checkout，正常停止本次 P/D；保留健康的 K。

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2
git switch v0.30.0rc2-develop
git pull --ff-only origin v0.30.0rc2-develop
```

## 2. 启动一个协调服务

K 已监听 24 号宿主机 `28997`；协调服务使用独立端口 `28998`，要求所有 D 容器可达。
在 **24 号宿主机**执行。若宿主机没有 Ascend 仓库，可以从更新后的容器复制单个标准库模块：

```bash
(
set -euo pipefail

# 替换为实际 D 容器名；不使用 docker cp，避免此前出现的 staging mount 问题。
docker exec vllm_ascend_dsa_pd cat \
  /home/c00612283/vllm-ascend-v0.30.0rc2/vllm_ascend/distributed/kv_transfer/kvaf/pd/initialization.py \
  > /home/c00612283/kvaf_pd_init_coordinator.py.part
test -s /home/c00612283/kvaf_pd_init_coordinator.py.part
mv /home/c00612283/kvaf_pd_init_coordinator.py.part \
   /home/c00612283/kvaf_pd_init_coordinator.py

python3 -u /home/c00612283/kvaf_pd_init_coordinator.py \
  --host 141.62.38.24 --port 28998 \
  2>&1 | tee /home/c00612283/kvaf-init-coordinator.log
)
```

保持终端运行，预期 `KVAF_INIT_COORDINATOR_LISTEN address=141.62.38.24:28998`。
此服务只依赖 Python 标准库，不加载 CANN、torch、vLLM 或 KVAF。若在源码容器内启动同样服务，
可使用 `python3 scripts/kvaf_pd_init_coordinator.py --host ... --port 28998`，但全部署只能使用一个实例。
这是可信部署控制网络上的 TCP 服务，与原生 K 控制服务一样不提供公网鉴权；绑定指定业务地址。

## 3. 先做八进程无模型回归

在 **24 号 D 容器**，P/D 保持停止，K 与协调服务保持运行：

```bash
(
set -euo pipefail
REPO=/home/c00612283/vllm-ascend-v0.30.0rc2
BUILD=/home/c00612283/KVAF-Devlop/build-pd-py312
export PYTHONPATH="$BUILD/python${PYTHONPATH:+:$PYTHONPATH}"

/usr/local/python3.12.13/bin/python3 -u \
  "$REPO/scripts/kvaf_pd_ddr_residency.py" \
  --cpu 141.62.38.24:28997 --build-dir "$BUILD" \
  --devices 0 1 2 3 4 5 6 7 --ddr-size-gb 16 \
  --parallel --initialization-coordinator 141.62.38.24:28998 \
  --init-timeout 600 \
  2>&1 | tee /home/c00612283/ddr-coordinated.log
)
```

此时八个进程同时启动，真正的 SDK 初始化受准入控制；每个成功后保持驻留，全部就绪再统一释放。
需要看到八条 READY、ALL_RESIDENT_PASS、八条 SHUTDOWN_PASS，以及 PARALLEL_RESIDENCY_PASS。
日志详情与安全退出约定见 [驻留测试说明](kvaf_pd_ddr_residency.md)。
这不等于 NPU 数据或完整模型精度验证；若失败，停止后续部署并提供 D/K/协调服务同期日志。

## 4. D 启动配置

在现有 D 脚本的 `--kv-transfer-config` 内合并以下字段，保留其他已有字段，避免重复 JSON 键：

```json
{
  "kv_connector": "PDKVoffloadConnector",
  "kv_connector_module_path": "vllm_ascend.distributed.kv_transfer.kvaf.pd.connector",
  "kv_role": "kv_consumer",
  "kv_port": 31000,
  "kv_connector_extra_config": {
    "timeout": 600,
    "initialization_coordinator": "141.62.38.24:28998"
  }
}
```

`additional-config.kvaf_config.cpu_address` 仍为 `141.62.38.24:28997`，
`ddr_size_gb=16`、DP8/TP1/EP8、模型与其余算子参数保持原配置。
其他 D 主机若访问同一 K，同样配置该协调地址；不必修改 P 的 decode_endpoints 或在 P 配置此字段。
`timeout=600` 同时用于原有 PD 控制 RPC 与等待准入，**不会改变 SDK 内部 socket 超时**。

无模型回归通过后，启动 D，待所有 D 的 DDR 初始化完成再启动 P，最后启动 Proxy 发请求。
双 D 部署在所有 D 就绪前不要发送推理流量，以免把运行期控制流量混入本次初始化验收。

## 5. 关闭、失败与恢复

默认 `initialization_coordinator=""`。删除 D 中此字段或设为空字符串即回到原有初始化路径，
不会建协调连接；未使用 PDKVoffloadConnector 的原生混部也不受影响。

正常初始化只有一个持有者；排队超时的 worker 会报错，且不会调用 SDK 初始化。
没有自动重试、租约过期接管、K 重启或杀进程行为。
若持有者初始化抛异常、退出或在报告成功之前断连，协调器会将该 K endpoint 置为 blocked，
拒绝后续初始化，而不是把可能未静止的资源交给下一个 worker。
无日志的长期持有者可能使等待者超时，这比自动抢占正在工作的 SDK 更保守。

出现 `previous initializer failed/disconnected` 时：

1. 停止新初始化和推理请求，保存失败 worker 的原生日志和 K 日志。
2. 正常停止相关 P/D，确认所有初始化和传输已静止、DDR 会话已清理；不能只等待超时时间后继续。
3. 确认 K 状态，必要时在所有客户停止后正常重启 K。
4. 正常停止并重启唯一协调服务，再重新启动 D/P。

协调状态保存在内存中。不要在有初始化持有者时重启协调服务，也不要以自动拉起的新实例替代旧实例；
服务重启必须遵循上述排空过程，避免旧持有者与新实例放行的 worker 同时初始化。

## 验证记录

本地 CPU UT 覆盖真实 TCP 多连接互斥、不同 endpoint 独立放行、等待超时不抢占、
持有者异常/断连拒绝后续初始化、关闭功能不建连接、PD Adapter 参数与调用顺序，
以及八个真实 spawn 进程配合模拟 SDK 的并发初始化与统一释放。
这些不是 Ascend NPU 实测；硬件修复效果以第 3、4 节复测及后续 KV/精度检查为准。
