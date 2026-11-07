# PD 控制连接排障

适用于 `v0.30.0rc2-develop2` 的独立 PDKVoffloadConnector。
这次修正仅调整外部 PD TCP 控制服务，不修改原生框架、KVAF SDK、K 服务或 KV 字节布局。

## 已知证据和边界

现场 P 在 `_send_arena -> _send_layer -> plan` 等待响应时出现连接重置；
请求路由到 D4 的 31004 端口，事后 8 个 D worker 的 hello 均正常。
宿主机累计 ListenOverflows/ListenDrops 为 144，未取得失败前后增量。
因此监听队列压力是排查方向，尚不能断定它是这次重置的唯一原因。

Python 3.12 的 ThreadingTCPServer 默认监听队列较小。PD 控制服务现在显式请求
128 个待接受连接的位置，以承接多个 TP worker 和 arena 的连接突发。
该值不是执行线程数、KV block 数或 SDK 队列数，实际还受内核 somaxconn 限制。
不会增加新的 KVAF transport，不修改系统参数，不自动重放失败 RPC。
这属于已选择 PD Connector 内部的缺陷修正，无额外实验开关；未选 PD 时不创建服务。

## 更新和重启

P、D 都更新 develop2。此项为 Python 修改，已使用 editable 安装时无需重新编译
vLLM、Ascend 算子、KVOP、KVAF binding 或 K 服务。
正常停止本次请求和 P/D，确认旧传输已经结束后再释放旧注册；不强制释放完成状态不明确的源。
K 和初始化协调器可保持运行，前提是没有未解决的初始化占用。

启动 D，等待 DDR 初始化和原生模型预热结束，再检查控制端口和 HTTP health。
D 的控制服务在原生预热结束后启动；仅出现 Initializing KVConnectorBase 不代表可接入。
P 容器验证所有 D 端口 hello 成功后启动 P，最后启动 Proxy 并发送请求。
两侧 diagnostics_requests 和 verify_kv 如已配置，仍须一致；本次错误日志不依赖这些开关。

## 失败日志

仅失败时打印 `KVAF_PD_RPC_FAIL`，成功请求不增加逐层 Info/Debug 日志。
使用原有 vLLM logger，启动 hello 轮询失败及正常 TCP 空探测不刷错误日志。
不会输出完整 metadata、block table、ticket、Tensor 或 KV 内容。
D 的 dispatch 异常附带 Python 堆栈；普通业务异常仍返回原有协议错误。

| 字段 | 含义 |
| --- | --- |
| side=client | 发起方；通常是 P，也可能是主动调用控制接口的检查程序 |
| side=server | D 控制服务 |
| endpoint | client 为目标端点，server 为本地监听端点 |
| phase=connect/send/receive | 建立连接、发送请求、等待响应失败 |
| phase=dispatch | D 已解码请求，但处理操作失败 |
| phase=respond | D 返回响应失败，调用方不能据此假定操作未执行 |
| op | hello/register/plan/pull/complete 等实际操作名 |
| producer/request/generation/layer/member | 请求及 TP 分片上下文；接收解码前不可用时为 `-` |
| error/detail | 异常类型与截断摘要 |

在 P/D 各自容器执行，填入本次确定的日志路径，避免误读其他运行轮次：

```bash
P_LOG=/home/c00612283/pd-logs/p/本次运行目录/server.log
D_LOG=/home/c00612283/pd-logs/d/本次运行目录/server.log
# 两端分别执行对应的命令；启动脚本也可能将日志写到仓库内 pd-logs。
grep -n -A 35 'KVAF_PD_RPC_FAIL' "$P_LOG" | tail -n 100
grep -n -A 35 'KVAF_PD_RPC_FAIL' "$D_LOG" | tail -n 100
```

`client phase=receive` 只证明未收到响应，不能证明 D 没执行操作。
不得通过捕获连接异常后重发 Ready/Pull 或提前释放源来绕过失败。
普通 TCP connect 和 hello 成功也不能替代 plan、双 Pull 和模型输出的验收。

## 受控复现：队列计数增量

在 D 控制服务所在网络命名空间采集；host network 容器可在宿主机采集。
非 host network 容器应在容器内采集，不能混用宿主机计数归因容器端口。
不要自动重试已经失败且完成状态不明的业务请求；先完成现场保留和有序清理。
在同一个终端定义函数：

```bash
listen_counts() {
python3 - <<'PY'
from pathlib import Path
lines = Path('/proc/net/netstat').read_text().splitlines()
for header, values in zip(lines, lines[1:]):
    if header.startswith('TcpExt:') and values.startswith('TcpExt:'):
        counters = dict(zip(header.split()[1:], values.split()[1:]))
        for key in ('ListenOverflows', 'ListenDrops', 'TCPBacklogDrop'):
            print(key, counters.get(key, '0'))
        break
PY
}
listen_counts > /tmp/pd-listen-before.txt
cat /proc/sys/net/core/somaxconn
cat /proc/sys/net/ipv4/tcp_abort_on_overflow
```

发送一条固定请求。返回或失败后立即执行：

```bash
listen_counts > /tmp/pd-listen-after.txt
awk '
NR==FNR {before[$1]=$2; next}
{printf "%s: before=%s after=%s delta=%d\n", $1,before[$1],$2,$2-before[$1]}
' /tmp/pd-listen-before.txt /tmp/pd-listen-after.txt
```

计数属于该网络命名空间的所有监听端口，增量仍需与时间及 RPC endpoint 对齐。
零增量且再次重置时，优先根据 server dispatch/respond 错误、worker 存活及
对应端点的连接记录继续定位，不宣称监听队列修正已经解决根因。

## 验证范围

CPU UT 覆盖 TP8×6 的 48 个同时发起的真实 TCP plan 请求、业务异常返回、
连接重置保留原异常且不重试、响应丢失不重放、空探测与截断协议区分。
这些测试不执行真实 SDK、NPU DMA 或全模型推理。
硬件验收仍需短请求、原 41510-token 长请求及固定采样配置下的基线输出对比。
