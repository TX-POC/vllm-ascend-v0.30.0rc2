# 已有 vLLM 容器：Ascend / KVAF 编译与 PD 联调教程

双机完整权重与非 PD 准确度对照请使用[双机全模型教程](kvaf_pd_two_node_full_model_accuracy.md)。

更新：2026-11-05。适用于当前单机四层 GLM5.3 W4A4C8 验证：机头 CPU 运行 K，P 使用卡 2、3（TP2/DP1），D 使用卡 4、5（TP1/DP2）。这是临时联调部署，不改变原始多机 P/DP/EP 与八台远端 K 的生产目标。

## 1. 本次问题与处理结论

| 现象 | 原因或诊断结论 | 正确处理 |
| --- | --- | --- |
| scripts/*.sh 不存在 | 在用户主目录或 KVAF 仓执行框架相对路径 | 先进入 Ascend 仓；或使用绝对脚本路径 |
| CMakeLists.txt、connector_pull_api_design.md 缺失 | 指向错误目录，或 checkout 不完整/分支不匹配 | 检查两个文件和实际分支，不创建空文件绕过 |
| 构建脚本无输出，日志 0 字节 | 脚本的 test -f 在 CMake 前退出；管道未启用 pipefail 时容易漏看失败 | 先做源码检查；构建使用 set -euo pipefail |
| git pull 的 HEAD/ref 错误 | 本地 Git 引用异常 | 保留旧目录和修改，另行正常 clone 匹配分支；不要删除 .git 强行修复 |
| import torch 失败 | 测试使用的 python3 不是容器已安装 torch 的 Python | 统一 /usr/local/python3.12.13/bin/python3 |
| kvaf 无法导入、提示 partially initialized | 曾用 Python 3.12 加载 cpython-311 扩展；也可能是未构建或动态库缺失 | 用同一 Python 在独立 build-pd-py312 中构建，检查真实 loader 错误 |
| kvaf_cpu_service 不存在 | K 服务没有成功构建；binding 成功不等于 K 编译成功 | 独立构建 build-hostcpu-k，检查可执行文件 |
| requires urma_api.h and liburma | CMake 未找到现场 UMDK 路径 | 显式传入 include/library 参数 |
| URMA_ACCESS_REMOTE_* 未定义 | 当前源码枚举名与现场 UMDK 26.06 头文件不一致 | 确认头文件后，K 构建使用本教程的枚举兼容参数 |
| Address already in use | 旧服务仍监听，或 Ctrl+Z 仅暂停了进程 | 确认监听进程，复用或正常停止；不要重复启动 |
| curl URL 无 host、Proxy host 为空 | 新终端未设置 LOCAL_IP | 每个终端设置，或采用本教程的固定 IP 命令 |
| --device-a 收到空字符串 | 新终端没有设置卡号变量 | 测试显式传 --device-a 4 --device-b 2 |
| JSONDecodeError、bash 空命令 | 粘贴了 HTML 转义、空 JSON 或错误的反斜杠 | 使用仓库 p_start.sh、d_start.sh；JSON 直接编辑字段 |
| run_dir: unbound variable | 使用了未定义日志变量，或重复添加 tee | 当前脚本使用 dir 并已自动保存 server.log，无需追加 tee |
| Worker exit -9 | 表示 SIGKILL，现有证据不能直接判定 OOM | 查宿主机内核、容器 cgroup v1 限额及同卡进程；宿主机 free 充足不能排除容器限额 |
| StoreKVBlock FP8 路径异常 | 现场该优化路径存在 dtype 支持限制 | 本轮 P/D 保持 c8_enable_reshape_optim=false，使用原生替代路径 |
| IndexerWrapper 无 write_cache | 外部诊断钩子曾访问包装层而非 impl | 开发分支 b3f5c282e 已修复并补 UT；更新 Python 代码后重启 |
| HTTP 400：8033+160 超过 8192 | 输入与输出总 token 数超限 | 缩短输入/输出；或同时调整 P/D 的 max-model-len 并评估内存 |
| HTTP 200 但输出乱码 | HTTP 成功不代表 KV 或模型精度通过 | 做同模型独立基线、开启有界 KV 诊断，不能据此认定联调成功 |
| pd-diag-p.log 不存在 | 启动脚本没有创建该固定文件 | 从实际 pd-logs/p、pd-logs/d 的 server.log 汇总 |

## 2. 固定路径与源码检查

在已有 CANN、ATB、torch、torch_npu、vLLM 的容器中操作，沿用容器原有环境初始化方式。下面不重新安装 vLLM，也不替换 torch/torch_npu。

```bash
export KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3
export ASCEND_REPO=/home/c00612283/vllm-ascend-v0.30.0rc2
export KVAF_PD_SRC=/home/c00612283/KVAF-Devlop
export KVAF_PD_BUILD=/home/c00612283/KVAF-Devlop/build-pd-py312
export MODEL=/home/models/glm5-mtp-4l
export LOCAL_IP=141.62.38.12
export NIC=enp34s0f1
cd "$ASCEND_REPO"

"$KVAF_PD_PYTHON" -c 'import sys, torch, torch_npu, vllm; print(sys.executable); print(torch.__version__, torch_npu.__version__); print(vllm.__version__, vllm.__file__)'
git status --short
git branch --show-current
git -C "$KVAF_PD_SRC" status --short
git -C "$KVAF_PD_SRC" branch --show-current
ls "$KVAF_PD_SRC/CMakeLists.txt" "$KVAF_PD_SRC/docs/connector_pull_api_design.md"
ls scripts/install_kvaf_pd_source.sh scripts/kvaf_pd_k.sh p_start.sh d_start.sh
```

Ascend 应为 v0.30.0rc2-develop；KVAF 使用包含 connector_pull_api_design.md 的 cpu-pull-hbm-reference 对应源码，注意远端实际分支名可能包含 codex/ 前缀。不要将 main 的不匹配源码当作已适配接口版本。目录名不代表分支正确。

Ascend 工作区无未保存修改时更新：

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2
git pull --ff-only origin v0.30.0rc2-develop
git log -1 --oneline
```

先保存现场手改脚本再更新；不要覆盖个人配置。应包含 b3f5c282e 或其后续提交。当前构建会对 KVAF 应用已有 RA 借用补丁，不要用 reset --hard 清除它。

## 3. 编译并安装 Ascend、KVOP 与 KVAF binding

在上述变量已设置的终端执行。第一个参数是字面量 -，表示保留镜像已有 vLLM，不需要 VLLM_SRC=/path/to/vllm。

```bash
(
set -euo pipefail
cd /home/c00612283/vllm-ascend-v0.30.0rc2
bash scripts/install_kvaf_pd_source.sh - "$KVAF_PD_SRC" "$KVAF_PD_BUILD" \
  2>&1 | tee /home/c00612283/kvaf-pd-build.log
)
```

脚本安装构建依赖、editable 安装本地 vLLM-Ascend、编译 KVOP、应用 KVAF RA 借用补丁并编译 Python binding。应看到 PD_SDK_IMPORT_OK 和 PD_SOURCE_INSTALL_OK。失败时先处理日志中的第一条编译错误，不继续启动。

验证当前 Python 加载的是新目录：

```bash
source /home/c00612283/vllm-ascend-v0.30.0rc2/third_party/KVOP/env.sh
export PYTHONPATH="$KVAF_PD_BUILD/python${PYTHONPATH:+:$PYTHONPATH}"
"$KVAF_PD_PYTHON" - <<'PY'
import sys
import sysconfig
import kvaf
import vllm_ascend
print(sys.executable, sysconfig.get_config_var("EXT_SUFFIX"))
print(vllm_ascend.__file__)
print(kvaf.__file__)
assert hasattr(kvaf.NpuRuntime, "submit_cpu_pull")
assert hasattr(kvaf.NpuRuntime, "batch_get_raw_addr")
assert kvaf._native.reuse_device_ra_supported
print("IMPORT_OK")
PY
```

本机 Python 3.12 的扩展应为 cpython-312-aarch64-linux-gnu.so；不要指向旧 build-hostcpu-pull 中的 cpython-311 文件。PYTHONPATH 加的是包含 kvaf 文件夹的 python 目录，不是 .so 文件路径。

## 4. 独立编译机头 K 服务

现场头文件 /usr/include/ub/umdk/urma/urma_api.h，库 /usr/lib64/liburma.so；已知安装包为 umdk-urma-devel/lib-26.06.0-B026。先确认访问权限枚举：

```bash
ls /usr/include/ub/umdk/urma/urma_api.h /usr/lib64/liburma.so
rg -n 'URMA_ACCESS_(REMOTE_)?(READ|WRITE|ATOMIC)' /usr/include/ub/umdk/urma
```

若容器没有 rg，改用 grep -R -n。下面兼容参数仅用于现场已确认提供 URMA_ACCESS_READ/WRITE/ATOMIC 的头文件；它只兼容名称，不保证其他 UMDK ABI 或 UB 数据面兼容。

```bash
(
set -euo pipefail
cd /home/c00612283/vllm-ascend-v0.30.0rc2
bash scripts/kvaf_pd_k.sh build \
  /home/c00612283/KVAF-Devlop \
  /home/c00612283/KVAF-Devlop/build-hostcpu-k \
  -DKVAF_URMA_INCLUDE_DIR=/usr/include/ub/umdk/urma \
  -DKVAF_URMA_LIBRARY=/usr/lib64/liburma.so \
  '-DCMAKE_CXX_FLAGS=-DURMA_ACCESS_REMOTE_READ=URMA_ACCESS_READ -DURMA_ACCESS_REMOTE_WRITE=URMA_ACCESS_WRITE -DURMA_ACCESS_REMOTE_ATOMIC=URMA_ACCESS_ATOMIC' \
  2>&1 | tee /home/c00612283/kvaf-k-build.log
test -x /home/c00612283/KVAF-Devlop/build-hostcpu-k/runtime/kvaf_cpu_service
ldd /home/c00612283/KVAF-Devlop/build-hostcpu-k/runtime/kvaf_cpu_service
)
```

ldd 不应有 not found。K 是独立 C++ 可执行文件，构建时关闭 Python；因此 binding 构建成功不能替代本节。容器内必须可访问实际 UB 设备、驱动和 UMDK；这里只占用机头 CPU，不要求鲲鹏远端服务器。

## 5. 启动 K（终端一）

先查端口；没有 ss 时可用 fuser 或 netstat：

```bash
fuser -n tcp 28997
# 或：netstat -lntp | grep ':28997'
```

无监听通常返回非零，不要因此判定编译失败。若已有服务，先 ps -fp 实际PID 确认归属，可复用匹配的 K。需要停止时对确认的 PID 使用 kill -TERM；不要照抄历史 PID，也不要杀掉不属于本次联调的进程。No such process 表示已退出，重新查端口即可。

```bash
bash /home/c00612283/vllm-ascend-v0.30.0rc2/scripts/kvaf_pd_k.sh start \
  /home/c00612283/KVAF-Devlop/build-hostcpu-k \
  141.62.38.12 --port 28997
```

保留前台。单个 K 服务可以供多个 NPU 客户端连接，各 D worker 独立分配 Runtime/region；本次 D 配置每 worker 1 GiB，共两份预算。监听成功仅证明控制端口可用，不证明 UB 传输正确。裸机或容器均需匹配设备和驱动环境；本教程统一使用已验证环境的容器，避免混用库。

## 6. 检查并启动 D、P

直接编辑仓库根目录两份脚本的字段，不传位置参数。当前配置：

| 字段 | P | D |
| --- | --- | --- |
| NPU / TP / DP | 2,3 / 2 / 1 | 4,5 / 1 / 2 |
| HTTP / DP RPC | 8100 / 13389 | 8200 / 13489 |
| Connector | kv_producer | kv_consumer |
| decode_endpoints | 141.62.38.12:31000、31001 | 按 D worker 建立控制监听 |
| KVAF 配置 | enabled=false，仍需 SDK 用于 PD 注册/供拉取 | enabled=true，cpu_address=141.62.38.12:28997 |

两侧保持同一模型路径、FP8 主 KV/indexer、block-size=128、MTP3、PP1、关闭 prefix caching、c8_enable_reshape_optim=false。不要照搬 Mooncake 模板的 Connector、KVPP 或并行参数。

如需排障，将两份脚本 kv_connector_extra_config 中的 diagnostics_requests 改为 2、verify_kv 改为 true（JSON 小写布尔值）。默认 0/false；诊断会同步和读回数据，仅用于有限请求检查，完成后关闭。

终端二：

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2
bash -n d_start.sh
./d_start.sh
```

终端三：

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2
bash -n p_start.sh
./p_start.sh
```

新终端不会继承其他终端临时 export；p_start.sh/d_start.sh 已内联现场配置。卡号、NIC、IP、模型或 build 路径变动时，手动同步修改对应字段。

退出用 Ctrl+C。Ctrl+Z 只暂停进程，仍占用端口和资源；当前 shell 可用 jobs -l 查看，再 fg 恢复后 Ctrl+C 正常退出。脚本已自动 tee 到当前工作目录 pd-logs/p或d/时间_PID/server.log，不要再追加包含未定义 run_dir 的日志管道。

## 7. 健康检查与 Proxy（终端四）

```bash
curl --fail --noproxy '*' http://141.62.38.12:8200/health
curl --fail --noproxy '*' http://141.62.38.12:8100/health
curl --fail --noproxy '*' http://141.62.38.12:8100/v1/models

KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3 \
bash /home/c00612283/vllm-ascend-v0.30.0rc2/scripts/kvaf_pd_proxy.sh \
  --host 141.62.38.12 --port 8000 \
  --prefiller-hosts 141.62.38.12 --prefiller-ports 8100 \
  --decoder-hosts 141.62.38.12 --decoder-ports 8200
```

P/D 健康检查失败时不继续发业务请求。DeepSelect 导入 warning 与后续致命 traceback 分开判断，不能仅凭 warning 认定启动失败。

## 8. POST 请求与 KV 检查

另开终端发送短请求：

```bash
curl --fail --noproxy '*' \
  -X POST http://141.62.38.12:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm","messages":[{"role":"user","content":"请简短介绍中国的首都。"}],"max_tokens":16,"temperature":0,"stream":false}'
```

先确认无异常、无乱码，再使用实际 tokenizer 计数超过 128、且输入加输出不超过 8192 的请求覆盖完整主 KV block 到 DDR 的路径。短请求可能只有 tail，不能验证 DDR offload。完整 block 主 KV 进入 DDR；不满 block 的 tail 主 KV 与 indexer 相关数据进入 D HBM。

本次启用诊断只选前两条请求；请让其中至少一条包含完整 block。超过诊断数量后要重新启动 P/D 才能重新采样。日志收集使用真实目录：

```bash
find /home/c00612283/vllm-ascend-v0.30.0rc2/pd-logs \
  -type f -name server.log -exec grep -H 'KVAF_PD_DIAG' {} + \
  > /home/c00612283/pd-diag-all.log

/usr/local/python3.12.13/bin/python3 \
  /home/c00612283/vllm-ascend-v0.30.0rc2/scripts/check_kvaf_pd_diagnostics.py \
  /home/c00612283/pd-diag-all.log --require-ddr --require-compute
```

应优先收集同一次启动的 P/D 日志；旧日志可先移到其他目录，避免不同实验混合。日志包括实际 Indexer 写入、地址核验、HBM/DDR 字节检查、Decode Gate、SFA 与 MTP 证据；INDEXER_UNSUPPORTED 表示该层打点不支持，不能当作算子路径通过。诊断通过不等于模型精度通过，还需与相同权重/量化/输入/采样条件的非 PD 基线比较。四层模型的回答质量也需用其自身基线判断。

## 9. 再次失败时收集什么

- 构建失败：保留 kvaf-pd-build.log / kvaf-k-build.log，提供第一条 error 前后上下文；gmake Error 2 本身不是根因。
- 启动或请求失败：同时提供本次 P、D server.log 与 Proxy、K 输出，保留最早 traceback。
- -9：宿主机查 dmesg 或 journalctl -k；现场是 cgroup v1，查容器实际 memory.limit_in_bytes、memory.failcnt、memory.oom_control。不要只查 v2 memory.events。
- 资源冲突：npu-smi info 与 ps 核对实际 PID、容器归属和卡号，避免伤及其他模型服务。
- 精度问题：保存原始请求/响应、模型配置与代码 SHA，提供同模型非 PD 基线和本次 KV 诊断结果。

当前诊断适配修复 b3f5c282e 已通过 218 项 CPU/mock UT，未代替本机 NPU 精度和稳定性实测。仅更新 PD Python 代码时，无需重编 KVAF C++，重启 P/D/Proxy 即可；首次安装、SDK/C++ 或 ABI 改动则按本教程重新编译。
