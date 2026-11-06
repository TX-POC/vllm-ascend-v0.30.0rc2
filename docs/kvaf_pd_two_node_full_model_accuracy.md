# 双机全模型 PD 安装、部署与准确度验收

更新：2026-11-06。以本次现场成功构建及后续启动排障过程刷新。两机已出现安装成功标志，但随后 D profiling 暴露原生扩展缺失、P 等待 D 就绪；尚未完成本轮完整模型推理及准确度验收。启动必须以实际 health 与 worker 就绪验证，不能仅凭进程开始加载或安装成功标志判定。

## 1. 本次部署配置

| 项目 | P 节点 | D 节点 |
| --- | --- | --- |
| 业务 IP | 141.62.38.16 | 141.62.38.24 |
| NPU | 8 张 950DT，卡 0～7 | 8 张 950DT，卡 0～7 |
| 并行 | TP8 / DP1 / EP8 | TP1 / DP8 / EP8 |
| HTTP / DP RPC | 8100 / 13389 | 8200 / 13489 |
| Connector 角色 | kv_producer | kv_consumer |
| Connector 控制端口 | 注册全部 D worker | 31000～31007 |
| KVAF Decode Backend | enabled=false，仍进行源注册 | enabled=true |
| 主机附属服务 | Proxy，端口 8000 | 机头 CPU K，端口 28997 |

共同配置：完整模型 /home/models/GLM-5.3-w4a4c8-mxfp4，网卡 enp34s0f1，Python /usr/local/python3.12.13/bin/python3；W4A4C8、MTP3、max-model-len=131072、block-size=128、eager、关闭 prefix caching、c8_enable_reshape_optim=false。首轮 max-num-seqs=1，max-num-batched-tokens=4096，gpu-memory-utilization=0.85。

D 的 ddr_size_gb=16 是每 worker 的预算，八 worker 合计约 128 GiB，另需元数据和系统余量。主 KV 基础预算：目标层数 × floor(上下文/128) × 128 × 656 × 每 worker 并发。例如 78 层、131072 tokens、单并发约 6.25 GiB/worker，不含对齐与其他开销；必须按实际 config 中层数计算。MTP 主 KV 仍驻 HBM。

这是两机验证，不替代原始两台 P、两台 D、八台 K 的生产验收。P/D 为独立引擎，各 EP8，不会合并成 EP16；不传 --nnodes 2。HCCL 本机集合通信和跨机 KVAF UB 数据链路分别验收。

## 2. 宿主机与容器操作边界

| 操作 | 执行环境 |
| --- | --- |
| docker exec、宿主 CA 导出、跨机 scp | 对应服务器裸机 |
| Ascend / KVOP / SDK 安装，P/D/Proxy 启动 | 已安装 CANN、torch、torch_npu、vLLM 的容器 |
| K 编译和启动 | 本次成功的 SDK、URMA、UB 设备齐全环境；可独立部署于24号裸机或容器 |
| curl 业务请求 | 任意可达客户端，@文件路径必须在该客户端实际存在 |

容器名示例为 vllm_ascend_dsa_pd，不同机器若名称不同需修改。/tmp 不跨裸机、容器或服务器共享；相同 root@localhost 提示符不能说明处在同一环境。启动和编译前先检查真实目录，不能创建空的同名源码目录应付错误。

裸机进入本机容器：

```bash
docker ps --format 'table {{.Names}}\t{{.Image}}'
docker exec -it vllm_ascend_dsa_pd bash
```

容器内检查：

```bash
ls -ld /home/c00612283/vllm-ascend-v0.30.0rc2 /home/c00612283/KVAF-Devlop
ls /home/c00612283/KVAF-Devlop/CMakeLists.txt \
   /home/c00612283/KVAF-Devlop/docs/connector_pull_api_design.md
/usr/local/python3.12.13/bin/python3 -c 'import sys, torch, torch_npu; print(sys.executable); print(torch.__version__, torch_npu.__version__)'
command -v cmake
command -v ninja
command -v g++
```

缺少 connector_pull_api_design.md 通常是 SDK 分支/目录不匹配，安装脚本会在 CMake 前退出，甚至留下零字节日志。使用实际包含该文件的 cpu-pull-hbm-reference 源码；本仓核对版本为 codex/cpu-pull-hbm-reference 的 a2f212b2811a40ea2df4351a451381897351dd49。先保存本地修改，再检查 git branch/status；不创建空文档、不删除检查、不 reset --hard。两端记录相同 Ascend/KVAF SHA 和匹配的构建补丁。

完整模型、tokenizer/chat template 需两端一致，不能沿用四层权重或其响应作为全模型基线。

## 3. CANN 环境和证书

本次构建日志的 CANN 路径为 /usr/local/Ascend/cann-9.2.0，Python 为 3.12.13，torch 2.10.0、torch_npu 2.10.0.post7、已有 vLLM 0.30.0。这些是现场版本记录，不是任意混用版本都兼容的声明。

16号曾先后出现 GitCode DNS 失败和经代理下载时证书链不受信任。24号构建成功后，将24号可信 CA bundle 传入16号并实际验证 GitCode 下载成功。CA 哈希相同只证明复制完整，curl 验证成功才证明其适用于当前连接。

在16号裸机执行（SSH 账户按现场权限调整）：

```bash
scp root@141.62.38.24:/etc/pki/tls/certs/ca-bundle.crt \
  /home/c00612283/ca-bundle-24.pem

docker exec -i vllm_ascend_dsa_pd sh -c '
set -e
cat > /tmp/ca-bundle-24.pem.new
test -s /tmp/ca-bundle-24.pem.new
mv /tmp/ca-bundle-24.pem.new /tmp/ca-bundle-24.pem
' < /home/c00612283/ca-bundle-24.pem

sha256sum /home/c00612283/ca-bundle-24.pem
docker exec vllm_ascend_dsa_pd sha256sum /tmp/ca-bundle-24.pem
```

本次 docker cp 曾报 remove staging directory: device or resource busy；上面的标准输入传输不使用 docker cp 的临时挂载，无需重启 Docker 或强制卸载目录。只有确定可信来源的 CA 才能使用，不从未验证的连接中自行获取证书并直接信任。

容器内验证：

```bash
curl --cacert /tmp/ca-bundle-24.pem \
  --fail --location --connect-timeout 20 --max-time 300 \
  'https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1/abseil-cpp-20230802.1.tar.gz' \
  -o /tmp/abseil-ca-check.tar.gz
tar -tzf /tmp/abseil-ca-check.tar.gz >/dev/null
```

## 4. 预备本地构建依赖

本次缺少 makeself、abseil、protobuf，均属于 Ascend 算子构建依赖，不是 KVAF Pull 错误。下载到 csrc/third_party/pkg，不能假定在裸机 /tmp 下载后容器就能看到。

下面在已验证 CA 的构建容器执行。其他正常环境也可使用其自身可信 CA；不全局关闭 TLS 校验。

```bash
(
set -euo pipefail
REPO=/home/c00612283/vllm-ascend-v0.30.0rc2
DEST="$REPO/csrc/third_party/pkg"
CA=/tmp/ca-bundle-24.pem
mkdir -p "$DEST"

curl --cacert "$CA" --fail --location --retry 3 \
  'https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1/abseil-cpp-20230802.1.tar.gz' \
  -o "$DEST/abseil-cpp-20230802.1.tar.gz.part"
tar -tzf "$DEST/abseil-cpp-20230802.1.tar.gz.part" >/dev/null
mv "$DEST/abseil-cpp-20230802.1.tar.gz.part" "$DEST/abseil-cpp-20230802.1.tar.gz"

curl --cacert "$CA" --fail --location --retry 3 \
  'https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz' \
  -o "$DEST/protobuf-25.1.tar.gz.part"
tar -tzf "$DEST/protobuf-25.1.tar.gz.part" >/dev/null
mv "$DEST/protobuf-25.1.tar.gz.part" "$DEST/protobuf-25.1.tar.gz"

curl --cacert "$CA" --fail --location --retry 3 \
  'https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz' \
  -o "$DEST/makeself-release-2.5.0-patch1.tar.gz"
printf '%s  %s\n' \
  bfa730a5763cdb267904a130e02b2e48e464986909c0733ff1c96495f620369a \
  "$DEST/makeself-release-2.5.0-patch1.tar.gz" | sha256sum -c -

WORK=$(mktemp -d /tmp/kvaf-makeself.XXXXXX)
tar -xzf "$DEST/makeself-release-2.5.0-patch1.tar.gz" -C "$WORK"
SCRIPT=$(find "$WORK" -type f -name makeself.sh -print -quit)
test -n "$SCRIPT"
SOURCE=$(dirname "$SCRIPT")
test -s "$SOURCE/makeself-header.sh"
mkdir -p "$REPO/csrc/third_party/makeself"
cp -a "$SOURCE/." "$REPO/csrc/third_party/makeself/"
chmod u+x "$REPO/csrc/third_party/makeself/"*.sh
echo "LOCAL_DEPENDENCIES_READY"
)
```

makeself 原生规则要求第三方目录中存在 makeself.sh 与 makeself-header.sh，仅保存压缩包不够。abseil/protobuf 的原生规则支持 pkg 目录中的上述文件名。补包后重新运行安装脚本使 CMake 重新配置；旧 Ninja 下载脚本仍可能保存 HTTPS URL，甚至删除旧包后再下载。

若从24号搬运已成功的依赖，打包这两个压缩包与 makeself 目录，传入16号对应 third_party；用传输前后 SHA256 验证搬运完整性。传输校验值不能替代依赖来源认证，需保留可信下载来源。

## 5. 保留已有 vLLM，安装 Ascend / KVOP / SDK

两台各自在构建容器执行：

```bash
(
set -euo pipefail
cd /home/c00612283/vllm-ascend-v0.30.0rc2
export KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3
export SOC_VERSION=ascend950dt_9582
export COMPILE_CUSTOM_KERNELS=1

# 本容器具备已验证 CA 时，供本次构建子进程使用。
if [ -s /tmp/ca-bundle-24.pem ]; then
    export SSL_CERT_FILE=/tmp/ca-bundle-24.pem
    export CURL_CA_BUNDLE=/tmp/ca-bundle-24.pem
    export REQUESTS_CA_BUNDLE=/tmp/ca-bundle-24.pem
    export PIP_CERT=/tmp/ca-bundle-24.pem
fi

set +u
if [ -f /usr/local/Ascend/cann-9.2.0/set_env.sh ]; then
    source /usr/local/Ascend/cann-9.2.0/set_env.sh
elif [ -f /usr/local/Ascend/cann-9.2.0/bin/setenv.bash ]; then
    source /usr/local/Ascend/cann-9.2.0/bin/setenv.bash
else
    echo "未找到 CANN 环境脚本"
    exit 1
fi
set -u

LOG="/home/c00612283/pd-build-$(date +%Y%m%d_%H%M%S).log"
echo "本轮日志：$LOG"
bash scripts/install_kvaf_pd_source.sh - \
  /home/c00612283/KVAF-Devlop \
  /home/c00612283/KVAF-Devlop/build-pd-py312 \
  2>&1 | tee "$LOG"
)
```

参数 - 只跳过 vLLM 源码安装，不跳过 vLLM-Ascend 编译。流程是检查环境 → 构建依赖 → Ascend 安装/算子构建 → KVOP → KVAF binding/RA 借用补丁 → 最终导入。期望 PD_SDK_IMPORT_OK、PD_SOURCE_INSTALL_OK；cpython-312 扩展匹配 Python3.12，不能加载旧 cpython-311 文件。阶段未通过就停止，不把 pip 最后的 failed-wheel-build 当作具体根因。

### 5.1 abseil 补丁找不到文件

本次出现过本地包已找到，但 Performing patch step 后 can't find file to patch。源码准备目录和步骤记录可能不一致；Stripping trailing CRs 本身不是根因。

先验证压缩包包含 CMakeLists.txt、absl/base/options.h、absl/hash/internal/hash.cc 与 hash.h。仅在确认是这个错误、已停止构建并保存本地修改后，备份 csrc/third_party/abseil-cpp 和 csrc/build/abseil_build_transformer-prefix，再从第 5 节重新配置。不要跳过符号隔离补丁，也不要仅手工解压未打补丁源码后让 CMake 将其当作已准备输入。

### 5.2 取得真正错误

完整日志保存到文件，不仅复制 pip 输出前几百行。本次多份附件只含弃用提示，实际 FAILED 任务不在其中。必要时在同一 CANN 环境中对现有目录单并发增量构建：

```bash
cmake --build /home/c00612283/vllm-ascend-v0.30.0rc2/csrc/build \
  --parallel 1 --verbose > /home/c00612283/ascend-ninja-debug.log 2>&1

grep -n -B 5 -A 60 -m 1 \
  -E 'FAILED:|fatal error:|CMake Error|Each download failed|status_code: 60|Could not resolve|Killed signal' \
  /home/c00612283/ascend-ninja-debug.log
```

CMP0135/DOWNLOAD_EXTRACT_TIMESTAMP、可选 ccache 未找到、许可证弃用和头文件 pragma message 不能单独判定为失败根因。

## 6. 启动前生成产物检查

PD_SOURCE_INSTALL_OK 不是后续源码目录永远完整的保证。若目录被替换/清理或 editable 状态变化，应检查实际导入位置。

### 6.1 KVOP 缺失或多 vendor

启动若显示 Build KVOP first，在同一 P/D 容器内独立恢复：

```bash
(
set -euo pipefail
cd /home/c00612283/vllm-ascend-v0.30.0rc2
bash third_party/KVOP/build.sh 8 /usr/local/python3.12.13/bin/python3 \
  2>&1 | tee /home/c00612283/kvop-build.log
source third_party/KVOP/env.sh
echo "KVOP_ENV_OK"
)
```

env.sh 要求 original/_custom_opp/vendors/*/op_api/lib/libcust_opapi.so 匹配唯一文件。零个或多个都可能触发提示；多个 vendor 时先核对产物，不反复重建或随意删除。

### 6.2 _build_info 缺失

若出现 cannot import name '_build_info'，通过原生生成流程恢复，不手写硬件信息、不改原生 device_config：

```bash
(
set -euo pipefail
cd /home/c00612283/vllm-ascend-v0.30.0rc2
export SOC_VERSION=ascend950dt_9582
PYTHON=/usr/local/python3.12.13/bin/python3
BUILD_PY_DIR=$(mktemp -d /tmp/ascend-build-py.XXXXXX)
"$PYTHON" setup.py build_py --build-lib "$BUILD_PY_DIR" \
  2>&1 | tee /home/c00612283/ascend-build-info.log
"$PYTHON" - <<'PY'
from vllm_ascend import _build_info
from vllm_ascend.device.device_config import check_ascend_device_type
print(_build_info.__file__, _build_info.__device_type__)
check_ascend_device_type()
print("BUILD_INFO_OK")
PY
)
```

本次950DT预期 A5，必须由实际 SOC 生成并与运行硬件校验。上述 build_py 不重编 C++ 算子。两机各自检查；不要从未知版本复制生成文件。

### 6.3 原生 vllm_ascend_C 扩展缺失

本次 D 在 profiling 的 layernorm.forward_oot 中报 No module named 'vllm_ascend.vllm_ascend_C'。它不是 KVAF 的 _kvaf，也不是 KVOP 的 libcust_opapi.so。build_py 只生成 Python 产物，不能恢复这个原生扩展；PD_SOURCE_INSTALL_OK 的现有检查也未覆盖该模块，必须补充显式导入。

两机先在各自实际运行容器检查：

```bash
/usr/local/python3.12.13/bin/python3 - <<'PY'
import torch
import torch_npu
import vllm_ascend.vllm_ascend_C as extension
print("ASCEND_EXTENSION_OK", extension.__file__)
PY
```

若 ModuleNotFoundError，检查 COMPILE_CUSTOM_KERNELS 是否为0、源码是否更换、产物是否位于另一 Python/目录。通过原生安装流程重建，仅在导入失败的环境执行，先正常停止本次模型服务：

```bash
(
set -euo pipefail
cd /home/c00612283/vllm-ascend-v0.30.0rc2
export SOC_VERSION=ascend950dt_9582
export COMPILE_CUSTOM_KERNELS=1
export MAX_JOBS=8
/usr/local/python3.12.13/bin/python3 -m pip install \
  --no-deps --no-build-isolation -v -e . \
  2>&1 | tee /home/c00612283/ascend-native-extension-build.log
)
```

沿用已加载的 CANN/可信 CA 与已准备的依赖，不运行在 UT 用的 COMPILE_CUSTOM_KERNELS=0 模式。安装后再次通过上述扩展导入检查，才启动 D。不要修改原生 layernorm 以跳过该算子，也不要把 build_py 成功当作扩展构建成功。

本次 P 的 No available shared memory broadcast block found 提示出现在 Connector 初始化后，没有独立异常堆栈；Sender 会等待 D 的控制端点，D 已失败时 P 可能一直等待直到超时。先修复 D，再排空并重新启动 D/P 注册；不要仅靠延长超时处理。若 D 已就绪而 P 仍卡住，再检查 P→D 端口、源注册、UB 和完整 worker 日志。

## 7. 独立构建并启动 K

K 不依赖 vLLM-Ascend 仓、Python 或 torch，只需要匹配的 KVAF 源码、UMDK、驱动和设备访问。24号曾在 K 环境中仅有 KVAF 仓，找不到 Ascend 仓；此时直接 CMake，不必强行切换到另一个容器。

在24号选定的 K 环境执行：

```bash
(
set -euo pipefail
SRC=/home/c00612283/KVAF-Devlop
BUILD="$SRC/build-hostcpu-k"
test -f "$SRC/CMakeLists.txt"
test -f /usr/include/ub/umdk/urma/urma_api.h
test -f /usr/lib64/liburma.so
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DKVAF_BUILD_REMOTE_CPU_SERVICE=ON -DKVAF_BUILD_PYTHON=OFF \
  -DKVAF_ENABLE_CPU_OFFLOAD=OFF -DKVAF_ENABLE_AIV_BATCH_GET=OFF \
  -DKVAF_ENABLE_AIV_BATCH_PUT=OFF -DBUILD_TESTING=OFF \
  -DKVAF_URMA_INCLUDE_DIR=/usr/include/ub/umdk/urma \
  -DKVAF_URMA_LIBRARY=/usr/lib64/liburma.so \
  '-DCMAKE_CXX_FLAGS=-DURMA_ACCESS_REMOTE_READ=URMA_ACCESS_READ -DURMA_ACCESS_REMOTE_WRITE=URMA_ACCESS_WRITE -DURMA_ACCESS_REMOTE_ATOMIC=URMA_ACCESS_ATOMIC'
cmake --build "$BUILD" --parallel 8
test -x "$BUILD/runtime/kvaf_cpu_service"
ldd "$BUILD/runtime/kvaf_cpu_service"
echo "K_SERVICE_BUILD_OK"
)
```

URMA 枚举兼容参数来自已确认的 UMDK 26.06 环境，其他版本先核对头文件，不代表通用 ABI 兼容。ldd 不应出现 not found。绑定头文件不等于已经有可执行文件；SDK binding 与 K 服务是两次独立构建。

端口占用先用 fuser -v -n tcp 28997 核对归属；缺少 ss 可使用 fuser/netstat。不自动杀进程，不照抄历史 PID。K 启动：

```bash
/home/c00612283/KVAF-Devlop/build-hostcpu-k/runtime/kvaf_cpu_service \
  --listen 141.62.38.24 --port 28997 --memory-page-mode default
```

K 若在裸机，D 容器使用实际可达的 141.62.38.24:28997；容器部署须同时提供 UB 设备和匹配库。单个 K 支持多个 D 客户端，但各 worker 独立 Runtime/region，不是共享一份请求目录。

## 8. P/D 启动配置与顺序

现场实际脚本为 /home/c00612283/p_dp1tp8start.sh（16号）、/home/c00612283/d_dp8tp1start.sh（24号）。这些是现场手动保存的脚本，Git 仓库不会自动生成同名文件。

仓库 examples/disaggregated_prefill_v1/kvaf_two_node 中仍保留原先同 TP 对照模板，不能不修改就当作本次实际配置运行。按以下表直接修改模板的原生字段，或使用本次已启动的现场脚本：

| 字段 | p_start.sh | d_start.sh | d_baseline.sh |
| --- | --- | --- | --- |
| local_ip | 141.62.38.16 | 141.62.38.24 | 141.62.38.24 |
| TP / DP / local DP | 8 / 1 / 1 | 1 / 8 / 8 | 1 / 8 / 8 |
| max-model-len | 131072 | 131072 | 131072 |
| max-num-seqs | 1 | 1 | 1 |
| K cpu_address | 无 | 141.62.38.24:28997 | 关闭 KVAF |
| ddr_size_gb | 无 | 16，每 worker | 无 |
| Worker / Connector | PDKVWorker / producer | PDKVWorker / consumer | 原生默认 / 不配置 |

共同保持第 1 节模型与参数，served-model-name=glm。P 的 kvaf_config.enabled=false 是关闭其 Decode 卸载 Backend，PD 源注册仍使用 SDK，不能删掉 build_dir。D 开启 KVAF。两侧 PYTHONPATH 和 build_dir 指向本机 build-pd-py312；source 本机 third_party/KVOP/env.sh。

P 的 decode_endpoints 必须填写全部八个 D worker：

```json
[
  "141.62.38.24:31000", "141.62.38.24:31001",
  "141.62.38.24:31002", "141.62.38.24:31003",
  "141.62.38.24:31004", "141.62.38.24:31005",
  "141.62.38.24:31006", "141.62.38.24:31007"
]
```

其字段为原生 kv-transfer-config.kv_connector_extra_config.decode_endpoints。D kv_port=31000，端口公式 kv_port+dp_rank*tp_size+tp_rank。本次 TP 比率为8；一次请求由一个 D DP worker 接收，八个 P TP rank 按贡献规则参与。初始化为全建链，不能只配置两个 endpoint。

新拓扑或 D 重启后，应排空并正常停止本次 Proxy/P/D，再 K → D → P → Proxy 重新注册。K 可以保留。Ctrl+Z 只暂停进程，会继续占端口/HBM，应 Ctrl+C 正常退出。

24号容器启动 D：

```bash
cd /home/c00612283
./d_dp8tp1start.sh
```

16号容器启动 P：

```bash
cd /home/c00612283
./p_dp1tp8start.sh
```

两端健康后，在16号另一个容器终端启动 Proxy：

```bash
curl --fail --noproxy '*' http://141.62.38.16:8100/health
curl --fail --noproxy '*' http://141.62.38.24:8200/health

KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3 \
bash /home/c00612283/vllm-ascend-v0.30.0rc2/scripts/kvaf_pd_proxy.sh \
  --host 141.62.38.16 --port 8000 \
  --prefiller-hosts 141.62.38.16 --prefiller-ports 8100 \
  --decoder-hosts 141.62.38.24 --decoder-ports 8200
```

业务网可达不代表 UB 可达：还需允许 P→D 控制端口、D/K→P 注册源的动态 TCP 回连，以及跨机 P NPU↔D NPU、P NPU↔K CPU 的 UB/EID 通信。现场容器/驱动/设备映射和防火墙按实际拓扑验证，不复制其他服务器 EID，不仅凭 health 判断数据面通过。

## 9. 完整模型准确度验收

诊断默认 diagnostics_requests=0、verify_kv=false。需要验收时两侧同步改为2/true，重启后优先让前两条请求覆盖短请求与非整 block 长请求。开启会同步、哈希和回读，不用于性能测量；达到采样预算后不再覆盖新请求。

建议先在24号运行 d_baseline.sh 的原生混部基线，保存完整模型同请求结果，再正常停止，启动 PD。若已启动 P/D，也可排空后先测基线，再恢复 PD。基线和 D 不能同时抢占八张卡。异构 TP 下还可补 P TP8 原生基线，区分 TP 数值差异；单机四层对照结果不能替代全模型结果。

客户端先保存请求文件，再对同一文件分别请求基线8200与 Proxy8000：

```bash
cat > /home/c00612283/full-model-request.json <<'JSON'
{
  "model": "glm",
  "messages": [{"role": "user", "content": "请简短介绍中国的首都。"}],
  "temperature": 0,
  "top_p": 1,
  "seed": 1024,
  "max_tokens": 128,
  "stream": false
}
JSON

# 原生基线已启动时执行
curl --fail --noproxy '*' --max-time 600 \
  http://141.62.38.24:8200/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @/home/c00612283/full-model-request.json \
  -o /home/c00612283/full-model-baseline-response.json

# 停止基线并启动完整 PD 后执行
curl --fail --noproxy '*' --max-time 600 \
  http://141.62.38.16:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @/home/c00612283/full-model-request.json \
  -o /home/c00612283/full-model-pd-response.json
```

/v1/models 只是查询。直接对启用 consumer 的 D:8200 发送普通请求不能替代通过 Proxy 的分离测试。41510-token 长请求需复用真实原始 JSON，而不是响应文件或同长度的另一段文本。

每组每条重复三次，核对输入、生成序列（有 token IDs 时优先）、文本、finish_reason 与 completion_tokens，记录首个分歧；id/created 不要求相同。文本相同不等于证明 token IDs 相同，概率比较必须在相同生成前缀下进行。异构 TP、量化和 batch 形状可能导致数值差异，不能以“浮点误差”直接放行，需按固定任务集和事先约定的容差判定。单条首都问答不代表通用准确率。

两机日志默认在仓库根目录 pd-logs/p、pd-logs/d 下，因现场脚本先 cd 到仓库。以实际打印路径为准，保存本轮 P/D、Proxy、K 原始日志和实际启动脚本。将两端本轮文件收集到同一审计机后，可执行：

```bash
python3 /home/c00612283/vllm-ascend-v0.30.0rc2/scripts/check_kvaf_pd_diagnostics.py \
  /home/c00612283/two-node-p.log /home/c00612283/two-node-d.log \
  --require-ddr --require-compute
```

two-node-p.log / two-node-d.log 是现场从本轮 server.log 收集的文件，不会自动生成。长请求应覆盖满块与非零 tail；仅短请求不强制 DDR。检查地址、HBM/DDR 字节、副本、tail→ABC 与完成门槛；任何 FAIL 不能由 HTTP200 覆盖。

已知边界仍保留：部分计算事件未携带 generation，SFA_DONE/MODEL_FORWARD_DONE 是调用返回证据，HOT_GET_SUBMITTED 不代表非零 miss；旧 DDR_REGION 字段读取可能打印 null。de01cdf55 已修复等长 chunk 副本 JSON 被误去重的问题，但未修复全部证据缺口。审计 PASS 仅作辅助，不能作为设备完成或端到端准确度证书。

## 10. 本次问题速查

| 现象 | 处理 |
| --- | --- |
| cd 源码目录不存在 | 核对宿主/容器；K 可直接从独立 SDK 构建 |
| 无输出退出、零字节日志 | 检查 set -e 前置条件，尤其 SDK 说明文件和 Python 路径 |
| GitCode DNS / SSL 60 | 检查当前环境代理与可信 CA，或搬运可信下载的依赖；不改核心代码 |
| docker cp staging busy | docker exec -i 经标准输入传入，不重启整个 Docker |
| /tmp 包消失 | 从实际容器最终 pkg 目录检查，不混用裸机和另一台的 /tmp |
| Ninja subcommand failed | 找第一条 FAILED/具体 error，保留完整日志；不要把后续 pragma warning 当根因 |
| abseil patch 找不到目标 | 检查包结构，备份生成源码与步骤记录后重新配置，不跳过补丁 |
| KVOP Build first | 同容器独立构建并 source，检查唯一 vendor 库 |
| _build_info 导入失败 | 原生 setup.py build_py 生成并核验实际设备 |
| vllm_ascend_C 缺失 | 原生 COMPILE_CUSTOM_KERNELS=1 安装并显式导入；build_py/KVOP/SDK 成功不能替代 |
| P shared memory broadcast 等待 | 先核实 D 是否失败及控制端点是否就绪，不直接判定为共享内存容量不足 |
| 端口占用 | 确认进程归属，正常停止本次服务；不用 Ctrl+Z |
| HTTP200 但乱码 | 同完整权重、同输入基线对照，不能只看长度或 KV 日志 PASS |

本次刷新只修改文档。现场构建及启动阶段信息来自用户日志，尚未确认本轮最终就绪；云环境未执行 NPU 构建、跨机推理或准确度实测；未改变框架核心或诊断开关默认值。
