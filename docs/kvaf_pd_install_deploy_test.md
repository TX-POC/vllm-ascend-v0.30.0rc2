# GLM5.3 W4A4C8：安装、部署与测试方法

适用版本：`v0.30.0rc2-develop` 的独立 PD Connector 接入实现；首批组件历史版本为 `cef996c1b0c6a7c0f6c85c51ceb91e2a53d9f937`。

## 1. 当前可执行范围

当前分支已接入独立 PDKVoffloadConnector、外部 Worker/Runner、真实逐层 arena、普通 tail 接收区、TCP 布局交换、全建链注册、双 Pull 与原生 Decode Backend。实际入口、原生配置和 SDK RA 补丁以 [Connector 部署说明](kvaf_pd_connector.md) 为准。当前云环境只执行 CPU/mock 逻辑验证，真实模型 PD 仍需现场验收。

按以下层次执行，避免把组件测试成功当成模型部署成功：

| 层次 | 环境 | 当前能做什么 | 通过意味着什么 |
| --- | --- | --- | --- |
| A：组件 UT | Linux CPU、Python 3.12 | 安装测试依赖并运行全部 KVAF UT | 地址、字节、事务及 Backend 关键逻辑通过 |
| B：独立 SDK | 950DT NPU + 鲲鹏 K + CANN/UB | 构建 KVAF，运行 CPU Pull、NPU raw READ、KVOP 测试 | 对应设备/链路/算子通过，不等于完整模型 PD 验收通过 |
| C：完整 PD | 两台 P、两台 D、八台 K、Proxy | 按新 Connector 配置启动，再按第 8 节验收 | 模型、跨节点路由、传输和释放均通过 |

无 NPU/CANN/UB 时先执行 A；这些硬件缺失不阻塞当前开发与逻辑验证。本文未执行容器启动、SDK 硬件构建或模型部署。实现状态详见 [实施记录](kvaf_pd_implementation.md)。

暂无独立鲲鹏服务器时，先按 [昇腾机头 CPU 作为临时 K 的联调测试](kvaf_pd_ascend_host_cpu_test.md) 部署 CPU 服务；该文档覆盖同机双卡及借另一台昇腾机头 CPU 的测试方法。

## 2. 版本与目录

| 内容 | 版本依据 |
| --- | --- |
| 框架仓 | `https://github.com/TX-POC/vllm-ascend-v0.30.0rc2.git`，`v0.30.0rc2-develop` |
| 本文验收代码 | 当前开发分支的 PD Connector 接入实现 |
| 新 Pull SDK | `https://github.com/TX-POC/KVAF-Devlop.git`，`codex/cpu-pull-hbm-reference` |
| SDK 源码核对提交 | `a2f212b2811a40ea2df4351a451381897351dd49` |

以下在**新目录**克隆。已有工作副本可直接使用，先核对 `git status` 和版本，不覆盖本地修改。`PD_WORKSPACE` 等是命令中的路径/设备替换变量，不是新增的 vLLM 配置项。

```bash
export PD_WORKSPACE="$PWD/pd-work"
export ASCEND_SRC="$PD_WORKSPACE/vllm-ascend"
export KVAF_PD_SRC="$PD_WORKSPACE/kvaf-pd-sdk"
mkdir -p "$PD_WORKSPACE"

git clone --branch v0.30.0rc2-develop --single-branch \
  https://github.com/TX-POC/vllm-ascend-v0.30.0rc2.git "$ASCEND_SRC"
git clone --branch codex/cpu-pull-hbm-reference --single-branch \
  https://github.com/TX-POC/KVAF-Devlop.git "$KVAF_PD_SRC"

git -C "$ASCEND_SRC" rev-parse HEAD
git -C "$KVAF_PD_SRC" rev-parse HEAD
```

分支会继续更新。如需精确复现本文，在上述新克隆中固定版本：

```bash
git -C "$ASCEND_SRC" switch v0.30.0rc2-develop
git -C "$KVAF_PD_SRC" checkout --detach a2f212b2811a40ea2df4351a451381897351dd49
```

重要版本差异：

- 框架的 `scripts/install_kvaf.sh` 固定旧 SDK `c4375e13...`，并应用旧版 RA 补丁；本方法的新 Pull SDK 使用独立目录，按第 5 节构建。
- 目标 SDK 的 `build.sh npu` 没有显式开启 `KVAF_ENABLE_CPU_OFFLOAD`。新建构建目录时默认 OFF；只执行它不足以准备 CPU Pull API。
- 新 PD 入口使用专用 Runtime 适配，不传旧 Adapter 的 reuse_device_ra 参数。进入 TP/EP 模型联调前，必须按 [Connector 说明](kvaf_pd_connector.md#2-sdk-源码构建) 构建显式借用 HCCL device RA 的 binding；独立 SDK PASS 不等于 HCCL 共存已实测。

## 3. A：当前 CPU 安装与 UT

### 3.1 建立独立测试环境

以下版本对应本云环境已通过的 Python 3.12 CPU 组合。测试只使用源码目录，不需要安装完整 vLLM、vLLM-Ascend、KVAF 二进制或模型权重。不要把 CPU torch wheel 安装进已有 torch_npu 推理环境。

```bash
python3.12 -m venv "$PD_WORKSPACE/venv-cpu"
source "$PD_WORKSPACE/venv-cpu/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.3.5 numba==0.68.1 llvmlite==0.50.0 ruff==0.14.0

python -c 'import sys, torch, numpy, numba; print(sys.version); print(torch.__version__, numpy.__version__, numba.__version__)'
```

这些固定版本用于复现 CPU 结果，不是 P/D 推理容器的依赖版本要求；推理容器继续使用经过配套验证的 torch/torch_npu/CANN。

### 3.2 新增 PD UT：43 项

在框架仓根目录执行。保留已有 `PYTHONPATH`，增加 KVOP 的纯 CPU 实现；无需编译 KVOP NPU 算子。使用 Bash 的 `pipefail` 保留测试失败退出码。

```bash
cd "$ASCEND_SRC"
mkdir -p "$PD_WORKSPACE/results"
set -o pipefail

PYTHONPATH="$ASCEND_SRC/third_party/KVOP${PYTHONPATH:+:$PYTHONPATH}" \
  python -m unittest discover -s tests/ut -p 'test_kvaf_pd*.py' -v \
  2>&1 | tee "$PD_WORKSPACE/results/pd-ut.log"
```

本版预期：`Ran 43 tests`，结尾 `OK`，退出码 0，且没有跳过这些测试。其中 `test_kvaf_pd.py` 为 36 项，`test_kvaf_pd_backend.py` 为 7 项。

### 3.3 KVAF 总回归：121 项

```bash
PYTHONPATH="$ASCEND_SRC/third_party/KVOP${PYTHONPATH:+:$PYTHONPATH}" \
  python -m unittest discover -s tests/ut -p 'test_kvaf_*.py' -v \
  2>&1 | tee "$PD_WORKSPACE/results/kvaf-all-ut.log"
```

本版预期：43 项新增 + 78 项已有 = 121 项，结尾 `OK`，退出码 0。该计数只适用于上述提交。UT 中设备边界由替身提供，布局、事务、原生目录发布及缓存策略代码实际执行；并不建立真实 UB/RDMA 链路。

### 3.4 语法与格式

```bash
python -m compileall -q \
  vllm_ascend/distributed/kv_transfer/kvaf/pd \
  tests/ut/test_kvaf_pd.py tests/ut/test_kvaf_pd_backend.py

ruff check vllm_ascend/distributed/kv_transfer/kvaf/pd tests/ut/test_kvaf_pd*.py
ruff format --check vllm_ascend/distributed/kv_transfer/kvaf/pd tests/ut/test_kvaf_pd*.py
git diff --check
```

仓库要求的全仓检查为 `bash format.sh ci`，需先安装 `requirements-lint.txt`。该脚本含自动修正，建议在可丢弃的文件副本运行。本版曾因基线格式、日志入口、禁止导入及 workflow 配置未整体通过；这些失败没有通过修改原生核心处理。新增组件的静态检查已通过，完整记录见实施文档。

## 4. P/D/K 部署准备

### 4.1 部署清单

| 节点 | 总体并行配置 | 进程/资源要求 |
| --- | --- | --- |
| P0、P1 | P 集群 TP8 / DP2 / EP16 | 合计 16 个 P NPU worker；通常每台承载一个 8-TP 的 DP 组 |
| D0、D1 | D 集群 TP1 / DP16 / EP16 | 合计 16 个 D worker；每台承载 8 个本地 DP worker |
| K0～K7 | KVAF CPU 服务 | 每个 D worker 绑定一个 K endpoint，持有独立 Runtime/DDR region |
| Proxy | 原生 PD 请求路由 | 接入实际 P/D endpoint、请求标识与 D 定位信息 |

EP16 不再乘出一套额外 worker。各 D worker 到 K 的实际映射逐一填写，不能默认每台 K 固定对应两个 D。每个 K 的 DDR 预算应覆盖分配到它的所有 D region。

部署清单至少保存：镜像 digest、vLLM SHA、框架/SDK SHA、权重与量化配置版本、节点 IP、物理 NPU ID、DP/TP 身份、K endpoint、UB 设备/EID、每 worker DDR 预算和端口。

当前 `kvaf_config.cpu_addresses` 的键是**宿主机物理 NPU ID**，不是全局 DP rank；同一物理 ID 在两台 D 主机上需要各自的配置。新 PD Worker 的路由接线完成后还须验证映射实际生效。

容量先按下式估算，最终以实际 region ticket 和内存预算为准：

```text
完整主 KV payload/request = floor(N / 128) × Target 层数 × 83,968 B
DDR 预留需求还包含 slot 对齐、并发、目录及已启用辅助缓存等开销。
若沿用既有 4 KiB slot 对齐，单层满块 slot 为 86,016 B。
```

`ddr_size_gb` 是每个 D worker 的总预算，不是每层预算。DDR 在 D 调用 Runtime 初始化时分配；K 服务启动本身不代表 DDR region 已建立。

### 4.2 P/D 容器和框架安装

有真实 950DT 环境时，使用现场已经验证的 A5 开发镜像，并保留 digest。仓库提供 `Dockerfile.a5` / `Dockerfile.a5.openEuler`；不要把 GLM 文档中的旧 A3/v0.23 镜像当成本方案的已验证镜像。

在各 P/D 宿主机、框架仓目录中执行，先将 `IMAGE` 设置为实际镜像：

```bash
cd "$ASCEND_SRC"
: "${IMAGE:?请设置现场验证的 A5 开发镜像}"
bash scripts/start_vllm_container.sh "$IMAGE" glm53-pd-dev
docker exec -it glm53-pd-dev bash
```

该脚本要求宿主机存在对应设备和挂载目录，并使用 host 网络、512 GiB shm 上限、特权模式和所有选中 NPU。按实际硬件与资源准备；同名容器存在时不会自动删除。

在容器内，镜像须已有 `/vllm-workspace/vllm` 源码和配套构建依赖：

```bash
cd /vllm-workspace/vllm-ascend
bash scripts/install_vllm_in_container.sh

python3 -c 'import torch, torch_npu; print(torch.__version__); print(torch_npu.__version__)'
python3 -m pip show vllm vllm-ascend
git -C /vllm-workspace/vllm rev-parse HEAD
git rev-parse HEAD
```

安装脚本加载 CANN/ATB 并做 editable 安装，使用镜像的配套依赖。记录镜像内实际 vLLM 提交；只安装框架源码不会启用新 PD 组件。

进入新容器后重新设置 `ASCEND_SRC=/vllm-workspace/vllm-ascend`，并将 `KVAF_PD_SRC` 设为容器内实际可见的 SDK 路径。脚本绑定了框架仓及 `/data`、`/mnt`、`/home` 等目录；若宿主机的 SDK 克隆不在这些挂载中，可在容器可写目录按第 2 节单独克隆同一 SDK 提交。各 K 也需准备相同源码，在各自架构上编译。

## 5. B：独立 KVAF SDK 安装

本节用于验证 SDK 数据面。先在独立测试环境执行，完成 RA/HCCL 兼容后再与 vLLM TP/EP 会话集成。CPU/NPU 两端使用同一 SDK 提交重新构建；原生协议为 KVRP/KVHR v4。

### 5.1 八台 K：CPU 服务构建和启动

在各鲲鹏 K 上使用同一 SDK 源码，准备已验证的 UB 驱动、UMDK/liburma/provider、C++17 编译器及 CMake。在 SDK 根目录构建：

```bash
cd "$KVAF_PD_SRC"
cmake -S . -B build-pd-cpu \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DKVAF_BUILD_REMOTE_CPU_SERVICE=ON \
  -DKVAF_BUILD_PYTHON=OFF \
  -DKVAF_ENABLE_CPU_OFFLOAD=OFF \
  -DKVAF_ENABLE_AIV_BATCH_GET=OFF \
  -DKVAF_ENABLE_AIV_BATCH_PUT=OFF \
  -DBUILD_TESTING=OFF
cmake --build build-pd-cpu -j8

./build-pd-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28999
```

liburma 不在默认位置时，给 **cmake 配置命令**增加实际 `-DKVAF_URMA_ROOT=...` 或 `-DKVAF_URMA_INCLUDE_DIR=... -DKVAF_URMA_LIBRARY=...`。运行进程保留相应动态库路径和 `/dev/uburma` 访问权限。

多 UB 设备时可在服务命令追加现场确认的 `--ub-device`、`--eid-index`。每台 K 的 `K_IP:28999` 是示例 endpoint，端口可改。用 `ss -lntp` 查看监听、保存 candidate 日志；控制端口可达之后还需实际 READ 校验。

### 5.2 NPU 测试环境：同时构建 CPU Pull 和 Get/Put/raw READ

先在当前终端加载现场匹配的 CANN 环境，使用已安装 torch/torch_npu 的 Python，并为该 Python 安装 pybind11。需要 A5 ccec/ld.lld 和配套 asc-devkit 的 `simt_api/device_atomic_functions.h`。`ASCEND_CANN_PACKAGE_PATH` 应指向包含 `include/acl/acl.h` 的 toolkit 根目录。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd "$KVAF_PD_SRC"
python3 -m pip install 'pybind11>=2.13'

cmake -S . -B build-pd-npu \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DKVAF_ENABLE_CPU_OFFLOAD=ON \
  -DKVAF_ENABLE_AIV_BATCH_GET=ON \
  -DKVAF_ENABLE_AIV_BATCH_PUT=ON \
  -DKVAF_BUILD_PYTHON=ON \
  -DKVAF_BUILD_REMOTE_CPU_SERVICE=OFF \
  -DBUILD_TESTING=OFF \
  -DPython_EXECUTABLE="$(command -v python3)"
cmake --build build-pd-npu -j8

export PYTHONPATH="$KVAF_PD_SRC/build-pd-npu/python${PYTHONPATH:+:$PYTHONPATH}"
ls build-pd-npu/bin/kvaf_execute_batch_get_kernels.o \
   build-pd-npu/bin/kvaf_execute_batch_put_kernels.o \
   build-pd-npu/bin/kvaf_execute_batch_get_raw_addr_kernels.o

python3 - <<'PY'
import kvaf
print(kvaf.__file__)
for name in ('NpuRuntime', 'CpuPullHbmSource', 'NpuPullHbmSource'):
    assert hasattr(kvaf, name), name
for name in ('prepare_put', 'submit_cpu_pull', 'wait_cpu_pull', 'commit_put',
             'attach_cpu_source', 'attach_hbm', 'batch_get_raw_addr', 'resolve_raw_addr'):
    assert hasattr(kvaf.NpuRuntime, name), name
print('KVAF_PULL_API_PRESENT')
PY
```

确认 `kvaf.__file__` 来自当前 `build-pd-npu/python/kvaf`。三项 AIV object 和 Python binding 使用同一次构建的版本；raw READ 需要 V2 ABI。该 API 检查不创建 Runtime，也不证明建链或数据正确。

## 6. B：SDK 数据面与算子测试

以下现有 KVAF 脚本通过 multiprocessing/Pipe 和本机 listener 交换元数据，适用于**同机双 NPU**，K 可在远端。它们不是跨主机 P/D 部署脚本。在空闲测试设备上执行；`--device-a` 是读取端，角色对应 D；`--device-b` 是生产端，角色对应 P。

先设置真实 endpoint，例如 `K_ENDPOINT=实际K地址:28999`；NPU ID 是当前进程可见的逻辑设备号：

```bash
cd "$KVAF_PD_SRC"
: "${K_ENDPOINT:?请设置真实 K endpoint}"
export PD_READER_DEVICE=0
export PD_SOURCE_DEVICE=1
```

### 6.1 主 KV 的 CPU Pull

```bash
python3 tests/python/test_remote_cpu_offload.py \
  --cpu "$K_ENDPOINT" \
  --device-a "$PD_READER_DEVICE" --device-b "$PD_SOURCE_DEVICE" \
  --rounds 3
```

预期出现每轮 `REMOTE_OFFLOAD_ROUND_PASS` 和最终 `REMOTE_OFFLOAD_PASS`，退出码 0。脚本检查 Prepare/Submit/Wait/Commit、DDR checksum、重复通知、目录及源复用确认。这里使用独立 SDK 测试缓冲区，没有运行 GLM。

### 6.2 CPU Pull + NPU raw READ + 普通 Get/Put

分别填入两张卡的 **fullmesh EID**。CPU↔NPU 与 NPU↔NPU 路由不是同一套 EID；脚本提示 fullmesh 不包含在当前自动发现中，不能套用源码示例机器的 EID。

```bash
: "${D_FULLMESH_EID:?请设置读取端 fullmesh EID}"
: "${P_FULLMESH_EID:?请设置生产端 fullmesh EID}"

python3 tests/python/test_hbm_read_raw_addr.py \
  --cpu "$K_ENDPOINT" \
  --device-a "$PD_READER_DEVICE" --device-b "$PD_SOURCE_DEVICE" \
  --npu-eid-a-raw "$D_FULLMESH_EID" --npu-eid-b-raw "$P_FULLMESH_EID" \
  --batch-get-kernel "$KVAF_PD_SRC/build-pd-npu/bin/kvaf_execute_batch_get_kernels.o" \
  --batch-put-kernel "$KVAF_PD_SRC/build-pd-npu/bin/kvaf_execute_batch_put_kernels.o" \
  --raw-kernel "$KVAF_PD_SRC/build-pd-npu/bin/kvaf_execute_batch_get_raw_addr_kernels.o" \
  --rounds 3
```

预期最终 `HBM_RAW_PASS`，退出码 0。必要时用脚本已有的 `--npu-eid-a`、`--npu-eid-b-cpu` 和 `--cpu-candidate` 固定现场 CPU 路由。成功标准是输出/DDR 数据比较通过，不能只看 attach 成功。

脚本出现 `REMOTE_OFFLOAD_RETAIN` 时表示未决资源仍被保留，应查询原事务/检查 K 日志并确认 DMA 停止，再退出相关进程。

### 6.3 KVOP/LIDU 硬件检查

在已安装框架的 P/D 测试容器、框架仓根目录执行：

```bash
cd /vllm-workspace/vllm-ascend
bash third_party/KVOP/build.sh
source third_party/KVOP/env.sh
python3 third_party/KVOP/tests/lidu_feedback_npu.py
```

此脚本使用逻辑 NPU 0，检查 eager/graph replay、1/4-query 混合与反馈，不需要权重或 K 服务；它仍是算子测试，不代表模型 MTP 精度验收。

## 7. B：独立传输性能方法

先通过第 6 节正确性测试，再测性能。开始用小批次确认，随后逐步增加；CPU 场景的 `--block-bytes 83968` 对应一个单层主 KV 满块 payload。

```bash
cd "$KVAF_PD_SRC"
python3 tests/python/bench_remote_cpu_offload.py \
  --cpu "$K_ENDPOINT" \
  --device-a "$PD_READER_DEVICE" --device-b "$PD_SOURCE_DEVICE" \
  --blocks 16 --block-bytes 83968 --warmup 3 --iterations 10

python3 tests/python/bench_hbm_read_raw_addr.py \
  --device-a "$PD_READER_DEVICE" --device-b "$PD_SOURCE_DEVICE" \
  --npu-eid-a-raw "$D_FULLMESH_EID" --npu-eid-b-raw "$P_FULLMESH_EID" \
  --raw-kernel "$KVAF_PD_SRC/build-pd-npu/bin/kvaf_execute_batch_get_raw_addr_kernels.o" \
  --packets 16 --packet-bytes 83968 --warmup 3 --iterations 10
```

分别检查 `CPU_PULL_BENCH_PASS` 和 `NPU_HBM_BENCH_PASS`。CPU 默认统计 PreparePut→Commit 的 Python 调用区间，NPU 默认统计 raw READ 调用区间；不将二者直接当纯链路带宽。需要阶段统计时按 SDK `docs/cpu_pull_offload_usage.md` 开启汇总，逐批日志保持关闭。记录实际字节数、批次数、并发、均值/P95、错误数及所用 EID/版本。

## 8. C：完整 PD 部署后的验收方法

本节用于真实硬件验收。启动采用 [新 Connector 的原生配置](kvaf_pd_connector.md#3-原生配置入口)；现有 scripts/start_vllm_server.sh 保持非 PD 基线，不会仅因 kvaf_enabled=true 自动选择新 Connector。

### 8.1 启动与就绪顺序

1. 启动 K0～K7，核实控制监听与 UB candidate。
2. P/D worker 可并行初始化实际 HBM；D 按映射初始化独立 Runtime/DDR region，取得 ticket。
3. D 将 ticket 发 P；P 按层、按 D peer 注册源，将 PA 布局和 SDK descriptor/ticket 发 D；KVAF 内部处理 UBA 映射同步。
4. D attach 所需源；核实所有 P NPU↔D NPU、P NPU↔D 所属外置 CPU 的建链/MR 导入。
5. 完成全部 worker 初始化后，将 P/D endpoint 加入 Proxy，再接收请求。

MR 全建链不表示每个请求从全部 P 副本重复传输。请求仍须匹配其 P DP/TP 贡献组和目标 D worker。

接线完成版本的配置须沿用原生 `--kv-transfer-config` / `--additional-config`：P producer、D consumer，显式选择真实注册的新 Connector；P TP8/DP2、D TP1/DP16，均 EP16；主 KV block size 128、C8、MTP3；D 本地 vLLM prefix 关闭，P 沿已验证配置。新 PD 的 DDR/Get 为逐层几何 `(656,128)`；非 PD 原生 `GROUP_WIDTH=4` 不变。

真实模块路径、worker 入口、ZMQ/源 listener 端口及外部接收区接口应随实现给出，当前没有可执行的对应字段组合。

### 8.2 请求正确性和字节核对

N 表示本次实际生产、交接的有效 KV token 数，以源 Meta 为准。不能用字符数代替，也应核对 MTP/末 token 处理后的实际交接长度。

| N | 主 KV DDR 满块数 | DDR payload B/层 | tail token | tail HBM B/层 | Indexer+Scale HBM B/层 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 1 | 656 | 132 |
| 127 | 0 | 0 | 127 | 83,312 | 16,764 |
| 128 | 1 | 83,968 | 0 | 0 | 16,896 |
| 129 | 1 | 83,968 | 1 | 656 | 17,028 |
| 256 | 2 | 167,936 | 0 | 0 | 33,792 |
| 300 | 2 | 167,936 | 44 | 28,864 | 39,600 |

N=0 是组件边界用例，不要求模型 HTTP 接受空 token 请求。N=300 的 84 个 padding token 不传输、不计入有效长度。上表是有效 payload，未包含 DDR slot 对齐或协议开销。

每个用例核对：P/D 各自 block table、PA 差和源 offset；Put 当前 entries 地址；主 KV 满块 DDR 字节、Indexer/Scale 与 tail 的 D HBM 字节；Wait/Commit/目录发布的先后；所有必需层、MTP 层、贡献者完成后才放行 D。源 offset 从对应 plane 的 block table 取物理 block，UBA 绝对地址由 SDK 解析。

另覆盖 chunked prefill、TP8→TP1 空贡献、短于 2048 token、tail 续写到满块、多请求并发、原生 graph/eager 模式和 MTP3。逐步扩展规模：单贡献组/单 D → TP8/单 D → 两 P/两 D → 八 K 映射；记录每步实际启用的 worker。

### 8.3 HTTP 冒烟和模型精度

待真实 PD 服务就绪，将 `PD_PROXY_URL` 设置为 Proxy origin，再使用仓库原有请求脚本：

```bash
cd "$ASCEND_SRC"
: "${PD_PROXY_URL:?请设置已就绪的 PD Proxy origin}"
DSA_BASE_URL="$PD_PROXY_URL" DSA_MODEL=glm DSA_MAX_TOKENS=64 \
  bash scripts/test_model_curl.sh short
```

脚本验证 HTTP 和非空输出并保存请求/响应。还应以相同权重、量化、prompt、temperature/seed 做原生基线对照，并运行既有模型精度任务；HTTP 200 或输出有文字不能替代 KV 字节及精度验证。

现有 `scripts/benchmark.py` 依赖 `/tokenize`、模型信息、流式 token IDs 和 DP 路由；当前通用 PD Proxy 示例未提供 `/tokenize`。后续确认这些接口被正确转发后再使用它压测 PD，不能直接把单机基线的 DP 参数改为 16 视为 PD 性能验收。PD 性能应记录端到端 TTFT、Decode TPOT、有效输出 token TPS、P/D 计算与两条 Pull 的耗时，并单独记录失败/超时。

### 8.4 故障与释放

| 场景 | 预期结果 |
| --- | --- |
| Submit 响应丢失 | 保留原 Put、Ready、源和目的，查询/等待原事务；不重新 PreparePut |
| Commit 确认丢失 | 重试原 Commit 的确认；不改 Abort、不重发 READ |
| raw READ 确认丢失 | 原 Ready 调用 resolve_raw_addr；成功后才标记 output 就绪 |
| 迟到/重复消息、旧 epoch | 不影响另一个活动批次，不误释放新源 |
| 取消、抢占、K 中断 | DMA 不确定时继续保活；按原生错误/无效块通道处理，不新增业务自动重新 Prefill |
| 请求结束 | 同一 key 池正确释放；继续考虑既有引用归属，避免重复释放 |

正常停止：Proxy 停止接收新请求 → 完成/恢复在途 Put/READ → release batch → D detach → P source shutdown → D Runtime/DDR 关闭 → K 最后退出。源、三数组和 output 的长期引用按 SDK 生命周期保留，不因超时就释放。

## 9. 排障与验收记录

| 现象 | 优先检查 |
| --- | --- |
| `ModuleNotFoundError: test_kvaf_adapter` | 使用仓库根目录的 unittest discover，保留 tests/ut 作为测试发现目录 |
| `No module named kvop` | `PYTHONPATH` 加入当前仓的 third_party/KVOP；CPU UT 不需要编译 NPU 算子 |
| 缺少 CpuPullHbmSource / submit_cpu_pull | SDK 提交、CPU_OFFLOAD CMake 开关、导入目录、旧 .pth 遮蔽 |
| raw kernel 缺失/ABI 拒绝 | 同一 SDK 构建的 V2 raw object 与绑定；避免旧构建目录混用 |
| `reuse_device_ra` 参数报错 | 旧 Adapter/目标 SDK 签名不兼容，按实施记录完成适配 |
| TCP 可达但注册/READ 失败 | UB provider、候选/EID、CPU 与 fullmesh 路由、MR 导入和设备可见性 |
| `Busy` / pending Put | 原事务仍持有 Runtime；协调已有 Decode Get，不另建事务绕过 |
| `unknown connector PDKVoffloadConnector` | 核对框架分支/导入路径及 kv_connector_module_path；同时选择外部 PDKVWorker，单独安装 SDK 不会切换框架代码 |

每轮保存：版本清单、启动配置、测试命令与退出码、UT 计数、SDK PASS 标记、按用例的有效字节数、故障恢复结果和性能原始数据。诊断数据在独立验证阶段采集，推理热路径不加逐步 Debug/Info 打印。

本文生成时复跑 A 层：新增 43 项、总回归 121 项均通过。16 段 Bash 示例通过语法检查（未执行安装/部署命令），4 条 SDK 测试/压测命令的参数与对应源码匹配；Markdown 格式和拼写检查通过。

B/C 层应在现场执行并填入真实结果；当前云环境的 CPU 结果不能标成 SDK 硬件或完整 PD 已通过。
