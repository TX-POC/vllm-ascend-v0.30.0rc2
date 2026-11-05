# PD 算子路径与 KV 字节诊断

适用于 PDKVoffloadConnector，所有改动位于外部 PD 扩展。默认关闭，不修改原生 Runner、attention 或算子注册。日志使用原生 vllm logger/格式，固定标记 KVAF_PD_DIAG，内容为单行 JSON；不打印 prompt、生成文本、原始 KV 数据或 P 源 UBA 基址。DDR 目的地址仅记录 KVAF Put 返回的权威值。

## 1. 开关与检查范围

两个字段放入原生 kv-transfer-config.kv_connector_extra_config：

```json
{"diagnostics_requests": 2, "verify_kv": true}
```

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| diagnostics_requests | 0 | 每 worker 最多选择的请求代次数，范围 0～16；0 不加载诊断对象或安装诊断装饰器 |
| verify_kv | false | 对选中传输的全部有效字节做 SHA256 对比；必须同时开启 diagnostics_requests |

只需要路径日志时设置 diagnostics_requests=2、verify_kv=false，不增加哈希、CPU 数据拷贝或 DDR 回读。字节校验要求两侧 verify_kv 一致，握手不匹配会拒绝注册。D 按本 worker 接收的前 N 个请求选取；P 配合对应 D 的选中请求进行校验，所以全建链环境中 P 可能配合多个 D 的各 N 个请求。诊断预算耗尽后不再覆盖新请求；重新启动 worker 可重新采集。

开启 verify_kv 会有同步、CPU 拷贝和一次一页的 DDR 回读，明显影响性能；不要用于性能测量或长期生产运行。CPU Pull 的回读发生在原 Put Commit 和目录发布之后，持有现有 Runtime 锁，使用独立单页临时 HBM，不改业务 output、不重放 Ready/Put。回读失败且完成状态不明时保留其 buffer/access table 并停止该 Runtime，禁止带着不确定读继续复用资源。

校验包括：P 全部 TP 贡献方主 KV 副本一致性（确认仅取 member 0 的前提）；P 源段与 D HBM 接收段；P 满块主 KV 与 Commit 后 DDR 回读；普通 tail 导入 ABC；原生末 prompt token 重算的 ABC/staging 拷贝。传输 SHA256 覆盖段内全部字节，不是抽样。只有有效 token 字节进入传输和哈希，padding 不计入。诊断失败沿原生 load-error 流程处理，不新增业务 rePrefill；原生 engine 可能进入其自有恢复逻辑，因此 HTTP 200 不能覆盖诊断 FAIL。

## 2. 当前现场完整启动示例

先正常停止本次 Proxy 新请求，排空并退出本次 P/D，更新开发分支后重启。Python 源码 editable 安装无需重编 SDK 或 C++；K 可以保持运行。下面使用 IP 141.62.38.12、K 28997、P TP2 物理卡 2/3、D TP1/DP2 物理卡 4/5。K 必须确实监听 28997；如果现场使用其他端口，改 D 的 cpu_address。

### D 终端

```bash
export KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3
cd /home/c00612283/vllm-ascend-v0.30.0rc2
set -o pipefail

D_TRANSFER='{"kv_connector":"PDKVoffloadConnector","kv_connector_module_path":"vllm_ascend.distributed.kv_transfer.kvaf.pd.connector","kv_role":"kv_consumer","kv_port":31000,"kv_connector_extra_config":{"diagnostics_requests":2,"verify_kv":true}}'
D_ADDITIONAL='{"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"kvaf_config":{"enabled":true,"build_dir":"/home/c00612283/KVAF-Devlop/build-pd-py312","cpu_address":"141.62.38.12:28997","ddr_size_gb":1}}'

VLLM_LOGGING_LEVEL=INFO ASCEND_RT_VISIBLE_DEVICES=4,5 \
bash /home/c00612283/vllm-ascend-v0.30.0rc2/scripts/start_vllm_pd.sh \
  d 141.62.38.12 enp34s0f1 -- /home/models/glm5-mtp-4l \
  --port 8200 --tensor-parallel-size 1 \
  --data-parallel-size 2 --data-parallel-size-local 2 \
  --data-parallel-address 141.62.38.12 --data-parallel-rpc-port 13489 \
  --max-num-seqs 2 --max-model-len 8192 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 \
  --kv-transfer-config "$D_TRANSFER" --additional-config "$D_ADDITIONAL" \
  2>&1 | tee /home/c00612283/pd-diag-d.log
```

### P 终端

待 D 控制监听就绪后执行：

```bash
export KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3
cd /home/c00612283/vllm-ascend-v0.30.0rc2
set -o pipefail

P_TRANSFER='{"kv_connector":"PDKVoffloadConnector","kv_connector_module_path":"vllm_ascend.distributed.kv_transfer.kvaf.pd.connector","kv_role":"kv_producer","kv_port":31000,"kv_connector_extra_config":{"decode_endpoints":["141.62.38.12:31000","141.62.38.12:31001"],"diagnostics_requests":2,"verify_kv":true}}'
P_ADDITIONAL='{"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"c8_enable_reshape_optim":false,"kvaf_config":{"enabled":false,"build_dir":"/home/c00612283/KVAF-Devlop/build-pd-py312"}}'

VLLM_LOGGING_LEVEL=INFO ASCEND_RT_VISIBLE_DEVICES=2,3 \
bash /home/c00612283/vllm-ascend-v0.30.0rc2/scripts/start_vllm_pd.sh \
  p 141.62.38.12 enp34s0f1 -- /home/models/glm5-mtp-4l \
  --port 8100 --tensor-parallel-size 2 \
  --data-parallel-size 1 --data-parallel-size-local 1 \
  --data-parallel-address 141.62.38.12 --data-parallel-rpc-port 13389 \
  --max-num-seqs 2 --max-model-len 8192 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 \
  --kv-transfer-config "$P_TRANSFER" --additional-config "$P_ADDITIONAL" \
  2>&1 | tee /home/c00612283/pd-diag-p.log
```

此脚本默认 eager；先用它收集算子证据。图模式会省略被捕获的 Python indexer 写钩子，不能把缺少 INDEXER_WRITE 日志直接当成图中未执行写入；MODEL_FORWARD 记录实际 runtime graph mode，MTP_PROPOSE_DONE 和 P/D 字节检查仍有效。排查图模式时复用原生 --no-enforce-eager/--compilation-config 参数，并单独与 eager 对照。

INDEXER_WRITE 的 dummy=true 表示原生 MTP 跳过时的占位执行，不代表请求 KV 有效。空闲 DP frame 不沿用上一请求的诊断上下文；图回放用实际 FULL runtime 的 MODEL_FORWARD_DONE 返回证据，日志审计不会要求图内重新执行逐层 Python hook。

### Proxy 终端

```bash
curl --fail --max-time 10 --noproxy '*' http://141.62.38.12:8200/health \
&& curl --fail --max-time 10 --noproxy '*' http://141.62.38.12:8100/health \
&& KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3 \
bash /home/c00612283/vllm-ascend-v0.30.0rc2/scripts/kvaf_pd_proxy.sh \
  --host 141.62.38.12 --port 8000 \
  --prefiller-hosts 141.62.38.12 --prefiller-ports 8100 \
  --decoder-hosts 141.62.38.12 --decoder-ports 8200
```

## 3. 请求与日志检查

第四个终端先发送短请求，验证只有 main tail 的场景：

```bash
curl --fail --max-time 900 --noproxy '*' \
  http://141.62.38.12:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm","messages":[{"role":"user","content":"请简短介绍中国的首都。"}],"max_tokens":16,"temperature":0,"stream":false}'
```

再发送长请求以覆盖主 KV 满块 DDR：

```bash
/usr/local/python3.12.13/bin/python3 - <<'PY' > /tmp/pd-diag-long.json
import json
prompt = "请阅读下面的材料，最后用一句中文概括：" + "北京是中国的首都，也是重要的文化中心。" * 80
print(json.dumps({"model":"glm", "messages":[{"role":"user", "content":prompt}],
                  "max_tokens":16, "temperature":0, "stream":False}))
PY
curl --fail --max-time 900 --noproxy '*' \
  http://141.62.38.12:8000/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @/tmp/pd-diag-long.json
```

以响应 usage.prompt_tokens 和 RECEIVE_SEED.tokens 为准，长请求须 >=128 且不超过 max-model-len。重复字符数量不等于实际 token 数。完成响应后检查：

```bash
grep 'KVAF_PD_DIAG' /home/c00612283/pd-diag-p.log /home/c00612283/pd-diag-d.log

/usr/local/python3.12.13/bin/python3 \
  /home/c00612283/vllm-ascend-v0.30.0rc2/scripts/check_kvaf_pd_diagnostics.py \
  /home/c00612283/pd-diag-p.log /home/c00612283/pd-diag-d.log \
  --require-ddr --require-compute
```

只有证据齐全且没有诊断 FAIL 才输出 PD_DIAGNOSTICS_PASS。缺日志、只开日志未开 verify、缺 DDR 路径、缺计算证据均返回非零；不要通过删去 FAIL 行来制造通过结果。日志跨多个 worker 交错是正常现象，按 worker/request/generation/layer/member/path/sequence 关联。每次重启保存独立日志，不混入其他版本或上次运行的记录。

## 4. 事件与原始需求对照

| 事件 | 检查内容 |
| --- | --- |
| WORKER、LAYOUT、REGISTERED、DDR_REGION | 物理设备/TP/DP、实际层数、PA/层内 offset、原生 Tensor dtype/shape/stride、源到全部 D peer 的注册、独立 region 代次 |
| ADDRESS_CHECK | 使用真实 Tensor.data_ptr、物理块表和 128×各 plane 字节宽度，独立核对 P 源及 D HBM 目的 offset；不调用被核对的 LayerLayout.offset 计算 |
| INDEXER_WRITE、INDEXER_WRITE_DONE | 实际 indexer 写函数入口/成功返回，scatter_cache 或 store_kv_block，输入与缓存 dtype；仅诊断实例装饰，不替换算子 |
| MODEL_FORWARD、MODEL_FORWARD_DONE、SFA_CALL、SFA_DONE | 实际模型 runtime graph mode、Target SFA 调用/返回、prefill replay 或 decode、缓存 dtype；FULL 图返回作为图路径证据，不冒充逐层 Python hook 执行 |
| MTP_PROPOSE_DONE、MTP_SKIP | 实际原生 MTP proposal 成功或原生长度跳过，不将占位执行当成有效请求 KV |
| RECEIVE_SEED、PULL_PLAN、SEND_PLAN | 请求代次、P/D 物理块表摘要、满块/tail、复制与分片贡献、空贡献及 4096 分批 |
| SOURCE_READY、PULL_DONE、SOURCE_RELEASED | 原 Ready 序列、raw READ DONE、CPU Pull COMMITTED_PUBLISHED、原源释放 |
| HOT_GET_SUBMITTED | 原生 Decode Get 入口已返回、设备 stream 顺序执行；不额外读取设备 miss_count，不将 enqueued 当成 DMA 完成 |
| KV_CHECK，path=npu/cpu | P 源与 D HBM/DDR 全段字节 SHA256，失败段序号；展示前两段 digest，但程序比较所有段 |
| MAIN_REPLICA_CHECK | 各 P TP 成员同一逻辑主 KV 块有效字节一致性；不一致会阻止该请求 PD 放行 |
| ABC_IMPORT、ABC_REPLAY_DDR、ABC_TO_STAGE、STAGE_TO_ABC | 接收期不预热 Hot，计算期导入 ABC、末 token 重算、保留后续续写；首次拷贝字节检查 |
| DECODE_READY | 本请求全部实际贡献完成、目录/数据门控满足；记录已核对批次数和副本证据数 |

校验通过可以证明选中运行中的这些路径和字节契约，不能仅凭它保证 GLM 全模型算子精度或证明四层删层模型具有正常语言能力。还需同一模型、tokenizer、量化和相同请求的非 PD 基线精度对照；关闭诊断后单独验证性能、并发和长稳。单机 TP2/DP2 不替代原始 TP8/DP2、TP1/DP16、8 K 的生产验收。

算子路径事件记录实际原生 Python/SDK 入口和返回边界；底层 CANN/AIV kernel 的指令级执行与耗时应使用原生 profiler。若需证明业务 Decode 的实际 Get miss，结合原有 KVAF BatchGet completed 的 miss_tokens 日志/指标；diagnostic DDR 回读通过和业务 Hot Get 提交是两类证据。

要做 127/128/129/255/256/300 token 边界、chunk、取消/恢复、MTP 长度上限、eager/图模式矩阵，将 diagnostics_requests 改为足够的数量（最多16），使用同一 tokenizer 生成实际长度并在日志核对，按用例分别保存证据。日志不会代替未执行的测试。

## 5. 关闭

停止新请求，正常退出本次 P/D；移除两侧 diagnostics_requests/verify_kv，或显式设为 0/false 后重启。默认关闭时不创建诊断对象，不安装 indexer 装饰器，不进行 SHA256、额外同步或 DDR 回读，也不产生 KVAF_PD_DIAG 日志。已有原生 KVAF 日志遵循其自身配置，不由这两个开关修改。

本轮验证：新增 15 项诊断 UT，全量 215 项 KVAF UT 通过。覆盖独立地址公式错误、HBM/DDR 字节损坏、非发送 TP 主 KV 副本差异、ABC/staging 交接、默认关闭、仅日志模式、诊断数量上限、未决回读保活及审计缺失证据。修改文件静态检查通过；全仓 CI 仍有此前基线问题。未在本云环境执行真实 NPU 诊断、生产全建链或模型精度验证。
