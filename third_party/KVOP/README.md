# KVOP: original LIDU and slot LRU

Build in the serving container after sourcing its CANN environment:

```bash
bash third_party/KVOP/build.sh
source third_party/KVOP/env.sh
```

This builds the standalone original LIDU package and installs the KVOP Python
package. It does not build or change KVAF. The server launch script sources the
vendor path only when `kvaf_enabled=true`; the disabled baseline needs no KVOP.

## Serving integration

See [the service guide](../../scripts/README.md) for configuration, grouped KV
layout, hot/ABC capacity, prefix reuse, DDR admission and validation.
The hardware feedback check uses the same hot-cache policy as serving; only
zero-task transport calls are stubbed. It does not retain a second HBM mode.

## Source and limits

See [original/SOURCE.md](original/SOURCE.md) for the exact source and changes.
The original Numba kernels are in `kvop/lru_core.py`; their algorithms are
unchanged. The native source includes the already-restored large-capacity and
stream-launch fixes. This integration adjusts query-layout validation for
single-query decode, empty graph padding rows, and unused trailing queries.
Native resident-map capacity is 133632, including the 2048 spare candidates.
It covers the current 131072-token model limit plus MTP/block alignment.

## Validation

```bash
PYTHONPATH=third_party/KVOP python tests/ut/test_kvaf_feedback.py
python third_party/KVOP/tests/lidu_feedback_npu.py
```

The last command requires the compiled native operator and uses logical NPU 0.
It checks eager and graph replay, a mixed 1/4-query batch, padding, zero native
misses, unchanged maps, and actual feedback-driven LRU updates. Four consecutive
replays use different TopK values to detect cross-round feedback overwrite. It requires
neither model weights nor a KVAF CPU service.

## Decode profiling and logs

`start_vllm_server.sh` configures on-demand CPU/NPU profiling by default;
`profiler_dir` at the top controls the trace directory. Capture starts only
when the benchmark sends `/start_profile`, and stops through `/stop_profile`.
Stack and memory profiling are disabled to reduce capture overhead.

```bash
python scripts/benchmark.py --levels 2 --profile-decode --profile-decode-batch 2 --profile-decode-tokens 32
```

The profile wave is separate from the reported performance wave. Its capture
window follows client-observed decode arrivals, not exact device iteration
boundaries. Compare LIDU device time with these CPU scopes:

- `kvaf: request_prepare_and_feedback_wait`: request maps and previous feedback wait.
- `kvaf: cache_input_upload`: LRU candidates and native input uploads.
- `kvaf: feedback_d2h_submit`: enqueue per-layer feedback copies.
- `kvaf: batch_put`: stable-block publication, including synchronous Put work.

KVAF native stderr is redirected throughout initialization, inference and
shutdown to the per-worker `kvaf_DP*_TP*_device*_pid*.log` path printed at
connection. vLLM INFO/error messages stay on the console. Neither KVAF nor the native LIDU kernel changes in this integration; no rebuild
is required if the previous version is already installed.

### Per-card hot miss log

The [service guide](../../scripts/README.md#验证与日志) defines the counters and
sampling window. They reuse existing D2H feedback; no additional device
operators or synchronizations are introduced for these statistics.

## Python service modules

The KVAF integration has nine modules under
`vllm_ascend/distributed/kv_transfer/kvaf` (plus `__init__.py`):

| Module | Responsibility |
| --- | --- |
| `config.py` | User configuration and validation |
| `adapter.py` | SDK/runtime lifecycle, CPU routing and native operator binding |
| `layout.py` | Static cache geometry and shared-index layer groups |
| `state.py` | Request progress, DDR ownership and device slot maps |
| `transfer.py` | Put, Get, address publication and byte verification |
| `cache_policy.py` | Prefill/decode residency, ABC tails and asynchronous LRU feedback |
| `prefix_cache.py` | Prefix catalog, admission/eviction, snapshots and connector protocol |
| `metrics.py` | Per-card cache miss summaries |
| `native_log.py` | Native SDK log redirection |

Feedback remains part of cache policy. Its existing stream, events, double
buffers and background executor preserve their execution order. Scheduler
catalog and worker snapshot classes share a module, not mutable process state.
Native LIDU and Numba LRU algorithms remain in KVOP.
