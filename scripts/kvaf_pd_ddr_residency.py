# SPDX-License-Identifier: Apache-2.0
"""Opt-in, model-free PD DDR initialization diagnostic; never force-kills workers."""

import argparse
import contextlib
import importlib.util
import json
import multiprocessing as mp
import signal
import time
import traceback
from pathlib import Path

GIB = 1024**3
SLOT_ALIGNMENT = 4096
TOKEN_BYTES = 656
BLOCK_TOKENS = 128


def emit(event, **fields):
    print("KVAF_DDR_DIAG " + json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def create_runtime(args, device):
    # Import in spawned workers only; no vLLM/model/platform initialization.
    import kvaf
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    return kvaf.NpuRuntime(
        batch_get_kernel_object_path=str(args.build_dir / "bin/kvaf_execute_batch_get_kernels.o"),
        batch_put_kernel_object_path=str(args.build_dir / "bin/kvaf_execute_batch_put_kernels.o"),
        batch_get_raw_addr_kernel_object_path=str(args.build_dir / "bin/kvaf_execute_batch_get_raw_addr_kernels.o"),
        batch_get_token_bytes=TOKEN_BYTES,
        batch_get_tokens_per_block=BLOCK_TOKENS,
        device_id=device,
        npu_core_count=args.npu_core_count,
        jettys_per_core=args.jettys_per_core,
        enable_cpu_offload=True,
    )


def admission(args):
    if not args.initialization_coordinator:
        return contextlib.nullcontext()
    path = Path(__file__).resolve().parents[1] / "vllm_ascend/distributed/kv_transfer/kvaf/pd/initialization.py"
    spec = importlib.util.spec_from_file_location("kvaf_pd_init_admission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.initialization_slot(args.initialization_coordinator, args.cpu, timeout=args.init_timeout)


def worker(args, device, pipe):
    # Parent handles Ctrl+C and requests orderly release, including pending init.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    runtime = None
    stage = "create_runtime"
    start = time.monotonic()
    try:
        runtime = create_runtime(args, device)
        stage = "admission"
        with admission(args):
            stage = "link_to"
            runtime.link_to(args.cpu)
            stage = "initialize"
            runtime.initialize()
            stage = "initialize_ddr"
            runtime.initialize_ddr(args.ddr_bytes, args.slot_size)
        ticket = runtime.offload_region()
        pipe.send({"event": "READY", "device": device, "ticket": ticket, "elapsed": round(time.monotonic() - start, 3)})
        # Keep the Runtime, region and all transport resources alive until release.
        stage = "wait_release"
        if pipe.recv() != "RELEASE":
            raise RuntimeError("unexpected coordinator command")
    except BaseException:
        traceback.print_exc()
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            pipe.send({"event": "FAIL", "device": device, "stage": stage})
    finally:
        try:
            if runtime is not None:
                runtime.shutdown()
        except BaseException:
            traceback.print_exc()
            emit(
                "RETAINED",
                device=device,
                pid=mp.current_process().pid,
                reason="shutdown failed; quiesce transport before manual termination",
            )
            # Do not release unresolved DMA resources through process teardown.
            while True:
                time.sleep(30)
        else:
            pipe.close()
            emit("SHUTDOWN_PASS", device=device)


def validate_tickets(records, ddr_bytes, endpoint):
    identities = set()
    ranges = []
    for record in records:
        ticket = record["ticket"]
        identity = (ticket["epoch"], ticket["region_id"])
        if ticket["address"] != endpoint or ticket["bytes"] != ddr_bytes:
            raise ValueError("DDR descriptor does not match requested endpoint/size")
        if identity in identities:
            raise ValueError("duplicate live DDR region identity")
        identities.add(identity)
        ranges.append((ticket["base"], ticket["base"] + ticket["bytes"]))
    for left, right in zip(sorted(ranges), sorted(ranges)[1:]):
        if left[1] > right[0]:
            raise ValueError("live DDR ranges overlap")


def run(args, context=None):
    context = context or mp.get_context("spawn")
    workers = []
    records = []
    passed = False

    def wait_ready(device, pipe):
        if not pipe.poll(args.init_timeout):
            raise TimeoutError(f"device {device}: initialization deadline exceeded")
        record = pipe.recv()
        emit(record["event"], **{key: value for key, value in record.items() if key != "event"})
        if record["event"] != "READY":
            raise RuntimeError(f"device {device}: initialization failed")
        records.append(record)
        if not all(process.is_alive() for _, process, _ in workers):
            raise RuntimeError("a previously started worker exited")

    try:
        for device in args.devices:
            parent, child = context.Pipe()
            process = context.Process(target=worker, args=(args, device, child))
            process.start()
            child.close()
            workers.append((device, process, parent))
            emit("START", device=device, pid=process.pid)
            if not args.parallel:
                wait_ready(device, parent)
        if args.parallel:
            for device, _, pipe in workers:
                wait_ready(device, pipe)
        validate_tickets(records, args.ddr_bytes, args.cpu)
        emit(
            "ALL_RESIDENT_PASS",
            count=len(records),
            total_bytes=args.ddr_bytes * len(records),
            note="initialization/residency only; not a KV accuracy test",
        )
        passed = True
    except (Exception, KeyboardInterrupt) as error:
        emit("RESIDENCY_FAIL", reason=str(error))
    finally:
        # Release only after every requested worker is ready, or on any failure.
        for _, _, pipe in workers:
            with contextlib.suppress(BrokenPipeError, EOFError, OSError):
                pipe.send("RELEASE")
        deadline = time.monotonic() + args.cleanup_timeout
        for _, process, _ in workers:
            process.join(max(0, deadline - time.monotonic()))
        pending = [process.pid for _, process, _ in workers if process.is_alive()]
        if pending:
            passed = False
            emit(
                "CLEANUP_PENDING",
                pids=pending,
                note="no force-kill; inspect K/worker logs; this coordinator remains waiting",
            )
            for _, process, _ in workers:
                process.join()
        for device, process, pipe in workers:
            if process.exitcode != 0:
                passed = False
                emit("WORKER_EXIT_FAIL", device=device, exitcode=process.exitcode)
            pipe.close()
    mode = "PARALLEL" if args.parallel else "SERIAL"
    emit(f"{mode}_RESIDENCY_PASS" if passed else f"{mode}_RESIDENCY_FAIL")
    return 0 if passed else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", required=True, help="K host:port")
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--devices", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--ddr-size-gb", type=int, default=16, help="GiB per worker")
    parser.add_argument("--parallel", action="store_true", help="start all workers before waiting for READY")
    parser.add_argument("--initialization-coordinator", default="", help="opt-in shared admission host:port")
    parser.add_argument("--npu-core-count", type=int, default=2)
    parser.add_argument("--jettys-per-core", type=int, default=2)
    parser.add_argument(
        "--init-timeout", type=int, default=180, help="parent/admission wait; does not change SDK timeouts"
    )
    parser.add_argument("--cleanup-timeout", type=int, default=180)
    args = parser.parse_args()
    if len(set(args.devices)) != len(args.devices) or any(device < 0 for device in args.devices):
        parser.error("devices must be distinct nonnegative SDK/torch device IDs")
    if min(args.ddr_size_gb, args.init_timeout, args.cleanup_timeout) <= 0:
        parser.error("size and deadlines must be positive integers")
    if not 1 <= args.npu_core_count <= 48 or not 1 <= args.jettys_per_core <= 64:
        parser.error("require cores in 1..48 and jettys per core in 1..64")
    args.build_dir = args.build_dir.expanduser().resolve()
    for kind in ("get", "put", "get_raw_addr"):
        path = args.build_dir / "bin" / f"kvaf_execute_batch_{kind}_kernels.o"
        if not path.is_file():
            parser.error(f"missing kernel: {path}")
    args.slot_size = ((TOKEN_BYTES * BLOCK_TOKENS + SLOT_ALIGNMENT - 1) // SLOT_ALIGNMENT) * SLOT_ALIGNMENT
    args.ddr_bytes = (args.ddr_size_gb * GIB // args.slot_size) * args.slot_size
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
