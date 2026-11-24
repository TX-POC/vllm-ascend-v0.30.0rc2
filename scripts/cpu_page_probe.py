# SPDX-License-Identifier: Apache-2.0
"""Per-process THP control and compact CPU-service memory evidence (Linux)."""

import argparse
import ctypes
import json
import os
import re
import time
from pathlib import Path

PR_SET_THP_DISABLE = 41


def memory_snapshot(pid):
    proc = Path("/proc") / str(pid)
    totals = dict(Anonymous=0, AnonHugePages=0, Rss=0)
    for line in (proc / "smaps").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in totals:
            totals[key] += int(value.split()[0])
    nodes = {}
    for line in (proc / "numa_maps").read_text().splitlines():
        size = re.search(r"kernelpagesize_kB=(\d+)", line)
        page_kib = int(size[1]) if size else os.sysconf("SC_PAGE_SIZE") // 1024
        for node, pages in re.findall(r"\bN(\d+)=(\d+)", line):
            nodes[node] = nodes.get(node, 0) + int(pages) * page_kib
    return dict(pid=pid, **totals, numa_kib=nodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run", help="Exec a command with process-local THP policy")
    run.add_argument("--mode", choices=("base", "thp"), required=True)
    run.add_argument("command", nargs=argparse.REMAINDER)
    watch = commands.add_parser("watch", help="Keep peak RSS evidence while the independent benchmark runs")
    watch.add_argument("--pids", nargs="+", type=int, required=True)
    watch.add_argument("--seconds", type=int, default=1800)
    watch.add_argument("--output", type=Path, default=Path("cpu-pages"))
    args = parser.parse_args()
    if args.action == "run":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            parser.error("command required after --")
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(PR_SET_THP_DISABLE, int(args.mode == "base"), 0, 0, 0):
            raise OSError(ctypes.get_errno(), "PR_SET_THP_DISABLE failed")
        print(f"THP process mode={args.mode}; system policy still applies; command={command}", flush=True)
        os.execvp(command[0], command)
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {}
    for name in ("enabled", "defrag", "hpage_pmd_size"):
        path = Path("/sys/kernel/mm/transparent_hugepage") / name
        settings[name] = path.read_text().strip() if path.exists() else "unavailable"
    peak = {}
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        for pid in args.pids:
            try:
                row = memory_snapshot(pid)
            except (OSError, ValueError):
                continue
            if row["Rss"] >= peak.get(pid, {}).get("Rss", 0):
                peak[pid] = row
        lines = [f"THP system policy: {settings}", "Peak RSS snapshot per process; sample while DDR is allocated:"]
        for pid, row in sorted(peak.items()):
            ratio = 100 * row["AnonHugePages"] / max(row["Anonymous"], 1)
            nodes = {n: round(kib / 2**20, 2) for n, kib in row["numa_kib"].items()}
            lines.append(
                f"pid={pid} RSS={row['Rss'] / 2**20:.2f}GiB anon={row['Anonymous'] / 2**20:.2f}GiB "
                f"AnonHugePages={row['AnonHugePages'] / 2**20:.2f}GiB coverage={ratio:.1f}% NUMA_GiB={nodes}"
            )
        (args.output / "summary.log").write_text("\n".join(lines) + "\n")
        (args.output / "memory.json").write_text(json.dumps(dict(settings=settings, processes=peak), indent=2))
        time.sleep(10)
    print((args.output / "summary.log").read_text())


if __name__ == "__main__":
    main()
