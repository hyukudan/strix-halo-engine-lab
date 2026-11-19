#!/usr/bin/env python3
"""Launch-geometry audit of one traced run: every kernel with its grid, block, registers, shared memory, launches,
time and median idle before it, against the device's multiprocessor count (SMs on CUDA, CUs on HIP).

Inputs: an Nsight Systems SQLite export (nsys export --type sqlite) or a rocprofv3 kernel trace CSV
(--kernel-trace --output-format csv: *_kernel_trace.csv). The multiprocessor count comes from the argument (read it
from the device: cudaDeviceProp.multiProcessorCount / hipDeviceProp_t.multiProcessorCount); nothing is assumed.

Flags: GRID<MP (fewer blocks than multiprocessors: at most one block per unit, latency-bound unless the block is
large and the work small), ONE-BLOCK (grid 1). Rows sort by total time, so the first flagged rows are the shapes worth
fixing. With --step-kernel NAME the totals are also divided by the launches of NAME (one per step) to give ms/step.

usage: launch_audit.py TRACE --mp N [--top 40] [--step-kernel lm_head_rows_kernel] [--flagged-only]
"""
import argparse
import csv
import sqlite3
import statistics


def kernel_name(n):
    """Drop the trailing parameter list (balanced parentheses from the end) and a leading 'void ': a demangled HIP/CUDA
    name keeps '(anonymous namespace)' and template arguments, so distinct kernels stay distinct rows."""
    n = n.strip()
    if n.endswith(")"):
        depth = 0
        for i in range(len(n) - 1, -1, -1):
            depth += {")": 1, "(": -1}.get(n[i], 0)
            if depth == 0:
                n = n[:i].rstrip()
                break
    return n[5:] if n.startswith("void ") else n


def nsys_rows(path):
    c = sqlite3.connect(path)
    names = dict(c.execute("select id, value from StringIds"))
    q = ("select start, end, streamId, gridX*gridY*gridZ, blockX*blockY*blockZ, registersPerThread, "
         "staticSharedMemory+dynamicSharedMemory, coalesce(shortName, demangledName) from CUPTI_ACTIVITY_KIND_KERNEL order by start")
    for s, e, stream, grid, block, regs, smem, n in c.execute(q):
        yield s, e, stream, grid, block, regs, smem, names.get(n, str(n))


def rocprof_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        wg = int(r.get("Workgroup_Size_X", 0) or 0) * int(r.get("Workgroup_Size_Y", 1) or 1) * int(r.get("Workgroup_Size_Z", 1) or 1)
        grid_items = int(r.get("Grid_Size_X", 0) or 0) * int(r.get("Grid_Size_Y", 1) or 1) * int(r.get("Grid_Size_Z", 1) or 1)
        blocks = grid_items // wg if wg else 0  # rocprof grid sizes are in work-items
        regs = int(r.get("Arch_VGPR_Count", 0) or r.get("VGPR_Count", 0) or 0)
        smem = int(r.get("LDS_Block_Size", 0) or r.get("LDS_Size", 0) or 0)
        yield (int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r.get("Queue_Id", r.get("Stream_Id", 0)), blocks, wg, regs,
               smem, r["Kernel_Name"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--mp", type=int, required=True, help="multiprocessor count of the traced device")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--step-kernel", default="")
    ap.add_argument("--flagged-only", action="store_true")
    a = ap.parse_args()
    rows = list(nsys_rows(a.trace) if a.trace.endswith((".sqlite", ".db")) else rocprof_rows(a.trace))
    agg, last_end = {}, {}
    steps = 0
    for s, e, stream, grid, block, regs, smem, name in rows:
        name = kernel_name(name)
        gap = s - last_end.get(stream, s)
        last_end[stream] = e
        k = agg.setdefault(name, {"n": 0, "t": 0, "grid": set(), "block": block, "regs": regs, "smem": smem, "gaps": []})
        k["n"] += 1
        k["t"] += e - s
        k["grid"].add(grid)
        if 0 <= gap < 50_000:
            k["gaps"].append(gap)
        if a.step_kernel and name.startswith(a.step_kernel):
            steps += 1
    per = f" | ms/step (/{steps})" if steps else ""
    print(f"kernels {len(rows)} names {len(agg)} multiprocessors {a.mp}")
    print("kernel | launches | total ms | mean us | blocks min..max | threads | regs | smem B | idle-before med us | flag" + per)
    shown = 0
    for name, k in sorted(agg.items(), key=lambda kv: -kv[1]["t"]):
        gmin, gmax = min(k["grid"]), max(k["grid"])
        flag = "ONE-BLOCK" if gmax == 1 else ("GRID<MP" if gmax < a.mp else "")
        if a.flagged_only and not flag:
            continue
        gap = statistics.median(k["gaps"]) / 1e3 if k["gaps"] else float("nan")
        tail = f" | {k['t'] / 1e6 / steps:.3f}" if steps else ""
        print(f"{name[:96]} | {k['n']} | {k['t'] / 1e6:.2f} | {k['t'] / k['n'] / 1e3:.1f} | {gmin}..{gmax} | {k['block']} | "
              f"{k['regs']} | {k['smem']} | {gap:.2f} | {flag}{tail}")
        shown += 1
        if shown >= a.top:
            break


if __name__ == "__main__":
    main()
