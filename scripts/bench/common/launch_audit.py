#!/usr/bin/env python3
"""Launch-geometry audit of one traced run: every kernel with its grid, block, registers, shared memory, launches,
time and idle before it, against the device's multiprocessor count (SMs on CUDA, CUs on HIP).

Inputs: an Nsight Systems SQLite export (nsys export --type sqlite) or a rocprofv3 kernel trace CSV
(--kernel-trace --output-format csv: *_kernel_trace.csv). The multiprocessor count comes from the argument (read it
from the device: cudaDeviceProp.multiProcessorCount / hipDeviceProp_t.multiProcessorCount); nothing is assumed.

Idle before a launch is the time its stream (queue) ran no kernel before it: rows are sorted by start (a rocprofv3
CSV is not in start order), the stream's busy end is the latest end seen so far, and the first launch of a stream has
no predecessor so it records nothing. Every gap is kept (n, p50, p95, max per kernel); --short-us T adds a separate
column with the median of the gaps below T. The header also gives the device busy time (union of kernel intervals
over all streams) and idle time over the traced span; copies are not in a kernel trace, so idle includes them.

Flags: GRID<MP (fewer blocks than multiprocessors: at most one block per unit, latency-bound unless the block is
large and the work small), ONE-BLOCK (grid 1). Rows sort by total time, so the first flagged rows are the shapes worth
fixing. With --step-kernel NAME the totals are also divided by the launches of NAME (one per step) to give ms/step.

usage: launch_audit.py TRACE --mp N [--top 40] [--step-kernel lm_head_rows_kernel] [--flagged-only] [--short-us T]
"""
import argparse
import csv
import sqlite3


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


def quantile(sorted_vals, q):
    """Linear interpolation between closest ranks (q=0.5 is the median); nan when empty."""
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def aggregate(rows, step_kernel=""):
    """Per kernel name: launches, total time, grids and the idle-before gaps (ns) of its launches. Rows are taken in
    start order; a stream's busy end is the latest end so far (an overlapping launch leaves no idle), and the first
    launch of each stream has no predecessor, so it adds no gap."""
    agg, last_end = {}, {}
    steps = 0
    for s, e, stream, grid, block, regs, smem, name in sorted(rows, key=lambda r: (r[0], r[1])):
        name = kernel_name(name)
        k = agg.setdefault(name, {"n": 0, "t": 0, "grid": set(), "block": block, "regs": regs, "smem": smem, "gaps": []})
        if stream in last_end:
            k["gaps"].append(max(0, s - last_end[stream]))
            last_end[stream] = max(last_end[stream], e)
        else:
            last_end[stream] = e
        k["n"] += 1
        k["t"] += e - s
        k["grid"].add(grid)
        if step_kernel and name.startswith(step_kernel):
            steps += 1
    return agg, steps


def device_busy(rows):
    """(span, busy) in ns: span from the first start to the last end, busy = union of all kernel intervals."""
    span = busy = 0
    cur_s = cur_e = None
    for s, e, *_ in sorted(rows, key=lambda r: (r[0], r[1])):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
        span = max(r[1] for r in rows) - min(r[0] for r in rows)
    return span, busy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--mp", type=int, required=True, help="multiprocessor count of the traced device")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--step-kernel", default="")
    ap.add_argument("--flagged-only", action="store_true")
    ap.add_argument("--short-us", type=float, default=0.0,
                    help="also print the median of the idle-before gaps below this many microseconds, as its own column")
    a = ap.parse_args()
    rows = list(nsys_rows(a.trace) if a.trace.endswith((".sqlite", ".db")) else rocprof_rows(a.trace))
    agg, steps = aggregate(rows, a.step_kernel)
    span, busy = device_busy(rows)
    short_ns = a.short_us * 1e3
    per = f" | ms/step (/{steps})" if steps else ""
    short = f" | idle<{a.short_us:g}us p50 (n)" if short_ns > 0 else ""
    print(f"kernels {len(rows)} names {len(agg)} multiprocessors {a.mp}")
    print(f"span ms {span / 1e6:.3f} | device busy ms {busy / 1e6:.3f} | device idle ms {(span - busy) / 1e6:.3f} "
          "(union of kernel intervals over all streams; copies are not in a kernel trace, so idle includes them)")
    print("kernel | launches | total ms | mean us | blocks min..max | threads | regs | smem B | "
          "idle-before n | p50 us | p95 us | max us" + short + " | flag" + per)
    shown = 0
    for name, k in sorted(agg.items(), key=lambda kv: -kv[1]["t"]):
        gmin, gmax = min(k["grid"]), max(k["grid"])
        flag = "ONE-BLOCK" if gmax == 1 else ("GRID<MP" if gmax < a.mp else "")
        if a.flagged_only and not flag:
            continue
        g = sorted(k["gaps"])
        idle = f"{len(g)} | {quantile(g, 0.5) / 1e3:.2f} | {quantile(g, 0.95) / 1e3:.2f} | {(g[-1] if g else float('nan')) / 1e3:.2f}"
        if short_ns > 0:
            low = [x for x in g if x < short_ns]
            idle += f" | {quantile(low, 0.5) / 1e3:.2f} ({len(low)})"
        tail = f" | {k['t'] / 1e6 / steps:.3f}" if steps else ""
        print(f"{name[:96]} | {k['n']} | {k['t'] / 1e6:.2f} | {k['t'] / k['n'] / 1e3:.1f} | {gmin}..{gmax} | {k['block']} | "
              f"{k['regs']} | {k['smem']} | {idle} | {flag}{tail}")
        shown += 1
        if shown >= a.top:
            break


if __name__ == "__main__":
    main()
