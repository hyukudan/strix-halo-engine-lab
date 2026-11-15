#!/usr/bin/env python3
"""Window-loop census of one prefill run from a rocprofv3 kernel trace (CSV).

Answers, before any scheduling change to a per-window launch loop (attention windows, index/select/flash/combine
sets): how long the loop spans, how much of it the GPU is idle, which kinds take the time, how the launches spread
over hardware queues, and how many compute units the CU-bound kernel (one CTA per CU) leaves free.

Usage:
  window_census.py TRACE.csv [--run N] [--split KERNEL] [--kinds a,b,...] [--fill KERNEL] [--cus N] [--edge-us U]
Runs are split at launches of --split (default embedding_kernel; --run -1 = the last one, the warm run of a
--repeat process). A cluster is a run of consecutive window-kind launches (by start); any other kernel ends it.
The fill sweep counts the CTAs of running --fill kernels capped at --cus: an over-count of resident CTAs, so the
reported deficit is a LOWER bound of the CU time that kernel leaves free inside the clusters.
"""
import argparse
import collections
import csv
import sys

DS41_KINDS = ("index_qprep", "index_scores", "select", "flash_rows", "attention_flash_combine", "attention_chunk",
              "attention_combine", "attention_f16wmma", "__amd_rocclr_fillBuffer")


def load(path):
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            n = r["Kernel_Name"].replace("(anonymous namespace)::", "")
            p = n.find("(")
            if p > 0:
                n = n[:p]
            wg = max(1, int(r["Workgroup_Size_X"]) * int(r["Workgroup_Size_Y"]) * int(r["Workgroup_Size_Z"]))
            grid = int(r["Grid_Size_X"]) * int(r["Grid_Size_Y"]) * int(r["Grid_Size_Z"])
            out.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), n[:60], int(r["Queue_Id"]), grid // wg,
                        int(r["LDS_Block_Size"])))
    out.sort()
    return out


def union(iv):
    total, s0, e = 0, None, -1
    for s, en in sorted(iv):
        if s > e:
            if s0 is not None:
                total += e - s0
            s0, e = s, en
        else:
            e = max(e, en)
    return total + (e - s0 if s0 is not None else 0)


def census(k, kinds, fill, cus, edge_ns, cu_lds=65536):
    def win(n):
        return any(n.startswith(w) or n.find("::" + w) >= 0 for w in kinds)

    clusters, cur = [], []
    for x in k:
        if win(x[2]):
            cur.append(x)
        elif cur:
            clusters.append(cur)
            cur = []
    if cur:
        clusters.append(cur)
    clusters = [c for c in clusters if len(c) >= 8]
    res = {"wall_ms": (k[-1][1] - k[0][0]) / 1e6, "launches": len(k), "clusters": len(clusters),
           "loop_launches": sum(len(c) for c in clusters)}
    res["span_ms"] = sum(max(x[1] for x in c) - c[0][0] for c in clusters) / 1e6
    res["busy_ms"] = sum(union([(x[0], x[1]) for x in c]) for c in clusters) / 1e6
    res["idle_ms"] = res["span_ms"] - res["busy_ms"]
    per = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    queues = collections.Counter()
    for c in clusters:
        for x in c:
            p = per[x[2]]
            p[0] += 1
            p[1] += x[1] - x[0]
            p[2][x[4]] += 1
            queues[x[3]] += 1
    res["kinds"] = sorted(((n, p[0], p[1] / 1e6, p[2].most_common(2)) for n, p in per.items()), key=lambda t: -t[2])
    res["queues"] = dict(queues)
    under = edge = inner = fill_busy = shared = 0.0
    for c in clusters:
        fk = [x for x in c if x[2].find(fill) >= 0]
        # CTAs that cannot share a CU with a fill CTA (LDS above what the fill CTA leaves) occupy CUs too
        fill_lds = max((x[5] for x in fk), default=0)
        ok = [x for x in c if x[2].find(fill) >= 0 or x[5] > cu_lds - fill_lds]
        ev2 = sorted([(x[0], x[4]) for x in ok] + [(x[1], -x[4]) for x in ok])
        level, t, e0 = 0, c[0][0], max(x[1] for x in c)
        for ts, dv in ev2 + [(e0, 0)]:
            if ts > t and level < cus:
                shared += (ts - t) * (cus - level) / float(cus)
            t = max(t, ts)
            level += dv
        fill_busy += union([(x[0], x[1]) for x in fk])
        ev = sorted([(x[0], x[4]) for x in fk] + [(x[1], -x[4]) for x in fk])
        s0, e0 = c[0][0], max(x[1] for x in c)
        level, t = 0, s0
        for ts, dv in ev + [(e0, 0)]:
            if ts > t and level < cus:
                w = (cus - level) / float(cus)
                under += ts - t
                for a0, a1, is_edge in ((t, min(ts, s0 + edge_ns), True), (max(t, e0 - edge_ns), ts, True),
                                        (max(t, s0 + edge_ns), min(ts, e0 - edge_ns), False)):
                    if a1 > a0:
                        if is_edge:
                            edge += (a1 - a0) * w
                        else:
                            inner += (a1 - a0) * w
            t = max(t, ts)
            level += dv
    res.update(fill_busy_ms=fill_busy / 1e6, under_ms=under / 1e6, deficit_edge_ms=edge / 1e6,
               deficit_inner_ms=inner / 1e6, deficit_ms=(edge + inner) / 1e6, free_ms=shared / 1e6)
    return res


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace")
    ap.add_argument("--run", type=int, default=-1)
    ap.add_argument("--split", default="embedding_kernel")
    ap.add_argument("--kinds", default=",".join(DS41_KINDS))
    ap.add_argument("--fill", default="flash_rows")
    ap.add_argument("--cus", type=int, default=40, help="compute units of the device (from its declaration)")
    ap.add_argument("--cu-lds", type=int, default=65536, help="LDS bytes per compute unit (from the device declaration)")
    ap.add_argument("--edge-us", type=int, default=300)
    a = ap.parse_args(argv)
    k = load(a.trace)
    cut = [i for i, x in enumerate(k) if x[2].find(a.split) >= 0] or [0]
    runs = [(cut[j], cut[j + 1] if j + 1 < len(cut) else len(k)) for j in range(len(cut))]
    lo, hi = runs[a.run]
    r = census(k[lo:hi], tuple(a.kinds.split(",")), a.fill, a.cus, a.edge_us * 1000, a.cu_lds)
    print("run %d of %d: wall %.1f ms, launches %d; window clusters %d, launches %d, span %.1f ms, busy-union %.1f, idle %.1f"
          % (a.run % len(runs), len(runs), r["wall_ms"], r["launches"], r["clusters"], r["loop_launches"], r["span_ms"],
             r["busy_ms"], r["idle_ms"]))
    print("%-58s %7s %9s %8s %s" % ("kind", "n", "incl ms", "us/each", "CTAs (top)"))
    for n, cnt, ms, ctas in r["kinds"][:14]:
        print("%-58s %7d %9.1f %8.1f %s" % (n, cnt, ms, ms * 1e3 / cnt, ctas))
    print("queues in window clusters:", r["queues"])
    print("%s busy-union %.1f ms; CTAs in flight < %d: %.1f ms; CU-idle equivalent >= %.1f ms (edges %d us: %.1f, interior %.1f)"
          % (a.fill, r["fill_busy_ms"], a.cus, r["under_ms"], r["deficit_ms"], a.edge_us, r["deficit_edge_ms"],
             r["deficit_inner_ms"]))
    print("CU-idle equivalent with the CTAs that cannot share a CU with %s counted (LDS rule, %d B/CU): >= %.1f ms"
          % (a.fill, a.cu_lds, r["free_ms"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
