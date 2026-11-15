#!/usr/bin/env python3
"""Summary of client.py records: per class and arm, median decode tok/s, tokens per verify cycle, wall, GPU clock,
the B/A ratio of each adjacent pair, and the identity verdict (every request of a class must return the same ids).

usage: summary.py OUT.jsonl [SERVER_STDERR]   (stderr: the server's "tool-source" lines, CPU cost and source counters)
"""
import json, re, statistics, sys


def rate(r):
    ms = r.get("decode_ms")
    return r["tokens"] / (ms / 1000.0) if ms else r["tokens"] / r["wall_s"]


def main():
    recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    recs = [r for r in recs if r.get("arm") in ("A", "B")]
    bad = [r for r in recs if "error" in r]
    ok = [r for r in recs if "error" not in r]
    print("records %d errors %d" % (len(recs), len(bad)))
    verdict = True
    for cls in dict.fromkeys(r["cls"] for r in ok):
        rows = [r for r in ok if r["cls"] == cls]
        shas = {r["ids_sha"] for r in rows}
        same = len(shas) == 1
        verdict &= same
        line = ["%-12s ids %s" % (cls, "identical" if same else "DIFFER %s" % sorted(shas))]
        for arm in ("A", "B"):
            a = [r for r in rows if r["arm"] == arm]
            if not a: continue
            cyc = [r["tokens"] / r["mtp_cycles"] for r in a if r.get("mtp_cycles")]
            clk = [r.get("gpu_after", {}).get("sclk", "?") for r in a]
            line.append("%s n%d tok %d %.2f tok/s E %.2f wall %.2f s sclk %s" % (
                arm, len(a), a[0]["tokens"], statistics.median(rate(r) for r in a),
                statistics.median(cyc) if cyc else float("nan"), statistics.median(r["wall_s"] for r in a), ",".join(clk)))
        ratios = []
        for rnd in sorted({r["round"] for r in rows}):
            pa = [r for r in rows if r["round"] == rnd and r["arm"] == "A"]
            pb = [r for r in rows if r["round"] == rnd and r["arm"] == "B"]
            if pa and pb: ratios.append(rate(pb[0]) / rate(pa[0]))
        if ratios:
            line.append("B/A pairs %s median %+.1f %%" % (",".join("%.3f" % x for x in ratios), (statistics.median(ratios) - 1) * 100))
        print(" | ".join(line))
    if len(sys.argv) > 2:
        build, req = [], []
        for l in open(sys.argv[2], errors="replace"):
            m = re.search(r"tool-source: .*build ([0-9.]+) ms", l)
            if m: build.append(float(m.group(1)))
            if l.startswith("tool-source request"): req.append(l.strip())
        if build:
            print("tool-source build: n %d median %.3f ms max %.3f ms" % (len(build), statistics.median(build), max(build)))
        for l in req[:12]: print(l[:200])
    print("IDENTITY", "PASS" if verdict and not bad else "FAIL")


if __name__ == "__main__":
    main()
