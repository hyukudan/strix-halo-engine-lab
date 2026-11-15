#!/usr/bin/env python3
"""mmflash summaries.

  summary.py timing RUNDIR        one line: TTFT per session (rank 0, first = cold, last = warm), per rank the mean busy and
                                  sclk over busy >= 50 samples, the max kfd evicted_ms, and GPU clocks per token
                                  (busy x sclk x warm TTFT / prompt tokens) of the warm session.
  summary.py table CMPDIR TIMING  one line per comparison JSON (floor verdict, failing checks with value > limit, absolute
                                  KL / top-1 / dNLL / row-KL quantiles / selection id differences), then the timing lines.
"""
import glob
import json
import os
import re
import sys


def mon(path):
    busy, sclk, ev = [], [], 0
    try:
        for line in open(path, encoding="utf-8", errors="replace"):
            b = re.search(r"busy=(\d+)", line)
            c = re.search(r"(\d+)Mhz \*", line)
            e = re.search(r"ev=(\d+)", line)
            if e:
                ev = max(ev, int(e.group(1)))
            if b and c and int(b.group(1)) >= 50:
                busy.append(int(b.group(1)))
                sclk.append(int(c.group(1)))
    except OSError:
        pass
    n = len(busy)
    return (sum(busy) / n if n else 0.0), (sum(sclk) / n if n else 0.0), ev


def timing(run):
    log = open(os.path.join(run, "r0.log"), encoding="utf-8", errors="replace").read()
    ttft = re.findall(r"TTFT ([0-9.]+) s", log)
    try:
        sess = open(os.path.join(run, "sessions"), encoding="utf-8").read().strip().split(",")
        toks = os.path.getsize(os.path.join(os.environ.get("MMF_PROMPTS", "."), sess[-1] + ".u32")) // 4
    except (OSError, IndexError):
        sess, toks = [], 0
    out = [os.path.basename(run), "ttft " + " ".join("%s=%s" % (s, t) for s, t in zip(sess, ttft)) if sess else " ".join(ttft)]
    for r in (0, 1):
        b, c, ev = mon(os.path.join(run, "mon%d.txt" % r))
        line = "r%d busy %.0f%% %.0fMHz ev %d" % (r, b, c, ev)
        if ttft and toks:
            line += " gclk/tok %.2fM" % (b / 100.0 * c * 1e6 * float(ttft[-1]) / toks / 1e6)
        out.append(line)
    try:
        out.append(open(os.path.join(run, "mem.txt"), encoding="utf-8").read().replace("\n", " ").strip())
    except OSError:
        pass
    print(" | ".join(out)[:300])


def table(cmpdir, tim):
    for p in sorted(glob.glob(os.path.join(cmpdir, "*.json"))):
        try:
            q = json.load(open(p, encoding="utf-8"))["pooled"]
        except (OSError, ValueError, KeyError) as e:
            print(os.path.basename(p), "unreadable", e)
            continue
        st, sel = q.get("stats", {}), q.get("selection", {})
        kq = st.get("kl_q", {})
        name = os.path.basename(p)[:-5]
        rule = q.get("dnll_rule") if isinstance(q.get("dnll_rule"), dict) else {}
        fails = ["%s %.4g>%.4g" % (c["name"], c.get("value", float("nan")), c.get("limit", float("nan")))
                 for c in rule.get("checks", []) if c.get("required") and not c.get("pass")]
        verdict = ("floor pass=%s" % q.get("pass")) if rule.get("checks") else "absolute"
        print("%s %s kl=%.4f top1=%.4f dnll=%+.4f q50=%.2e q90=%.3f selid=%.3f fails: %s" % (
            name, verdict, q.get("kl", 0), q.get("top1", 0), q.get("dnll", 0), kq.get("q50", 0), kq.get("q90", 0),
            sel.get("id_diff", 0) / max(1, sel.get("slots", 1)), "; ".join(fails) or "-"))
    print("timing:")
    try:
        for line in open(tim, encoding="utf-8"):
            print(" ", line.strip()[:230])
    except OSError:
        print("  none")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "timing":
        timing(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "table":
        table(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
