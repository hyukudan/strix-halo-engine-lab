#!/usr/bin/env python3
"""Offline scorer of speculative rounds: would the drafts a rejected round dropped have been a free proposal next round?

Reads round traces written by strix::spec::RoundTraceWriter (BlockSpeculator::set_round_sink; the CUDA Qwen dense spec CLI
writes them with --spec-trace DIR). For every rejected round (accepted a < k) with a successor round it builds candidates
placed after the emitted correction c = greedy[a] and counts R = the number of leading candidate ids equal to the true
stream after c (the emitted ids are serial greedy, so the stream is the reference):
  sub   drafts[a+1:]   one-token substitution (the main candidate)
  skip  drafts[a+2:]   the draft had one token too many
  ins   drafts[a:]     the target inserted c before the draft
  ctrl  greedy[a+1:]   target ids of the rejected rows (conditioned on the wrong draft: a control, expected ~0)
f = fraction of rounds (with a successor) where the sub candidate has R >= 1 and R >= M, M = the accepted count the next
round actually reached: there a free lookup block would have matched the drafter at 0 drafter ms.
Predicted effect (PREDICTION, cost model of the trace itself: per-round draft_ms, verify V(r) = a + b (r - 1) fitted on the
trace's verify_ms): oracle = time saved on the f rounds at equal tokens; always = use the sub candidate after every
rejected round (tokens R + 1 instead of M + 1; stream shift ignored: approximate). Host time is not in the trace.

usage: recycle_score.py TRACE.srt... [--expect key=value,...]   |   recycle_score.py --selftest
"""
import struct
import sys

REC = struct.Struct("<4sQIIIIIIBxxxff")  # 48 bytes
assert REC.size == 48
VARIANTS = ("sub", "skip", "ins", "ctrl")
MIN_REJECTED = 30


def parse(data):
    if data[:4] != b"SRT1":
        raise ValueError("not a round trace (magic)")
    version, ll = struct.unpack_from("<II", data, 4)
    if version != 1:
        raise ValueError("round trace version %d" % version)
    o = 12 + ll
    label = data[12:o].decode("utf-8", "replace")
    recs = []
    while o < len(data):
        magic, ctx, anchor, k, rows, acc, cp, ext, src, vms, dms = REC.unpack_from(data, o)
        if magic != b"RREC" or rows != k + 1 or acc > k:
            raise ValueError("bad record at byte %d" % o)
        o += REC.size
        drafts = list(struct.unpack_from("<%dI" % k, data, o))
        o += 4 * k
        greedy = list(struct.unpack_from("<%dI" % rows, data, o))
        o += 4 * rows
        recs.append(dict(context=ctx, anchor=anchor, k=k, rows=rows, a=acc, copy=cp, ext=ext, source=src, verify_ms=vms,
                         draft_ms=dms, drafts=drafts, greedy=greedy, emitted=drafts[:acc] + [greedy[acc]]))
    return label, recs


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def fit_verify(recs):
    xs = [r["rows"] - 1 for r in recs]
    ys = [r["verify_ms"] for r in recs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else 0.113
    return (my - b * mx), b


def score(recs):
    """Returns a dict of counts and predictions for one stream of rounds (one run, in order)."""
    stream, ends = [], []
    for r in recs:
        stream += r["emitted"]
        ends.append(len(stream))
    va, vb = fit_verify(recs) if recs else (0.0, 0.0)
    V = lambda rows: va + vb * (rows - 1)  # noqa: E731
    out = dict(rounds=len(recs), succ=max(len(recs) - 1, 0), rejected=0, hits={v: 0 for v in VARIANTS},
               runs={v: 0 for v in VARIANTS}, ties0=0, saved_oracle=0.0, saved_always=0.0, dtok_always=0,
               T=sum(r["verify_ms"] + r["draft_ms"] for r in recs), draft=sum(r["draft_ms"] for r in recs),
               N=sum(len(r["emitted"]) for r in recs))
    for i, r in enumerate(recs[:-1]):
        a, k = r["a"], r["k"]
        if a >= k:
            continue
        out["rejected"] += 1
        after = stream[ends[i]:]
        nxt = recs[i + 1]
        M = nxt["a"]
        cands = dict(sub=r["drafts"][a + 1:], skip=r["drafts"][a + 2:], ins=r["drafts"][a:], ctrl=r["greedy"][a + 1:])
        for v, c in cands.items():
            R = lcp(c, after)
            out["runs"][v] += R
            if R >= 1 and R >= M:
                out["hits"][v] += 1
        c = cands["sub"]
        R = lcp(c, after)
        if R == 0 and M == 0:
            out["ties0"] += 1
        saved = nxt["draft_ms"] + V(nxt["rows"]) - V(len(c) + 1)
        if R >= 1 and R >= M:
            out["saved_oracle"] += saved
        if c:
            out["saved_always"] += saved
            out["dtok_always"] += R - M
    return out


def merge(a, b):
    m = dict(a)
    for key in ("rounds", "succ", "rejected", "ties0", "saved_oracle", "saved_always", "dtok_always", "T", "draft", "N"):
        m[key] = a[key] + b[key]
    m["hits"] = {v: a["hits"][v] + b["hits"][v] for v in VARIANTS}
    m["runs"] = {v: a["runs"][v] + b["runs"][v] for v in VARIANTS}
    return m


def report(name, s):
    succ, rej = max(s["succ"], 1), max(s["rejected"], 1)
    f = s["hits"]["sub"] / succ
    share = s["draft"] / s["T"] if s["T"] > 0 else 0.0
    g_or = s["T"] / (s["T"] - s["saved_oracle"]) - 1 if s["T"] > s["saved_oracle"] else 0.0
    t_al = s["T"] - s["saved_always"]
    g_al = ((s["N"] + s["dtok_always"]) / t_al) / (s["N"] / s["T"]) - 1 if t_al > 0 and s["N"] > 0 else 0.0
    if s["rejected"] < MIN_REJECTED:
        verdict = "inconclusive(rejected<%d)" % MIN_REJECTED
    elif f >= 0.03:
        verdict = "f>=3%%:check-policy-net(oracle %+.1f%%, always %+.1f%%)" % (100 * g_or, 100 * g_al)
    else:
        verdict = "NEGATIVE(f<3%)"
    print("recycle: content=%s rounds=%d rejected_rounds=%d f=%.4f f_rej=%.4f mean_run=%.2f skip_f=%.4f ins_f=%.4f ctrl_f=%.4f "
          "ties0=%d draft_share=%.3f predicted_gain_oracle=%+.2f%% predicted_gain_always=%+.2f%% verdict=%s"
          % (name, s["rounds"], s["rejected"], f, s["hits"]["sub"] / rej, s["runs"]["sub"] / rej, s["hits"]["skip"] / succ,
             s["hits"]["ins"] / succ, s["hits"]["ctrl"] / succ, s["ties0"], share, 100 * g_or, 100 * g_al, verdict))
    return dict(rounds=s["rounds"], rejected=s["rejected"], f=f, f_rej=s["hits"]["sub"] / rej, mean_run=s["runs"]["sub"] / rej)


def bins_of(recs):
    edges = [(0, 1024, "<1K"), (1024, 4096, "1-4K"), (4096, 16384, "4-16K"), (16384, 1 << 62, ">=16K")]
    out = {}
    for lo, hi, name in edges:
        idx = [i for i, r in enumerate(recs) if lo <= r["context"] < hi]
        if idx:
            out[name] = idx
    return out


def run(paths, expect=None):
    by_label, results = {}, {}
    for p in paths:
        with open(p, "rb") as fh:
            label, recs = parse(fh.read())
        s = score(recs)
        by_label[label] = merge(by_label[label], s) if label in by_label else s
        bins = bins_of(recs)
        if len(bins) > 1:  # context split: scored on contiguous runs of each bin's rounds
            for name, idx in bins.items():
                report("%s@%s" % (label, name), score([recs[i] for i in idx]))
    for label in sorted(by_label):
        results[label] = report(label, by_label[label])
    if expect:
        got = results[sorted(results)[0]]
        for kv in expect.split(","):
            key, val = kv.split("=")
            if abs(got[key] - float(val)) > 1e-6:
                print("EXPECT FAIL: %s = %s, expected %s" % (key, got[key], val))
                return 1
        print("EXPECT PASS")
    return 0


def write_trace(path, label, rounds):
    """Test helper: rounds = [(anchor, drafts, greedy, accepted)] as model rounds, the C++ writer's format."""
    b = bytearray(b"SRT1" + struct.pack("<II", 1, len(label)) + label.encode())
    for anchor, d, g, a in rounds:
        k = len(d)
        b += REC.pack(b"RREC", 64, anchor, k, k + 1, a, 0, 0, 2, 13.3 + 0.113 * k, 0.51 * k)
        b += struct.pack("<%dI" % k, *d) + struct.pack("<%dI" % (k + 1), *g)
    with open(path, "wb") as fh:
        fh.write(bytes(b))


def selftest():
    import os
    import tempfile
    s = lambda i: 100 + i  # noqa: E731
    rounds = [(s(0), [s(1), s(2), 9, s(4), s(5), s(6)], [s(1), s(2), s(3), 1, 1, 1, 1], 2),
              (s(3), [s(4), 9], [s(4), s(5), 1], 1),
              (s(5), [s(6)], [s(6), s(7)], 1),
              (s(7), [9, s(9), s(10)], [s(8), 1, 1, 1], 0),
              (s(8), [s(9), s(10), s(11), s(12)], [s(9), s(10), s(11), s(12), s(13)], 4)]
    fd, path = tempfile.mkstemp(suffix=".srt")
    os.close(fd)
    try:
        write_trace(path, "fixture", rounds)
        rc = run([path], "rounds=5,rejected=3,f=0.25,f_rej=0.3333333333333333,mean_run=1.6666666666666667")
    finally:
        os.remove(path)
    print("recycle_score selftest %s" % ("ok" if rc == 0 else "FAIL"))
    return rc


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--selftest"]:
        sys.exit(selftest())
    exp = None
    if "--expect" in args:
        i = args.index("--expect")
        exp = args[i + 1]
        del args[i:i + 2]
    if not args:
        print(__doc__)
        sys.exit(2)
    sys.exit(run(args, exp))
