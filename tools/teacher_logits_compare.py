#!/usr/bin/env python3
"""Compare teacher-forced logit spools (TMLLOG01, --teacher-logits-file) row by row.

usage: klcmp.py REF.spool CAND.spool [--segments FILE.json] [--skip N]
  KL(ref || cand) per row (full vocabulary, float64), top-1 agreement, KL p99/max, and the
  NLL of the actual next token under each model. --segments: JSON list of [name, first_row,
  rows] to report per segment. --skip N ignores the first N rows of every segment.
"""
import argparse, json, struct, sys
import numpy as np


def open_spool(path):
    f = open(path, "rb")
    assert f.read(8) == b"TMLLOG01", path
    vocab, rows, n = struct.unpack("<QQQ", f.read(24))
    inputs = np.frombuffer(f.read(4 * n), np.uint32)
    return f, vocab, rows, inputs


def rows_of(f, vocab, rows):
    for r in range(rows):
        inp, pos = struct.unpack("<IQ", f.read(12))
        assert pos == r
        yield np.frombuffer(f.read(4 * vocab), np.float32)


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - (m + np.log(np.exp(x - m).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("cand")
    ap.add_argument("--segments")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--json")
    a = ap.parse_args()
    fr, vr, rr, ir = open_spool(a.ref)
    fc, vc, rc, ic = open_spool(a.cand)
    assert vr == vc and rr == rc and np.array_equal(ir, ic), "spools differ in vocabulary, rows or inputs"
    kl = np.empty(rr)
    top = np.empty(rr, bool)
    nll_r = np.empty(rr)
    nll_c = np.empty(rr)
    for r, (x, y) in enumerate(zip(rows_of(fr, vr, rr), rows_of(fc, vc, rc))):
        lp, lq = log_softmax(x), log_softmax(y)
        p = np.exp(lp)
        kl[r] = float(np.sum(p * (lp - lq)))
        top[r] = int(np.argmax(x)) == int(np.argmax(y))
        nxt = int(ir[r + 1])
        nll_r[r], nll_c[r] = -lp[nxt], -lq[nxt]
    segs = json.load(open(a.segments)) if a.segments else [["all", 0, int(rr)]]
    out = {}
    for name, first, n in segs + ([["all", 0, int(rr)]] if a.segments else []):
        sl = slice(first + a.skip, first + n)
        k = kl[sl]
        out[name] = {"rows": int(k.size), "kl_mean": float(k.mean()), "kl_p99": float(np.percentile(k, 99)),
                     "kl_max": float(k.max()), "top1": float(top[sl].mean()),
                     "ppl_ref": float(np.exp(nll_r[sl].mean())), "ppl_cand": float(np.exp(nll_c[sl].mean()))}
        o = out[name]
        print(f"{name:12s} rows {o['rows']:6d}  KL {o['kl_mean']:.5f}  p99 {o['kl_p99']:.4f}  max {o['kl_max']:.3f}  "
              f"top1 {100 * o['top1']:.2f}%  ppl ref {o['ppl_ref']:.3f} cand {o['ppl_cand']:.3f}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
