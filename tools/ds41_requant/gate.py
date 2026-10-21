#!/usr/bin/env python3
"""Combines the two ranks' per-row statistics of strix-deepseek41-requant-eval --compare-ref into the gate metrics.

usage: gate.py STATS_PREFIX SEQ.u32 SEQ.segs [--json OUT] [--greedy]
  STATS_PREFIX.r0 / .r1: 48-byte rows (lse_p, lse_q, s: f64; max_p, max_q: f32; arg_p, arg_q: u32; tgt_p, tgt_q: f32)
Per row: KL(ref || cand) = w0 s0 + w1 s1 - lse_P + lse_Q (w_s = exp(lse_P_s - lse_P)); top-1 = global argmax equality;
NLL of the next token from the slice that holds it. Rows predict token row+1; the last row of a segment (which predicts
the next segment's BOS) is not scored. --greedy: segments are "name start n_prompt n_gen" (built by --greedy); a
segment's match length is the number of reference greedy tokens the candidate reproduces before its first miss.
"""
import json, sys
import numpy as np

ROW = np.dtype([("lse_p", "<f8"), ("lse_q", "<f8"), ("s", "<f8"), ("max_p", "<f4"), ("max_q", "<f4"), ("arg_p", "<u4"),
                ("arg_q", "<u4"), ("tgt_p", "<f4"), ("tgt_q", "<f4")])


def rows(prefix):
    a, b = np.fromfile(prefix + ".r0", ROW), np.fromfile(prefix + ".r1", ROW)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    lp = np.logaddexp(a["lse_p"], b["lse_p"])
    lq = np.logaddexp(a["lse_q"], b["lse_q"])
    kl = np.exp(a["lse_p"] - lp) * a["s"] + np.exp(b["lse_p"] - lp) * b["s"] - lp + lq
    kl = np.maximum(kl, 0.0)
    ap = np.where(b["max_p"] > a["max_p"], b["arg_p"], a["arg_p"])  # ties: rank 0 (lower ids), as the engine
    aq = np.where(b["max_q"] > a["max_q"], b["arg_q"], a["arg_q"])
    tp = np.where(np.isnan(a["tgt_p"]), b["tgt_p"], a["tgt_p"]).astype(np.float64)
    tq = np.where(np.isnan(a["tgt_q"]), b["tgt_q"], a["tgt_q"]).astype(np.float64)
    return kl, ap, aq, lp - tp, lq - tq


def summarize(kl, ap, aq, nllp, nllq, mask):
    k = kl[mask]
    return {"n": int(mask.sum()), "kl": float(k.mean()), "kl_p99": float(np.percentile(k, 99)), "kl_max": float(k.max()),
            "top1": float((ap[mask] == aq[mask]).mean()), "ppl_ref": float(np.exp(np.nanmean(nllp[mask]))),
            "ppl_cand": float(np.exp(np.nanmean(nllq[mask])))}


def main():
    prefix, seq_path, segs_path = sys.argv[1:4]
    out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    greedy = "--greedy" in sys.argv
    seq = np.fromfile(seq_path, "<u4")
    kl, ap, aq, nllp, nllq = rows(prefix)
    n = len(kl)
    res = {"rows": n}
    if greedy:
        matches = {}
        for line in open(segs_path):
            name, start, npr, ngen = line.split()
            start, npr, ngen = int(start), int(npr), int(ngen)
            first = start + npr - 1  # row predicting the first generated token
            m = 0
            while m < ngen and first + m < n and aq[first + m] == seq[first + m + 1]:
                m += 1
            matches[name] = m
            assert all(ap[first + j] == seq[first + j + 1] for j in range(min(ngen, n - first - 1))), "reference mismatch"
        v = np.array(list(matches.values()))
        res["greedy"] = {"per_prompt": matches, "mean_match": float(v.mean()), "full": int((v == 64).sum()), "prompts": len(v)}
        print(f"greedy: mean match {v.mean():.1f} / 64, full {int((v == 64).sum())}/{len(v)}  " +
              " ".join(f"{k}:{x}" for k, x in matches.items()))
    else:
        cats = {}
        valid = np.zeros(n, bool)
        for line in open(segs_path):
            cat, doc, start, length = line.split()
            start, length = int(start), int(length)
            m = np.zeros(n, bool)
            m[start:min(start + length - 1, n)] = True
            cats.setdefault(cat, np.zeros(n, bool))
            cats[cat] |= m
            valid |= m
        if "long" in cats:  # position buckets of the long stream
            for lo, hi in [(0, 2048), (2048, 4096), (4096, 8192)]:
                m = valid.copy()
                m[:lo] = False
                m[hi:] = False
                cats[f"long{lo // 1024}-{hi // 1024}K"] = m
            del cats["long"]
        res["all"] = summarize(kl, ap, aq, nllp, nllq, valid)
        for c, m in cats.items():
            res[c] = summarize(kl, ap, aq, nllp, nllq, m)
        for c in ["all"] + list(cats):
            r = res[c]
            print(f"{c:12s} n {r['n']:5d}  KL {r['kl']:.5f}  p99 {r['kl_p99']:.4f}  max {r['kl_max']:.3f}  top1 {100 * r['top1']:.2f}%  "
                  f"ppl {r['ppl_ref']:.3f} -> {r['ppl_cand']:.3f}")
    if out_json:
        json.dump(res, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
