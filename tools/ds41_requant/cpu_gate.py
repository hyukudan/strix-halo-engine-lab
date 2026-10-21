#!/usr/bin/env python3
"""KL(base || variant), top-1 and NLL per category from the CPU sweep's full logits (cpu/<variant>/<doc>.f32)."""
import sys, os
import numpy as np
V = 129280
segs = [l.split() for l in open("data/eval.segs")]
seq = np.fromfile("data/eval.u32", "<u4")
for v in sys.argv[1:]:
    cats = {}
    for cat, doc, start, length in segs:
        a, b = f"cpu/base/{doc}.f32", f"cpu/{v}/{doc}.f32"
        if not (os.path.exists(a) and os.path.exists(b)):
            continue
        p = np.fromfile(a, "<f4").reshape(-1, V).astype(np.float64)
        q = np.fromfile(b, "<f4").reshape(-1, V).astype(np.float64)
        n = min(len(p), len(q)) - 1  # the last row predicts past the window
        p, q = p[:n], q[:n]
        lp = p - np.logaddexp.reduce(p, axis=1, keepdims=True)
        lq = q - np.logaddexp.reduce(q, axis=1, keepdims=True)
        kl = np.sum(np.exp(lp) * (lp - lq), axis=1)
        tgt = seq[int(start) + 1:int(start) + 1 + n]
        r = cats.setdefault(cat, [[], [], [], []])
        r[0] += kl.tolist(); r[1] += (p.argmax(1) == q.argmax(1)).tolist()
        r[2] += (-lp[np.arange(n), tgt]).tolist(); r[3] += (-lq[np.arange(n), tgt]).tolist()
    allr = [sum((c[i] for c in cats.values()), []) for i in range(4)]
    out = [f"{v:16s}"]
    for name, r in [("all", allr)] + sorted(cats.items()):
        if r[0]:
            out.append(f"{name} n{len(r[0])} KL {np.mean(r[0]):.4f} p99 {np.percentile(r[0], 99):.3f} top1 {100*np.mean(r[1]):.1f} "
                       f"dNLL {np.mean(r[3]) - np.mean(r[2]):+.4f}")
    print(" | ".join(out))
