#!/usr/bin/env python3
"""Relative reconstruction error of the candidate formats on real tensors (no imatrix), plus a per-tensor Lloyd
codebook of 16 int8 entries fitted on the block-normalized weights (a 'learned nl4')."""
import sys
import numpy as np
import gguf
sys.path.insert(0, __import__("os").path.dirname(__file__))
import requant as R

M = sys.argv[1]
names = sys.argv[2:]
rd = gguf.GGUFReader(M)
T = {t.name: t for t in rd.tensors}


def lloyd16(x, iters=12):
    # block-normalized values in [-1, 1] (by the block's max magnitude), 16-level Lloyd-Max on a sample
    xb = x.reshape(-1, 32)
    am = np.abs(xb).max(axis=1, keepdims=True)
    v = (xb / np.where(am > 0, am, 1)).ravel()
    rng = np.random.default_rng(0)
    v = rng.choice(v, size=min(v.size, 2_000_000), replace=False)
    c = np.quantile(v, (np.arange(16) + 0.5) / 16)
    for _ in range(iters):
        k = np.abs(v[:, None] - c[None, :]).argmin(axis=1)
        for j in range(16):
            sel = v[k == j]
            if sel.size:
                c[j] = sel.mean()
    c = np.clip(np.round(c / np.abs(c).max() * 127), -127, 127).astype(np.int8)
    c = np.unique(c)
    while c.size < 16:
        c = np.append(c, c[-1])
    return np.sort(c)


for n in names:
    t = T[n]
    cols, rows = int(t.shape[0]), int(t.shape[1])
    with open(M, "rb") as f:
        f.seek(int(t.data_offset))
        raw = f.read(int(t.n_bytes))
    x, _ = R.dequant_q8(raw, rows, cols)
    if rows > 8192:  # sample rows of big tensors
        x = x[:: rows // 8192][:8192]
    xb = x.reshape(-1, 32)
    w = np.ones_like(xb)
    out = []
    learned = lloyd16(x)
    for fmt in ["u3", "nl3", "u4", "nl4", "lm4", "u5", "u6"]:
        cb = learned if fmt == "lm4" else R.codebook(fmt)
        d, k = R.quantize_blocks(xb, w, cb, 6)
        y = d.astype(np.float32)[:, None] * cb[k].astype(np.float32)
        out.append(f"{fmt} {np.sum((xb - y) ** 2) / np.sum(xb * xb):.2e}")
    print(n, " | ".join(out), "| lm4 table", learned.tolist(), flush=True)
