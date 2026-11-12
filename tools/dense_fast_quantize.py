#!/usr/bin/env python3
"""Build a dense FAST pack: 4/5-bit K-quant copies of a checkpoint's dense Q8_0 tensors.

usage: dense_fast_quantize.py SRC.gguf PACK.gguf [--spec REGEX=FMT ...] [--default FMT]
           [--only RE] [--skip RE] [--imatrix F] [--no-refine] [--target-sha256 HEX]
           [--jobs N] [--stats F]

Every dense Q8_0 matrix of SRC that the 1..8-row dense A8 decode windows read (GDN, QSA and
shared-expert gate/up projections; input width a multiple of 256, at most 8192) is re-quantised
from its Q8_0 values and stored in PACK under its own name as a standard GGUF Q4_K (4.5 bits per
weight), Q5_K (5.5) or Q6_K (6.5625) tensor. FMT is q4_k, q5_k, q6_k or q8 (q8 = leave the tensor out, the engine
keeps its Q8_0 bytes); the first --spec whose REGEX matches the tensor name wins, then
--default (q4_k). The pack header pins SRC by SHA-256 and content identity; the engine loads
it with --dense-fast-pack (see docs/engine/kernels/dense-fast-pack.md).

Scale rule: per 32-weight block a weighted least-squares (scale, min) search over 21
rounding grids (the K-quant reference rule; weights sqrt(mean x^2) + |x|, times the column
importance when --imatrix is given), then the 6-bit scale/min codes of each 256-weight
super-block are refined one step either way against the same weighted error (--no-refine
keeps the plain rounding). The pack changes the model's numerics: it is a FAST artifact and
needs its own quality gate.
"""
import argparse, hashlib, json, mmap, os, re, struct, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q6p_quantize import gguf_open, nbytes, write_pack, load_imatrix, dequant_q8, Q8_0  # noqa: E402

Q4_K, Q5_K, Q6_K = 12, 13, 14
# Tensors with a typed dense A8 window form in the engine (the pack loader refuses any other).
WINDOW = re.compile(r"\.(attn_qkv|attn_gate|ssm_out|attn_q|attn_k|attn_v|attn_output|ffn_gate_shexp|ffn_up_shexp)\.weight$")
FORMATS = {"q4_k": (Q4_K, 15, 144), "q5_k": (Q5_K, 31, 176), "q6_k": (Q6_K, 31, 210)}
CHUNK_BLOCKS = 1 << 15


def f16(x):
    return x.astype(np.float16).astype(np.float32)


def fit_blocks(x, w, nmax, nstep=20, rmin=-1.0, rdelta=0.1):
    """x, w: (B, 32). Returns (scale, min magnitude) per block for y = scale * q - min, q in 0..nmax."""
    mn = np.minimum(x.min(1), 0.0)
    rng = x.max(1) - mn
    flat = rng <= 0
    rng = np.where(flat, 1.0, rng).astype(np.float32)
    xs = x - mn[:, None]
    sum_w, sum_x = w.sum(1), (w * x).sum(1)
    best = np.full(x.shape[0], np.inf, np.float32)
    scale = np.zeros(x.shape[0], np.float32)
    minv = np.zeros(x.shape[0], np.float32)
    for step in range(-1, nstep + 1):
        iscale = (nmax if step < 0 else rmin + rdelta * step + nmax) / rng
        q = np.clip(np.rint(iscale[:, None] * xs), 0, nmax)
        wq = w * q
        sum_l, sum_l2, sum_xl = wq.sum(1), (wq * q).sum(1), (wq * x).sum(1)
        det = sum_w * sum_l2 - sum_l * sum_l
        ok = det > 0
        safe = np.where(ok, det, 1.0)
        s = (sum_w * sum_xl - sum_x * sum_l) / safe
        m = (sum_l2 * sum_x - sum_l * sum_xl) / safe
        pos = m > 0
        s = np.where(pos, sum_xl / np.where(sum_l2 > 0, sum_l2, 1.0), s)
        m = np.where(pos, 0.0, m)
        err = (w * (s[:, None] * q + m[:, None] - x) ** 2).sum(1)
        take = ok & (err < best)
        best = np.where(take, err, best)
        scale = np.where(take, s, scale).astype(np.float32)
        minv = np.where(take, m, minv).astype(np.float32)
    scale[flat] = 0.0
    minv[flat] = mn[flat]
    return np.maximum(scale, 0.0), -minv


def block_error(x, w, dd, dm, nmax):
    """Weighted error and codes of blocks (..., 32) at effective scale dd and min magnitude dm."""
    q = np.where(dd[..., None] > 0, np.rint((x + dm[..., None]) / np.where(dd > 0, dd, 1.0)[..., None]), 0.0)
    q = np.clip(q, 0, nmax)
    return (w * (dd[..., None] * q - dm[..., None] - x) ** 2).sum(-1), q


def quantize_super(x, w, nmax, refine=True):
    """x, w: (S, 8, 32) float32. Returns d, dmin (S,), ls, lm (S, 8) uint8, q (S, 8, 32) uint8."""
    S = x.shape[0]
    scale, minv = fit_blocks(x.reshape(-1, 32), w.reshape(-1, 32), nmax)
    scale, minv = scale.reshape(S, 8), minv.reshape(S, 8)
    max_s, max_m = scale.max(1), minv.max(1)
    d = f16(max_s / 63.0)
    dmin = f16(max_m / 63.0)
    ls = np.where(max_s[:, None] > 0, np.clip(np.rint(scale * (63.0 / np.where(max_s > 0, max_s, 1.0))[:, None]), 0, 63), 0)
    lm = np.where(max_m[:, None] > 0, np.clip(np.rint(minv * (63.0 / np.where(max_m > 0, max_m, 1.0))[:, None]), 0, 63), 0)
    err, q = block_error(x, w, d[:, None] * ls, dmin[:, None] * lm, nmax)
    if refine:
        best_ls, best_lm = ls.copy(), lm.copy()
        for ds in (-1, 0, 1):
            for dl in (-1, 0, 1):
                if ds == 0 and dl == 0:
                    continue
                cs, cm = np.clip(ls + ds, 0, 63), np.clip(lm + dl, 0, 63)
                e, cq = block_error(x, w, d[:, None] * cs, dmin[:, None] * cm, nmax)
                take = e < err
                err = np.where(take, e, err)
                q = np.where(take[..., None], cq, q)
                best_ls, best_lm = np.where(take, cs, best_ls), np.where(take, cm, best_lm)
        ls, lm = best_ls, best_lm
    return d, dmin, ls.astype(np.uint8), lm.astype(np.uint8), q.astype(np.uint8)


def pack_super(d, dmin, ls, lm, q, fmt):
    """GGUF block_q4_K (144 bytes) / block_q5_K (176 bytes) per 256 weights."""
    S = d.shape[0]
    size = FORMATS[fmt][2]
    out = np.zeros((S, size), np.uint8)
    out[:, 0:2] = d.astype("<f2").view(np.uint8).reshape(S, 2)
    out[:, 2:4] = dmin.astype("<f2").view(np.uint8).reshape(S, 2)
    out[:, 4:8] = (ls[:, :4] & 63) | ((ls[:, 4:] >> 4) << 6)
    out[:, 8:12] = (lm[:, :4] & 63) | ((lm[:, 4:] >> 4) << 6)
    out[:, 12:16] = (ls[:, 4:] & 15) | ((lm[:, 4:] & 15) << 4)
    pairs = q.reshape(S, 4, 2, 32)
    lo, hi = pairs[:, :, 0, :], pairs[:, :, 1, :]
    if fmt == "q5_k":
        qh = np.zeros((S, 32), np.uint8)
        for p in range(4):
            qh |= ((lo[:, p, :] >> 4) << (2 * p)) | ((hi[:, p, :] >> 4) << (2 * p + 1))
        out[:, 16:48] = qh
        out[:, 48:176] = ((lo & 15) | ((hi & 15) << 4)).reshape(S, 128)
    else:
        out[:, 16:144] = (lo | (hi << 4)).reshape(S, 128)
    return out


def unpack_super(raw, fmt):
    """Reference decode of block_q4_K / block_q5_K bytes: (S, 256) float32, y = d * sc * q - dmin * m."""
    size = FORMATS[fmt][2]
    b = np.frombuffer(raw, np.uint8).reshape(-1, size)
    S = b.shape[0]
    d = b[:, 0:2].copy().view("<f2").astype(np.float32).reshape(S)
    dmin = b[:, 2:4].copy().view("<f2").astype(np.float32).reshape(S)
    s = b[:, 4:16]
    sc = np.empty((S, 8), np.float32)
    mm = np.empty((S, 8), np.float32)
    for j in range(8):
        if j < 4:
            sc[:, j], mm[:, j] = s[:, j] & 63, s[:, j + 4] & 63
        else:
            sc[:, j] = (s[:, j + 4] & 15) | ((s[:, j - 4] >> 6) << 4)
            mm[:, j] = (s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)
    if fmt == "q5_k":
        qh, ql = b[:, 16:48], b[:, 48:176].reshape(S, 4, 32)
    else:
        qh, ql = None, b[:, 16:144].reshape(S, 4, 32)
    q = np.empty((S, 8, 32), np.float32)
    for p in range(4):
        lo, hi = ql[:, p, :] & 15, ql[:, p, :] >> 4
        if qh is not None:
            lo = lo + (((qh >> (2 * p)) & 1) << 4)
            hi = hi + (((qh >> (2 * p + 1)) & 1) << 4)
        q[:, 2 * p, :], q[:, 2 * p + 1, :] = lo, hi
    return ((d[:, None] * sc)[..., None] * q - (dmin[:, None] * mm)[..., None]).reshape(S, 256)


def quantize_super_q6(x, w):
    """Q6_K: x, w (S, 16, 16). Symmetric y = d * sc * q, q in [-32, 31], sc int8 per 16 weights.
    Returns block_q6_K bytes (S, 210): ql[128], qh[64], scales[16], d."""
    S = x.shape[0]
    amax = np.abs(x).max(-1)
    safe = np.where(amax > 0, amax, 1.0)
    best = np.full(amax.shape, np.inf, np.float32)
    scale = np.zeros(amax.shape, np.float32)
    for step in range(-9, 10):
        iscale = -(32.0 + 0.1 * step) / safe
        q = np.clip(np.rint(iscale[..., None] * x), -32, 31)
        wq = w * q
        den = (wq * q).sum(-1)
        s = np.where(den > 0, (wq * x).sum(-1) / np.where(den > 0, den, 1.0), 0.0)
        err = (w * (s[..., None] * q - x) ** 2).sum(-1)
        take = err < best
        best = np.where(take, err, best)
        scale = np.where(take, s, scale).astype(np.float32)
    # Codes carry the sign of the scale: keep the reference convention (negative stored scales).
    mag = np.abs(scale).max(1)
    pick = np.take_along_axis(scale, np.abs(scale).argmax(1)[:, None], 1)[:, 0]
    iscale = np.where(mag > 0, -128.0 / np.where(pick != 0, pick, 1.0), 0.0)
    sc = np.clip(np.rint(iscale[:, None] * scale), -128, 127)
    d = f16(np.where(iscale != 0, 1.0 / np.where(iscale != 0, iscale, 1.0), 0.0))
    dd = d[:, None] * sc
    q = np.where(dd[..., None] != 0, np.rint(x / np.where(dd != 0, dd, 1.0)[..., None]), 0.0)
    code = (np.clip(q, -32, 31) + 32).astype(np.uint8).reshape(S, 2, 4, 32)  # half, quarter, l
    out = np.zeros((S, 210), np.uint8)
    lo, hi = code & 15, code >> 4
    ql = np.empty((S, 2, 64), np.uint8)
    ql[:, :, 0:32] = lo[:, :, 0] | (lo[:, :, 2] << 4)
    ql[:, :, 32:64] = lo[:, :, 1] | (lo[:, :, 3] << 4)
    out[:, 0:128] = ql.reshape(S, 128)
    out[:, 128:192] = (hi[:, :, 0] | (hi[:, :, 1] << 2) | (hi[:, :, 2] << 4) | (hi[:, :, 3] << 6)).reshape(S, 64)
    out[:, 192:208] = sc.astype(np.int8).view(np.uint8)
    out[:, 208:210] = d.astype("<f2").view(np.uint8).reshape(S, 2)
    return out


def unpack_super_q6(raw):
    """Reference decode of block_q6_K bytes: (S, 256) float32."""
    b = np.frombuffer(raw, np.uint8).reshape(-1, 210)
    S = b.shape[0]
    ql, qh = b[:, 0:128].reshape(S, 2, 64), b[:, 128:192].reshape(S, 2, 32)
    sc = b[:, 192:208].copy().view(np.int8).astype(np.float32)
    d = b[:, 208:210].copy().view("<f2").astype(np.float32).reshape(S)
    q = np.empty((S, 2, 4, 32), np.float32)
    q[:, :, 0] = (ql[:, :, 0:32] & 15) | ((qh & 3) << 4)
    q[:, :, 1] = (ql[:, :, 32:64] & 15) | (((qh >> 2) & 3) << 4)
    q[:, :, 2] = (ql[:, :, 0:32] >> 4) | (((qh >> 4) & 3) << 4)
    q[:, :, 3] = (ql[:, :, 32:64] >> 4) | (((qh >> 6) & 3) << 4)
    return ((d[:, None] * sc)[..., None] * (q.reshape(S, 16, 16) - 32.0)).reshape(S, 256)


def quantize_matrix(wm, fmt, imp=None, refine=True):
    """wm: (rows, cols) float32, cols % 256 == 0. Returns (bytes, sum sq error, sum sq value)."""
    rows, cols = wm.shape
    nmax = FORMATS[fmt][1]
    x = wm.reshape(-1, 8, 32)
    out = []
    num = den = 0.0
    per = cols // 256
    for a in range(0, x.shape[0], CHUNK_BLOCKS):
        xc = np.ascontiguousarray(x[a:a + CHUNK_BLOCKS])
        w = np.sqrt((xc * xc).mean(-1, keepdims=True)) + np.abs(xc)
        if imp is not None:
            col = (np.arange(a, a + xc.shape[0]) % per)[:, None, None] * 256 + np.arange(256).reshape(8, 32)[None]
            w = w * imp[col]
        w = np.maximum(w, 1e-12).astype(np.float32)
        if fmt == "q6_k":
            raw = quantize_super_q6(xc.reshape(-1, 16, 16), w.reshape(-1, 16, 16))
            y = unpack_super_q6(raw.tobytes()).reshape(xc.shape)
        else:
            raw = pack_super(*quantize_super(xc, w, nmax, refine), fmt)
            y = unpack_super(raw.tobytes(), fmt).reshape(xc.shape)
        num += float(((y - xc).astype(np.float64) ** 2).sum())
        den += float((xc.astype(np.float64) ** 2).sum())
        out.append(raw)
    return np.concatenate(out).tobytes(), num, den


def init(path, refine):
    global SRC, REFINE
    f = open(path, "rb")
    SRC = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    REFINE = refine


def job(args):
    name, dims, off, fmt, imp = args
    cols, rows = dims[0], int(np.prod(dims[1:]))
    t0 = time.time()
    wm = dequant_q8(SRC[off:off + nbytes(dims, Q8_0)], rows, cols)
    payload, num, den = quantize_matrix(wm, fmt, imp, REFINE)
    return name, fmt, payload, num, den, time.time() - t0


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(64 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("pack")
    ap.add_argument("--spec", action="append", default=[], help="REGEX=q4_k|q5_k|q6_k|q8, first match wins")
    ap.add_argument("--default", default="q4_k", choices=["q4_k", "q5_k", "q6_k", "q8"])
    ap.add_argument("--only")
    ap.add_argument("--skip")
    ap.add_argument("--imatrix")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--target-sha256", help="SHA-256 of SRC when already known (skips hashing the file)")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--stats")
    a = ap.parse_args()
    specs = []
    for s in a.spec:
        rx, _, fmt = s.rpartition("=")
        if fmt not in ("q4_k", "q5_k", "q6_k", "q8") or not rx:
            sys.exit(f"bad --spec {s}")
        specs.append((re.compile(rx), fmt))
    only = re.compile(a.only) if a.only else None
    skip = re.compile(a.skip) if a.skip else None
    imatrix = load_imatrix(a.imatrix) if a.imatrix else {}
    m, kv, tensors, order, data = gguf_open(a.src)
    todo, left = [], []
    for n in order:
        dims, tt, off = tensors[n]
        if not WINDOW.search(n) or tt != Q8_0 or len(dims) != 2 or (only and not only.search(n)) or (skip and skip.search(n)):
            continue
        fmt = next((f for rx, f in specs if rx.search(n)), a.default)
        if fmt == "q8" or dims[0] % 256 or dims[0] > 8192:
            left.append(n)
            continue
        todo.append((n, dims, data + off, fmt, imatrix.get(n)))
    size = os.path.getsize(a.src)
    with ProcessPoolExecutor(1) as hx:
        sha_future = None if a.target_sha256 else hx.submit(file_sha256, a.src)
        entries, stats, replaced = [], [], 0
        with ProcessPoolExecutor(a.jobs, initializer=init, initargs=(a.src, not a.no_refine)) as ex:
            for (name, dims, off, _, _), (_, fmt, payload, num, den, dt) in zip(todo, ex.map(job, todo)):
                entries.append((name, FORMATS[fmt][0], dims, payload))
                replaced += nbytes(dims, Q8_0)
                stats.append({"name": name, "format": fmt, "rel_rms": (num / max(den, 1e-30)) ** 0.5, "seconds": round(dt, 1)})
        sha = a.target_sha256 or sha_future.result()
    content = hashlib.sha256(b"STRIX-MODEL-CONTENT-V1" + struct.pack("<Q", 5) + b"model" + struct.pack("<Q", size) +
                             bytes.fromhex(sha)).hexdigest()
    spec = ",".join(a.spec + ["default=" + a.default]) + (",imatrix" if imatrix else "") + (",no-refine" if a.no_refine else "")
    write_pack(a.pack, entries, {"strix.dense_fast.format": "DENSE-FAST-v1", "strix.dense_fast.spec": spec,
                                 "strix.dense_fast.target_content": content, "strix.dense_fast.target_file_sha256": sha})
    packed = sum(len(e[3]) for e in entries)
    by = {}
    for s in stats:
        k = re.sub(r"\d+", "N", s["name"]) + ":" + s["format"]
        by.setdefault(k, []).append(s["rel_rms"])
    for k, v in sorted(by.items()):
        print(f"{k} n={len(v)} rel_rms mean {sum(v) / len(v):.5f} max {max(v):.5f}")
    print(f"pack tensors {len(entries)} bytes {packed} replace {replaced} (left on Q8_0: {len(left)})")
    print(f"target_file_sha256 {sha}\ntarget_content {content}")
    if a.stats:
        with open(a.stats, "w", encoding="utf-8") as f:
            json.dump({"spec": spec, "packed_bytes": packed, "replaced_bytes": replaced, "left": left, "tensors": stats}, f, indent=1)


if __name__ == "__main__":
    main()
