#!/usr/bin/env python3
"""DeepSeek V4.1 Flash: requantize Q8_0 tensors into Q8_0-embeddable codebook formats (weight overlay).

Every format here keeps the Q8_0 block geometry: 32 weights, one f16 scale d, and each weight is d * c[k] where c
is an int8 codebook of 2^bits entries. So every block is also an exact Q8_0 block (same d, codes c[k]) and the
engine's A8 x Q8_0 product of the emulated block is bitwise the product a packed kernel computes from (d, k).
The overlay GGUF written here holds the emulated Q8_0 tensors; the runners read it with --overlay.

Formats (bits per weight with the f16 scale):
  u3 u4 u5 u6   uniform symmetric codes [-2^(b-1), 2^(b-1)-1]        3.5 / 4.5 / 5.5 / 6.5
  nl3 nl4       non-uniform int8 codebooks (fixed, symmetric-ish)      3.5 / 4.5
  q8            keep the tensor (no-op; useful in specs)
  bf16          router (F32): values rounded to BF16, still stored as F32 (a BF16 copy computes the same)
  mx8           engram_kv (F16): int8 x power-of-two scale per 32 weights (8.25 bpw), exact in the F16 container

usage: requant.py SRC.gguf OUT.gguf --identity ID --spec CLASS=FMT[,CLASS=FMT...] [--imatrix F] [--jobs N]
                  [--layers A-B] [--stats F] [--search N]
classes: q_a q_b kv wo_a wo_b sh_gu sh_down head (and the shorthands attn = q_a,q_b,kv,wo_a,wo_b, sh = sh_gu,sh_down)
"""
import argparse, json, os, re, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

CLASSES = {
    "q_a": r"blk\.(\d+)\.attn_q_a\.weight",
    "q_b": r"blk\.(\d+)\.attn_q_b\.weight",
    "kv": r"blk\.(\d+)\.attn_kv\.weight",
    "wo_a": r"blk\.(\d+)\.attn_output_a\.weight",
    "wo_b": r"blk\.(\d+)\.attn_output_b\.weight",
    "sh_gu": r"blk\.(\d+)\.ffn_(?:gate|up)_shexp\.weight",
    "sh_down": r"blk\.(\d+)\.ffn_down_shexp\.weight",
    "head": r"output\.weight",
    "router": r"blk\.(\d+)\.ffn_gate_inp\.weight",  # F32: fmt bf16 (values rounded to BF16, stored as F32)
    "ekv": r"blk\.(\d+)\.engram_kv\.weight",       # F16: fmt mx8 (int8 x power-of-two scale per 32, exact in F16)
}
# formats of the non-Q8_0 classes: (source GGML type, output GGML type)
FLOAT_FORMATS = {"bf16": (0, 0), "mx8": (1, 1)}
SHORT = {"attn": ["q_a", "q_b", "kv", "wo_a", "wo_b"], "sh": ["sh_gu", "sh_down"]}
# calibration input of each class (labels written by strix-deepseek41-requant-eval --calib)
CALIB = {"q_a": "rms_norm@{l}/attn:x", "kv": "rms_norm@{l}/attn:x", "q_b": "rms_norm@{l}/q:qr",
         "wo_a": "attention@{l}:o", "wo_b": "o_a@{l}:o_low", "sh_gu": "rms_norm@{l}/ffn:x"}

NL4 = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.int8)
NL3 = np.array([-127, -82, -45, -14, 14, 45, 82, 127], np.int8)  # placeholder symmetric grid; see --codebook


def codebook(fmt):
    if fmt.startswith("u"):
        b = int(fmt[1:])
        return np.arange(-(1 << (b - 1)), 1 << (b - 1), dtype=np.int8)
    if fmt == "nl4":
        return NL4
    if fmt == "nl3":
        return NL3
    raise ValueError(fmt)


def bpw(fmt):
    return 8.5 if fmt == "q8" else (int(fmt[-1]) * 32 + 16) / 32


Q8 = np.dtype([("d", "<f2"), ("q", "i1", (32,))])


def dequant_q8(raw, rows, cols):
    b = np.frombuffer(raw, dtype=Q8).reshape(rows, cols // 32)
    return (b["d"].astype(np.float32)[..., None] * b["q"].astype(np.float32)).reshape(rows, cols), b


def assign(x, cb_sorted, mids):
    """Index into cb_sorted of the nearest codebook value of x (ties to the lower value)."""
    return np.searchsorted(mids, x, side="left")


def quantize_blocks(x, w, cb, search):
    """x, w: [nblocks, 32] float32 (w >= 0 importance). Returns (d f16 [nblocks], k uint8 [nblocks, 32] indices into cb)."""
    cbs = np.sort(cb.astype(np.float32))
    order = np.argsort(cb.astype(np.float32), kind="stable")
    mids = (cbs[1:] + cbs[:-1]) * 0.5
    amax_i = np.argmax(np.abs(x), axis=1)
    xmax = x[np.arange(len(x)), amax_i]
    # scales that put the block's largest-magnitude weight on the outermost code of its sign
    edge = np.where(xmax >= 0, cbs[-1], cbs[0])
    base = np.where(edge != 0, xmax / edge, 0.0).astype(np.float32)
    best_err = np.full(len(x), np.inf, np.float32)
    best_d = base.copy()
    for t in range(-search, search + 1):
        d = base * np.float32(1.0 + 0.02 * t) if search else base
        d = d.astype(np.float16).astype(np.float32)  # the stored scale
        inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1), 0).astype(np.float32)
        k = assign(x * inv[:, None], cbs, mids)
        v = cbs[k] * d[:, None]
        err = np.sum(w * (x - v) ** 2, axis=1)
        # least-squares refit of d for these assignments, then re-assign once
        c = cbs[k]
        num, den = np.sum(w * x * c, axis=1), np.sum(w * c * c, axis=1)
        d2 = np.where(den > 0, num / np.where(den > 0, den, 1), d).astype(np.float16).astype(np.float32)
        inv2 = np.where(d2 != 0, 1.0 / np.where(d2 != 0, d2, 1), 0).astype(np.float32)
        k2 = assign(x * inv2[:, None], cbs, mids)
        err2 = np.sum(w * (x - cbs[k2] * d2[:, None]) ** 2, axis=1)
        use2 = err2 < err
        d = np.where(use2, d2, d)
        err = np.where(use2, err2, err)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_d = np.where(better, d, best_d)
    d = best_d.astype(np.float16)
    df = d.astype(np.float32)
    inv = np.where(df != 0, 1.0 / np.where(df != 0, df, 1), 0).astype(np.float32)
    k_sorted = assign(x * inv[:, None], cbs, mids)
    return d, order[k_sorted].astype(np.uint8)


def emulate_q8(d, k, cb):
    out = np.zeros(d.shape, dtype=Q8)
    out["d"] = d
    out["q"] = cb[k]
    return out


def importance(cls, layer, rows, cols, imat):
    """Per-weight importance [rows, cols] broadcastable, from the calibration sums (mean x^2), or None."""
    if imat is None or cls not in CALIB:
        return None
    key = CALIB[cls].format(l=layer)
    if key not in imat:
        return None
    m = imat[key]
    if cls == "wo_a":  # grouped: output rows [g*1024, (g+1)*1024) read o[g*cols, (g+1)*cols)
        groups = m.size // cols
        per = rows // groups
        return np.repeat(m.reshape(groups, 1, cols), per, axis=0).reshape(rows, cols)
    if m.size != cols:
        raise ValueError(f"{key}: {m.size} values for {cols} columns")
    return m[None, :]


def bf16_round(x):
    """Round-to-nearest-even of float32 values to BF16, returned as float32 (NaN/Inf untouched)."""
    u = x.astype("<f4").view("<u4").astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    finite = np.isfinite(x)
    return np.where(finite, r.view("<f4"), x).astype("<f4")


def mx8(x):
    """Per 32 weights: d = 2^e (smallest with |x| / d <= 127, e >= -24), q = rint(x / d) in [-127, 127]; d * q is exact
    in F16 (7 significant bits, multiples of 2^-24), so the F16 tensor holds exactly the 8-bit values."""
    xb = x.astype(np.float64).reshape(-1, 32)
    amax = np.abs(xb).max(axis=1)
    e = np.where(amax > 0, np.ceil(np.log2(np.maximum(amax, 1e-300) / 127.0)), -24).astype(np.int64)
    e = np.maximum(e, -24)
    d = np.ldexp(1.0, e)[:, None]
    q = np.clip(np.rint(xb / d), -127, 127)
    y = (q * d).astype(np.float16)
    assert np.array_equal(y.astype(np.float64), q * d), "mx8 value not exact in F16"
    return y.reshape(x.shape)


def work(job):
    src, name, offset, nbytes, rows, cols, fmt, cls, layer, imat_path, search, wmode = job
    with open(src, "rb") as f:
        f.seek(offset)
        raw = f.read(nbytes)
    if fmt in FLOAT_FORMATS:
        x = np.frombuffer(raw, dtype="<f4" if fmt == "bf16" else "<f2").reshape(rows, cols).astype(np.float32)
        y = bf16_round(x) if fmt == "bf16" else mx8(x)
        e = x - y.astype(np.float32)
        out = y.astype("<f4" if fmt == "bf16" else "<f2").tobytes()
        return name, out, {"name": name, "fmt": fmt, "rel_mse": float(np.sum(e * e) / max(np.sum(x * x), 1e-30))}
    x, _ = dequant_q8(raw, rows, cols)
    imat = load_imatrix(imat_path) if imat_path else None
    imp = importance(cls, layer, rows, cols, imat)
    if imp is None:
        w = np.ones_like(x)
    else:
        imp = imp / max(float(np.mean(imp)), 1e-30)
        w = np.broadcast_to(imp, x.shape).astype(np.float32)
        if wmode == "imx":
            w = w * np.sqrt(np.mean(x * x) + x * x)
        w = w + 1e-6
    cb = codebook(fmt)
    xb, wb = x.reshape(-1, 32), np.ascontiguousarray(w).reshape(-1, 32)
    d, k = quantize_blocks(xb, wb, cb, search)
    q = emulate_q8(d, k, cb)
    y = (d.astype(np.float32)[:, None] * cb[k].astype(np.float32)).reshape(rows, cols)
    e = x - y
    stats = {"name": name, "fmt": fmt, "rel_mse": float(np.sum(e * e) / max(np.sum(x * x), 1e-30))}
    if imp is not None:
        wi = np.broadcast_to(imp, x.shape)
        stats["rel_wmse"] = float(np.sum(wi * e * e) / max(np.sum(wi * x * x), 1e-30))
    return name, q.tobytes(), stats


_IMAT = {}


def load_imatrix(path):
    if path in _IMAT:
        return _IMAT[path]
    out = {}
    with open(path, "rb") as f:
        data = f.read()
    p = 0
    while p < len(data):
        nl = data.index(b"\n", p)
        label, n, cnt = data[p:nl].decode().split()
        n, cnt = int(n), int(cnt)
        v = np.frombuffer(data, dtype="<f8", count=n, offset=nl + 1)
        out[label] = (v / max(cnt, 1)).astype(np.float64)
        p = nl + 1 + 8 * n
    _IMAT[path] = out
    return out


def parse_spec(spec):
    """CLASS[@A-B]=FMT items; for each tensor the last matching item wins."""
    out = []
    for item in spec.split(","):
        lhs, fmt = item.split("=")
        cls, _, rng = lhs.partition("@")
        lo, hi = (int(v) for v in rng.split("-")) if rng else (-1, 10**9)
        codebook(fmt) if fmt not in ("q8", *FLOAT_FORMATS) else None
        for c in SHORT.get(cls, [cls]):
            if c not in CLASSES:
                raise ValueError(f"unknown class {c}")
            out.append((c, lo, hi, fmt))
    return out


def rule(spec, cls, layer):
    fmt = None
    for c, lo, hi, f in spec:
        if c == cls and (layer < 0 or lo <= layer <= hi):
            fmt = f
    return fmt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--identity", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--imatrix")
    ap.add_argument("--wmode", default="imp", choices=["imp", "imx"])
    ap.add_argument("--layers", help="A-B inclusive (default all)")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--search", type=int, default=6)
    ap.add_argument("--stats")
    a = ap.parse_args()
    import gguf

    spec = parse_spec(a.spec)
    lo, hi = (int(v) for v in a.layers.split("-")) if a.layers else (0, 10**9)
    rd = gguf.GGUFReader(a.src)
    jobs = []
    for t in rd.tensors:
        for cls, rx in CLASSES.items():
            m = re.fullmatch(rx, t.name)
            if not m:
                continue
            layer = int(m.group(1)) if m.groups() else -1
            fmt = rule(spec, cls, layer)
            if fmt is None or fmt == "q8" or (layer >= 0 and not (lo <= layer <= hi)):
                continue
            want = FLOAT_FORMATS[fmt][0] if fmt in FLOAT_FORMATS else 8
            if int(t.tensor_type) != want:
                raise ValueError(f"{t.name} has GGML type {int(t.tensor_type)}, {fmt} needs {want}")
            cols, rows = int(t.shape[0]), int(t.shape[1])
            jobs.append((a.src, t.name, int(t.data_offset), int(t.n_bytes), rows, cols, fmt, cls, layer,
                         a.imatrix, a.search, a.wmode))
    if not jobs:
        sys.exit("no tensor selected")
    t0 = time.time()
    w = gguf.GGUFWriter(a.out, "deepseek41")
    w.add_string("ds41.requant.identity", a.identity)
    w.add_string("ds41.requant.spec", a.spec + (f" imatrix={os.path.basename(a.imatrix)} wmode={a.wmode}" if a.imatrix else ""))
    results = {}
    stats = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, data, st in ex.map(work, jobs, chunksize=1):
            results[name] = data
            stats.append(st)
            print(f"{name} {st['fmt']} rel_mse {st['rel_mse']:.3e}" + (f" rel_wmse {st['rel_wmse']:.3e}" if "rel_wmse" in st else ""),
                  f"({time.time() - t0:.0f} s)", flush=True)
    for j in jobs:
        name, rows, cols, fmt = j[1], j[4], j[5], j[6]
        if fmt == "bf16":
            w.add_tensor(name, np.frombuffer(results[name], dtype="<f4").reshape(rows, cols))
        elif fmt == "mx8":
            w.add_tensor(name, np.frombuffer(results[name], dtype="<f2").reshape(rows, cols))
        else:
            arr = np.frombuffer(results[name], dtype=np.uint8).reshape(rows, cols // 32 * 34)
            w.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    if a.stats:
        with open(a.stats, "w") as f:
            json.dump({"spec": a.spec, "identity": a.identity, "tensors": stats}, f, indent=1)
    print(f"wrote {a.out}: {len(jobs)} tensors in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
