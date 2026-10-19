#!/usr/bin/env python3
"""Dense Q8_0 -> Q6P32 quantiser (values only).

Q6P32 (6.5 bpw): blocks of 32 weights, w = d * q, q in [-32, 31], d an f16 per block
(byte layout in include/strix/q6p_registry.hpp, written by q6p_pack.py).

Because d is an f16 and |q| <= 32, every Q6P32 block is also an exact Q8_0 block (same d,
codes q). This script writes the "emulated" GGUF: a copy of the source GGUF whose dense
Q8_0 tensors hold the Q6P32 values in Q8_0 containers (and whose F32 router holds
BF16-rounded values). q6p_pack.py then builds the Q6P32 pack bound to that file; the engine
reading the pack computes exactly what it computes from the emulated GGUF alone.

usage: q6p_quantize.py SRC.gguf OUT_EMULATED.gguf [--imatrix F] [--only REGEX] [--skip REGEX]
                    [--no-emulated] [--jobs N] [--stats FILE]
"""
import argparse, json, mmap, os, re, struct, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

Q8_0, F32, BF16, U8 = 8, 0, 30, 1  # U8 is written as GGUF type I8 (24) with byte dims
GGML_I8 = 24

DENSE = re.compile(r"\.(attn_qkv|attn_gate|ssm_out|attn_q|attn_k|attn_v|attn_output|hc_attn_down|hc_attn_up|hc_ffn_down|"
                   r"hc_ffn_up|ffn_gate_shexp|ffn_up_shexp|ffn_down_shexp)\.weight$|^output(_hc_down|_hc_up)?\.weight$")
ROUTER = re.compile(r"\.ffn_gate_inp\.weight$")

# ---------------------------------------------------------------- GGUF reading
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class Reader:
    def __init__(self, buf):
        self.b, self.p = buf, 0

    def take(self, fmt):
        v = struct.unpack_from("<" + fmt, self.b, self.p)
        self.p += struct.calcsize(fmt)
        return v

    def str(self):
        (n,) = self.take("Q")
        s = bytes(self.b[self.p:self.p + n]).decode("utf-8", "replace")
        self.p += n
        return s

    def val(self, t):
        if t in SCALAR:
            return self.take(SCALAR[t])[0]
        if t == 8:
            return self.str()
        if t == 9:
            (at,) = self.take("I")
            (n,) = self.take("Q")
            if at in SCALAR:
                v = self.take(SCALAR[at] * n) if n else ()
                return list(v)
            return [self.val(at) for _ in range(n)]
        raise ValueError(t)


def gguf_open(path):
    f = open(path, "rb")
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    r = Reader(m)
    assert bytes(m[:4]) == b"GGUF"
    r.p = 4
    (ver,) = r.take("I")
    (nt,) = r.take("Q")
    (nkv,) = r.take("Q")
    kv = {}
    for _ in range(nkv):
        k = r.str()
        (t,) = r.take("I")
        kv[k] = r.val(t)
    tensors = {}
    order = []
    for _ in range(nt):
        name = r.str()
        (nd,) = r.take("I")
        dims = r.take("Q" * nd)
        (tt,) = r.take("I")
        (off,) = r.take("Q")
        tensors[name] = (list(dims), tt, off)
        order.append(name)
    align = kv.get("general.alignment", 32)
    data = (r.p + align - 1) // align * align
    return m, kv, tensors, order, data


def nbytes(dims, tt):
    n = int(np.prod(dims))
    return {Q8_0: n // 32 * 34, F32: 4 * n, BF16: 2 * n, GGML_I8: n}[tt]


# ---------------------------------------------------------------- imatrix
def load_imatrix(path):
    """Importance matrix GGUF: per tensor "<name>.in_sum2" (sum of squared inputs per column) and
    "<name>.counts"; returns mean squared input per column for 1-D entries (dense tensors)."""
    m, kv, tensors, order, data = gguf_open(path)
    out = {}
    for name, (dims, tt, off) in tensors.items():
        if not name.endswith(".in_sum2"):
            continue
        base = name[:-len(".in_sum2")]
        cdims, ctt, coff = tensors[base + ".counts"]
        s = np.frombuffer(m, np.float32, int(np.prod(dims)), data + off).reshape(dims[::-1]).astype(np.float64)
        c = np.frombuffer(m, np.float32, int(np.prod(cdims)), data + coff).astype(np.float64)
        if s.ndim == 2:  # per-expert rows: not used for dense tensors
            continue
        out[base] = (s / max(c.reshape(-1)[0], 1.0)).astype(np.float32)
    return out


# ---------------------------------------------------------------- quantisation
def f16(x):
    return x.astype(np.float16).astype(np.float32)


WEIGHTING = "imp-sqrt"


def quantize_q6(w, imp):
    """w: (B, 32) float32 blocks, imp: (B, 32) importance. Returns q int8 (B, 32), d float16 (B,)."""
    w = w.astype(np.float32)
    # Importance-weighted error: imatrix column importance * sqrt(sigma2 + x^2); plain x^2 without one.
    sigma2 = 2.0 * np.mean(w * w, axis=1, keepdims=True)
    # "imp": mean squared input per column, the output-error weight of independent inputs.
    if imp is None:
        weight = w * w + 1e-12
    elif WEIGHTING == "imp":
        weight = imp + 1e-12 * np.max(imp)
    else:
        weight = imp * np.sqrt(sigma2 + w * w)
    idx = np.argmax(np.abs(w), axis=1)
    m = w[np.arange(w.shape[0]), idx][:, None]  # signed max-abs element
    best_err = np.full(w.shape[0], np.inf, np.float64)
    best_q = np.zeros(w.shape, np.int8)
    best_d = np.zeros(w.shape[0], np.float32)
    nz = (m[:, 0] != 0)
    candidates = [-32.0 + 0.25 * s for s in range(-6, 7)] + [31.0 + 0.25 * s for s in range(-4, 3)]
    for t in candidates:
        with np.errstate(divide="ignore", invalid="ignore"):
            d0 = np.where(nz[:, None], m / t, 1.0)
            q = np.clip(np.rint(w / d0), -32, 31)
            sxq = np.sum(weight * w * q, axis=1)
            sqq = np.sum(weight * q * q, axis=1)
            d = np.where(sqq > 0, sxq / sqq, 0.0)
        for _ in range(2):  # f16 rounding of d, then re-round the codes against the stored scale
            d = f16(d)
            with np.errstate(divide="ignore", invalid="ignore"):
                q = np.where(d[:, None] != 0, np.clip(np.rint(w / d[:, None]), -32, 31), 0.0)
            sxq = np.sum(weight * w * q, axis=1)
            sqq = np.sum(weight * q * q, axis=1)
            dn = np.where(sqq > 0, sxq / np.maximum(sqq, 1e-30), 0.0)
            err = np.sum(weight * (w - d[:, None] * q) ** 2, axis=1)
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_q[better] = q[better].astype(np.int8)
            best_d[better] = d[better]
            d = dn
    return best_q, best_d.astype(np.float16)


def q8_blocks(q, d, rows, cols):
    """Q8_0 container of the same values: per block f16 d then 32 int8 codes."""
    nb = cols // 32
    out = np.empty((rows, nb, 34), np.uint8)
    out[:, :, 0:2] = np.ascontiguousarray(d).astype("<f2").view(np.uint8).reshape(rows, nb, 2)
    out[:, :, 2:] = q.reshape(rows, nb, 32).view(np.uint8)
    return out.tobytes()


def dequant_q8(raw, rows, cols):
    blocks = np.frombuffer(raw, np.uint8).reshape(rows, cols // 32, 34)
    d = blocks[:, :, 0:2].copy().view("<f2").astype(np.float32)
    q = blocks[:, :, 2:].view(np.int8).astype(np.float32)
    return (q * d).reshape(rows, cols)


def bf16_round(x):
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


# ---------------------------------------------------------------- per-tensor job
SRC = None


def job(args):
    name, dims, tt, off, imp = args
    m = SRC
    cols, rows = dims[0], int(np.prod(dims[1:]))
    t0 = time.time()
    raw = m[off:off + nbytes(dims, tt)]
    if tt == F32:  # router: BF16 values
        w = np.frombuffer(raw, np.float32).reshape(rows, cols)
        r = bf16_round(w)
        bf = (r.view(np.uint32) >> 16).astype("<u2")
        rel = float(np.sqrt(np.mean((r - w) ** 2) / max(np.mean(w * w), 1e-30)))
        return name, r.tobytes(), None, {"kind": "bf16", "rel_rmse": rel, "s": time.time() - t0}
    assert tt == Q8_0 and cols % 32 == 0
    emu = []
    se = sw = see_rtn = 0.0
    chunk = max(1, (1 << 22) // cols)
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        w = dequant_q8(raw[r0 * cols // 32 * 34:r1 * cols // 32 * 34], r1 - r0, cols)
        blocks = w.reshape(-1, 32)
        ib = None if imp is None else np.broadcast_to(imp.reshape(1, cols), (r1 - r0, cols)).reshape(-1, 32)
        q, d = quantize_q6(blocks, ib)
        q = q.reshape(r1 - r0, cols)
        d = d.reshape(r1 - r0, cols // 32)
        emu.append(q8_blocks(q, d, r1 - r0, cols))
        deq = q.astype(np.float32).reshape(-1, 32) * d.astype(np.float32).reshape(-1, 1)
        wimp = 1.0 if ib is None else ib
        se += float(np.sum(wimp * (deq - blocks) ** 2))
        sw += float(np.sum(wimp * blocks * blocks))
        # round-to-nearest reference with amax/31.5 scale for the stats
        a = np.max(np.abs(blocks), axis=1, keepdims=True) / 31.5
        a = f16(a)
        with np.errstate(divide="ignore", invalid="ignore"):
            qr = np.where(a != 0, np.clip(np.rint(blocks / a), -32, 31), 0)
        see_rtn += float(np.sum(wimp * (qr * a - blocks) ** 2))
    stats = {"kind": "q6p", "rel_wrmse": (se / max(sw, 1e-30)) ** 0.5, "rel_wrmse_rtn": (see_rtn / max(sw, 1e-30)) ** 0.5,
             "imatrix": imp is not None, "s": time.time() - t0}
    return name, b"".join(emu), None, stats


def init(path, weighting="imp-sqrt"):
    global SRC, WEIGHTING
    WEIGHTING = weighting
    f = open(path, "rb")
    SRC = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)


# ---------------------------------------------------------------- GGUF writing (used by q6p_pack.py)
def w_str(out, s):
    b = s.encode()
    out += struct.pack("<Q", len(b)) + b


def write_pack(path, entries, meta):
    """entries: list of (name, ggml_type, dims, bytes). Alignment 256."""
    align = 256
    hdr = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(entries), len(meta) + 1))
    w_str(hdr, "general.alignment")
    hdr += struct.pack("<II", 4, align)
    for k, v in meta.items():
        w_str(hdr, k)
        hdr += struct.pack("<I", 8)
        w_str(hdr, v)
    off = 0
    offs = []
    for name, tt, dims, data in entries:
        offs.append(off)
        off = (off + len(data) + align - 1) // align * align
    for (name, tt, dims, data), o in zip(entries, offs):
        w_str(hdr, name)
        hdr += struct.pack("<I", len(dims)) + struct.pack("<" + "Q" * len(dims), *dims) + struct.pack("<IQ", tt, o)
    pad = (-len(hdr)) % align
    with open(path + ".part", "wb") as f:
        f.write(hdr + b"\0" * pad)
        for (name, tt, dims, data), o in zip(entries, offs):
            f.write(data)
            f.write(b"\0" * ((-len(data)) % align))
    os.replace(path + ".part", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("emulated")
    ap.add_argument("--imatrix")
    ap.add_argument("--only")
    ap.add_argument("--skip")
    ap.add_argument("--no-emulated", action="store_true")
    ap.add_argument("--no-router", action="store_true")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--stats")
    ap.add_argument("--weighting", choices=["imp-sqrt", "imp"], default="imp-sqrt")
    a = ap.parse_args()
    m, kv, tensors, order, data = gguf_open(a.src)
    imat = load_imatrix(a.imatrix) if a.imatrix else {}
    todo = []
    for name in order:
        dims, tt, off = tensors[name]
        if not (DENSE.search(name) and tt == Q8_0) and not (ROUTER.search(name) and tt == F32 and not a.no_router):
            continue
        if a.only and not re.search(a.only, name):
            continue
        if a.skip and re.search(a.skip, name):
            continue
        imp = imat.get(name)
        if imp is not None and imp.shape[0] != dims[0]:
            imp = None
        todo.append((name, dims, tt, data + off, imp))
    print(f"{len(todo)} tensors, imatrix entries {len(imat)}, with imatrix "
          f"{sum(t[4] is not None for t in todo)}", flush=True)
    if not a.no_emulated:
        t0 = time.time()
        os.system(f"cp --sparse=never '{a.src}' '{a.emulated}.part'")
        print(f"copied in {time.time() - t0:.0f} s", flush=True)
        emu = open(a.emulated + ".part", "r+b")
    stats = {}
    with ProcessPoolExecutor(a.jobs, initializer=init, initargs=(a.src, a.weighting)) as ex:
        for name, emu_bytes, _, st in ex.map(job, todo):
            dims, tt, off = tensors[name]
            if not a.no_emulated:
                assert len(emu_bytes) == nbytes(dims, tt)
                emu.seek(data + off)
                emu.write(emu_bytes)
            stats[name] = st
            print(name, json.dumps(st), flush=True)
    if not a.no_emulated:
        emu.close()
        os.replace(a.emulated + ".part", a.emulated)
    if a.stats:
        json.dump(stats, open(a.stats, "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
