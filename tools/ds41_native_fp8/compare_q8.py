#!/usr/bin/env python3
"""Requantization error of the model file's Q8_0 dense sites against the original FP8 weights (read-only on both).

For each layer and site: the original weight e4m3(code) * 2^(scale - 127) (32 x 32 blocks) against the Q8_0 block
values d * q of the model file (GGUF), as the relative RMS error ||Q - W|| / ||W|| and the share of exactly equal values.
SRC_DIR may be the original checkpoint or an extract (extract.py). Needs numpy.

usage: compare_q8.py SRC_DIR MODEL.gguf LAYERS (e.g. 0,1,2)
"""
import json, os, struct, sys
import numpy as np

SITES = [("attn.wq_a", "attn_q_a"), ("attn.wq_b", "attn_q_b"), ("attn.wkv", "attn_kv"), ("attn.wo_a", "attn_output_a"),
         ("attn.wo_b", "attn_output_b"), ("ffn.shared_experts.w1", "ffn_gate_shexp"), ("ffn.shared_experts.w3", "ffn_up_shexp"),
         ("ffn.shared_experts.w2", "ffn_down_shexp")]


def gguf_tensors(path):
    f = open(path, "rb")
    rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))
    rstr = lambda: f.read(rd("Q")[0]).decode("utf-8", "replace")
    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def value(t):
        if t == 8:
            return rstr()
        if t == 9:
            et, n = rd("I")[0], rd("Q")[0]
            return [value(et) for _ in range(n)]
        return rd(scalar[t])[0]
    assert f.read(4) == b"GGUF"
    _, nt, nkv = rd("I")[0], rd("Q")[0], rd("Q")[0]
    kv = {}
    for _ in range(nkv):
        k = rstr()
        kv[k] = value(rd("I")[0])
    ts = {}
    for _ in range(nt):
        name = rstr()
        nd = rd("I")[0]
        dims = rd("Q" * nd)
        typ, off = rd("I")[0], rd("Q")[0]
        ts[name] = (dims, typ, off)
    align = kv.get("general.alignment", 32)
    return ts, (f.tell() + align - 1) // align * align


def main():
    src, gguf, layers = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
    wm = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    e4m3 = np.zeros(256)
    for c in range(256):
        e, m = (c >> 3) & 15, c & 7
        v = np.nan if (c & 0x7F) == 0x7F else (m * 2.0 ** -9 if e == 0 else (8 + m) * 2.0 ** (e - 10))
        e4m3[c] = -v if c & 0x80 else v
    ts, data = gguf_tensors(gguf)

    def st(name):
        p = os.path.join(src, wm[name])
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            e = json.loads(f.read(n))[name]
            a, b = e["data_offsets"]
            f.seek(8 + n + a)
            return e, np.frombuffer(f.read(b - a), dtype=np.uint8)
    for L in layers:
        for orig, ours in SITES:
            ew, codes = st(f"layers.{L}.{orig}.weight")
            es, sc = st(f"layers.{L}.{orig}.scale")
            R, C = ew["shape"]
            w = e4m3[codes].reshape(R, C) * np.repeat(np.repeat(np.ldexp(1.0, sc.astype(np.int32) - 127).reshape(es["shape"]), 32, 0), 32, 1)
            dims, typ, off = ts[f"blk.{L}.{ours}.weight"]
            assert typ == 8 and dims == (C, R), (ours, dims, typ)
            raw = np.fromfile(gguf, dtype=np.uint8, count=R * C // 32 * 34, offset=data + off).reshape(R, C // 32, 34)
            q = (raw[:, :, 2:].view(np.int8).astype(np.float64) * raw[:, :, :2].copy().view(np.float16).astype(np.float64)).reshape(R, C)
            d = q - w
            print(f"L{L} {ours:16s} [{R}, {C}] exponents {sc.min()}..{sc.max()} exact {np.mean(d == 0) * 100:6.2f}% "
                  f"relRMS {np.sqrt((d ** 2).sum() / (w ** 2).sum()):.3e} max|d|/max|w| {np.abs(d).max() / np.abs(w).max():.3e}", flush=True)


if __name__ == "__main__":
    main()
