#!/usr/bin/env python3
"""Build the Q6P32 pack of an emulated checkpoint (see include/strix/q6p_registry.hpp).

usage: q6p_pack.py EMULATED.gguf PACK.gguf [--source-content HEX] [--jobs N]

Every dense Q8_0 tensor whose codes all lie in [-32, 31] becomes "<name>.q6p" (GGUF I8,
rows * cols/32 * 26 bytes, chunked row layout); every F32 router whose values are exact
BF16 becomes "<name>.bf16". The pack header pins the emulated file: its SHA-256 and the
engine's content identity (STRIX-MODEL-CONTENT-V1 over logical name "model").
"""
import argparse, hashlib, mmap, os, re, struct, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q6p_quantize import gguf_open, nbytes, write_pack, DENSE, ROUTER, Q8_0, F32, BF16, GGML_I8  # noqa: E402

ORIGINAL_CONTENT = "3726d6f0d9ba30b9d655ba303595118941baeca5e6945fc8785f938e0f5455e4"


def pack_rows(q8_raw, rows, cols):
    """Q8_0 bytes -> Q6P32 chunked rows; None if a code lies outside [-32, 31]."""
    nb = cols // 32
    blocks = np.frombuffer(q8_raw, np.uint8).reshape(rows, nb, 34)
    q = blocks[:, :, 2:].view(np.int8)
    if q.min() < -32 or q.max() > 31:
        return None
    c = (q.astype(np.int16) + 32).astype(np.uint8)
    lo, hi = c & 0x0F, (c >> 4) & 0x03
    qs = np.empty((rows, nb, 16), np.uint8)
    for k in range(4):
        qs[:, :, 4 * k:4 * k + 4] = lo[:, :, 8 * k:8 * k + 4] | (lo[:, :, 8 * k + 4:8 * k + 8] << 4)
    qh = np.zeros((rows, nb, 8), np.uint8)
    for j in range(2):
        for f in range(4):
            qh[:, :, 4 * j:4 * j + 4] |= hi[:, :, 16 * j + 4 * f:16 * j + 4 * f + 4] << (2 * f)
    d = blocks[:, :, 0:2]
    parts = []
    for b0 in range(0, nb, 8):
        b1 = min(nb, b0 + 8)
        parts += [qs[:, b0:b1].reshape(rows, -1), qh[:, b0:b1].reshape(rows, -1), d[:, b0:b1].reshape(rows, -1)]
    out = np.concatenate(parts, axis=1)
    assert out.shape[1] == nb * 26
    return out.tobytes()


def unpack_rows(buf, rows, cols):
    """Independent reference decoder of the chunked layout -> Q8_0 bytes."""
    nb = cols // 32
    data = np.frombuffer(buf, np.uint8).reshape(rows, nb * 26)
    out = np.empty((rows, nb, 34), np.uint8)
    for b in range(nb):
        chunk, inn = b // 8, b % 8
        n = min(256, cols - 256 * chunk)
        at = 208 * chunk
        qs = data[:, at + 16 * inn:at + 16 * inn + 16]
        qh = data[:, at + n // 2 + 8 * inn:at + n // 2 + 8 * inn + 8]
        out[:, b, 0:2] = data[:, at + n // 2 + n // 4 + 2 * inn:at + n // 2 + n // 4 + 2 * inn + 2]
        for e in range(32):
            k, i, up = e // 8, e % 4, 4 if (e % 8) >= 4 else 0
            lo = (qs[:, 4 * k + i] >> up) & 0x0F
            hi = (qh[:, 4 * (e // 16) + i] >> (2 * ((e % 16) // 4))) & 3
            out[:, b, 2 + e] = ((lo | (hi << 4)).astype(np.int16) - 32).astype(np.int8).view(np.uint8)
    return out.tobytes()


SRC = None


def init(path):
    global SRC
    f = open(path, "rb")
    SRC = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)


def job(args):
    name, dims, tt, off = args
    raw = SRC[off:off + nbytes(dims, tt)]
    cols, rows = dims[0], int(np.prod(dims[1:]))
    if tt == F32:
        w = np.frombuffer(raw, np.uint32)
        if np.any(w & 0xFFFF):
            return name, None, "not-bf16"
        return name, (w >> 16).astype("<u2").tobytes(), "bf16"
    out = []
    chunk = max(1, (1 << 22) // cols)
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        part = raw[r0 * (cols // 32) * 34:r1 * (cols // 32) * 34]
        p = pack_rows(part, r1 - r0, cols)
        if p is None:
            return name, None, "not-six-bit"
        if r0 == 0:
            assert unpack_rows(p, r1 - r0, cols) == bytes(part), name
        out.append(p)
    return name, b"".join(out), "q6p"


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(64 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest(), os.path.getsize(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emulated")
    ap.add_argument("pack")
    ap.add_argument("--source-content", default=ORIGINAL_CONTENT)
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    m, kv, tensors, order, data = gguf_open(a.emulated)
    todo = [(n, tensors[n][0], tensors[n][1], data + tensors[n][2]) for n in order
            if (DENSE.search(n) and tensors[n][1] == Q8_0) or (ROUTER.search(n) and tensors[n][1] == F32)]
    with ProcessPoolExecutor(1) as hx:
        sha_future = hx.submit(file_sha256, a.emulated)
        entries, skipped = [], []
        with ProcessPoolExecutor(a.jobs, initializer=init, initargs=(a.emulated,)) as ex:
            for name, payload, kind in ex.map(job, todo):
                dims = tensors[name][0]
                if payload is None:
                    skipped.append((name, kind))
                elif kind == "bf16":
                    entries.append((name + ".bf16", BF16, dims, payload))
                else:
                    entries.append((name + ".q6p", GGML_I8, [len(payload)], payload))
        sha, size = sha_future.result()
    content = hashlib.sha256(b"STRIX-MODEL-CONTENT-V1" + struct.pack("<Q", 5) + b"model" + struct.pack("<Q", size) +
                             bytes.fromhex(sha)).hexdigest()
    write_pack(a.pack, entries, {"strix.q6p.format": "Q6P32-v1", "strix.q6p.source_content": a.source_content,
                                 "strix.q6p.target_content": content, "strix.q6p.target_file_sha256": sha})
    print(f"pack tensors {len(entries)} (skipped {len(skipped)}: {skipped[:5]})")
    print(f"emulated_file_sha256 {sha}\nemulated_content {content}\nemulated_bytes {size}")


if __name__ == "__main__":
    main()
