#!/usr/bin/env python3
"""Independent CPU oracle and bounded fixture builder for CUDA trellis tests.

Requires NumPy; no GPU/runtime imports or downloads. Payloads stay outside Git.
The scalar circular-bit reader intentionally differs from CUDA's word funnel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
from pathlib import Path
import struct

import numpy as np

MARKER = 0x83DCD12D
HN = np.float32(1 / np.sqrt(128.0))


def codebook() -> np.ndarray:
    values = []
    for state in range(65536):
        p = (state * MARKER) & 0xFFFFFFFF
        b = sum((p >> shift) & 255 for shift in (0, 8, 16, 24))
        # Exact dyadic rational in binary64, followed by just one RN16.
        values.append(((1024 + b) * 1774 - 1329 * 2048) / 262144)
    return np.asarray(values, dtype="<f2")


BOOK = codebook()


def decode(packed: np.ndarray, bits: int) -> np.ndarray:
    """Slow, explicit circular bit stream + logical fragment permutation."""
    if not 1 <= bits <= 8 or packed.ndim != 3 or packed.shape[2] != 16 * bits:
        raise ValueError("unsupported integer rate/layout")
    ni, no, _ = packed.shape
    q = np.empty((ni * 16, no * 16), dtype="<f2")
    for it in range(ni):
        for ot in range(no):
            words16 = packed[it, ot]
            stream = []
            for j in range(0, len(words16), 2):
                word = int(words16[j]) | (int(words16[j + 1]) << 16)
                stream.extend((word >> b) & 1 for b in range(31, -1, -1))
            for pos in range(256):
                end = (pos + 1) * bits
                state = 0
                for b in range(end - 16, end):
                    state = 2 * state + stream[b % len(stream)]
                lane, f = divmod(pos, 8)
                i = 2 * (lane % 4) + f % 2 + 8 * ((f // 2) % 2)
                o = lane // 4 + 8 * (f // 4)
                q[it * 16 + i, ot * 16 + o] = BOOK[state]
    return q


def decode_bulk(packed: np.ndarray, bits: int) -> np.ndarray:
    """Bounded bit-matrix reader for large fixtures (no word-funnel algorithm)."""
    if not 1 <= bits <= 8 or packed.ndim != 3 or packed.shape[2] != 16 * bits:
        raise ValueError("unsupported integer rate/layout")
    ni, no, _ = packed.shape
    flat = np.asarray(packed, dtype=np.uint16).reshape(ni * no, 16 * bits)
    tiles = np.empty((ni * no, 16, 16), dtype="<f2")
    pos = np.arange(256)
    lane, f = pos // 8, pos % 8
    rr = 2 * (lane % 4) + f % 2 + 8 * ((f // 2) % 2)
    cc = lane // 4 + 8 * (f // 4)
    indices = ((pos[:, None] + 1) * bits - np.arange(16, 0, -1)) % (256 * bits)
    significance = np.left_shift(np.uint32(1), np.arange(15, -1, -1, dtype=np.uint32))
    for b in range(0, len(flat), 64):
        region = flat[b:b+64].astype(np.uint32)
        words = region[:, ::2] | (region[:, 1::2] << 16)
        stream = np.unpackbits(words.astype(">u4").view(np.uint8), axis=1, bitorder="big")
        states = np.sum(stream[:, indices] * significance, axis=2, dtype=np.uint32)
        tiles[b:b+64, rr, cc] = BOOK[states]
    return tiles.reshape(ni, no, 16, 16).transpose(0, 2, 1, 3).reshape(ni*16, no*16)


def hadamard(x: np.ndarray) -> np.ndarray:
    h = np.asarray(x, dtype=np.float32).copy()
    if h.shape[-1] % 128:
        raise ValueError("H128 shape")
    blocks = h.reshape(-1, 128)
    for stride in (1, 2, 4, 8, 16, 32, 64):
        v = blocks.reshape(-1, 2 * stride)
        lo, hi = v[:, :stride].copy(), v[:, stride:].copy()
        v[:, :stride], v[:, stride:] = lo + hi, lo - hi
    return h * HN


def input_stage(x: np.ndarray, suh: np.ndarray) -> np.ndarray:
    scaled = (x.astype(np.float32) * suh.astype(np.float32)).astype("<f2")
    return hadamard(scaled.astype(np.float32)).astype("<f2")


def core_gemv(a: np.ndarray, q: np.ndarray, splits: int) -> np.ndarray:
    """CUDA pair-sum identity: split/k/p/lane order is explicit, no BLAS."""
    rows, width = a.shape
    output = q.shape[1]
    result = np.empty((splits, rows, output), dtype="<f4")
    af, qf = a.astype(np.float32), q.astype(np.float32)
    for s in range(splits):
        acc = np.zeros((rows, output, 4), dtype=np.float32)
        for k in range((width // 16) * s // splits, (width // 16) * (s + 1) // splits):
            for p in (0, 1):
                for lane in range(4):
                    i = 16 * k + 2 * lane + 8 * p
                    pair = af[:, i, None] * qf[i] + af[:, i + 1, None] * qf[i + 1]
                    acc[:, :, lane] = acc[:, :, lane] + pair
        result[s] = (acc[:, :, 0] + acc[:, :, 2]) + (acc[:, :, 1] + acc[:, :, 3])
    return result


def output_stage(partial: np.ndarray, svh: np.ndarray, bias: np.ndarray) -> np.ndarray:
    z = np.zeros(partial.shape[1:], dtype=np.float32)
    for part in partial:
        z = z + part
    z = hadamard(z) * svh.astype(np.float32)
    if bias.size:
        z = z + bias.astype(np.float32)
    return z


def write_case(path: Path, packed, suh, svh, x, splits=4, bias=None, padded=True):
    i, o = len(suh), len(svh)
    rows, bits = len(x), packed.shape[2] // 16
    if i % 128 or o % 128 or not 1 <= splits <= min(32, i // 16):
        raise ValueError("invalid case dimensions")
    q = decode_bulk(packed, bits)
    a = input_stage(x, suh)
    p = core_gemv(a, q, splits)
    bias = np.array([], dtype="<f4") if bias is None else np.asarray(bias, dtype="<f4")
    y = output_stage(p, svh, bias)
    xs, ys = i + (7 if padded else 0), o + (11 if padded else 0)
    xp = np.full((rows, xs), np.nan, dtype="<f4")
    xp[:, :i] = x
    yp = np.full((rows, ys), np.float32(-123456.0), dtype="<f4")
    yp[:, :o] = y
    payloads = [np.asarray(packed, dtype="<u2"), np.asarray(suh, dtype="<f2"),
                np.asarray(svh, dtype="<f2"), bias, xp, q, a, p, yp]
    size = 40 + sum(v.nbytes for v in payloads)
    if size > 256 * 1024**2:
        raise ValueError("fixture exceeds 256 MiB")
    with path.open("wb") as f:
        f.write(b"TRLCASE1")
        f.write(struct.pack("<8I", i, o, bits, rows, splits, bool(bias.size), xs, ys))
        for payload in payloads:
            f.write(payload.tobytes())
    return {"file": path.name, "bytes": size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "input": i, "output": o, "bits": bits, "rows": rows, "splits": splits}


def synthetic(directory: Path):
    rng = np.random.default_rng(20261021)
    records = []
    for bits in range(1, 9):
        for i, o in ((128, 256), (256, 128)):
            packed = rng.integers(0, 65536, (i // 16, o // 16, 16 * bits), dtype=np.uint16)
            # Tile edges include zero, all-one, and the wraparound MSB impulse.
            packed[0, 0] = 0
            packed[0, 1] = 65535
            packed[1, 0] = 0
            packed[1, 0, -1] = 0x8000
            suh = rng.uniform(-0.25, 0.25, i).astype("<f2")
            svh = rng.uniform(-0.25, 0.25, o).astype("<f2")
            suh[:6] = np.array([0, -0.0, 2**-24, -2**-24, 0.5, -0.5], dtype="<f2")
            svh[:4] = np.array([0, -0.0, 0.5, -0.5], dtype="<f2")
            x = rng.normal(0, 0.2, (33, i)).astype("<f4")
            x[0] = 0
            x[0, [0, 15, 16, 127]] = [0.25, -0.5, 0.125, -0.25]
            bias = rng.normal(0, 0.01, o).astype("<f4") if bits % 2 else None
            path = directory / f"k{bits}-{i}-{o}.trlcase"
            records.append(write_case(path, packed, suh, svh, x, splits=3, bias=bias))
    (directory / "mul1.bin").write_bytes(BOOK.tobytes())
    (directory / "manifest.json").write_text(json.dumps({"seed": 20261021, "numpy": np.__version__,
        "mul1_sha256": hashlib.sha256(BOOK.tobytes()).hexdigest(), "cases": records}, indent=2), encoding="utf-8")


def from_safetensors(source: Path, prefix: str, directory: Path, inputs: int, outputs: int, rows: int):
    """Read just one modern tensor group, with explicit optional H128 crop.

    No implementation from a runtime/loader is imported. Source attribution is
    recorded in the external manifest alongside the fixture, never in Git.
    """
    if not 1 <= rows <= 64:
        raise ValueError("rows must be 1..64")
    with source.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        hsize = struct.unpack_from("<Q", mm)[0]
        if not 2 <= hsize <= min(16 * 1024**2, len(mm) - 8):
            raise ValueError("safetensors header extent")
        def unique_pairs(pairs):
            d = {}
            for k, v in pairs:
                if k in d:
                    raise ValueError("duplicate header key")
                d[k] = v
            return d
        header = json.loads(mm[8:8+hsize], object_pairs_hook=unique_pairs)
        types = {"I16": "<i2", "U16": "<u2", "F16": "<f2", "F32": "<f4", "I32": "<i4"}
        def get(suffix):
            desc = header[prefix + "." + suffix]
            dtype = np.dtype(types[desc["dtype"]])
            shape = desc["shape"]
            if any(not isinstance(v, int) or v < 0 for v in shape):
                raise ValueError("tensor shape")
            count = 1
            for v in shape:
                count *= v
            lo, hi = desc["data_offsets"]
            if not 0 <= lo <= hi <= len(mm) - 8 - hsize or hi - lo != count * dtype.itemsize:
                raise ValueError("tensor extent")
            return np.frombuffer(mm, dtype=dtype, count=count, offset=8+hsize+lo).reshape(shape).copy()
        for suffix in ("mcg", "su", "sv", "out_scales", "3inst"):
            if prefix + "." + suffix in header:
                raise ValueError("unsupported/ambiguous scale or codebook convention")
        marker = get("mul1")
        if marker.dtype != np.dtype("<i4") or marker.size != 1 or int(marker.reshape(-1)[0]) & 0xFFFFFFFF != MARKER:
            raise ValueError("mul1 marker")
        packed, suh, svh = get("trellis"), get("suh"), get("svh")
        full_output = len(svh)
        if packed.dtype.kind not in "iu" or packed.dtype.itemsize != 2 or packed.ndim != 3:
            raise ValueError("packed storage")
        if suh.dtype != np.dtype("<f2") or svh.dtype != np.dtype("<f2") or suh.ndim != 1 or svh.ndim != 1:
            raise ValueError("scale dtype/shape")
        if packed.shape[:2] != (len(suh) // 16, len(svh) // 16) or packed.shape[2] % 16:
            raise ValueError("tensor group shape")
        i, o = inputs or len(suh), outputs or len(svh)
        if i % 128 or o % 128 or not 0 < i <= len(suh) or not 0 < o <= len(svh):
            raise ValueError("crop must preserve whole H128 blocks")
        packed, suh, svh = packed[:i//16, :o//16].copy(), suh[:i], svh[:o]
        if not np.all(np.isfinite(suh)) or not np.all(np.isfinite(svh)):
            raise ValueError("nonfinite scales")
        bias = get("bias") if prefix + ".bias" in header else None
        if bias is not None:
            if bias.dtype != np.dtype("<f4") or bias.shape != (full_output,):
                raise ValueError("bias dtype/shape")
            bias = bias[:o]
            if not np.all(np.isfinite(bias)):
                raise ValueError("nonfinite bias")
        rng = np.random.default_rng(20261021)
        x = rng.normal(0, 0.05, (rows, i)).astype("<f4")
        record = write_case(directory / "real.trlcase", packed, suh, svh, x, bias=bias)
        record.update(source=str(source.resolve()), prefix=prefix, crop=[i, o])
        (directory / "manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--prefix")
    p.add_argument("--inputs", type=int, default=0)
    p.add_argument("--outputs", type=int, default=0)
    p.add_argument("--rows", type=int, default=8)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.source:
        if not args.prefix:
            p.error("--source requires --prefix")
        from_safetensors(args.source, args.prefix, args.out, args.inputs, args.outputs, args.rows)
    else:
        synthetic(args.out)


if __name__ == "__main__":
    main()
