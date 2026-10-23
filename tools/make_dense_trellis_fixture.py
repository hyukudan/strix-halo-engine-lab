#!/usr/bin/env python3
"""Create an external dense projection fixture; never qualifies model quality.

The source is a standard tensor container with trellis/suh/svh/mul1 fields.
An optional materialized F16 matrix is [output,input]. Without it, the reference
is explicitly a CPU formula proxy, not the builder's GPU reconstruction.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_container(path, base):
    data = path.read_bytes()
    length, = struct.unpack_from("<Q", data)
    if length > len(data) - 8:
        raise ValueError("container header exceeds file")
    header = json.loads(data[8:8 + length])
    def get(suffix, dtype):
        item = header[base + "." + suffix]
        begin, end = item["data_offsets"]
        if begin < 0 or end < begin or 8 + length + end > len(data):
            raise ValueError("tensor range exceeds file")
        expected_type = {"<u2": "I16", "<f2": "F16", "<u4": "I32"}[dtype]
        if item["dtype"] != expected_type:
            raise ValueError("tensor storage type differs")
        result = np.frombuffer(data, dtype=dtype, count=(end - begin) // np.dtype(dtype).itemsize,
                               offset=8 + length + begin)
        if result.nbytes != end - begin or result.size != math.prod(item["shape"]):
            raise ValueError("tensor byte extent differs")
        return result.reshape(item["shape"])
    tiles, su, sv, marker = get("trellis", "<u2"), get("suh", "<f2"), get("svh", "<f2"), get("mul1", "<u4")
    if marker.size != 1 or int(marker.flat[0]) != 0x83DCD12D or tiles.ndim != 3 or tiles.shape[-1] not in (80, 96):
        raise ValueError("unsupported codebook or integer rate")
    inputs, outputs, bits = tiles.shape[0] * 16, tiles.shape[1] * 16, tiles.shape[2] // 16
    if su.shape != (inputs,) or sv.shape != (outputs,) or inputs % 128 or outputs % 128:
        raise ValueError("scale geometry differs")
    if not np.isfinite(su).all() or not np.isfinite(sv).all():
        raise ValueError("nonfinite signed scales")
    return data, tiles, su, sv, inputs, outputs, bits


def codebook(states):
    product = states.astype(np.uint32) * np.uint32(0x83DCD12D)
    total = sum((product >> np.uint32(8 * b)) & np.uint32(255) for b in range(4))
    numerator = (total.astype(np.int64) + 1024) * 1774 - 1329 * 2048
    return (numerator.astype(np.float64) / 262144).astype(np.float16)


def rotated(tiles, bits):
    # Independent bit-array oracle: no sliding-window integer shifts from HIP.
    count = tiles.shape[0] * tiles.shape[1]
    words = np.ascontiguousarray(tiles).view("<u4").reshape(count, 8 * bits)
    indices = ((np.arange(256)[:, None] + 1) * bits - 16 + np.arange(16)[None, :]) % (256 * bits)
    lane, fragment = np.arange(256) // 8, np.arange(256) % 8
    row = 2 * (lane % 4) + fragment % 2 + 8 * ((fragment // 2) % 2)
    col = lane // 4 + 8 * (fragment // 4)
    decoded = np.empty((count, 16, 16), dtype=np.float16)
    for begin in range(0, count, 256):
        w = words[begin:begin + 256]
        stream = ((w[:, :, None] >> np.arange(31, -1, -1, dtype=np.uint32)) & 1).astype(np.uint8).reshape(len(w), -1)
        windows = np.packbits(stream[:, indices], axis=-1, bitorder="big")
        states = np.ascontiguousarray(windows).view(">u2").reshape(len(w), 256)
        decoded[begin:begin + len(w), row, col] = codebook(states)
    return decoded.reshape(tiles.shape[0], tiles.shape[1], 16, 16).transpose(0, 2, 1, 3).reshape(tiles.shape[0] * 16, tiles.shape[1] * 16)


def hadamard(values, axis):
    moved = np.moveaxis(values, axis, -1)
    shape = moved.shape
    work = np.array(moved.reshape(-1, 128), dtype=np.float64, copy=True)
    for stride in (1, 2, 4, 8, 16, 32, 64):
        blocks = work.reshape(-1, 128 // (2 * stride), 2, stride)
        left, right = blocks[:, :, 0].copy(), blocks[:, :, 1].copy()
        blocks[:, :, 0], blocks[:, :, 1] = left + right, left - right
    work *= 1 / math.sqrt(128)
    return np.moveaxis(work.reshape(shape), -1, axis)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("tensor")
    ap.add_argument("output", type=Path)
    ap.add_argument("--reference-f16", type=Path)
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise ValueError("fixture outputs must be new")
    lut = codebook(np.arange(65536, dtype=np.uint32)).astype("<f2").tobytes()
    if sha(lut) != "bc48d02cb1c14939dc47b90f870dd689d63e3b1ab69a157e65538db68677a6c8":
        raise ValueError("complete codebook digest differs")
    source, tiles, su, sv, inputs, outputs, bits = read_container(args.source, args.tensor)
    q = rotated(tiles, bits)
    if args.reference_f16:
        reference_bytes = args.reference_f16.read_bytes()
        if len(reference_bytes) != inputs * outputs * 2:
            raise ValueError("materialized F16 matrix extent differs")
        weight = np.frombuffer(reference_bytes, "<f2").reshape(outputs, inputs).T
    else:
        matrix = hadamard(hadamard(q, 0), 1)
        matrix *= su.astype(np.float64)[:, None]
        matrix *= sv.astype(np.float64)[None, :]
        weight = matrix.astype(np.float16)
        del matrix
    if not np.isfinite(weight).all():
        raise ValueError("nonfinite materialized reference")
    rows = 8
    x = np.random.default_rng(args.seed).standard_normal((rows, inputs), dtype=np.float32)
    xb = x.view(np.uint32)
    bf = ((xb + 0x7fff + ((xb >> 16) & 1)) >> 16).astype("<u2")
    exact_x = (bf.astype(np.uint32) << 16).view(np.float32)
    expected = (exact_x.astype(np.float64) @ weight.astype(np.float64)).astype("<f4")
    # Shape-matched Q8 control, never a claim of source-checkpoint identity.
    groups = np.ascontiguousarray(weight.T).astype(np.float32).reshape(outputs, inputs // 32, 32)
    d = (np.max(np.abs(groups), axis=-1) / 127).astype("<f2")
    codes = np.rint(groups / np.where(d == 0, 1, d).astype(np.float32)[:, :, None]).clip(-127, 127).astype(np.int8)
    q8 = np.empty((outputs, inputs // 32, 34), dtype=np.uint8)
    q8[:, :, :2] = np.ascontiguousarray(d).view(np.uint8).reshape(outputs, inputs // 32, 2)
    q8[:, :, 2:] = codes.view(np.uint8)
    packed = tiles.astype("<u2").tobytes()
    header = struct.pack("<8sIIIIQQQ16s", b"TRLDNS1\0", inputs, outputs, bits, rows, len(packed), q8.nbytes,
                         int(bool(args.reference_f16)), bytes(16))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        for part in (header, packed, su.astype("<f2").tobytes(), sv.astype("<f2").tobytes(), bf.tobytes(),
                     q.astype("<f2").tobytes(), expected.tobytes(), q8.tobytes()):
            output.write(part)
    metadata = dict(source=str(args.source), source_sha256=sha(source), tensor=args.tensor, inputs=inputs, outputs=outputs,
                    bits=bits, rows=rows, seed=args.seed, oracle="materialized-f16" if args.reference_f16 else "cpu-formula-f16-proxy",
                    reference_sha256=sha(reference_bytes) if args.reference_f16 else None,
                    fixture_sha256=sha(args.output.read_bytes()), performance_qualified=False, model_quality_qualified=False)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
