#!/usr/bin/env python3
"""Lossless ratio and single-thread throughput of zstd level 1 on a real hand-off state, per region.

Usage: state_ratio.py STATE.rtxs [--out DIR] [--level 1]
  STATE.rtxs: a strix-qwen-rtx-state-v1 file ("STRXRTX1", u32 json bytes, u32 0, JSON header, payload).
  Groups the row components by id prefix (kv, index, mtp, ...) and the fixed components by id, and prints one JSON
  line per group: plain bytes, zstd ratio and MB/s on the plain bytes and on the byte-plane split (plane j = byte j of
  every element, width 2 for row groups, 4 for the FP32 fixed matrices). With --out it writes each group to
  DIR/<group>.bin so strix-xfer-probe codec --data file:DIR/<group>.bin --width W measures bp-v1 on the same bytes.
Needs Python >= 3.14 (compression.zstd). No GPU.
"""
import argparse
import json
import os
import struct
import sys
import time

try:
    from compression import zstd
except ImportError:  # pragma: no cover
    sys.exit("state_ratio.py needs Python 3.14 (compression.zstd)")


def planes(b, w):
    n = len(b) // w
    return [b[j : n * w : w] for j in range(w)] + ([b[n * w :]] if len(b) % w else [])


def measure(name, b, w, level):
    t = time.perf_counter()
    c = zstd.compress(b, level=level)
    t_plain = time.perf_counter() - t
    t = time.perf_counter()
    cp = sum(len(zstd.compress(p, level=level)) for p in planes(b, w))
    t_planes = time.perf_counter() - t
    mb = len(b) / 1e6
    return {
        "group": name, "width": w, "plain_bytes": len(b),
        "zstd_ratio": round(len(b) / max(1, len(c)), 4), "zstd_MBps_1t": round(mb / t_plain, 1),
        "zstd_planes_ratio": round(len(b) / max(1, cp), 4), "zstd_planes_MBps_1t": round(mb / t_planes, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--out")
    ap.add_argument("--level", type=int, default=1)
    a = ap.parse_args()
    with open(a.state, "rb") as f:
        data = f.read()
    if data[:8] != b"STRXRTX1":
        sys.exit("not a STRXRTX1 file")
    (jlen,) = struct.unpack("<I", data[8:12])
    head = json.loads(data[16 : 16 + jlen])
    base = 16 + jlen
    p = head["position"]
    groups, off = {}, base
    for c in head["rows"]:
        n = max(0, p - c.get("shift", 0)) * c["bytes_per_row"]
        groups.setdefault(c["id"].split(".")[0], []).append(data[off : off + n])
        off += n
    fixed0 = off
    for c in head["fixed"]:
        groups.setdefault("fixed." + c["id"], []).append(data[fixed0 + c["offset"] : fixed0 + c["offset"] + c["bytes"]])
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    print(json.dumps({"position": p, "row_bytes": head["row_bytes"], "fixed_bytes": head["fixed_bytes"]}))
    for name, parts in groups.items():
        b = b"".join(parts)
        if len(b) < 4096:
            continue
        w = 4 if name.startswith("fixed.") else 2
        if a.out:
            with open(os.path.join(a.out, name + ".bin"), "wb") as f:
                f.write(b)
        print(json.dumps(measure(name, b, w, a.level)))


if __name__ == "__main__":
    main()
