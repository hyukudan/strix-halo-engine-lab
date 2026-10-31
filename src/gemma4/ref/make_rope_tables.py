"""Offline g4-v1 rotary tables. Requires MPFR; inference and ctest do not need Python.

Generate with --positions 65536 for the complete contract. MPFR's error bound is checked
with can_round before exporting each binary32 result. Output is little-endian raw FP32,
[position,pair,cos/sin], with SHA-256 receipts. No weights or prompts are embedded.
"""
import argparse
import concurrent.futures
import gmpy2 as mp
import hashlib
import json
import pathlib
import struct


def shard(args):
    start, stop = args
    mp.get_context().precision = 96
    def rounded(v):
        if v and not mp.can_round(v, 94, mp.RoundToNearest, mp.RoundToNearest, 24):
            raise RuntimeError("MPFR rounding unresolved; increase generation precision")
        with mp.context(mp.ieee(32)):
            return float(+v)
    def f32(v): return struct.unpack('<f', struct.pack('<f', v))[0]
    frequencies = []
    for dim, active, theta in ((256, 128, 10000), (512, 64, 1000000)):
        row = []
        for i in range(active):
            z = mp.mpfr(theta) ** mp.mpfr(f32(2*i/dim))
            row.append(f32(1/rounded(z)))
        frequencies.append(row)
    buffers = [bytearray(), bytearray()]
    for p in range(start, stop):
        for route, pairs in enumerate((128, 256)):
            for i in range(pairs):
                if i >= len(frequencies[route]):
                    buffers[route] += struct.pack('<ff', 1, 0)
                    continue
                angle = f32(p * frequencies[route][i])
                sn, cs = mp.sin_cos(mp.mpfr(angle))
                buffers[route] += struct.pack('<ff', rounded(cs), rounded(sn))
    return buffers


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=pathlib.Path)
    ap.add_argument('--positions', type=int, default=65536)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    if not 1 <= a.positions <= 65536 or not 1 <= a.workers <= 16:
        ap.error('provide valid position/worker limits')
    a.output.mkdir(parents=True, exist_ok=True)
    # Keep resident buffers bounded; tasks return at most 3 MiB each.
    tasks = [(b, min(a.positions, b+1024)) for b in range(0, a.positions, 1024)]
    paths = [a.output / 'sliding.f32', a.output / 'global.f32']
    with concurrent.futures.ProcessPoolExecutor(a.workers) as pool:
        with open(paths[0], 'wb') as sliding, open(paths[1], 'wb') as global_:
            for buffers in pool.map(shard, tasks):
                sliding.write(buffers[0]); global_.write(buffers[1])
    receipt = dict(identity='g4-v1', positions=a.positions, layout='position,pair,cos/sin; f32 little endian',
                   rounding='MPFR 96-bit, can_round(error=94,RNE,24)', mpfr=mp.mpfr_version(), gmpy2=mp.version(),
                   sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (a.output/'tables.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
