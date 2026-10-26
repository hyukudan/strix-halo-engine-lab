#!/usr/bin/env python3
"""Compare two MiniMax prompt-state blobs (Engine::export_rows layout, one rank): per layer K [n][256], V [n][256],
index key [n][128], BF16. Prints identity or the first differing layer / component / row and a per-layer count.


usage: rtxpf_state_compare.py A.bin B.bin
"""
import sys

import numpy as np


def main():
    a = np.fromfile(sys.argv[1], dtype=np.uint16)
    b = np.fromfile(sys.argv[2], dtype=np.uint16)
    if a.size != b.size:
        print(f'SIZE DIFFERS: {a.size * 2} vs {b.size * 2} bytes')
        return 2
    n = a.size // (60 * 640)
    if n * 60 * 640 != a.size:
        print('not a 60-layer state blob')
        return 2
    if np.array_equal(a, b):
        print(f'IDENTICAL: {n} rows x 60 layers ({a.size * 2} bytes)')
        return 0
    a = a.reshape(60, -1)
    b = b.reshape(60, -1)
    parts = [('K', 0, n * 256, 256), ('V', n * 256, n * 512, 256), ('ik', n * 512, n * 640, 128)]
    first = None
    for L in range(60):
        line = []
        for name, lo, hi, w in parts:
            d = np.nonzero(a[L, lo:hi] != b[L, lo:hi])[0]
            line.append(f'{name} {d.size}')
            if d.size and first is None:
                first = (L, name, int(d[0]) // w, int(d[0]) % w)
        if any(not s.endswith(' 0') for s in line):
            print(f'layer {L:2d}: ' + ', '.join(line))
    print(f'DIFFERENT: first at layer {first[0]} {first[1]} row {first[2]} element {first[3]}')
    return 1


if __name__ == '__main__':
    sys.exit(main())
