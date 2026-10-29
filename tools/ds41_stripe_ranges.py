#!/usr/bin/env python3
"""Byte ranges of a DeepSeek V4.1 GGUF to place on a second drive (tierstream/stripe-ranges-v1).

The ranges file feeds `ts-stripe make`. All routed-expert tensors and Engram tables are
interleaved on the global chunk grid by default (odd chunks on the secondary). Individual expert slices
need not balance evenly; complete layers and random Engram jobs exercise both locations.

usage: python -I ds41_stripe_ranges.py MODEL.gguf > ranges.json
"""
import argparse
import json
import os
import re
import struct
import sys

# ggml type -> (block elements, block bytes)
TYPES = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144),
         13: (256, 176), 14: (256, 210), 16: (256, 66), 18: (256, 98), 24: (1, 1), 26: (1, 4), 28: (1, 8), 30: (1, 2),
         39: (32, 17)}
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def rd(f, fmt):
    return struct.unpack('<' + fmt, f.read(struct.calcsize(fmt)))


def rstr(f):
    (n,) = rd(f, 'Q')
    return f.read(n).decode('utf-8', 'replace')


def value(f, t):
    if t in SCALAR:
        raw = f.read(SCALAR[t])
        return int.from_bytes(raw, 'little') if t != 6 and t != 12 else None
    if t == 8:
        return rstr(f)
    if t == 9:
        (at,) = rd(f, 'I')
        (n,) = rd(f, 'Q')
        for _ in range(n):
            value(f, at)
        return None
    raise ValueError(f'gguf value type {t}')


def tensors(path):
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            raise SystemExit('not a GGUF file')
        (version,) = rd(f, 'I')
        if version not in (2, 3):
            raise ValueError(f'unsupported GGUF version {version}')
        n_tensors, n_kv = rd(f, 'QQ')
        alignment = 32
        for _ in range(n_kv):
            key = rstr(f)
            (t,) = rd(f, 'I')
            v = value(f, t)
            if key == 'general.alignment':
                alignment = v
        out = []
        for _ in range(n_tensors):
            name = rstr(f)
            (nd,) = rd(f, 'I')
            dims = rd(f, 'Q' * nd)
            (t,) = rd(f, 'I')
            (off,) = rd(f, 'Q')
            if t not in TYPES:
                raise SystemExit(f'{name}: unknown ggml type {t}')
            elems = 1
            for d in dims:
                elems *= d
            be, bb = TYPES[t]
            if not dims or any(d == 0 for d in dims) or dims[0] % be:
                raise ValueError(f'{name}: invalid quantized dimensions')
            out.append((name, off, elems // be * bb))
        if not isinstance(alignment, int) or alignment < 1 or alignment & (alignment - 1):
            raise ValueError('invalid GGUF alignment')
        data = (f.tell() + alignment - 1) // alignment * alignment
    result = [(n, data + o, b) for n, o, b in out]
    size = os.path.getsize(path)
    if any(o > size or b > size - o for _, o, b in result):
        raise ValueError('tensor beyond end of GGUF')
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--no-engram', action='store_true', help='omit Engram tables')
    ap.add_argument('--no-experts', action='store_true', help='do not interleave the routed experts')
    ap.add_argument('--policy', choices=('interleave', 'interleave-even', 'all'), default='interleave',
                    help='placement policy for every selected tensor (default: interleave)')
    a = ap.parse_args()
    ranges = []
    for name, off, size in tensors(a.model):
        if not a.no_experts and re.fullmatch(r'blk\.\d+\.ffn_(gate|up|down)_exps\.weight', name):
            ranges.append({'name': name, 'offset': off, 'bytes': size, 'policy': a.policy})
        m = re.fullmatch(r'blk\.(\d+)\.engram_embd\.weight', name)
        if m and not a.no_engram:
            ranges.append({'name': name, 'offset': off, 'bytes': size, 'policy': a.policy})
    ranges.sort(key=lambda r: r['offset'])
    json.dump({'schema': 'tierstream/stripe-ranges-v1', 'primary': os.path.basename(a.model),
               'size': os.path.getsize(a.model), 'ranges': ranges}, sys.stdout, indent=1)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
