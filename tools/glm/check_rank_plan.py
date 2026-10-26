"""CPU check of the C++ TP2 rank-slice plan (src/glm/load/glm_rank_plan.cpp) on real EXL3 tensors (design notes).

For every split entry present in the fixture, both ranks: the bytes the C++ plan's pieces select from the file must
equal, bit for bit, the rank slice built with numpy from the split JSON ranges in test_split.py's convention
(out: trellis[:, b/16:e/16, :] per range, suh whole, svh[b:e]; in: trellis[b/16:e/16], suh[b:e], svh whole;
mul1 whole; plain rows: w[b:e] per range; replicate: whole).
usage: check_rank_plan.py MODEL_DIR SPLIT_JSON PIECES_JSON
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

model, split_path, pieces_path = sys.argv[1:4]
split = json.loads(Path(split_path).read_text())['tensors']
plan = json.loads(Path(pieces_path).read_text())
DT = {'F32': np.float32, 'F16': np.float16, 'BF16': np.uint16, 'I16': np.int16, 'U16': np.uint16, 'I32': np.int32, 'U8': np.uint8}

tensors, files = {}, plan['files']
for fi, f in enumerate(files):
    with open(f, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        h = json.loads(fh.read(n))
    for k, v in h.items():
        if k == '__metadata__':
            continue
        b, e = v['data_offsets']
        tensors[k] = (fi, 8 + n + b, e - b, v['dtype'], v['shape'])
maps = [np.memmap(f, dtype=np.uint8, mode='r') for f in files]


def raw(name):
    fi, off, nb, dt, shape = tensors[name]
    return np.frombuffer(maps[fi][off:off + nb].tobytes(), dtype=DT[dt]).reshape(shape)


def reference(g, p, r):
    rr = p['ranges'][r]
    if p['kind'] == 'plain':
        w = raw(g)
        if p['mode'] == 'replicate':
            return w.tobytes()
        return b''.join(np.ascontiguousarray(w[b:e]).tobytes() for b, e in rr)
    tr, suh, svh = raw(g + '.trellis'), raw(g + '.suh'), raw(g + '.svh')
    mul = raw(g + '.mul1').tobytes() if g + '.mul1' in tensors else b''
    if p['mode'] == 'replicate':
        return tr.tobytes() + suh.tobytes() + svh.tobytes() + mul
    if p['mode'] == 'out':
        t = np.concatenate([tr[:, b // 16:e // 16, :] for b, e in rr], axis=1)
        return np.ascontiguousarray(t).tobytes() + suh.tobytes() + b''.join(svh[b:e].tobytes() for b, e in rr) + mul
    t = np.concatenate([tr[b // 16:e // 16] for b, e in rr], axis=0)
    return np.ascontiguousarray(t).tobytes() + b''.join(suh[b:e].tobytes() for b, e in rr) + svh.tobytes() + mul


fails, checked, modes = [], 0, {}
for r in (0, 1):
    for g, sl in plan['ranks'][r].items():
        got = b''.join(maps[f][s:s + n].tobytes() for f, s, n in sl['pieces'])
        ref = reference(g, split[g], r)
        checked += 1
        key = (split[g]['kind'], split[g]['mode'])
        modes[key] = modes.get(key, 0) + 1
        if got != ref or len(got) != sl['bytes']:
            fails.append(f'r{r} {g}')
            print(f'FAIL r{r} {g}: {len(got)} vs {len(ref)} B')
print('modes checked:', {f'{k[0]}/{k[1]}': v for k, v in sorted(modes.items())})
print(f'check_rank_plan: {checked} slices, {len(fails)} failures: {"PASS" if not fails and checked else "FAIL"}')
sys.exit(1 if fails or not checked else 0)
