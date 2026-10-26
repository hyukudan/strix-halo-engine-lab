"""Gate D (b), CPU side: digests of the logical FP16 Q of every EXL3 rank slice the engine loads, decoded with the
strix-trellis library's own independent Python codec (dequantize_rotated), for comparison with strix-glm5-gate-d-digest
(the production device decode primitive over the bytes staged by the engine's own Lib::stage_out / stage_in).

Inventory = the split contract (split-4.05bpw.json): every 'exl3' tensor of the selected layers (+ 'head' = lm_head),
for each rank and each range: 'out' -> Q[:, a:b], 'in' -> Q[a:b, :], 'replicate' -> Q. --expect N is required and the
run fails unless exactly N unique items are written.
Digest of a slice Q_s [R, C] (local coordinates): band sums B_t = sum over the 16 rows of tile row t and all C columns of
splitmix64(((r * C + c) << 16) | q) mod 2^64; digest = FNV-1a 64 over B_0..B_{R/16-1} (8 bytes little-endian each).
A single changed element always changes its band sum (splitmix64 is a bijection); the compared value is the 64-bit
FNV of the bands, which can still collide: probabilistic evidence, not a proof. Every item's '.mul1' must be 0x83DCD12D.
usage: gate_d_digest.py MODEL_DIR TRELLIS_PY_DIR SPLIT_JSON OUT.tsv --layers 0,1,2,3,4,5,45,head --expect N [--jobs J]
Output: '<prefix> <rank> <mode> <a> <b> <bits> <R> <C> <digest hex16>' sorted.
"""
import argparse, json, re, struct, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

CHUNK_TILES = 256  # output tiles per decode chunk (bounds memory for lm_head)


def splitmix64(z):
    with np.errstate(over='ignore'):
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def fnv_bands(bands):
    h = 14695981039346656037
    for b in bands:
        for byte in int(b).to_bytes(8, 'little'):
            h = ((h ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def headers(model):
    idx = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    files = set(idx.values()) | {p.name for p in model.glob('*.safetensors')}  # + unlisted shards (mtp.safetensors)
    out = {}
    for f in sorted(files):
        with open(model / f, 'rb') as fh:
            n = struct.unpack('<Q', fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != '__metadata__':
                if k in out:
                    raise ValueError(f'tensor defined twice: {k}')
                out[k] = (f, 8 + n + v['data_offsets'][0], v['data_offsets'][1] - v['data_offsets'][0], v['dtype'], v['shape'])
    return out


def slice_digests(job):
    """One tensor: digests of all its distinct (row range, col range) slices. job = (model, trellis_py, prefix, hdr, slices)."""
    model, trellis_py, prefix, hdr, mul1, slices = job
    sys.path.insert(0, trellis_py)
    from strix_trellis.codec import dequantize_rotated
    f, off, nb, dt, shape = hdr
    if dt != 'I16' or len(shape) != 3 or shape[2] % 16 or not shape[0] or not shape[1] or nb != shape[0] * shape[1] * shape[2] * 2:
        raise ValueError(f'bad trellis {prefix}: {dt} {shape} {nb}')
    mf, moff, mnb, mdt, mshape = mul1
    mv = np.fromfile(Path(model) / mf, dtype=np.int32, count=1, offset=moff)
    if mdt != 'I32' or mnb != 4 or mv.size != 1 or int(mv[0]) & 0xFFFFFFFF != 0x83DCD12D:
        raise ValueError(f'codebook marker is not mul1: {prefix}')
    raw = np.fromfile(Path(model) / f, dtype=np.int16, count=nb // 2, offset=off).reshape(shape)
    if raw.size * 2 != nb:
        raise ValueError(f'short read {prefix}')
    out = {}
    for (r0, r1, c0, c1) in slices:
        R, C = r1 - r0, c1 - c0
        if R <= 0 or C <= 0 or R % 16 or C % 16 or r0 % 16 or c0 % 16 or r1 > shape[0] * 16 or c1 > shape[1] * 16:
            raise ValueError(f'bad slice {prefix} {r0}:{r1} {c0}:{c1}')
        if R * C >= 1 << 47:
            raise ValueError(f'slice too large for the digest key {prefix}')
        bands = np.zeros(R // 16, dtype=np.uint64)
        for t0 in range(c0 // 16, c1 // 16, CHUNK_TILES):
            t1 = min(t0 + CHUNK_TILES, c1 // 16)
            q = dequantize_rotated(raw[r0 // 16:r1 // 16, t0:t1, :]).view(np.uint16).astype(np.uint64)  # [R, w]
            cols = np.arange(t0 * 16 - c0, t1 * 16 - c0, dtype=np.uint64)
            rows = np.arange(R, dtype=np.uint64)
            key = ((rows[:, None] * np.uint64(C) + cols[None, :]) << np.uint64(16)) | q
            with np.errstate(over='ignore'):
                bands += splitmix64(key).reshape(R // 16, -1).sum(axis=1, dtype=np.uint64)
        out[(r0, r1, c0, c1)] = (shape[2] // 16, R, C, fnv_bands(bands))
    return prefix, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('trellis_py')
    ap.add_argument('split')
    ap.add_argument('out')
    ap.add_argument('--layers', required=True)
    ap.add_argument('--expect', type=int, required=True)
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    model = Path(a.model)
    hs = headers(model)
    sel = set(a.layers.split(','))
    split = json.loads(Path(a.split).read_text())['tensors']
    items, jobs = [], {}
    for prefix, v in sorted(split.items()):
        if v['kind'] != 'exl3':
            continue
        m = re.search(r'\.layers\.(\d+)\.', prefix)
        if not ((m and m.group(1) in sel) or (prefix == 'lm_head' and 'head' in sel)):
            continue
        tr = hs[prefix + '.trellis']
        I, N = tr[4][0] * 16, tr[4][1] * 16
        for rank, ranges in enumerate(v['ranges']):
            for (ra, rb) in ranges:
                if v['mode'] == 'replicate':
                    s, ab = (0, I, 0, N), (0, N)
                elif v['mode'] == 'out':
                    s, ab = (0, I, ra, rb), (ra, rb)
                elif v['mode'] == 'in':
                    s, ab = (ra, rb, 0, N), (ra, rb)
                else:
                    raise ValueError(f'unknown mode {prefix} {v["mode"]}')
                items.append((prefix, rank, v['mode'], ab, s))
                jobs.setdefault(prefix, set()).add(s)
    keys = {(p, r, m, ab) for p, r, m, ab, _ in items}
    if len(keys) != len(items) or len(items) != a.expect:
        sys.exit(f'inventory {len(items)} items ({len(keys)} unique) != expected {a.expect}')
    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        work = [(str(model), a.trellis_py, p, hs[p + '.trellis'], hs[p + '.mul1'], sorted(s)) for p, s in jobs.items()]
        for i, (p, d) in enumerate(ex.map(slice_digests, work, chunksize=4)):
            res[p] = d
            if i % 100 == 0:
                print(f'{i}/{len(work)} tensors {time.time() - t0:.0f} s', flush=True)
    lines = []
    for p, r, m, (ab0, ab1), s in items:
        bits, R, C, d = res[p][s]
        lines.append(f'{p} {r} {m} {ab0} {ab1} {bits} {R} {C} {d:016x}\n')
    lines.sort()
    Path(a.out).write_text(''.join(lines))
    print(f'done {len(lines)} items ({len(jobs)} tensors) {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
