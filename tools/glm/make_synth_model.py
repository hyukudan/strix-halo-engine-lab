"""Synthetic GLM-5.3-Flash EXL3 model directory for engine smoke tests before the real weights land.

Exact tensor names / shapes / dtypes of the EXL3 4.05 bpw pack (validate_exl3_pack.expected_manifest + the pinned
bits per role), random contents at small scales so activations stay finite: trellis words uniform int16 (every word
decodes to a valid mul1 codebook value), suh/svh ~ +-0.02 F16, plain F16/BF16 weights N(0, 0.02), norms 1.0,
mHC fn N(0, 1e-3) with base/scale from the real layer-3 values' order of magnitude, A_log 0, dt_bias 0, router N(0, 0.02),
bias 0, embedding N(0, 0.5). Layers [0, N) (+ optional MTP layer 45) and the globals; NOT a model: smoke tests only.
usage: make_synth_model.py CONFIG_JSON OUT_DIR [--layers N] [--mtp] [--seed S]
"""
import argparse, json, struct, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from validate_exl3_pack import expected_manifest  # noqa: E402

BITS = {  # identity-4.05bpw.json roles
    'experts': 4, 'mlp.gate_proj': 5, 'mlp.up_proj': 5, 'mlp.down_proj': 5, 'eh_proj': 5,
}
SHARD_LIMIT = 4 << 30


def bits_of(name):
    if '.experts.' in name:
        return 4
    for k, b in BITS.items():
        if name.endswith(k) and 'shared' not in name:
            return b
    return 6


def f16(a):
    return np.asarray(a, np.float16)


def bf16(a):
    u = np.asarray(a, np.float32).view(np.uint32).astype(np.uint64)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16)).astype(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('out')
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--mtp', action='store_true')
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    nl = cfg['text_config']['num_hidden_layers']
    man = expected_manifest(cfg)
    keep = lambda k: ('.layers.' not in k) or any(f'.layers.{l}.' in k for l in range(a.layers)) or (a.mtp and f'.layers.{nl}.' in k)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tensors = []  # (name, dtype str, shape, bytes)
    for key, spec in man.items():
        if not keep(key):
            continue
        if spec[0] == 'exl3':
            _, k_in, n_out = spec
            b = bits_of(key)
            tensors.append((key + '.trellis', 'I16', [k_in // 16, n_out // 16, 16 * b], lambda s: rng.integers(-32768, 32767, s, dtype=np.int16)))
            tensors.append((key + '.suh', 'F16', [k_in], lambda s: f16(rng.choice([-0.02, 0.02], s))))
            tensors.append((key + '.svh', 'F16', [n_out], lambda s: f16(rng.choice([-0.02, 0.02], s))))
            tensors.append((key + '.mul1', 'I32', [], lambda s: np.array([0x83DCD12D - (1 << 32)], np.int32)))
            continue
        _, dtypes, shape = spec
        dt = sorted(dtypes)[0] if len(dtypes) > 1 else next(iter(dtypes))
        if key.endswith(('hc_attn_fn', 'hc_ffn_fn')):
            dt, gen = 'F32', (lambda s: rng.normal(0, 1e-3, s).astype(np.float32))
        elif key.endswith(('hc_attn_base', 'hc_ffn_base')):
            dt, gen = 'F32', (lambda s: rng.normal(0, 0.5, s).astype(np.float32))
        elif key.endswith(('hc_attn_scale', 'hc_ffn_scale')):
            dt, gen = 'F32', (lambda s: np.full(s, 0.5, np.float32))
        elif key.endswith(('A_log', 'dt_bias', 'e_score_correction_bias')):
            dt, gen = 'F32', (lambda s: np.zeros(s, np.float32))
        elif key.endswith('index_kpool_compress_ape'):
            dt, gen = 'F32', (lambda s: rng.normal(0, 0.1, s).astype(np.float32))
        elif '.indexer.k_norm.' in key:
            dt, gen = 'F16', (lambda s: f16(np.ones(s)) if key.endswith('weight') else f16(np.zeros(s)))
        elif 'norm' in key and len(shape) == 1:
            dt = 'BF16' if 'BF16' in dtypes else 'F16'
            gen = (lambda s, d=dt: bf16(np.ones(s)) if d == 'BF16' else f16(np.ones(s)))
        elif 'embed_tokens' in key:
            dt, gen = 'BF16', (lambda s: bf16(rng.normal(0, 0.5, s)))
        elif key.endswith('conv1d.weight') or key.endswith('o_norm.weight'):
            dt, gen = 'BF16', (lambda s: bf16(rng.normal(0, 0.3, s)))
        else:
            dt = 'F16' if 'F16' in dtypes else sorted(dtypes)[0]
            gen = (lambda s, d=dt: f16(rng.normal(0, 0.02, s)) if d == 'F16' else bf16(rng.normal(0, 0.02, s)))
        tensors.append((key, dt, list(shape), gen))
    size = {'I16': 2, 'F16': 2, 'BF16': 2, 'F32': 4, 'I32': 4}
    shards, cur, cur_bytes = [], [], 0
    for t in tensors:
        n = size[t[1]] * int(np.prod(t[2])) if t[2] else size[t[1]]
        if cur and cur_bytes + n > SHARD_LIMIT:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append((t, n))
        cur_bytes += n
    if cur:
        shards.append(cur)
    wmap = {}
    for si, sh in enumerate(shards):
        fname = f'model-{si + 1:05d}-of-{len(shards):05d}.safetensors'
        header, off = {}, 0
        for (name, dt, shape, _), n in sh:
            header[name] = {'dtype': dt, 'shape': shape, 'data_offsets': [off, off + n]}
            off += n
            wmap[name] = fname
        hb = json.dumps(header, separators=(',', ':')).encode()
        hb += b' ' * (-len(hb) % 8)
        with open(out / fname, 'wb') as f:
            f.write(struct.pack('<Q', len(hb)) + hb)
            for (name, dt, shape, gen), n in sh:
                arr = np.ascontiguousarray(gen(tuple(shape) if shape else (1,)))
                data = arr.tobytes()[:n] if shape else arr.tobytes()[:size[dt]]
                assert len(data) == n, (name, len(data), n)
                f.write(data)
        print(fname, off, flush=True)
    (out / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {'synthetic': True}, 'weight_map': wmap}))
    (out / 'config.json').write_text(json.dumps(cfg))
    print('tensors', len(tensors), 'shards', len(shards))


if __name__ == '__main__':
    main()
