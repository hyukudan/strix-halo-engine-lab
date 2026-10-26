#!/usr/bin/env python3
"""R1b identity experiment: export the plain tensors of the R1b allowlist for a set of layers, for
hip_glm_r1b_tests' per-layer error report (R1 / R1b vs float64). Per layer OUT/L<l>/:
  hc_attn_fn.bin hc_ffn_fn.bin (f32 [24][16384]), hc_*_base.bin (f32 [24]), hc_*_scale.bin (f32 [3]),
  in_norm.bin post_norm.bin (f32 from BF16, exact), MoE layers: router.bin (f16 [288][4096]) bias.bin (f32 [288]),
  KDA layers: f_a.bin g_a.bin (f16 [128][4096]), b.bin (f16 [64][4096], all heads; rank r uses rows r*32..).
usage: export_r1b_cases.py MODEL_DIR OUT_DIR L1,L2,...
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cpu_reference import SafeDir, P, is_kda, is_moe  # noqa: E402


def main():
    model, out, layers = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(',')]
    s = SafeDir(model)

    def get(k, dt):
        if not s.has(k):
            raise SystemExit(f'missing tensor {k}')  # fail closed
        f, base, v = s.idx[k]
        a, _ = v['data_offsets']
        arr = np.memmap(f, dtype=s.DT[v['dtype']], mode='r', offset=base + a, shape=tuple(v['shape']) if v['shape'] else (1,))
        if dt == 'f16':
            if v['dtype'] != 'F16':
                raise SystemExit(f'{k}: dtype {v["dtype"]}, expected F16')
            return np.ascontiguousarray(arr).view(np.uint16)
        if v['dtype'] == 'BF16':
            return (np.asarray(arr).astype(np.uint32) << 16).view(np.float32)
        if v['dtype'] != 'F32':
            raise SystemExit(f'{k}: dtype {v["dtype"]}, expected F32/BF16')
        return np.ascontiguousarray(arr, dtype=np.float32)

    for L in layers:
        d = os.path.join(out, f'L{L}')
        os.makedirs(d, exist_ok=True)
        pre = f'{P}layers.{L}.'
        items = []
        for site in ('attn', 'ffn'):
            for part in ('fn', 'base', 'scale'):
                items.append((f'hc_{site}_{part}', pre + f'hc_{site}_{part}', 'f32'))
        items += [('in_norm', pre + 'input_layernorm.weight', 'f32'), ('post_norm', pre + 'post_attention_layernorm.weight', 'f32')]
        if is_moe(L):
            items += [('router', pre + 'mlp.gate.weight', 'f16'), ('bias', pre + 'mlp.gate.e_score_correction_bias', 'f32')]
        if is_kda(L):
            items += [('f_a', pre + 'self_attn.f_a_proj.weight', 'f16'), ('g_a', pre + 'self_attn.g_a_proj.weight', 'f16'),
                      ('b', pre + 'self_attn.b_proj.weight', 'f16')]
        with open(os.path.join(d, 'manifest.txt'), 'w') as m:
            for name, key, dt in items:
                a = get(key, dt)
                a.tofile(os.path.join(d, name + '.bin'))
                m.write(f'{name} {key} {dt} {a.size}\n')
        print(f'L{L}: {len(items)} tensors -> {d} (kda {is_kda(L)}, moe {is_moe(L)})')


if __name__ == '__main__':
    main()
