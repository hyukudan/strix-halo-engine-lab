#!/usr/bin/env python3
"""R1b identity experiment: per-layer residual error of the device (strix-glm5-rank --g3 TOKENS --layers N, world 1, R = 1
teacher forcing) in both identities vs the float64 CPU oracle dump (cpu_reference.py, omath-real-4.npz: t<k>.L<l> =
residual [4][4096] after layer l for token k). Prints per layer: rel-L2 (over all tokens and streams) of R1 and R1b vs
float64, max over tokens, and R1b vs R1.
usage: r1b_g3_layers.py NPZ PREFIX_R1|FILE.f32 PREFIX_R1B|FILE.f32 NLAYERS   (PREFIX-L<N>.f32 = [T][4][4096] after N layers,
       or one --g3-sweep file [T][NLAYERS][4][4096])
"""
import sys
import numpy as np


def main():
    npz, p1, pb, nl = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    ref = np.load(npz)
    T = len({k.split('.')[0] for k in ref.files})
    for n in range(1, nl + 1):
        o = np.stack([ref[f't{t}.L{n - 1}'] for t in range(T)]).reshape(T, -1)
        out = {}
        for tag, p in (('R1', p1), ('R1b', pb)):
            try:
                if p.endswith('.f32'):  # --g3-sweep file [T][NL][4][4096]
                    out[tag] = np.fromfile(p, dtype=np.float32).reshape(T, nl, -1)[:, n - 1, :].astype(np.float64)
                else:
                    out[tag] = np.fromfile(f'{p}-L{n}.f32', dtype=np.float32).reshape(T, -1).astype(np.float64)
            except (OSError, ValueError) as e:
                print(f'layer {n - 1}: {tag} missing ({e})')
        if len(out) < 2:
            continue
        msg = f'layer {n - 1} (after {n} layers):'
        for tag in ('R1', 'R1b'):
            d = out[tag] - o
            rel = np.linalg.norm(d) / np.linalg.norm(o)
            rmax = max(np.linalg.norm(d[t]) / np.linalg.norm(o[t]) for t in range(T))
            msg += f' {tag} vs fp64 rel-L2 {rel:.3e} (max token {rmax:.3e});'
        d = out['R1b'] - out['R1']
        msg += f' R1b vs R1 rel-L2 {np.linalg.norm(d) / np.linalg.norm(out["R1"]):.3e}, bitwise equal tokens {sum(np.array_equal(out["R1"][t], out["R1b"][t]) for t in range(T))}/{T}'
        print(msg)


if __name__ == '__main__':
    main()
