#!/usr/bin/env python3
"""R1b identity experiment: end logits of the device runs (strix-glm5-rank --logits-out, both rank slices) for the
identities R1 and R1b vs the full-depth float64 O-math logits of the same position (om_run.py logits-last1.f64).
Prints per identity: rel-L2, max |d|, KL(P64 || P), top-1 equal, top-10 overlap, and R1b vs R1.
usage: r1b_endlogits.py OMATH.f64 PREFIX_R1 PREFIX_R1B [PROMPT_INDEX]   (files PREFIX-p<i>-r0.f32 / -r1.f32)
"""
import sys
import numpy as np


def logsm(x):
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def load(prefix, i):
    return np.concatenate([np.fromfile(f'{prefix}-p{i}-r{r}.f32', dtype=np.float32) for r in (0, 1)]).astype(np.float64)


def main():
    om = np.fromfile(sys.argv[1], dtype=np.float64)
    i = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    runs = {'R1': load(sys.argv[2], i), 'R1b': load(sys.argv[3], i)}
    if om.size != runs['R1'].size:
        raise SystemExit(f'size mismatch: omath {om.size} vs device {runs["R1"].size}')
    lo = logsm(om)
    po = np.exp(lo)
    top10 = set(np.argsort(-om)[:10])
    for k, x in runs.items():
        d = x - om
        kl = float((po * (lo - logsm(x))).sum())
        ov = len(top10 & set(np.argsort(-x)[:10]))
        print(f'{k:4s} vs float64: rel-L2 {np.linalg.norm(d) / np.linalg.norm(om):.3e} max |d| {np.abs(d).max():.3e} '
              f'KL(P64||P) {kl:.3e} nats, top-1 {"equal" if x.argmax() == om.argmax() else "DIFFERS"} '
              f'({int(x.argmax())} vs {int(om.argmax())}), top-10 overlap {ov}/10')
    d = runs['R1b'] - runs['R1']
    print(f'R1b vs R1: rel-L2 {np.linalg.norm(d) / np.linalg.norm(runs["R1"]):.3e} max |d| {np.abs(d).max():.3e}')


if __name__ == '__main__':
    main()
