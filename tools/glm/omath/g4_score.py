#!/usr/bin/env python3
"""G4 quality scoring (design notes section 6.2, design notes section 2): device (or C-prod) logits vs O-math on the
scored positions of each G4 prompt. Inputs per prompt: the token file (T + 1 tokens), O-math logits (float64
[NS][154880], omath_layer.py --logits) and the candidate's two rank slices (float32 [NS][vocab_slice], rank 0 = ids
0..V/2-1, rank 1 = the rest; strix-glm5-rank --tf ... -r0.f32 / -r1.f32). Prints per prompt and pooled: KL(P_O || P_x)
mean / p99 (nats, float64 log-softmax over the full vocab), top-1 agreement with O-math (ties -> lower id) and top-1 vs
the real next token, each in % of the scored positions.

usage: g4_score.py 'NAME|TOKENS.u32|OMATH.f64|CAND-r0.f32|CAND-r1.f32' [...]
"""
import sys
import numpy as np

V = 154880


def log_softmax(x):
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def top1(x):
    return x.argmax(axis=1)  # numpy argmax returns the first (lowest) index of the maximum


def main():
    pooled = {'kl': [], 'agree': [], 'real': []}
    for spec in sys.argv[1:]:
        name, tok, om, r0, r1 = spec.split('|')
        toks = np.fromfile(tok, dtype=np.uint32)
        O = np.fromfile(om, dtype=np.float64)
        ns = O.size // V
        O = O.reshape(ns, V)
        a, b = np.fromfile(r0, dtype=np.float32), np.fromfile(r1, dtype=np.float32)
        if a.size + b.size != ns * V or a.size % ns or b.size % ns:
            raise SystemExit(f'{name}: logits sizes do not match {ns} x {V}')
        Xc = np.concatenate([a.reshape(ns, -1), b.reshape(ns, -1)], axis=1).astype(np.float64)
        lo, lx = log_softmax(O), log_softmax(Xc)
        kl = (np.exp(lo) * (lo - lx)).sum(axis=1)
        T = toks.size - 1
        real = toks[T - ns + 1:T + 1]                                   # targets of the scored positions T-ns .. T-1
        agree = top1(O) == top1(Xc)
        hit = top1(Xc) == real
        print(f'{name}: {ns} scored | KL mean {kl.mean():.5f} p99 {np.percentile(kl, 99):.5f} max {kl.max():.5f} nats | '
              f'top-1 vs O-math {100 * agree.mean():.2f} % | top-1 vs real {100 * hit.mean():.2f} % '
              f'(O-math vs real {100 * (top1(O) == real).mean():.2f} %)')
        pooled['kl'].append(kl)
        pooled['agree'].append(agree)
        pooled['real'].append(hit)
    kl = np.concatenate(pooled['kl'])
    print(f'pooled: {kl.size} scored | KL mean {kl.mean():.5f} p99 {np.percentile(kl, 99):.5f} | top-1 vs O-math '
          f'{100 * np.concatenate(pooled["agree"]).mean():.2f} % | top-1 vs real {100 * np.concatenate(pooled["real"]).mean():.2f} %')


if __name__ == '__main__':
    main()
