#!/usr/bin/env python3
"""R1b identity experiment: compare two strix-glm5-rank decode runs (identity R1 = A, R1b = B) on the same prompts.
Per prompt: greedy token agreement (first divergence, % equal positions), and on the first generated token's logits
(full vocab = rank 0 slice ++ rank 1 slice, --logits-out files): rel-L2 |B - A| / |A|, max |d|, top-1 equal, KL(P_A || P_B).
usage: r1b_compare.py LOG_A_R0 LOG_B_R0 LOGITS_PREFIX_A LOGITS_PREFIX_B [NAMES]
"""
import os, sys
import numpy as np


def tokens(log):
    out = []
    for line in open(log, errors='replace'):
        if line.startswith('rank 0 tokens:'):
            out.append([int(t) for t in line.split(':', 1)[1].split()])
    return out


def logits(prefix, i):
    parts = [f'{prefix}-p{i}-r{r}.f32' for r in (0, 1)]
    if not all(os.path.exists(p) for p in parts):
        return None
    return np.concatenate([np.fromfile(p, dtype=np.float32) for p in parts]).astype(np.float64)


def main():
    la, lb, pa, pb = sys.argv[1:5]
    names = sys.argv[5].split(',') if len(sys.argv) > 5 else None
    ta, tb = tokens(la), tokens(lb)
    for i, (a, b) in enumerate(zip(ta, tb)):
        n = min(len(a), len(b))
        first = next((k for k in range(n) if a[k] != b[k]), None)
        eq = sum(1 for k in range(n) if a[k] == b[k])
        name = names[i] if names and i < len(names) else f'p{i}'
        msg = f'prompt {name}: tokens {n}, equal {eq} ({100.0 * eq / n:.1f} %), first divergence {first if first is not None else "none"}'
        A, B = logits(pa, i), logits(pb, i)
        if A is not None and B is not None:
            d = B - A
            sa = A - A.max()
            sb = B - B.max()
            lpa = sa - np.log(np.exp(sa).sum())
            lpb = sb - np.log(np.exp(sb).sum())
            kl = float((np.exp(lpa) * (lpa - lpb)).sum())
            msg += (f'; first-token logits rel-L2 {np.linalg.norm(d) / np.linalg.norm(A):.3e} max |d| {np.abs(d).max():.3e}'
                    f' top-1 {"equal" if A.argmax() == B.argmax() else "DIFFERS"} KL {kl:.3e} nats')
        print(msg)


if __name__ == '__main__':
    main()
