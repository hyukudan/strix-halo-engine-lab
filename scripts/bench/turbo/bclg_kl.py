#!/usr/bin/env python3
"""KL(ref||cand), top-1 agreement and dNLL between two STRIX_BITCALC_LOGITS_OUT files (records 'BCLG', full vocabulary).

  bclg_kl.py REF CAND [CAND ...]    per cand: records compared (step-aligned), KL mean/max, top-1 %, dNLL on the ref token
Record 0 is the prefill's last row (the prompt-conditioned distribution); later records are decode steps and are
comparable only while both runs emitted the same tokens (stop at the first step whose emitted token differs).
"""
import struct, sys
import numpy as np


def records(path):
    with open(path, "rb") as f:
        while True:
            h = f.read(24)
            if len(h) < 24:
                return
            magic, step, pos, argmax, emitted, n = struct.unpack("<6I", h)
            assert magic == 0x42434C47, path
            yield step, argmax, emitted, np.frombuffer(f.read(4 * n), np.float32).astype(np.float64)


def lse(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())


def main():
    ref = list(records(sys.argv[1]))
    for cp in sys.argv[2:]:
        cand = list(records(cp))
        kl, t1, dn = [], 0, []
        n = 0
        for (s, a, e, lr), (s2, a2, e2, lc) in zip(ref, cand):
            lpr, lpc = lr - lse(lr), lc - lse(lc)
            kl.append(float((np.exp(lpr) * (lpr - lpc)).sum()))
            t1 += int(a == a2)
            dn.append(float(-lpc[e] + lpr[e]))
            n += 1
            if e != e2:
                break
        if not n:
            print(cp, "no records"); continue
        print("%s rows=%d KL mean %.5f max %.5f top1 %.1f%% dNLL %+.5f" % (cp, n, np.mean(kl), max(kl), 100 * t1 / n, np.mean(dn)))


if __name__ == "__main__":
    main()
