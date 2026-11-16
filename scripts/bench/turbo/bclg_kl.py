#!/usr/bin/env python3
"""KL(ref||cand), top-1 agreement and dNLL between two STRIX_BITCALC_LOGITS_OUT files (records 'BCLG', full vocabulary).

  bclg_kl.py REF CAND [CAND ...]    per cand: records compared (step-aligned), KL mean/max, top-1 %, dNLL on the ref token
Record 0 is the prefill's last row (the prompt-conditioned distribution); later records are decode steps and are
comparable only while both runs emitted the same tokens (stop at the first step whose emitted token differs).
"""
import struct, sys
from itertools import zip_longest
import numpy as np


def records(path):
    with open(path, "rb") as f:
        while True:
            h = f.read(24)
            if not h:
                return
            if len(h) != 24:
                raise ValueError(f"{path}: incomplete BCLG header")
            magic, step, pos, argmax, emitted, n = struct.unpack("<6I", h)
            if magic != 0x42434C47 or not n or argmax >= n or emitted >= n:
                raise ValueError(f"{path}: invalid BCLG header")
            payload = f.read(4 * n)
            if len(payload) != 4 * n:
                raise ValueError(f"{path}: incomplete BCLG payload")
            yield step, pos, argmax, emitted, np.frombuffer(payload, '<f4').astype(np.float64)


def lse(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())


def main():
    ref = list(records(sys.argv[1]))
    for cp in sys.argv[2:]:
        cand = list(records(cp))
        kl, t1, dn = [], 0, []
        n = 0
        for rr, cr in zip_longest(ref, cand):
            if rr is None or cr is None:
                raise ValueError(f"{cp}: unmatched records before token divergence")
            s, pos, a, e, lr = rr
            s2, pos2, a2, e2, lc = cr
            if (s, pos, len(lr)) != (s2, pos2, len(lc)):
                raise ValueError(f"{cp}: step/position/vocabulary mismatch")
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
