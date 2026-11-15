#!/usr/bin/env python3
"""Summary of a ds41_prefill_matrix.sh output dir: per length cold (first request) and warm (last request) prefill,
identity (sha256 / logits fnv / next token over all reps), the --timing accounting when present, and the GPU SM clock
(mean of busy samples, max) with GPU clocks per token = busy x clock x prefill / tokens.

  python ds41_prefill_summary.py OUT_DIR
"""
import collections
import json
import os
import sys


def samples(out):
    s = []
    p = os.path.join(out, 'smi.samples')
    if os.path.exists(p):
        for line in open(p, encoding='utf-8', errors='replace'):
            f = line.split()
            if len(f) == 2:
                try:
                    t = float(f[0])
                    mem, util, clk = (float(x) for x in f[1].split(','))
                    s.append((t, mem, util, clk))
                except ValueError:
                    pass
    return s


def marks(out):
    m = {}
    p = os.path.join(out, 'marks.txt')
    if os.path.exists(p):
        for line in open(p, encoding='utf-8'):
            f = line.split()
            if len(f) >= 4:
                m.setdefault(int(f[1]), {})[f[2]] = float(f[3])
    return m


def fmt_acct(a):
    r, d = a.get('run', {}), a.get('dma', {})
    keys = ('host_setup_s', 'embed_s', 'compute_s', 'exposed_wait_s', 'acquire_wait_s', 'gaps_s', 'head_s',
            'host_tail_s', 'engram_join_s', 'unattributed_s')
    run = ' '.join(f"{k[:-2]} {r.get(k, 0):.3f}" for k in keys)
    dma = (f"dma run {d.get('run_gb', 0):.1f} GB busy {d.get('copy_busy_s', 0):.2f} s ({d.get('run_gbps', 0):.1f} GB/s) "
           f"preload {d.get('preload_gb', 0):.1f} GB {d.get('preload_s', 0):.2f} s hidden {d.get('hidden_s', 0):.2f} s")
    return f"    run: wall {r.get('wall_s', 0):.3f} device {r.get('device_s', 0):.3f} | {run}\n    {dma}"


def main():
    out = sys.argv[1]
    reqs = [json.loads(x) for x in open(os.path.join(out, 'requests.jsonl'), encoding='utf-8') if x.strip()]
    by = collections.OrderedDict()
    for j in reqs:
        by.setdefault(j.get('tokens', 0), []).append(j)
    smp, mk = samples(out), marks(out)
    for n, xs in by.items():
        ids = {(x.get('sha256'), x.get('logits_fnv'), x.get('next_token')) for x in xs}
        pf = [x.get('seconds', {}).get('prefill', 0.0) for x in xs]
        st = {x.get('http_status') for x in xs}
        sha = str(next(iter(ids))[0])[:16]
        line = (f"L={n} reps {len(xs)} status {sorted(st)} cold prefill {pf[0]:.3f} s wall {xs[0]['wall_s']:.3f} | "
                f"warm prefill {pf[-1]:.3f} s wall {xs[-1]['wall_s']:.3f} = {n / pf[-1]:.0f} tok/s | "
                f"{'IDENTICAL' if len(ids) == 1 else 'DIFFER'} sha {sha} next {next(iter(ids))[2]}")
        if n in mk and 'begin' in mk[n] and 'end' in mk[n]:
            w = [s for s in smp if mk[n]['begin'] <= s[0] <= mk[n]['end']]
            busy = [s for s in w if s[2] > 0]
            if busy:
                clk = sum(s[3] for s in busy) / len(busy)
                frac = sum(s[2] for s in w) / (100.0 * len(w))
                line += (f" | SM {clk:.0f} MHz busy-mean, max {max(s[3] for s in w):.0f}, util {frac:.2f}, "
                         f"Mclk/tok(warm) {frac * clk * pf[-1] / n:.3f}")
        print(line)
        for tag, x in (('cold', xs[0]), ('warm', xs[-1])):
            if 'accounting' in x:
                print(f"  {tag}:")
                print(fmt_acct(x['accounting']))
    if smp:
        print(f"GPU{os.environ.get('GPU_INDEX', '')} peak MiB {max(s[1] for s in smp):.0f}, max SM MHz {max(s[3] for s in smp):.0f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
