#!/usr/bin/env python3
"""Combine the 'ab' lines of strix-glm5-rank --ab (rank 0 + rank 1 logs) into R1 vs R1b statistics per position and per
prompt: KL(P_R1 || P_R1b) (full vocab, from the per-rank softmax partials), top-1 and top-5 agreement (teacher-forced on
R1's tokens), route flips per layer, and the free-run greedy agreement / first divergence (ab-summary lines).
usage: ab_stats.py RANK0.log RANK1.log [NAMES]
"""
import math, re, sys
from collections import defaultdict


def parse(path):
    rows = {}
    summ = {}
    for line in open(path, errors='replace'):
        if line.startswith('ab rank '):
            m = re.match(r'ab rank (\d+) p (\d+) i (\d+) tokA (\d+) tokB (\d+) ma (\S+) sa (\S+) mb (\S+) sb (\S+) t (\S+) flips (\d+)/(\d+) \[([^\]]*)\] topA(.*) topB(.*)$', line.strip())
            if not m:
                continue
            g = m.groups()
            key = (int(g[1]), int(g[2]))
            ta = [(int(x.split(':')[0]), float(x.split(':')[1])) for x in g[13].split()]
            tb = [(int(x.split(':')[0]), float(x.split(':')[1])) for x in g[14].split()]
            rows[key] = dict(tokA=int(g[3]), tokB=int(g[4]), ma=float(g[5]), sa=float(g[6]), mb=float(g[7]), sb=float(g[8]), t=float(g[9]),
                             flips=int(g[10]), moe=int(g[11]), fl=[int(x) for x in g[12].split(',') if x], ta=ta, tb=tb)
        elif line.startswith('ab-summary rank '):
            m = re.search(r' p (\d+) prompt (\d+) tokens, N (\d+): TF top-1 .*? (\d+)/\d+, free-run equal (\d+)/\d+, first divergence (\S+);', line)
            if m:
                summ[int(m.group(1))] = dict(plen=int(m.group(2)), n=int(m.group(3)), tf=int(m.group(4)), eq=int(m.group(5)), first=m.group(6))
    return rows, summ


def lse(m, s):
    return m + math.log(s)


def main():
    r0, s0 = parse(sys.argv[1])
    r1, _ = parse(sys.argv[2])
    names = sys.argv[3].split(',') if len(sys.argv) > 3 else None
    per = defaultdict(list)
    flips_layer = defaultdict(int)
    tot_pos = 0
    for key in sorted(r0):
        if key not in r1:
            continue
        a, b = r0[key], r1[key]
        Ma = max(a['ma'], b['ma']); Sa = a['sa'] * math.exp(a['ma'] - Ma) + b['sa'] * math.exp(b['ma'] - Ma); La = Ma + math.log(Sa)
        Mb = max(a['mb'], b['mb']); Sb = a['sb'] * math.exp(a['mb'] - Mb) + b['sb'] * math.exp(b['mb'] - Mb); Lb = Mb + math.log(Sb)
        kl = math.exp(a['ma'] - La) * a['t'] + math.exp(b['ma'] - La) * b['t'] - La + Lb
        ta = sorted(a['ta'] + b['ta'], key=lambda x: (-x[1], x[0]))[:5]
        tb = sorted(a['tb'] + b['tb'], key=lambda x: (-x[1], x[0]))[:5]
        top1 = ta[0][0] == tb[0][0]
        top5 = len({x[0] for x in ta} & {x[0] for x in tb})
        per[key[0]].append((kl, top1, top5, a['flips'], a['moe']))
        for L in a['fl']:
            flips_layer[L] += 1
        tot_pos += 1
    allkl, all1, all5, allf, allm = [], 0, 0, 0, 0
    print('prompt | len | N | KL mean / p99 / max (nats) | TF top-1 agree | top-5 overlap mean | route flips (layer-positions) | free-run equal | first divergence')
    for p in sorted(per):
        v = per[p]
        kls = sorted(x[0] for x in v)
        s = s0.get(p, {})
        name = names[p] if names and p < len(names) else f'p{p}'
        f = sum(x[3] for x in v); mo = sum(x[4] for x in v)
        print(f'{name} | {s.get("plen", "?")} | {len(v)} | {sum(kls) / len(kls):.2e} / {kls[min(len(kls) - 1, int(0.99 * len(kls)))]:.2e} / {kls[-1]:.2e} | '
              f'{sum(x[1] for x in v)}/{len(v)} | {sum(x[2] for x in v) / len(v):.2f} | {f}/{mo} | {s.get("eq", "?")}/{s.get("n", "?")} | {s.get("first", "?")}')
        allkl += kls; all1 += sum(x[1] for x in v); all5 += sum(x[2] for x in v); allf += f; allm += mo
    allkl.sort()
    if allkl:
        print(f'POOLED: {len(per)} prompts, {tot_pos} positions: KL mean {sum(allkl) / len(allkl):.3e} p99 {allkl[int(0.99 * (len(allkl) - 1))]:.3e} max {allkl[-1]:.3e}; '
              f'TF top-1 agree {all1}/{tot_pos} ({100.0 * all1 / tot_pos:.2f} %); top-5 overlap {all5 / tot_pos:.3f}/5; route flips {allf}/{allm} layer-positions ({100.0 * allf / max(allm, 1):.3f} %)')
        firsts = [s0[p]['first'] for p in s0]
        nd = sum(1 for x in firsts if x == 'none')
        print(f'free-run: {nd}/{len(firsts)} prompts identical over N; first divergence positions: {sorted(int(x) for x in firsts if x != "none")}')
        print('route flips by layer: ' + ', '.join(f'L{L}:{c}' for L, c in sorted(flips_layer.items())))


if __name__ == '__main__':
    main()
