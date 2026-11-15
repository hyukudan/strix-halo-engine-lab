#!/usr/bin/env python3
"""hetpf summary (<= 60 lines): python summarize.py RUN.jsonl [--mon LABEL=DIR ...] [--smi PC.samples] [--link LINK.json]

Groups the het_ab.py records by (pair mode, arm). Per process the first request of an arm is "cold", the last "warm"; the
table gives the median over processes of each, never mixing them. A (RTX -> pair) is broken down into the service's
prefill / export / sha256 / transfer / import seconds (import = the pair's server time: sha, parse, queue, rank-1 relay,
device) plus the pair's first token, and decode tok/s of the tokens after it. R's TTFT is the service's queue + prefill
(its first token needs no export). GPU clocks per token = mean busy x mean clock x interval / tokens: the pair nodes from
the ticket's mon0/mon1.txt (--mon LABEL=DIR, LABEL = the record's label), the RTX from --smi ("epoch_ns, util, sm_mhz").
Continuations: common prefix of the generated ids with the reference (first exact B, else first fast B) and with the
first record of the same group.
"""
import argparse
import json
import re
import statistics as st
from collections import defaultdict


def load_mon(path):
    rows = []
    try:
        for line in open(path, errors='replace'):
            m = re.match(r'(\d+) busy=(\d+) sclk=.*?(\d+)Mhz', line)
            if m:
                rows.append((int(m.group(1)) / 1e9, float(m.group(2)) / 100.0, float(m.group(3))))
    except OSError:
        pass
    return rows


def load_smi(path):
    rows = []
    try:
        for line in open(path, errors='replace'):
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 3 and parts[0].isdigit():
                rows.append((int(parts[0]) / 1e9, float(parts[1]) / 100.0, float(parts[2])))
    except (OSError, ValueError):
        pass
    return rows


def cycles(rows, t0, t1):
    """G-cycles (busy x clock x time) and mean clock over [t0, t1]."""
    sel = [r for r in rows if t0 <= r[0] <= t1]
    if not sel or t1 <= t0:
        return None, None
    busy = st.mean(r[1] for r in sel)
    mhz = st.mean(r[2] for r in sel)
    return busy * mhz * 1e6 * (t1 - t0) / 1e9, mhz


def prefix(a, b):
    n = 0
    for x, y in zip(a or [], b or []):
        if x != y:
            break
        n += 1
    return n


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def f(x, d=3):
    return '-' if x is None else f'{x:.{d}f}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('jsonl')
    p.add_argument('--mon', action='append', default=[])
    p.add_argument('--smi')
    p.add_argument('--link')
    a = p.parse_args()
    recs = [json.loads(line) for line in open(a.jsonl) if line.strip()]
    mons = {}
    for m in a.mon:
        lab, d = m.split('=', 1)
        mons[lab] = (load_mon(d + '/mon0.txt'), load_mon(d + '/mon1.txt'))
    smi = load_smi(a.smi) if a.smi else []
    if a.link:
        L = json.load(open(a.link))
        print(f"link: rtt median {L['rtt_us']['median']:.0f} us (p90 {L['rtt_us']['p90']:.0f}); upload {L['upload']['median_MB_s']:.1f} MB/s, "
              f"download {L['download']['median_MB_s']:.1f} MB/s ({L['upload']['median_Gbit_s']:.3f} / {L['download']['median_Gbit_s']:.3f} Gbit/s)")
    errs = [r for r in recs if r.get('error')]
    for r in errs[:4]:
        print(f"ERROR {r['label']} {r['arm']}#{r['nth_in_process']}: {r['error'][:160]}")
    for r in recs:  # derived per record
        s = (r.get('service') or {}).get('seconds') or {}
        if r['arm'] == 'R' and s:
            r['ttft_s'] = (s.get('queue') or 0) + (s.get('prefill') or 0)
        t0 = r['epoch_start']
        n = r['tokens']
        if r['arm'] in ('A', 'R') and smi and s:
            c, mhz = cycles(smi, t0, t0 + (s.get('queue') or 0) + (s.get('prefill') or 0))
            r['rtx_mclk_tok'] = c * 1e3 / n if c else None
            r['rtx_mhz'] = mhz
        mon = mons.get(r['label'])
        if mon and r['arm'] in ('A', 'B') and r.get('first_s') is not None:
            first_abs = (r['epoch_end'] - (r.get('decode_s') or 0) - 0.0) if r.get('decode_s') else None
            if r['arm'] == 'B':
                cs = [cycles(m, t0, t0 + r['first_s'])[0] for m in mon]
                r['pair_pf_mclk_tok'] = sum(c for c in cs if c) * 1e3 / n if any(cs) else None
            if first_abs and r.get('n_chunks', 0) > 1:
                cs = [cycles(m, first_abs, first_abs + r['decode_s'])[0] for m in mon]
                r['pair_dec_mclk_tok'] = sum(c for c in cs if c) * 1e3 / (r['n_chunks'] - 1) if any(cs) else None
                r['pair_mhz'] = cycles(mon[0], first_abs, first_abs + r['decode_s'])[1]
    groups = defaultdict(list)
    for r in recs:
        if not r.get('error'):
            groups[(r['mode'] if r['arm'] != 'R' else 'rtx', r['arm'])].append(r)
    print('group | n | TTFT cold | TTFT warm (median) | all TTFT | decode tok/s warm | Mclk/token RTX pf | pair pf | pair dec')
    for (mode, arm), rs in sorted(groups.items()):
        per = defaultdict(list)
        for r in rs:
            per[r['label']].append(r)
        cold = [v[0] for v in per.values()]
        warm = [v[-1] for v in per.values() if len(v) > 1]
        print(f"{mode}/{arm} | {len(rs)} | {f(med(r['ttft_s'] for r in cold))} | {f(med(r['ttft_s'] for r in warm))} | "
              f"{' '.join(f(r['ttft_s'], 2) for r in rs)} | {f(med(r.get('decode_tok_s') for r in warm), 2)} | "
              f"{f(med(r.get('rtx_mclk_tok') for r in warm), 2)} | {f(med(r.get('pair_pf_mclk_tok') for r in warm), 2)} | "
              f"{f(med(r.get('pair_dec_mclk_tok') for r in warm), 1)}")
    A = [r for r in recs if r['arm'] == 'A' and not r.get('error')]
    if A:
        print('A breakdown (s): label#n prefill export sha256 transfer import[sha parse queue relay device] first ttft | bytes | continued')
        for r in A:
            s = r['service']['seconds']
            t = ((r['service'].get('import') or {}).get('timing')) or {}
            print(f"  {r['label']}#{r['nth_in_process']} {f(s.get('prefill'), 2)} {f(s.get('export'))} {f(s.get('sha256'))} "
                  f"{f(s.get('transfer'))} {f(s.get('import'))}[{f(t.get('sha256_s'))} {f(t.get('parse_s'))} {f(t.get('queue_s'))} "
                  f"{f(t.get('replicate_s'))} {f(t.get('device_s'))}] {f(r.get('first_s'))} {f(r['ttft_s'], 2)} | "
                  f"{r['service'].get('state_bytes')} | {r.get('continued')}")
    ref = next((r for r in recs if r['arm'] == 'B' and r['mode'] == 'exact' and r.get('gen_ids')), None) or \
        next((r for r in recs if r['arm'] == 'B' and r.get('gen_ids')), None)
    if ref:
        print(f"continuation vs reference {ref['mode']}/B {ref['label']}#{ref['nth_in_process']} ({len(ref['gen_ids'])} ids, first {ref['gen_ids'][:4]}):")
        for (mode, arm), rs in sorted(groups.items()):
            rs = [r for r in rs if r.get('gen_ids')]
            if rs:
                print(f"  {mode}/{arm}: prefix vs ref {[prefix(r['gen_ids'], ref['gen_ids']) for r in rs]} | vs own first "
                      f"{[prefix(r['gen_ids'], rs[0]['gen_ids']) for r in rs]} of {[len(r['gen_ids']) for r in rs]}")
    nxt = {r['service'].get('next_token') for r in recs if r.get('service') and r['service'].get('next_token') is not None}
    shas = {str(r['service'].get('sha256'))[:16] for r in recs if r['arm'] in ('A', 'R') and r.get('service')}
    print(f"RTX next_token {sorted(nxt)} | state sha256 {sorted(shas)}")


if __name__ == '__main__':
    main()
