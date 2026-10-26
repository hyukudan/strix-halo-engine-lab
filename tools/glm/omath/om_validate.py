#!/usr/bin/env python3
"""O-math validation step 7 (design notes section 6.7): the layer-major Hadamard-domain oracle (omath_layer.py) vs
the token-by-token CPU oracle dump (cpu_reference.py, reconstruct_weight + dense float64) on the same tokens, layers
0..N-1: residual rel-L2 per layer <= 1e-12 and identical router top-8 ids. Per layer, the layer's EXL3 tensors are
decoded to Q by strix-glm5-decode-q into QDIR (this run's own scratch, emptied after the layer); decode digests are printed.

usage: om_validate.py --model DIR --trellis-python DIR --decode-q BIN --qdir DIR --dump OMATH.npz --tokens 1,2,3,4 --layers N
"""
import argparse, os, subprocess, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import omath_layer as ol  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--trellis-python', required=True)
    ap.add_argument('--decode-q', required=True)
    ap.add_argument('--qdir', required=True)
    ap.add_argument('--dump', required=True)
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--layers', type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.qdir, exist_ok=True)
    W = ol.Weights(a.model, a.qdir, a.trellis_python)
    ref = np.load(a.dump)
    toks = [int(t) for t in a.tokens.split(',')]
    T = len(toks)
    emb = np.stack([W.s.row(ol.P + 'embed_tokens.weight', t) for t in toks])
    X = np.repeat(emb[:, None, :], ol.HC, axis=1)
    worst, ok = 0.0, True
    for L in range(a.layers):
        pfx = f'{ol.P}layers.{L}.'
        names = sorted(k[:-len('.trellis')] for k in W.s.idx if k.startswith(pfx) and k.endswith('.trellis'))
        lst = os.path.join(a.qdir, 'list.txt')
        with open(lst, 'w') as f:
            f.write('\n'.join(names) + '\n')
        t0 = time.time()
        r = subprocess.run([a.decode_q, a.model, a.qdir, '@' + lst], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-2000:])
            raise SystemExit(f'decode-q failed on layer {L}')
        dig = r.stdout.strip().splitlines()
        t1 = time.time()
        sel = {}
        X = ol.layer(W, L, X, sel)
        t2 = time.time()
        R = np.stack([ref[f't{t}.L{L}'] for t in range(T)])
        rel = float(np.linalg.norm(X - R) / np.linalg.norm(R))
        worst = max(worst, rel)
        msg = f'L{L}: {len(dig)} tensors decoded {t1 - t0:.1f} s, layer {t2 - t1:.1f} s, residual rel-L2 vs CPU oracle {rel:.3e}'
        ok &= rel <= 1e-12
        if 'experts' in sel:
            same = all(np.array_equal(np.sort(ref[f't{t}.L{L}.experts']), sel['experts'][t]) for t in range(T))
            ok &= same
            msg += f', router top-8 {"EQUAL" if same else "DIFFER"} (min boundary gap {sel["router_gap"].min():.2e})'
        print(msg, flush=True)
        print('   digests: ' + ' | '.join(d.split()[0].split('.')[-1] + ' ' + d.split()[-1] for d in dig[:4]) + (' ...' if len(dig) > 4 else ''), flush=True)
        for n in names:                                                     # this run's own scratch Q files
            os.remove(os.path.join(a.qdir, n + '.q16'))
    print(f'om_validate: worst rel-L2 {worst:.3e} -> {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
