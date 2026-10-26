#!/usr/bin/env python3
"""Resumable full-depth float64 O-math run (layer-major, omath_layer.py semantics) for one prompt, within a deadline.

State lives in OUT/: X-L<l>.f64 = residual [T][4][4096] after layer l (written atomically). A run starts after the last
existing layer file, decodes each layer's EXL3 tensors to Q with strix-glm5-decode-q into QDIR (this run's scratch,
emptied per layer, as om_validate.py), runs the layer, and stops cleanly when fewer than --reserve seconds remain before
--deadline (epoch seconds). After layer 44 it decodes lm_head and writes logits-last<NS>.f64 = float64 logits [NS][vocab]
of the last NS input positions (TOKENS holds T + 1 tokens: inputs 0..T-1).

usage: om_run.py --model DIR --trellis-python DIR --decode-q BIN --qdir DIR --tokens T.u32 --out DIR --deadline EPOCH
       [--reserve 45] [--score 1]
"""
import argparse, os, subprocess, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import omath_layer as ol  # noqa: E402

NL = 45


def decode(a, names):
    lst = os.path.join(a.qdir, 'list.txt')
    with open(lst, 'w') as f:
        f.write('\n'.join(names) + '\n')
    r = subprocess.run([a.decode_q, a.model, a.qdir, '@' + lst], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise SystemExit(f'decode-q failed on {names[:2]}')
    return r.stdout.strip().splitlines()


def drop(a, names):
    for n in names:  # this run's own scratch Q files
        p = os.path.join(a.qdir, n + '.q16')
        if os.path.exists(p):
            os.remove(p)


def main():
    ap = argparse.ArgumentParser()
    for k in ('--model', '--trellis-python', '--decode-q', '--qdir', '--tokens', '--out'):
        ap.add_argument(k, required=True)
    ap.add_argument('--deadline', type=float, required=True)
    ap.add_argument('--reserve', type=float, default=45.0)
    ap.add_argument('--score', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.qdir, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    W = ol.Weights(a.model, a.qdir, a.trellis_python)
    toks = np.fromfile(a.tokens, dtype=np.uint32)
    T = toks.size - 1
    done = [L for L in range(NL) if os.path.exists(os.path.join(a.out, f'X-L{L}.f64'))]
    L0 = max(done) + 1 if done else 0
    if L0 == 0:
        emb = np.stack([W.s.row(ol.P + 'embed_tokens.weight', int(t)) for t in toks[:T]])
        X = np.repeat(emb[:, None, :], ol.HC, axis=1)
    else:
        X = np.fromfile(os.path.join(a.out, f'X-L{L0 - 1}.f64'), dtype=np.float64).reshape(T, ol.HC, ol.H)
    print(f'om_run: T {T}, resuming at layer {L0}', flush=True)
    for L in range(L0, NL):
        if time.time() > a.deadline - a.reserve:
            print(f'om_run: STOP before layer {L} (deadline), resume later', flush=True)
            return 0
        pfx = f'{ol.P}layers.{L}.'
        names = sorted(k[:-len('.trellis')] for k in W.s.idx if k.startswith(pfx) and k.endswith('.trellis'))
        t0 = time.time()
        dig = decode(a, names)
        t1 = time.time()
        sel = {}
        X = ol.layer(W, L, X, sel)
        t2 = time.time()
        p = os.path.join(a.out, f'X-L{L}.f64')
        X.tofile(p + '.tmp')
        os.replace(p + '.tmp', p)
        if 'experts' in sel:
            np.save(os.path.join(a.out, f'experts-L{L}.npy'), np.asarray(sel['experts']))
        drop(a, names)
        print(f'L{L}: {len(dig)} tensors decoded {t1 - t0:.1f} s, layer {t2 - t1:.1f} s, residual rms {float(np.sqrt(np.mean(X * X))):.6e}'
              + (f', router min boundary gap {sel["router_gap"].min():.2e}' if 'router_gap' in sel else ''), flush=True)
    lp = os.path.join(a.out, f'logits-last{a.score}.f64')
    if not os.path.exists(lp):
        if time.time() > a.deadline - a.reserve:
            print('om_run: STOP before the head (deadline)', flush=True)
            return 0
        dig = decode(a, ['lm_head'])
        h = ol.rms(X[T - a.score:].mean(axis=1), W.plain(ol.P + 'norm.weight'))
        lg = ol.lin(W, 'lm_head', h)
        lg.tofile(lp + '.tmp')
        os.replace(lp + '.tmp', lp)
        drop(a, ['lm_head'])
        print(f'om_run: head done, logits {lg.shape} argmax {int(np.argmax(lg[-1]))} -> {lp}', flush=True)
    print('om_run: COMPLETE', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
