#!/usr/bin/env python3
"""GLM-5.3-Flash O-math at G4 scale (design notes): float64 oracle, LAYER-MAJOR teacher forcing.

Same semantics as tools/glm/cpu_reference.py with REF off (design notes; declared choices: absorbed MLA, F32 hc
base/scale as stored, conv state = last 3 pre-conv qkv columns, top-k ties -> lower index), but every layer runs over
all T positions of a prompt before the next layer, and every EXL3 linear is the float64 Hadamard-domain product of
strix_trellis.linear.matmul, y = H(H(x * suh) @ Q) * svh, with Q = the logical FP16 trellis decode written by
strix-glm5-decode-q (QDIR/<tensor>.q16, [inputs][outputs]) instead of a materialized W (plan section 11).

One call = one layer for one prompt: reads the residual X [T][4][4096] (float64) of the previous layer (or embeds the
tokens for layer 0), writes the next residual, the layer's selections (router top-8 ids, indexer pools with the
boundary score gap) and, after layer 44, the float64 logits of the last NS positions.

usage: omath_layer.py --model DIR --qdir DIR --trellis-python DIR --layer L --tokens T.u32 --inp X.f64|- --out Y.f64
       [--sel SEL.npz] [--score NS --logits LOG.f64]
"""
import argparse, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cpu_reference import (SafeDir, P, H, NH, HD, QKV, HC, HC_EPS, EPS, NE, TOPK, CLAMP, SCALE_R, KVL, MLA_HD, IDX_H,  # noqa: E402
                           IDX_D, KPOOL, IDX_TOPK, is_kda, is_moe, hc_weights)


# ---------------------------------------------------------------- weights
class Weights:
    """Plain tensors from the safetensors (float64); EXL3 linears as Hadamard-domain products over decoded Q."""

    def __init__(self, model, qdir, trellis_python):
        sys.path.insert(0, trellis_python)
        from strix_trellis import linear as tl
        self.tl, self.s, self.qdir = tl, SafeDir(model), qdir

    def plain(self, k):
        name = k if self.s.has(k) else k + '.weight'
        return np.asarray(self.s.raw(name), dtype=np.float64)

    def is_exl3(self, k):
        return self.s.has(k + '.trellis')

    def q(self, k):
        tr = self.s.idx[k + '.trellis'][2]['shape']                                    # [in/16, out/16, words]
        i, o = tr[0] * 16, tr[1] * 16
        q = np.fromfile(os.path.join(self.qdir, k + '.q16'), dtype=np.float16)
        if q.size != i * o:
            raise RuntimeError(f'{k}: decoded Q has {q.size} values, expected {i} x {o}')          # fail closed
        return q.reshape(i, o)

    def linear(self, k, x):
        """x [..., in] -> [..., out] (float64), strix_trellis.linear.matmul's order of operations."""
        Q = self.q(k).astype(np.float64)
        left, right = self.tl._scales(np.asarray(self.s.raw(k + '.suh')), np.asarray(self.s.raw(k + '.svh')), Q.shape)
        return self.tl.hadamard(self.tl.hadamard(x * left) @ Q) * right


def lin(W, k, x):
    """EXL3 product or plain [out, in] weight."""
    return W.linear(k, x) if W.is_exl3(k) else x @ W.plain(k).T


# ---------------------------------------------------------------- helpers (float64, vectorized over positions)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def rms(x, w=None, eps=EPS):
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y if w is None else y * w


def topk_lower_index(scores, k):
    order = np.lexsort((np.arange(scores.size), -scores))
    return order[:k]


def hc_pre_all(X, fn, base, scale):
    """Per position: pre [T,4], post [T,4], comb [T,4,4] (cpu_reference.hc_weights, row by row: Sinkhorn is tiny)."""
    T = X.shape[0]
    pre, post, comb = np.empty((T, 4)), np.empty((T, 4)), np.empty((T, 4, 4))
    for t in range(T):
        pre[t], post[t], comb[t] = hc_weights(X[t], fn, base, scale)
    return pre, post, comb


def hc_apply_all(X, post, comb, y):
    """out_i = post_i * y + sum_j comb[j, i] * X_j, per position."""
    return post[:, :, None] * y[:, None, :] + np.einsum('tji,tjd->tid', comb, X)


# ---------------------------------------------------------------- blocks
def kda(W, pre, x):
    a = pre + 'self_attn.'
    T = x.shape[0]
    qkv = lin(W, a + 'qkv_proj', x) if W.is_exl3(a + 'qkv_proj') else np.concatenate(
        [lin(W, a + f'{c}_proj', x) for c in 'qkv'], axis=1)                           # [T, 24576]
    conv = np.concatenate([W.plain(a + f'{c}_conv1d')[:, 0, :] for c in 'qkv']) if W.s.has(a + 'q_conv1d.weight') \
        else W.plain(a + 'conv1d')[:, 0, :]                                             # [24576, 4] oldest first
    padded = np.concatenate([np.zeros((3, 3 * QKV)), qkv])                             # zero initial conv state
    y = silu(sum(padded[j:j + T] * conv[:, j] for j in range(4)))
    q, k, v = (y[:, i * QKV:(i + 1) * QKV].reshape(T, NH, HD) for i in range(3))
    fa = lin(W, a + 'f_a_proj', x)
    g = -5.0 * sigmoid(np.exp(W.plain(a + 'A_log'))[None, :, None] *
                       ((lin(W, a + 'f_b_proj', fa) + W.plain(a + 'dt_bias')).reshape(T, NH, HD)))
    beta = sigmoid(lin(W, a + 'b_proj', x))                                             # [T, 64]
    q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) / math.sqrt(HD)
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    dec = np.exp(g)
    S = np.zeros((NH, HD, HD))
    o = np.empty((T, NH, HD))
    for t in range(T):                                                                  # the recurrence: sequential
        S = S * dec[t][:, :, None]
        u = np.einsum('hkv,hk->hv', S, k[t])
        S = S + k[t][:, :, None] * ((v[t] - u) * beta[t][:, None])[:, None, :]
        o[t] = np.einsum('hkv,hk->hv', S, q[t])
    gate = lin(W, a + 'g_b_proj', lin(W, a + 'g_a_proj', x)).reshape(T, NH, HD)
    o = rms(o, None, EPS) * W.plain(a + 'o_norm.weight') * sigmoid(gate)
    return lin(W, a + 'o_proj', o.reshape(T, -1)), S


def mla(W, pre, x, sel):
    a = pre + 'self_attn.'
    i = a + 'indexer.'
    T = x.shape[0]
    q_resid = rms(lin(W, a + 'q_a_proj', x), W.plain(a + 'q_a_layernorm.weight'))
    q = lin(W, a + 'q_b_proj', q_resid).reshape(T, NH, MLA_HD)
    lat = rms(lin(W, a + 'kv_a_proj_with_mqa', x), W.plain(a + 'kv_a_layernorm.weight'))  # [T, 512]
    if W.is_exl3(a + 'kv_b_proj'):                                                      # [out, in] via the identity product
        kvb = W.linear(a + 'kv_b_proj', np.eye(KVL)).T
    else:
        kvb = W.plain(a + 'kv_b_proj')
    kvb = kvb.reshape(NH, 2 * MLA_HD, KVL)
    wkb, wvb = kvb[:, :MLA_HD, :], kvb[:, MLA_HD:, :]
    kI = lin(W, i + 'wk', x)
    kI = (kI - kI.mean(-1, keepdims=True)) / np.sqrt(kI.var(-1, keepdims=True) + 1e-6) * W.plain(i + 'k_norm.weight') \
        + W.plain(i + 'k_norm.bias')
    gts = lin(W, i + 'index_kpool_compress_gate', x)
    ape = W.plain(i + 'index_kpool_compress_ape')
    NP = T // KPOOL
    Kp = kI[:NP * KPOOL].reshape(NP, KPOOL, IDX_D)
    G = gts[:NP * KPOOL].reshape(NP, KPOOL, IDX_D) + ape[None]
    pr = np.exp(G - G.max(axis=1, keepdims=True))
    pr = pr / pr.sum(axis=1, keepdims=True)
    poolK = (pr * Kp).sum(axis=1)                                                       # [P, 128]
    qI = lin(W, i + 'wq_b', q_resid).reshape(T, IDX_H, IDX_D)
    wI = lin(W, i + 'weights_proj', x) / math.sqrt(IDX_H)                               # [T, 32]
    qt = np.einsum('thd,hdl->thl', q, wkb)                                              # absorbed [T, 64, 512]
    out = np.empty((T, NH, MLA_HD))
    pools_sel, gaps = np.full((T, IDX_TOPK // KPOOL), -1, np.int32), np.full(T, np.inf)
    for t in range(T):
        n = (t + 1) // KPOOL
        idx = []
        if n:
            s = np.maximum(qI[t] @ poolK[:n].T / math.sqrt(IDX_D), 0.0)                 # [32, n]
            I = wI[t] @ s
            k = min(IDX_TOPK // KPOOL, n)
            pools = topk_lower_index(I, k)
            if k < n:                                                                   # boundary gap (plan 6.3)
                srt = np.sort(I)[::-1]
                gaps[t] = (srt[k - 1] - srt[k]) / max(abs(srt[k - 1]), 1e-300)
            pools = np.sort(pools)
            pools_sel[t, :pools.size] = pools
            idx = [int(p) * KPOOL + j for p in pools for j in range(KPOOL)]
        tail = (t + 1) % KPOOL
        idx = np.array(idx + list(range(t + 1 - tail, t + 1)), dtype=np.int64)
        C = lat[idx]                                                                    # [n_tok, 512]
        sc = qt[t] @ C.T / math.sqrt(MLA_HD)                                            # [64, n_tok]
        sc = np.exp(sc - sc.max(axis=1, keepdims=True))
        sc = sc / sc.sum(axis=1, keepdims=True)
        out[t] = np.einsum('hl,hdl->hd', sc @ C, wvb)
    sel['pools'] = pools_sel                                                          # [T, 512], -1 = unused
    sel['pool_gap'] = gaps
    return lin(W, a + 'o_proj', out.reshape(T, -1))


def swiglu(W, pfx, x):
    g = np.minimum(lin(W, pfx + 'gate_proj', x), CLAMP)
    u = np.clip(lin(W, pfx + 'up_proj', x), -CLAMP, CLAMP)
    return lin(W, pfx + 'down_proj', silu(g) * u)


def moe(W, pre, x, sel):
    T = x.shape[0]
    s = sigmoid(x @ W.plain(pre + 'mlp.gate.weight').T)                                 # [T, 288]
    choice = s + W.plain(pre + 'mlp.gate.e_score_correction_bias')
    ids = np.empty((T, TOPK), np.int32)
    wts = np.empty((T, TOPK))
    gaps = np.empty(T)
    for t in range(T):
        k = topk_lower_index(choice[t], TOPK)
        srt = np.sort(choice[t])[::-1]
        gaps[t] = (srt[TOPK - 1] - srt[TOPK]) / max(abs(srt[TOPK - 1]), 1e-300)
        w = s[t, k] / (s[t, k].sum() + 1e-20) * SCALE_R
        o = np.argsort(k)
        ids[t], wts[t] = k[o], w[o]
    out = np.zeros((T, H))
    for e in range(NE):                                                                 # ascending expert id
        rows, slot = np.nonzero(ids == e)
        if rows.size:
            y = swiglu(W, f'{pre}mlp.experts.{e}.', x[rows])
            out[rows] = out[rows] + wts[rows, slot][:, None] * y
    sel['experts'], sel['router_gap'] = ids, gaps
    return out + swiglu(W, pre + 'mlp.shared_experts.', x)


def layer(W, L, X, sel):
    pre = f'{P}layers.{L}.'
    hc = [(W.plain(pre + f'hc_{s}_fn'), W.plain(pre + f'hc_{s}_base'), W.plain(pre + f'hc_{s}_scale')) for s in ('attn', 'ffn')]
    p0, q0, c0 = hc_pre_all(X, *hc[0])
    x = rms(np.einsum('tj,tjd->td', p0, X), W.plain(pre + 'input_layernorm.weight'))
    if is_kda(L):
        y, S = kda(W, pre, x)
        sel['kda_state_last'] = S
    else:
        y = mla(W, pre, x, sel)
    X = hc_apply_all(X, q0, c0, y)
    p1, q1, c1 = hc_pre_all(X, *hc[1])
    x = rms(np.einsum('tj,tjd->td', p1, X), W.plain(pre + 'post_attention_layernorm.weight'))
    y = moe(W, pre, x, sel) if is_moe(L) else swiglu(W, pre + 'mlp.', x)
    return hc_apply_all(X, q1, c1, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--qdir', required=True)
    ap.add_argument('--trellis-python', required=True)
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--inp', required=True, help="previous residual X [T][4][4096] f64, or '-' for layer 0 (embed)")
    ap.add_argument('--out', required=True)
    ap.add_argument('--sel', default='')
    ap.add_argument('--score', type=int, default=0)
    ap.add_argument('--logits', default='')
    a = ap.parse_args()
    W = Weights(a.model, a.qdir, a.trellis_python)
    toks = np.fromfile(a.tokens, dtype=np.uint32)
    T = toks.size - 1                                                                   # inputs 0..T-1
    t0 = time.time()
    if a.inp == '-':
        emb = np.stack([W.s.row(P + 'embed_tokens.weight', int(t)) for t in toks[:T]])
        X = np.repeat(emb[:, None, :], HC, axis=1)
    else:
        X = np.fromfile(a.inp, dtype=np.float64).reshape(T, HC, H)
    sel = {}
    X = layer(W, a.layer, X, sel)
    tmp = a.out + '.tmp'
    X.tofile(tmp)
    os.replace(tmp, a.out)
    if a.sel:
        np.savez(a.sel + '.tmp.npz', **{k: v for k, v in sel.items()})
        os.replace(a.sel + '.tmp.npz', a.sel)
    if a.score and a.logits:
        h = rms(X[T - a.score:].mean(axis=1), W.plain(P + 'norm.weight'))
        lg = lin(W, 'lm_head', h)
        lg.tofile(a.logits + '.tmp')
        os.replace(a.logits + '.tmp', a.logits)
    print(f'omath layer {a.layer} T {T} {time.time() - t0:.1f} s, residual rms {float(np.sqrt(np.mean(X * X))):.6e}', flush=True)


if __name__ == '__main__':
    main()
