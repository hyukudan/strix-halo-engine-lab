#!/usr/bin/env python3
"""GLM-5.3-Flash (glm5_next) text decoder: float64 mathematical oracle (O-math), token-by-token decode.

Semantics: design notes (transformers modeling_glm5_next.py, pinned in design notes).
Everything is float64 with no intermediate rounding: this is the oracle the G3/G4 quality bounds are measured
against, not the engine's production arithmetic (C-prod).  Two weight providers with one canonical naming
(the original checkpoint's tensor names):
  fp8:  zai-org original FP8 e4m3 + 128x128 weight_scale_inv (subset files from fetch_fp8_subset.py)
  exl3: turboderp EXL3 (strix-trellis reconstruct_weight; fused qkv_proj / conv1d split back to q/k/v)
--ref-rounding emulates the reference's bf16 module-boundary roundings and its expanded MLA, to measure
the declared deviations D0/D1/D4 (P1 design 1.3).  Declared choices (plan section 6.1): absorbed MLA (exact in float64), F32 hc base/scale as stored, conv state =
last 3 pre-conv qkv columns, top-k ties -> lower index.  The MTP layer is not implemented until
the MTP arithmetic contract is reconciled.

usage: cpu_reference.py --provider fp8|exl3 --model DIR [--trellis-python DIR] --layers 0-5 --tokens 1,2,3
       [--dump OUT.npz]   (residual streams per layer per token; logits when the full model + head are present)
"""
import argparse, glob, json, math, os, re, struct, sys
import numpy as np

H, NH, HD, QKV = 4096, 64, 128, 8192          # hidden, KDA heads, KDA head dim, KDA q/k/v width
HC, HC_ITERS, HC_EPS, EPS = 4, 20, 1e-6, 1e-5
NE, TOPK, I_MOE, I_DENSE, SCALE_R, CLAMP = 288, 8, 2048, 12288, 2.5, 10.0
QL, KVL, MLA_HD, IDX_H, IDX_D, KPOOL, IDX_TOPK = 1536, 512, 256, 32, 128, 4, 2048
N_LAYERS, VOCAB = 45, 154880
P = 'model.language_model.'


def is_kda(L):
    return L % 4 != 3


def is_moe(L):
    return L >= 3


# ---------------------------------------------------------------- safetensors / providers
class SafeDir:
    """Lazy reader over a directory of safetensors files (header index, numpy memmap per tensor)."""
    DT = {'F32': np.float32, 'F16': np.float16, 'I16': np.int16, 'I32': np.int32, 'F8_E4M3': np.uint8,
          'BF16': np.uint16, 'U8': np.uint8}

    def __init__(self, d):
        self.idx = {}
        for f in sorted(glob.glob(os.path.join(d, '*.safetensors'))):
            with open(f, 'rb') as fh:
                n = struct.unpack('<Q', fh.read(8))[0]
                h = json.loads(fh.read(n))
            for k, v in h.items():
                if k != '__metadata__':
                    self.idx[k] = (f, 8 + n, v)

    def has(self, k):
        return k in self.idx

    def row(self, k, i):
        """One row of a 2-D tensor as float64 (BF16/F16/F32), without materializing the whole tensor."""
        f, base, v = self.idx[k]
        a = v['data_offsets'][0]
        arr = np.memmap(f, dtype=self.DT[v['dtype']], mode='r', offset=base + a, shape=tuple(v['shape']))
        r = np.asarray(arr[i])
        if v['dtype'] == 'BF16':
            r = (r.astype(np.uint32) << 16).view(np.float32)
        return r.astype(np.float64)

    def raw(self, k):
        f, base, v = self.idx[k]
        a, b = v['data_offsets']
        arr = np.memmap(f, dtype=self.DT[v['dtype']], mode='r', offset=base + a,
                        shape=tuple(v['shape']) if v['shape'] else (1,))
        if v['dtype'] == 'BF16':
            return (arr.astype(np.uint32) << 16).view(np.float32)
        return arr


def e4m3_lut():
    u = np.arange(256, dtype=np.uint32)
    s, e, m = (u >> 7) & 1, (u >> 3) & 15, u & 7
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e.astype(np.int64) - 7))
    v = np.where((e == 15) & (m == 7), np.nan, v)
    return np.where(s == 1, -v, v)


class Fp8Provider:
    """Original checkpoint (FP8 e4m3 + block scales, BF16, F32). get(name) -> float64, Linear layout [out, in]."""
    LUT = e4m3_lut()

    def __init__(self, d):
        self.s = SafeDir(d)

    def has(self, k):
        return self.s.has(k)

    def get(self, k):
        if self.s.has(k + '.weight_scale_inv'):
            w = self.LUT[np.asarray(self.s.raw(k + '.weight'))]
            sc = np.asarray(self.s.raw(k + '.weight_scale_inv'), dtype=np.float64)
            o, i = w.shape
            return w * np.repeat(np.repeat(sc, 128, 0), 128, 1)[:o, :i]
        name = k if self.s.has(k) else k + '.weight'
        return np.asarray(self.s.raw(name), dtype=np.float64)


class Exl3Provider:
    """turboderp EXL3: trellis groups reconstructed on CPU (strix-trellis), mapped to the original names."""

    def __init__(self, d, trellis_python):
        sys.path.insert(0, trellis_python)
        from strix_trellis.linear import reconstruct_weight
        self.rw = reconstruct_weight
        self.s = SafeDir(d)
        self.cache = {}

    def _group(self, k):
        if k in self.cache:
            return self.cache[k]
        tr = np.asarray(self.s.raw(k + '.trellis'))
        w = np.asarray(self.rw(tr, np.asarray(self.s.raw(k + '.suh')), np.asarray(self.s.raw(k + '.svh')),
                               k=tr.shape[-1] // 16, codebook='mul1'), dtype=np.float64).T   # -> [out, in]
        if len(self.cache) >= int(os.environ.get('ORACLE_TENSOR_CACHE', '12')):  # float64 matrices: ~67 MB each
            self.cache.clear()
        self.cache[k] = w
        return w

    def has(self, k):
        try:
            self._name(k)
            return True
        except KeyError:
            return False

    def _name(self, k):
        for cand in (k, k + '.weight', k + '.trellis'):
            if self.s.has(cand):
                return cand
        m = re.match(r'(.*\.self_attn)\.([qkv])_(proj|conv1d)$', k)
        if m and self.s.has(m.group(1) + ('.qkv_proj.trellis' if m.group(3) == 'proj' else '.conv1d.weight')):
            return k
        raise KeyError(k)

    def get(self, k):
        if self.s.has(k + '.trellis'):
            return self._group(k)
        m = re.match(r'(.*\.self_attn)\.([qkv])_(proj|conv1d)$', k)
        if m and not self.s.has(k + '.weight'):
            c = 'qkv'.index(m.group(2))
            w = (self._group(m.group(1) + '.qkv_proj') if m.group(3) == 'proj'
                 else np.asarray(self.s.raw(m.group(1) + '.conv1d.weight'), dtype=np.float64))
            return w[c * QKV:(c + 1) * QKV]
        name = k if self.s.has(k) else k + '.weight'
        return np.asarray(self.s.raw(name), dtype=np.float64)


# ---------------------------------------------------------------- math blocks (float64)
REF = False  # --ref-rounding: bf16 rounding at the reference's module boundaries + expanded MLA (deviations D0/D1/D4)


def B(x):
    """bf16 round-to-nearest-even of a float64 array (via fp32) when REF is on; identity otherwise."""
    if not REF:
        return x
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return u.astype(np.uint32).view(np.float32).astype(np.float64)


def rms(x, w=None, eps=EPS):
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y if w is None else B(B(y) * w)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def hc_weights(X, fn, base, scale):
    """mHC: X [4, H] -> pre [4], post [4], comb [4, 4] (rows j = input stream, cols i = output stream)."""
    F = rms(X.reshape(-1))
    z = fn @ F
    pre = sigmoid(z[0:4] * scale[0] + base[0:4]) + HC_EPS
    post = 2.0 * sigmoid(z[4:8] * scale[1] + base[4:8])
    c = z[8:24].reshape(4, 4) * scale[2] + base[8:24].reshape(4, 4)
    c = np.exp(c - c.max(axis=1, keepdims=True))
    c = c / c.sum(axis=1, keepdims=True) + HC_EPS
    c = c / (c.sum(axis=0, keepdims=True) + HC_EPS)
    for _ in range(HC_ITERS - 1):
        c = c / (c.sum(axis=1, keepdims=True) + HC_EPS)
        c = c / (c.sum(axis=0, keepdims=True) + HC_EPS)
    return pre, post, c


def hc_apply(X, post, comb, y):
    """out_i = post_i * y + sum_j comb[j, i] * X_j."""
    return B(B(post)[:, None] * y[None, :] + B(comb).T @ X)


def topk_lower_index(scores, k):
    """Top-k by value, ties broken by the lower index (declared rule); returns indices in descending score order."""
    order = np.lexsort((np.arange(scores.size), -scores))
    return order[:k]


class Layer:
    def __init__(self, prov, L):
        self.p, self.L, self.pre = prov, L, f'{P}layers.{L}.'
        g = lambda n: prov.get(self.pre + n)
        self.hc = [(g(f'hc_{s}_fn'), g(f'hc_{s}_base'), g(f'hc_{s}_scale')) for s in ('attn', 'ffn')]
        self.in_norm, self.post_norm = g('input_layernorm.weight'), g('post_attention_layernorm.weight')
        a = 'self_attn.'
        if is_kda(L):
            self.wq, self.wk, self.wv = g(a + 'q_proj'), g(a + 'k_proj'), g(a + 'v_proj')
            self.conv = np.concatenate([g(a + f'{c}_conv1d')[:, 0, :] for c in 'qkv'])      # [24576, 4]
            self.fa, self.fb, self.ga, self.gb = g(a + 'f_a_proj'), g(a + 'f_b_proj'), g(a + 'g_a_proj'), g(a + 'g_b_proj')
            self.dt_bias, self.A_log, self.bp = g(a + 'dt_bias'), g(a + 'A_log'), g(a + 'b_proj')
            self.o_norm, self.wo = g(a + 'o_norm.weight'), g(a + 'o_proj')
            self.S = np.zeros((NH, HD, HD))
            self.conv_state = np.zeros((3 * QKV, 3))
        else:
            self.qa, self.qa_norm, self.qb = g(a + 'q_a_proj'), g(a + 'q_a_layernorm.weight'), g(a + 'q_b_proj')
            self.kva, self.kva_norm = g(a + 'kv_a_proj_with_mqa'), g(a + 'kv_a_layernorm.weight')
            kvb = g(a + 'kv_b_proj').reshape(NH, 2 * MLA_HD, KVL)
            self.wkb, self.wvb = kvb[:, :MLA_HD, :], kvb[:, MLA_HD:, :]                      # [64, 256, 512]
            self.wo = g(a + 'o_proj')
            i = a + 'indexer.'
            self.iq, self.ik = g(i + 'wq_b'), g(i + 'wk')
            self.ik_w, self.ik_b = g(i + 'k_norm.weight'), g(i + 'k_norm.bias')
            self.iw, self.igate, self.iape = g(i + 'weights_proj'), g(i + 'index_kpool_compress_gate'), g(i + 'index_kpool_compress_ape')
            self.lat, self.ikeys, self.igs = [], [], []                                      # caches per position
        if is_moe(L):
            self.router, self.rbias = g('mlp.gate.weight'), g('mlp.gate.e_score_correction_bias')
            self.sh = [g(f'mlp.shared_experts.{n}_proj') for n in ('gate', 'up', 'down')]
            self.exp_cache = {}
        else:
            self.dense = [g(f'mlp.{n}_proj') for n in ('gate', 'up', 'down')]
        self.last = {}

    # -- attention blocks
    def kda(self, x):
        qkv = B(np.concatenate([self.wq @ x, self.wk @ x, self.wv @ x]))
        buf = np.concatenate([self.conv_state, qkv[:, None]], axis=1)                       # [24576, 4] oldest first
        self.conv_state = buf[:, 1:]
        y = B(silu((buf * self.conv).sum(axis=1)))
        q, k, v = (y[i * QKV:(i + 1) * QKV].reshape(NH, HD) for i in range(3))
        g = -5.0 * sigmoid(np.exp(self.A_log)[:, None] * ((self.fb @ (self.fa @ x)) + self.dt_bias).reshape(NH, HD))
        beta = B(sigmoid(B(self.bp @ x)))
        q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) / math.sqrt(HD)
        k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
        S = self.S * np.exp(g)[:, :, None]
        u = np.einsum('hkv,hk->hv', S, k)
        S = S + k[:, :, None] * ((v - u) * beta[:, None])[:, None, :]
        self.S = S
        o = B(np.einsum('hkv,hk->hv', S, q))
        gate = B(self.gb @ B(self.ga @ x)).reshape(NH, HD)
        o = B(rms(o, None, EPS) * self.o_norm * sigmoid(gate))
        return B(self.wo @ o.reshape(-1))

    def indexer_select(self, x, q_resid):
        t = len(self.lat) - 1                                                                 # query position
        kI = B(self.ik @ x)
        kI = B((kI - kI.mean()) / np.sqrt(kI.var() + 1e-6) * self.ik_w + self.ik_b)           # LayerNorm eps 1e-6
        self.ikeys.append(kI)
        self.igs.append(B(self.igate @ x))
        n_pools = (t + 1) // KPOOL                                                            # complete pools only
        sel = []
        if n_pools:
            K = np.array(self.ikeys[:n_pools * KPOOL]).reshape(n_pools, KPOOL, IDX_D)
            G = np.array(self.igs[:n_pools * KPOOL]).reshape(n_pools, KPOOL, IDX_D) + self.iape[None]
            pr = np.exp(G - G.max(axis=1, keepdims=True))
            pr = B(pr / pr.sum(axis=1, keepdims=True))
            poolK = B((B(pr * K)).sum(axis=1))                                                # [P, 128]
            qI = B(self.iq @ q_resid).reshape(IDX_H, IDX_D)
            s = np.maximum(qI @ poolK.T / math.sqrt(IDX_D), 0.0)                              # [32, P]
            w = B(self.iw @ x) / math.sqrt(IDX_H)
            I = w @ s
            pools = topk_lower_index(I, min(IDX_TOPK // KPOOL, n_pools))
            sel = [p * KPOOL + j for p in sorted(pools) for j in range(KPOOL)]
            self.last['pools'] = np.sort(pools)
        tail = (t + 1) % KPOOL
        return np.array(sel + list(range(t + 1 - tail, t + 1)), dtype=np.int64)

    def mla(self, x):
        q_resid = rms(B(self.qa @ x), self.qa_norm)
        q = B(self.qb @ q_resid).reshape(NH, MLA_HD)
        self.lat.append(rms(B(self.kva @ x), self.kva_norm))
        idx = self.indexer_select(x, q_resid)
        C = np.array(self.lat)[idx]                                                           # [T, 512]
        if REF:  # expanded form, as the reference: K/V from kv_b, bf16 scores and probabilities
            Kx, Vx = B(np.einsum('tl,hdl->htd', C, self.wkb)), B(np.einsum('tl,hdl->htd', C, self.wvb))
            sc = B(B(np.einsum('hd,htd->ht', q, Kx)) * (1.0 / math.sqrt(MLA_HD)))
            sc = np.exp(sc - sc.max(axis=1, keepdims=True))
            sc = B(sc / sc.sum(axis=1, keepdims=True))
            o = B(np.einsum('ht,htd->hd', sc, Vx))
        else:
            qt = np.einsum('hd,hdl->hl', q, self.wkb)                                        # absorbed: [64, 512]
            sc = qt @ C.T / math.sqrt(MLA_HD)                                                 # [64, T]
            sc = np.exp(sc - sc.max(axis=1, keepdims=True))
            sc = sc / sc.sum(axis=1, keepdims=True)
            o = np.einsum('hl,hdl->hd', sc @ C, self.wvb)                                     # [64, 256]
        return B(self.wo @ o.reshape(-1))

    # -- FFN blocks
    @staticmethod
    def swiglu(x, wg, wu, wd):
        g, u = np.minimum(B(wg @ x), CLAMP), np.clip(B(wu @ x), -CLAMP, CLAMP)
        return B(wd @ B(silu(g) * u))

    def expert(self, e):
        if e not in self.exp_cache:
            if len(self.exp_cache) >= int(os.environ.get('ORACLE_EXPERT_CACHE', '6')):  # ~200 MB per expert (float64)
                self.exp_cache.clear()
            self.exp_cache[e] = [self.p.get(f'{self.pre}mlp.experts.{e}.{n}_proj') for n in ('gate', 'up', 'down')]
        return self.exp_cache[e]

    def moe(self, x):
        s = sigmoid(self.router @ x)
        idx = topk_lower_index(s + self.rbias, TOPK)
        w = s[idx] / (s[idx].sum() + 1e-20) * SCALE_R
        self.last['experts'] = np.sort(idx)
        out = np.zeros(H)
        for e, we in sorted(zip(idx, w)):                                                     # ascending expert id
            out = B(out + B(we * self.swiglu(x, *self.expert(int(e)))))
        return B(out + self.swiglu(x, *self.sh))

    def step(self, X):
        pre, post, comb = hc_weights(X, *self.hc[0])
        x = rms(B(pre @ X), self.in_norm)
        y = self.kda(x) if is_kda(self.L) else self.mla(x)
        X = hc_apply(X, post, comb, y)
        pre, post, comb = hc_weights(X, *self.hc[1])
        x = rms(B(pre @ X), self.post_norm)
        y = self.moe(x) if is_moe(self.L) else self.swiglu(x, *self.dense)
        return hc_apply(X, post, comb, y)


class Model:
    def __init__(self, prov, layers):
        self.p = prov
        self.layers = [Layer(prov, L) for L in layers]
        self.full = list(layers) == list(range(N_LAYERS)) and prov.has('lm_head')
        if self.full:
            self.norm, self.head = prov.get(P + 'norm.weight'), prov.get('lm_head')

    def step(self, tok):
        X = np.repeat(self.p.s.row(P + 'embed_tokens.weight', tok)[None], HC, axis=0)
        streams = []
        for ly in self.layers:
            X = ly.step(X)
            streams.append(X.copy())
        logits = B(self.head @ rms(B(X.mean(axis=0)), self.norm)) if self.full else None
        return streams, logits


def parse_layers(s):
    a, b = s.split('-') if '-' in s else (s, s)
    return range(int(a), int(b) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--provider', choices=['fp8', 'exl3'], required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--trellis-python', default='')
    ap.add_argument('--layers', default='0-5')
    ap.add_argument('--tokens', required=True, help='comma list or @file.json')
    ap.add_argument('--dump', default='')
    ap.add_argument('--ref-rounding', action='store_true', help='bf16 module-boundary rounding + expanded MLA')
    a = ap.parse_args()
    global REF
    REF = a.ref_rounding
    prov = Fp8Provider(a.model) if a.provider == 'fp8' else Exl3Provider(a.model, a.trellis_python)
    toks = json.load(open(a.tokens[1:])) if a.tokens.startswith('@') else [int(t) for t in a.tokens.split(',')]
    m = Model(prov, parse_layers(a.layers))
    out = {}
    for t, tok in enumerate(toks):
        streams, logits = m.step(tok)
        for ly, X in zip(m.layers, streams):
            out[f't{t}.L{ly.L}'] = X
            for k, v in ly.last.items():
                out[f't{t}.L{ly.L}.{k}'] = v
        if logits is not None:
            out[f't{t}.logits'] = logits
        print(t, tok, 'last-layer stream rms', float(np.sqrt(np.mean(streams[-1] ** 2))), flush=True)
    if a.dump:
        np.savez_compressed(a.dump, **out)


if __name__ == '__main__':
    main()
