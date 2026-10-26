"""G0 CPU test of the TP2 split rules (split_plan.py) on real EXL3 tensors, float64.

1. Byte level: for every out/in EXL3 group, the rank slice taken from the split plan's ranges (trellis tiles, suh/svh
   slices) reconstructs exactly the corresponding columns (out) or rows (in) of the full reconstructed weight.
2. Block level (oracle math, cpu_reference.Layer methods): KDA (layer 4) and MLA + indexer (layer 3) attention, one
   routed expert, the shared expert and the dense MLP (layer 0): rank 0 + rank 1 partial outputs (in-split o/down sum)
   equal the full-weight output to float64 rounding (max rel error <= 1e-10), over several decode steps.
usage: test_split.py FIXTURE_DIR SPLIT_JSON TRELLIS_PYTHON
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import cpu_reference as R  # noqa: E402

fix, split, tp = sys.argv[1], json.loads(Path(sys.argv[2]).read_text())['tensors'], sys.argv[3]
prov = R.Exl3Provider(fix, tp)
s = prov.s
fails = []


def check(name, a, b, tol=1e-10):
    rel = float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))
    ok = rel <= tol
    print(f'{"OK  " if ok else "FAIL"} {name}: max rel {rel:.3e}')
    if not ok:
        fails.append(name)


# ---- 1. byte-level EXL3 slices
for g, p in split.items():
    if p['kind'] != 'exl3' or p['mode'] == 'replicate' or not s.has(g + '.trellis'):
        continue
    full = prov.get(g)                                          # [out, in]
    tr, suh, svh = (np.asarray(s.raw(g + x)) for x in ('.trellis', '.suh', '.svh'))
    k = tr.shape[-1] // 16
    for r in (0, 1):
        for b, e in p['ranges'][r]:
            if p['mode'] == 'out':
                w = np.asarray(prov.rw(tr[:, b // 16:e // 16, :], suh, svh[b:e], k=k, codebook='mul1'), np.float64).T
                ref = full[b:e]
            else:
                w = np.asarray(prov.rw(tr[b // 16:e // 16], suh[b:e], svh, k=k, codebook='mul1'), np.float64).T
                ref = full[:, b:e]
            check(f'slice {p["mode"]} r{r} [{b},{e}) {g}', w, ref, 1e-12)


# ---- 2. block level
class RankProv:
    """Rank view: plain rows / EXL3 out/in slices per split plan, applied to the full float64 weights."""

    def __init__(self, r):
        self.r = r

    def get(self, k):
        w = prov.get(k)
        g = k if k in split else (k + '.weight' if k + '.weight' in split else None)
        if g is None:  # q/k/v_proj and q/k/v_conv1d come from fused qkv_proj / conv1d rows
            m = __import__('re').match(r'(.*\.self_attn)\.([qkv])_(proj|conv1d)$', k)
            c = 'qkv'.index(m.group(2))
            return w[self.r * R.QKV // 2:(self.r + 1) * R.QKV // 2]          # per-head rows within q, k or v
        p = split[g]
        if p['mode'] == 'replicate':
            return w
        rr = p['ranges'][self.r]
        if p['mode'] in ('out', 'rows'):
            return np.concatenate([w[b:e] for b, e in rr])
        return np.concatenate([w[:, b:e] for b, e in rr], axis=1)


def stub(provider, L, kind):
    st = R.Layer.__new__(R.Layer)
    g = lambda n: provider.get(f'{R.P}layers.{L}.{n}')
    a = 'self_attn.'
    if kind == 'kda':
        nh = g(a + 'A_log').shape[0]
        st.wq, st.wk, st.wv = g(a + 'q_proj'), g(a + 'k_proj'), g(a + 'v_proj')
        st.conv = np.concatenate([g(a + f'{c}_conv1d')[:, 0, :] for c in 'qkv'])
        st.fa, st.fb, st.ga, st.gb = g(a + 'f_a_proj'), g(a + 'f_b_proj'), g(a + 'g_a_proj'), g(a + 'g_b_proj')
        st.dt_bias, st.A_log, st.bp = g(a + 'dt_bias'), g(a + 'A_log'), g(a + 'b_proj')
        st.o_norm, st.wo = g(a + 'o_norm.weight'), g(a + 'o_proj')
        st.S = np.zeros((nh, R.HD, R.HD))
        st.conv_state = np.zeros((st.conv.shape[0], 3))
    elif kind == 'mla':
        st.qa, st.qa_norm, st.qb = g(a + 'q_a_proj'), g(a + 'q_a_layernorm.weight'), g(a + 'q_b_proj')
        st.kva, st.kva_norm = g(a + 'kv_a_proj_with_mqa'), g(a + 'kv_a_layernorm.weight')
        kvb = g(a + 'kv_b_proj')
        nh = kvb.shape[0] // (2 * R.MLA_HD)
        kvb = kvb.reshape(nh, 2 * R.MLA_HD, R.KVL)
        st.wkb, st.wvb, st.wo = kvb[:, :R.MLA_HD, :], kvb[:, R.MLA_HD:, :], g(a + 'o_proj')
        i = a + 'indexer.'
        st.iq, st.ik, st.ik_w, st.ik_b = g(i + 'wq_b'), g(i + 'wk'), g(i + 'k_norm.weight'), g(i + 'k_norm.bias')
        st.iw, st.igate, st.iape = g(i + 'weights_proj'), g(i + 'index_kpool_compress_gate'), g(i + 'index_kpool_compress_ape')
        st.lat, st.ikeys, st.igs, st.last = [], [], [], {}
    return st


def kda_heads(st):
    """Rank-local KDA: same math with the rank's head count (q/k/v rows already sliced)."""
    nh = st.A_log.shape[0]
    saved = R.NH, R.QKV
    R.NH, R.QKV = nh, nh * R.HD
    return saved


rng = np.random.default_rng(0)
xs = [rng.standard_normal(R.H) for _ in range(9)]   # 9 decode steps: KDA recurrence, MLA up to 9 tokens + 1 pool

# KDA layer 4
full = stub(prov, 4, 'kda')
ranks = [stub(RankProv(r), 4, 'kda') for r in (0, 1)]
for t, x in enumerate(xs):
    yf = full.kda(x)
    ys = []
    for st in ranks:
        saved = kda_heads(st)
        ys.append(st.kda(x))
        R.NH, R.QKV = saved
    check(f'KDA L4 step {t}: r0 + r1 == full', ys[0] + ys[1], yf)

# MLA + indexer layer 3 (MLA heads 64 -> 32 per rank: patch NH for the einsum shapes via the sliced tensors)
full = stub(prov, 3, 'mla')
ranks = [stub(RankProv(r), 3, 'mla') for r in (0, 1)]
for t, x in enumerate(xs):
    yf = full.mla(x)
    ys = []
    for st in ranks:
        saved = R.NH
        R.NH = st.wkb.shape[0]
        ys.append(st.mla(x))
        R.NH = saved
    check(f'MLA L3 step {t}: r0 + r1 == full', ys[0] + ys[1], yf)

# expert 0 / 287, shared expert (layer 3), dense MLP (layer 0)
for name in ('mlp.experts.0', 'mlp.experts.287', 'mlp.shared_experts', 'mlp'):
    L = 0 if name == 'mlp' else 3
    pre = f'{R.P}layers.{L}.{name}.'
    w = lambda p: [p.get(pre + f'{n}_proj') for n in ('gate', 'up', 'down')]
    x = xs[0] * 0.05
    yf = R.Layer.swiglu(x, *w(prov))
    ys = [R.Layer.swiglu(x, *w(RankProv(r))) for r in (0, 1)]
    check(f'{name} L{L}: r0 + r1 == full', ys[0] + ys[1], yf)

print('FAILS:', fails if fails else 'none')
sys.exit(1 if fails else 0)
