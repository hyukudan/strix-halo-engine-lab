"""Export real-weight cases for the Tier A host emulation check (tests/glm_host_cpu_tests.cpp).

Each case is a directory of raw little-endian .bin arrays + manifest.txt ("name dtype count" per line):
weights in their file dtype (bf16/f16 as u16, f32), inputs as f32, float64 oracle outputs as f64
(cpu_reference.py math, O-math). The C++ test runs the host functions on the f32 inputs and reports the relative
error vs the oracle per block (a semantic check of the declared arithmetic, not a bitwise gate).
usage: export_host_case.py FIXTURE_DIR TRELLIS_PYTHON OUT_DIR
"""
import math, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import cpu_reference as R  # noqa: E402

fix, tp, out = sys.argv[1:4]
prov = R.Exl3Provider(fix, tp)
s = prov.s
rng = np.random.default_rng(1234)
P3, P4 = f'{R.P}layers.3.', f'{R.P}layers.4.'


class Case:
    def __init__(self, name):
        self.dir = Path(out) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lines = []

    def raw(self, name, tensor):          # file dtype, as stored
        f, base, v = s.idx[tensor]
        a, b = v['data_offsets']
        with open(f, 'rb') as fh:
            fh.seek(base + a)
            data = fh.read(b - a)
        dt = {'BF16': 'u16', 'F16': 'u16', 'F32': 'f32'}[v['dtype']]
        (self.dir / f'{name}.bin').write_bytes(data)
        self.lines.append(f'{name} {dt} {(b - a) // (2 if dt == "u16" else 4)} {v["dtype"]}')

    def arr(self, name, x, dt):
        x = np.ascontiguousarray(x, dtype={'f32': np.float32, 'f64': np.float64, 'u32': np.uint32}[dt])
        x.tofile(self.dir / f'{name}.bin')
        self.lines.append(f'{name} {dt} {x.size} -')

    def close(self):
        (self.dir / 'manifest.txt').write_text('\n'.join(self.lines) + '\n')


f32 = lambda x: np.asarray(x, np.float32).astype(np.float64)   # what the C++ side receives

# ---- mHC (layer 3 attention site): random streams
c = Case('hc_l3')
c.raw('fn', P3 + 'hc_attn_fn'); c.raw('base', P3 + 'hc_attn_base'); c.raw('scale', P3 + 'hc_attn_scale')
X = f32(rng.standard_normal((4, R.H)))
pre, post, comb = R.hc_weights(X, prov.get(P3 + 'hc_attn_fn'), prov.get(P3 + 'hc_attn_base'), prov.get(P3 + 'hc_attn_scale'))
y = f32(rng.standard_normal(R.H))
c.arr('X', X, 'f32'); c.arr('y', y, 'f32')
c.arr('pre', pre, 'f64'); c.arr('post', post, 'f64'); c.arr('comb', comb, 'f64')
c.arr('collapse', pre @ X, 'f64'); c.arr('expand', R.hc_apply(X, post, comb, y), 'f64')
c.close()

# ---- KDA layer 4, rank 0 (32 heads) over 12 steps: inputs = pre-conv qkv, f, gate, b of the rank
c = Case('kda_l4_r0')
H = 32
a = 'self_attn.'
qkv_full = [prov.get(P4 + a + f'{n}_proj') for n in 'qkv']
conv = np.asarray(s.raw(P4 + a + 'conv1d.weight'))          # f64 view of BF16 [24576,1,4]
fa, fb, ga, gb = (prov.get(P4 + a + n) for n in ('f_a_proj', 'f_b_proj', 'g_a_proj', 'g_b_proj'))
bp, A_log, dt_bias, o_norm = prov.get(P4 + a + 'b_proj'), prov.get(P4 + a + 'A_log'), prov.get(P4 + a + 'dt_bias'), prov.get(P4 + a + 'o_norm.weight')
# rank-0 slices of the plain weights as stored (u16): conv rows q 0..4095, k 8192.., v 16384..
f_, base_, v_ = s.idx[P4 + a + 'conv1d.weight']
convraw = np.fromfile(f_, dtype=np.uint16, count=24576 * 4, offset=base_ + v_['data_offsets'][0]).reshape(24576, 4)
np.concatenate([convraw[0:4096], convraw[8192:12288], convraw[16384:20480]]).tofile(c.dir / 'conv.bin')
c.lines.append('conv u16 49152 BF16')
c.arr('A_log', A_log[:H], 'f32'); c.arr('dt_bias', dt_bias[:H * 128], 'f32')
c.raw('o_norm', P4 + a + 'o_norm.weight')
st = R.Layer.__new__(R.Layer)
st.conv = np.concatenate([conv[0:4096, 0, :], conv[8192:12288, 0, :], conv[16384:20480, 0, :]])
st.S, st.conv_state = np.zeros((H, 128, 128)), np.zeros((3 * 4096, 3))
steps = 12
ins = {k: [] for k in ('qkv', 'f', 'gate', 'b')}
outs = []
saved = R.NH, R.QKV
R.NH, R.QKV = H, H * 128
for t in range(steps):
    x = rng.standard_normal(R.H) * 0.5
    qkv = f32(np.concatenate([w[:H * 128] @ x for w in qkv_full]))
    f = f32((fb @ (fa @ x))[:H * 128]); gate = f32((gb @ (ga @ x))[:H * 128]); b = f32((bp @ x)[:H])
    for k, v in (('qkv', qkv), ('f', f), ('gate', gate), ('b', b)):
        ins[k].append(v)
    # oracle KDA core on these exact inputs (the part after the projections)
    buf = np.concatenate([st.conv_state, qkv[:, None]], axis=1)
    st.conv_state = buf[:, 1:]
    yv = R.silu((buf * st.conv).sum(axis=1))
    q, k, v = (yv[i * R.QKV:(i + 1) * R.QKV].reshape(H, 128) for i in range(3))
    g = -5.0 * R.sigmoid(np.exp(A_log[:H])[:, None] * (f + dt_bias[:H * 128]).reshape(H, 128))
    beta = R.sigmoid(b)
    q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) / math.sqrt(128)
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    S = st.S * np.exp(g)[:, :, None]
    u = np.einsum('hkv,hk->hv', S, k)
    S = S + k[:, :, None] * ((v - u) * beta[:, None])[:, None, :]
    st.S = S
    o = np.einsum('hkv,hk->hv', S, q)
    outs.append((R.rms(o, None, R.EPS) * o_norm * R.sigmoid(gate.reshape(H, 128))).reshape(-1))
R.NH, R.QKV = saved
for k, v in ins.items():
    c.arr(k, np.array(v), 'f32')
c.arr('y', np.array(outs), 'f64')
c.arr('S_final', st.S, 'f64')
c.close()

# ---- MLA + indexer, layer 3, rank 0 (32 heads) over 9 tokens of real projections, plus a long synthetic case
c = Case('mla_l3_r0')
for n in ('kv_b_proj.weight', 'indexer.wk.weight', 'indexer.index_kpool_compress_gate', 'indexer.weights_proj.weight',
          'indexer.k_norm.weight', 'indexer.k_norm.bias', 'indexer.index_kpool_compress_ape'):
    c.raw(n[:-len('.weight')] if n.endswith('.weight') else n, P3 + a + n)
lay = R.Layer.__new__(R.Layer)
g3 = lambda n: prov.get(P3 + a + n)
lay.qa, lay.qa_norm, lay.qb = g3('q_a_proj'), g3('q_a_layernorm.weight'), g3('q_b_proj')
lay.kva, lay.kva_norm = g3('kv_a_proj_with_mqa'), g3('kv_a_layernorm.weight')
kvb = g3('kv_b_proj').reshape(64, 512, 512)
lay.wkb, lay.wvb, lay.wo = kvb[:, :256, :], kvb[:, 256:, :], g3('o_proj')
i_ = 'indexer.'
lay.iq, lay.ik, lay.ik_w, lay.ik_b = g3(i_ + 'wq_b'), g3(i_ + 'wk'), g3(i_ + 'k_norm.weight'), g3(i_ + 'k_norm.bias')
lay.iw, lay.igate, lay.iape = g3(i_ + 'weights_proj'), g3(i_ + 'index_kpool_compress_gate'), g3(i_ + 'index_kpool_compress_ape')
lay.lat, lay.ikeys, lay.igs, lay.last = [], [], [], {}
T = 9
X_, LAT, Q, QI, O, OB, SEL = [], [], [], [], [], [], []
bf = lambda v: (((np.asarray(v, np.float32).view(np.uint32).astype(np.uint64) + 0x7FFF + ((np.asarray(v, np.float32).view(np.uint32).astype(np.uint64) >> 16) & 1)) >> 16) << 16).astype(np.uint32).view(np.float32).astype(np.float64)
for t in range(T):
    x = f32(rng.standard_normal(R.H) * 0.5)
    q_resid = R.rms(lay.qa @ x, lay.qa_norm)
    q = f32((lay.qb @ q_resid).reshape(64, 256)[:32])
    latent = f32(R.rms(lay.kva @ x, lay.kva_norm))
    qI = f32(lay.iq @ q_resid)
    X_.append(x); LAT.append(latent); Q.append(q); QI.append(qI)
    lay.lat.append(latent)
    idx = lay.indexer_select(x, q_resid)
    C = np.array(lay.lat)[idx]
    qt = np.einsum('hd,hdl->hl', q, lay.wkb[:32])
    sc = qt @ C.T / 16.0
    sc = np.exp(sc - sc.max(axis=1, keepdims=True)); sc = sc / sc.sum(axis=1, keepdims=True)
    O.append(np.einsum('hl,hdl->hd', sc @ C, lay.wvb[:32]).reshape(-1))
    Cb = bf(C)  # the same attention on bf16-rounded cached latents (the declared cache rounding site)
    sb = qt @ Cb.T / 16.0
    sb = np.exp(sb - sb.max(axis=1, keepdims=True)); sb = sb / sb.sum(axis=1, keepdims=True)
    OB.append(np.einsum('hl,hdl->hd', sb @ Cb, lay.wvb[:32]).reshape(-1))
    SEL.append(len(idx))
for n, v in (('x', X_), ('latent', LAT), ('q', Q), ('qI', QI)):
    c.arr(n, np.array(v), 'f32')
c.arr('o', np.array(O), 'f64')
c.arr('o_bf16lat', np.array(OB), 'f64')
c.close()

# ---- router + expert 0 + shared (layer 3)
c = Case('moe_l3')
c.raw('router', P3 + 'mlp.gate.weight'); c.raw('bias', P3 + 'mlp.gate.e_score_correction_bias')
x = f32(rng.standard_normal(R.H) * 0.3)
s_ = R.sigmoid(prov.get(P3 + 'mlp.gate.weight') @ x)
idx = R.topk_lower_index(s_ + prov.get(P3 + 'mlp.gate.e_score_correction_bias'), 8)
ids = np.sort(idx)
w = s_[ids] / (s_[ids].sum() + 1e-20) * 2.5
c.arr('x', x, 'f32'); c.arr('ids', ids, 'u32'); c.arr('w', w, 'f64')
c.close()
print('exported to', out)
