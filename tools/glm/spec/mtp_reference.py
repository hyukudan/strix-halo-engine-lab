#!/usr/bin/env python3
"""GLM-5.3-Flash MTP draft layer (layer 45): float64 oracle (O-math), per design notes (consensus of
llama.cpp / vLLM / SGLang; the transformers reference does not implement it).

  z  = eh_proj([enorm(embed(x_{p+1})) | hnorm(h_p)])        h_p = the trunk's final-normed hidden (4-stream mean, norm)
  r  = z + MLA_DSA(input_layernorm(z))                       own latent/indexer cache, NoPE, plain residual (no mHC)
  y  = r + MoE(post_attention_layernorm(r))                  288 experts top-8 + shared, sigmoid+bias router x2.5
  h' = shared_head.norm(y);  logits = lm_head(h')            shared head; h' is the next draft step's h
Declared choices (same as the engine host reference src/glm/spec/glm_spec_host.cpp): no position-0 embedding zeroing
(llama.cpp/SGLang, F1); a fresh indexer selection per draft row (llama.cpp; identical to vLLM's index sharing up to
2,051 visible tokens, F2); MTP row j = the pair (h_j, x_{j+1}) at MTP cache index j.

The blocks come from cpu_reference.py (same float64 math as the trunk oracle). Uses: (1) the oracle the engine MTP is
measured against once the model lands (G-MTP, design notes section 5); (2) offline acceptance with real trunk taps
(spec_accept.py); (3) smoke test now on the FP8 L45 subset with synthetic h (finite, deterministic).

usage: mtp_reference.py --provider fp8|exl3 --model DIR [--trellis-python DIR] --tokens 1,2,3 [--h-npz taps.npz | --h-seed 1]
       [--steps 3]   (teacher-forced pairs over the token list, then a free-running draft chain of --steps)
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import cpu_reference as R  # noqa: E402

P = R.P + 'layers.45.'


class Mtp:
    def __init__(self, prov):
        g = lambda n: prov.get(P + n)
        self.prov = prov
        self.enorm, self.hnorm, self.eh = g('enorm.weight'), g('hnorm.weight'), g('eh_proj')
        self.head_norm = g('shared_head.norm.weight')
        self.head = prov.get('lm_head')
        # the DSA + MoE blocks of the trunk oracle, without mHC (layer object built field by field)
        L = R.Layer.__new__(R.Layer)
        L.p, L.L, L.pre, L.last = prov, 45, P, {}
        L.in_norm, L.post_norm = g('input_layernorm.weight'), g('post_attention_layernorm.weight')
        a, i = 'self_attn.', 'self_attn.indexer.'
        L.qa, L.qa_norm, L.qb = g(a + 'q_a_proj'), g(a + 'q_a_layernorm.weight'), g(a + 'q_b_proj')
        L.kva, L.kva_norm = g(a + 'kv_a_proj_with_mqa'), g(a + 'kv_a_layernorm.weight')
        kvb = g(a + 'kv_b_proj').reshape(R.NH, 2 * R.MLA_HD, R.KVL)
        L.wkb, L.wvb = kvb[:, :R.MLA_HD, :], kvb[:, R.MLA_HD:, :]
        L.wo = g(a + 'o_proj')
        L.iq, L.ik = g(i + 'wq_b'), g(i + 'wk')
        L.ik_w, L.ik_b = g(i + 'k_norm.weight'), g(i + 'k_norm.bias')
        L.iw, L.igate, L.iape = g(i + 'weights_proj'), g(i + 'index_kpool_compress_gate'), g(i + 'index_kpool_compress_ape')
        L.lat, L.ikeys, L.igs = [], [], []
        L.router, L.rbias = g('mlp.gate.weight'), g('mlp.gate.e_score_correction_bias')
        L.sh = [g(f'mlp.shared_experts.{n}_proj') for n in ('gate', 'up', 'down')]
        L.exp_cache = {}
        self.L = L

    def truncate(self, n):  # rollback of the MTP cache to MTP positions [0, n)
        del self.L.lat[n:], self.L.ikeys[n:], self.L.igs[n:]

    def step(self, h, tok):
        """One MTP row: the pair (h_p, x_{p+1}) appended at the next MTP position. Returns (h', logits)."""
        e = self.prov.s.row(R.P + 'embed_tokens.weight', tok)
        z = R.B(self.eh @ np.concatenate([R.rms(e, self.enorm), R.rms(h, self.hnorm)]))
        r = R.B(z + self.L.mla(R.rms(z, self.L.in_norm)))
        y = R.B(r + self.L.moe(R.rms(r, self.L.post_norm)))
        hp = R.rms(y, self.head_norm)
        return hp, R.B(self.head @ hp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--provider', choices=['fp8', 'exl3'], required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--trellis-python', default='')
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--h-npz', default='', help="trunk taps: array 'h' [n, 4096] (final-normed hidden per position)")
    ap.add_argument('--h-seed', type=int, default=1)
    ap.add_argument('--steps', type=int, default=3)
    a = ap.parse_args()
    prov = R.Fp8Provider(a.model) if a.provider == 'fp8' else R.Exl3Provider(a.model, a.trellis_python)
    toks = [int(t) for t in a.tokens.split(',')]
    n = len(toks) - 1
    if a.h_npz:
        hs = np.load(a.h_npz)['h'].astype(np.float64)
    else:  # synthetic, unit-RMS like a final-normed hidden (smoke only)
        hs = np.random.default_rng(a.h_seed).standard_normal((n, R.H))
    m = Mtp(prov)
    for j in range(n):  # teacher-forced rows (h_j, x_{j+1})
        hp, lg = m.step(hs[j], toks[j + 1])
        print(f'row {j}: pair (h_{j}, x_{j+1}={toks[j + 1]}) -> draft {int(np.argmax(lg))} '
              f'(max logit {lg.max():.4f}, finite {bool(np.isfinite(lg).all())})', flush=True)
    tok = int(np.argmax(lg))
    for s in range(1, a.steps):  # free-running chain: h' recycled
        hp, lg = m.step(hp, tok)
        tok = int(np.argmax(lg))
        print(f'chain step {s}: draft {tok} (max logit {lg.max():.4f})', flush=True)


if __name__ == '__main__':
    main()
