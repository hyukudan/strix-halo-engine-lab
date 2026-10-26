"""CPU per-layer reference for MiniMax-M3.1-Flash (EXL3 pack), phase 1 of the bring-up.

Two independent implementations of the same layers, compared on real activations:

A. Oracle ("sglang semantics", float64). Literal transcription of the SGLang model
   (python/sglang/srt/models/minimax_m3.py) and its MSA backend
   (layers/attention/minimax_sparse_backend.py + kernels/ops/attention/minimax_sparse,
   revision 299a75ac5d962087f8f09e19b23ff43548fc64c8). Weights are reconstructed in the
   ORIGINAL basis, W = diag(suh) H W_rot H diag(svh) (strix-trellis reconstruct
   formula), applied with dense matmuls; sparse attention is evaluated as a dense
   softmax under the block mask chosen by torch.topk.

B. Ours ("engine style"). Linear layers through the rotated basis
   (x*suh -> H128 -> x W_rot -> H128 -> *svh); MSA as an explicit indexer -> per-block
   max -> declared top-16 (score desc, block id asc, local block forced) -> per-group
   block-streaming online softmax over the selected blocks only. Two numerics:
     * exact: float64 everywhere (must equal A to rounding noise);
     * prod: the strix-trellis production contract for EXL3 linears (FP16 scaled input,
       FP32 Hadamard, FP16 core input, FP16 x FP16 products in FP32, FP32 output stages),
       FP32 elsewhere, K/V and index-K stored as BF16 (the SGLang default cache dtype).

Reads only the tensors of the requested layers from the local pack.
"""
import argparse
import concurrent.futures as cf
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

PFX = "language_model.model."


# --------------------------------------------------------------------------- pack access
class Pack:
    def __init__(self, root, trellis_python):
        sys.path.insert(0, trellis_python)
        import strix_trellis as st
        self.st = st
        self.trellis_python = trellis_python
        self.root = Path(root)
        self.cfg = json.loads((self.root / "config.json").read_text())["text_config"]
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self.files = {}

    def _file(self, name):
        fn = self.index[name]
        if fn not in self.files:
            self.files[fn] = self.st.SafetensorsFile(self.root / fn)
        return self.files[fn]

    def plain(self, name):
        return np.array(self._file(name).tensor_float32(name), dtype=np.float64)

    def exl3(self, prefix):
        return self.st.load_exl3(self._file(prefix + ".trellis"), prefix)

    def embed_rows(self, ids):
        name = PFX + "embed_tokens.weight"
        raw = self._file(name).tensor(name, raw_bf16=True)
        rows = np.asarray(raw[np.asarray(ids)])
        return (rows.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def _decode_job(args):
    root, trellis_python, prefix = args
    sys.path.insert(0, trellis_python)
    import strix_trellis as st
    index = json.loads((Path(root) / "model.safetensors.index.json").read_text())["weight_map"]
    w = st.load_exl3(Path(root) / index[prefix + ".trellis"], prefix)
    rot = st.dequantize_rotated_vectorized(np.asarray(w.trellis), w.bits)
    return prefix, rot, np.array(w.suh), np.array(w.svh)


class Linear:
    """One EXL3 linear: rotated FP16 matrix [in, out] plus folded FP16 scales."""

    def __init__(self, rot, suh, svh, st):
        self.rot, self.suh, self.svh, self.st = rot, suh, svh, st

    def original_weight(self):
        # Oracle path: W in the original basis, float64 ([in, out]).
        H = self.st.hadamard
        w = H(H(self.rot.astype(np.float64), axis=0), axis=1)
        return self.suh.astype(np.float64)[:, None] * w * self.svh.astype(np.float64)[None, :]

    def apply_exact(self, x):
        H = self.st.hadamard
        a = H(np.asarray(x, np.float64) * self.suh.astype(np.float64))
        core = torch.from_numpy(a) @ torch.from_numpy(self.rot.astype(np.float64))
        return H(core.numpy()) * self.svh.astype(np.float64)

    def apply_prod(self, x):
        st = self.st
        stages = st.input_stages_fp32(np.asarray(x, np.float32), self.suh)
        a = torch.from_numpy(stages["core_input_fp16"].astype(np.float32))
        # FP16 x FP16 products are exact in FP32; only the summation order differs
        # from core_dot_fp32 (BLAS blocking instead of increasing input index).
        core = (a @ torch.from_numpy(self.rot.astype(np.float32))).numpy()
        return st.output_stages_fp32(core, self.svh)["output_fp32"].astype(np.float32)


def load_linears(pack, prefixes, workers):
    jobs = [(str(pack.root), pack.trellis_python, p) for p in prefixes]
    out = {}
    with cf.ProcessPoolExecutor(workers) as ex:
        for prefix, rot, suh, svh in ex.map(_decode_job, jobs):
            out[prefix] = Linear(rot, suh, svh, pack.st)
    return out


# --------------------------------------------------------------------------- shared math
def gemma_rmsnorm(x, w, eps, dtype):
    x = np.asarray(x, dtype)
    var = np.mean(x * x, axis=-1, keepdims=True, dtype=dtype)
    return (x / np.sqrt(var + dtype(eps))) * (dtype(1.0) + w.astype(dtype))


def rope_tables(positions, rotary_dim, theta, dtype):
    inv = 1.0 / (theta ** (np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim))
    ang = np.asarray(positions, np.float64)[:, None] * inv[None, :]
    return np.cos(ang).astype(dtype), np.sin(ang).astype(dtype)


def rope_neox(x, cos, sin, rotary_dim):
    # x [T, heads, D]; neox halves over the first rotary_dim dims, rest unchanged.
    half = rotary_dim // 2
    x1, x2 = x[..., :half], x[..., half:rotary_dim]
    c, s = cos[:, None, :], sin[:, None, :]
    out = x.copy()
    out[..., :half] = x1 * c - x2 * s
    out[..., half:rotary_dim] = x2 * c + x1 * s
    return out


def swiglu_oai(gate, up, alpha, limit):
    gate = np.minimum(gate, limit)
    up = np.clip(up, -limit, limit)
    return gate * (1.0 / (1.0 + np.exp(-alpha * gate))) * (up + 1.0)


def to_bf16(x):
    # Round-to-nearest-even FP32 -> BF16 -> FP32 (cache storage of the prod variant).
    b = np.asarray(x, np.float32).view(np.uint32)
    b = ((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return b.view(np.float32)


# --------------------------------------------------------------------------- oracle (A)
class Oracle:
    def __init__(self, pack, cfg):
        self.pack, self.cfg = pack, cfg
        sa = cfg["sparse_attention_config"]
        self.bs, self.topk = sa["sparse_block_size"], sa["sparse_topk_blocks"]
        self.init_blocks, self.local_blocks = sa["sparse_init_block"], sa["sparse_local_block"]

    def select_blocks(self, idx_q, idx_k):
        """SGLang prefill indexer semantics: per token t, per index head h."""
        T, nih, d = idx_q.shape
        bs = self.bs
        nb_total = (T + bs - 1) // bs
        scale = d ** -0.5
        q = torch.from_numpy(idx_q)
        k = torch.from_numpy(idx_k[:, 0, :])
        sel = np.zeros((nih, T, nb_total), dtype=bool)
        block_scores = np.full((nih, T, nb_total), -np.inf)
        for h in range(nih):
            s = (q[:, h, :] @ k.T) * scale  # [T, T]
            s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
            pad = nb_total * bs - T
            sp = torch.nn.functional.pad(s, (0, pad), value=float("-inf"))
            bmax = sp.view(T, nb_total, bs).amax(-1)  # max score per block (causal)
            block_scores[h] = bmax.numpy()
            t_idx = torch.arange(T)
            valid = t_idx // bs + 1  # blocks up to and including the query's block
            score = bmax.clone()
            blk = torch.arange(nb_total)[None, :]
            causal = blk < valid[:, None]
            local = causal & (blk >= (valid[:, None] - self.local_blocks).clamp(min=0))
            init = causal & (blk < self.init_blocks)
            score = torch.where(causal, score, torch.full_like(score, -1e30))
            score = torch.where(init, torch.full_like(score, 1e30), score)
            score = torch.where(local, torch.full_like(score, 1e29), score)
            k_sel = min(self.topk, nb_total)
            top = torch.topk(score, k_sel, dim=-1).indices
            mask = torch.zeros(T, nb_total, dtype=torch.bool)
            mask.scatter_(1, top, True)
            mask &= causal
            sel[h] = mask.numpy()
        return sel, block_scores

    def attention(self, L, x, positions, lin):
        c = self.cfg
        eps, hd, nh, nkv = c["rms_norm_eps"], c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"]
        a = f"{PFX}layers.{L}.self_attn."
        T = x.shape[0]
        h = gemma_rmsnorm(x, self.pack.plain(f"{PFX}layers.{L}.input_layernorm.weight"), eps, np.float64)
        q = h @ lin[a + "q_proj"].W
        k = h @ lin[a + "k_proj"].W
        v = h @ lin[a + "v_proj"].W
        q_raw = q.reshape(T, nh, hd)
        q = gemma_rmsnorm(q_raw, self.pack.plain(a + "q_norm.weight"), eps, np.float64)
        k = gemma_rmsnorm(k.reshape(T, nkv, hd), self.pack.plain(a + "k_norm.weight"), eps, np.float64)
        v = v.reshape(T, nkv, hd)
        cos, sin = rope_tables(positions, c["rotary_dim"], c["rope_theta"], np.float64)
        q, k = rope_neox(q, cos, sin, c["rotary_dim"]), rope_neox(k, cos, sin, c["rotary_dim"])
        sa = c["sparse_attention_config"]
        nih, idim = sa["sparse_num_index_heads"], sa["sparse_index_dim"]
        iq = h @ self.pack.plain(a + "index_q_proj.weight").T
        ik = h @ self.pack.plain(a + "index_k_proj.weight").T
        iq = gemma_rmsnorm(iq.reshape(T, nih, idim), self.pack.plain(a + "index_q_norm.weight"), eps, np.float64)
        ik = gemma_rmsnorm(ik.reshape(T, 1, idim), self.pack.plain(a + "index_k_norm.weight"), eps, np.float64)
        iq, ik = rope_neox(iq, cos, sin, c["rotary_dim"]), rope_neox(ik, cos, sin, c["rotary_dim"])
        sel, bscores = self.select_blocks(iq, ik)
        # idx head g selects for kv head g (num_idx_heads // num_kv_heads == 1).
        assert nih == nkv
        group = nh // nkv
        out = np.empty((T, nh, hd))
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
        for g in range(nkv):
            tokmask = torch.from_numpy(np.repeat(sel[g], self.bs, axis=1)[:, :T]) & causal
            qg = torch.from_numpy(q[:, g * group:(g + 1) * group, :]).permute(1, 0, 2)  # [16,T,D]
            s = qg @ torch.from_numpy(k[:, g, :]).T * hd ** -0.5
            s = s.masked_fill(~tokmask[None], float("-inf"))
            p = torch.softmax(s, dim=-1)
            out[:, g * group:(g + 1) * group, :] = (p @ torch.from_numpy(v[:, g, :])).permute(1, 0, 2).numpy()
        o = out.reshape(T, nh * hd) @ lin[a + "o_proj"].W
        return x + o, {"sel": sel, "block_scores": bscores, "q": q, "k": k, "v": v, "iq": iq, "ik": ik,
                       "q_raw": q_raw, "attn": out}

    def ffn(self, L, x, lin):
        c = self.cfg
        h = gemma_rmsnorm(x, self.pack.plain(f"{PFX}layers.{L}.post_attention_layernorm.weight"),
                          c["rms_norm_eps"], np.float64)
        alpha, limit = c["swiglu_alpha"], c["swiglu_limit"]
        if not c["moe_layer_freq"][L]:
            f = f"{PFX}layers.{L}.mlp."
            act = swiglu_oai(h @ lin[f + "gate_proj"].W, h @ lin[f + "up_proj"].W, alpha, limit)
            return x + act @ lin[f + "down_proj"].W, {}
        b = f"{PFX}layers.{L}.block_sparse_moe."
        logits = h @ self.pack.plain(b + "gate.weight").T  # FP32 router weight, float64 math
        scores = 1.0 / (1.0 + np.exp(-logits))
        choice = scores + self.pack.plain(b + "e_score_correction_bias")[None, :]
        ids = torch.topk(torch.from_numpy(choice), c["num_experts_per_tok"], dim=-1, sorted=False).indices.numpy()
        wts = np.take_along_axis(scores, ids, axis=1)
        wts = wts / (wts.sum(-1, keepdims=True) + 1e-20) * c["routed_scaling_factor"]
        y = np.zeros_like(h)
        for e in np.unique(ids):
            rows, slot = np.nonzero(ids == e)
            pre = f"{b}experts.{e}."
            act = swiglu_oai(h[rows] @ lin[pre + "w1"].original_weight(),
                             h[rows] @ lin[pre + "w3"].original_weight(), alpha, limit)
            y[rows] += wts[rows, slot][:, None] * (act @ lin[pre + "w2"].original_weight())
        s = b + "shared_experts."
        act = swiglu_oai(h @ lin[s + "gate_proj"].W, h @ lin[s + "up_proj"].W, alpha, limit)
        y += act @ lin[s + "down_proj"].W
        return x + y, {"ids": ids, "weights": wts}


# --------------------------------------------------------------------------- ours (B)
class Ours:
    def __init__(self, pack, cfg, mode):
        assert mode in ("exact", "prod")
        self.pack, self.cfg, self.mode = pack, cfg, mode
        self.dt = np.float64 if mode == "exact" else np.float32
        sa = cfg["sparse_attention_config"]
        self.bs, self.topk = sa["sparse_block_size"], sa["sparse_topk_blocks"]
        self.local_blocks, self.init_blocks = sa["sparse_local_block"], sa["sparse_init_block"]

    def lin(self, linear, x):
        return linear.apply_exact(x) if self.mode == "exact" else linear.apply_prod(x)

    def plain_mm(self, x, w_out_in):
        # Unquantized FP16/FP32 projections (indexer, router): weights kept at source precision.
        return (torch.from_numpy(np.asarray(x, self.dt)) @ torch.from_numpy(w_out_in.astype(self.dt)).T).numpy()

    def store(self, x):
        return to_bf16(x) if self.mode == "prod" else x

    def declared_topk(self, block_scores, t):
        """Top-k blocks for query position t: local forced, then score desc, block id asc."""
        valid = t // self.bs + 1
        if valid <= self.topk:
            return np.arange(valid)
        local_start = max(0, valid - self.local_blocks)
        forced = list(range(min(self.init_blocks, valid))) + list(range(local_start, valid))
        forced = sorted(set(forced))
        rest = [b for b in range(valid) if b not in forced]
        sc = block_scores[rest]
        order = np.lexsort((np.array(rest), -sc))  # primary: score desc; ties: lower id
        chosen = forced + [rest[i] for i in order[: self.topk - len(forced)]]
        return np.sort(np.array(chosen))

    def attention(self, L, x, positions, lin):
        c, dt = self.cfg, self.dt
        eps, hd, nh, nkv = c["rms_norm_eps"], c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"]
        a = f"{PFX}layers.{L}.self_attn."
        T = x.shape[0]
        h = gemma_rmsnorm(x, self.pack.plain(f"{PFX}layers.{L}.input_layernorm.weight"), eps, dt)
        q = self.lin(lin[a + "q_proj"], h).astype(dt)
        k = self.lin(lin[a + "k_proj"], h).astype(dt)
        v = self.lin(lin[a + "v_proj"], h).astype(dt)
        q = gemma_rmsnorm(q.reshape(T, nh, hd), self.pack.plain(a + "q_norm.weight"), eps, dt)
        k = gemma_rmsnorm(k.reshape(T, nkv, hd), self.pack.plain(a + "k_norm.weight"), eps, dt)
        cos, sin = rope_tables(positions, c["rotary_dim"], c["rope_theta"], dt)
        q, k = rope_neox(q, cos, sin, c["rotary_dim"]), rope_neox(k, cos, sin, c["rotary_dim"])
        k, v = self.store(k), self.store(v.reshape(T, nkv, hd))
        sa = c["sparse_attention_config"]
        nih, idim = sa["sparse_num_index_heads"], sa["sparse_index_dim"]
        iq = self.plain_mm(h, self.pack.plain(a + "index_q_proj.weight"))
        ik = self.plain_mm(h, self.pack.plain(a + "index_k_proj.weight"))
        iq = gemma_rmsnorm(iq.reshape(T, nih, idim), self.pack.plain(a + "index_q_norm.weight"), eps, dt)
        ik = gemma_rmsnorm(ik.reshape(T, 1, idim), self.pack.plain(a + "index_k_norm.weight"), eps, dt)
        iq, ik = rope_neox(iq, cos, sin, c["rotary_dim"]), self.store(rope_neox(ik, cos, sin, c["rotary_dim"]))
        # Indexer: per index head, per block max of q.k (scale is monotonic; applied as declared).
        nb = (T + self.bs - 1) // self.bs
        bscores = np.full((nih, T, nb), -np.inf, dtype=dt)
        kt = torch.from_numpy(ik[:, 0, :].astype(dt))
        for h_ in range(nih):
            for b in range(nb):
                lo, hi = b * self.bs, min(T, (b + 1) * self.bs)
                s = (torch.from_numpy(iq[lo:, h_, :]) @ kt[lo:hi].T).numpy() * dt(idim ** -0.5)
                tpos = np.arange(lo, T)[:, None]
                s = np.where(np.arange(lo, hi)[None, :] <= tpos, s, -np.inf)
                bscores[h_, lo:, b] = s.max(axis=1)
        sel = np.zeros((nih, T, nb), dtype=bool)
        for h_ in range(nih):
            for t in range(T):
                sel[h_, t, self.declared_topk(bscores[h_, t], t)] = True
        # Main attention: per kv group, stream only the selected blocks, online softmax.
        group = nh // nkv
        out = np.zeros((T, nh, hd), dtype=dt)
        scale = dt(hd ** -0.5)
        chunk = 256
        for g in range(nkv):
            kg, vg = torch.from_numpy(k[:, g, :].astype(dt)), torch.from_numpy(v[:, g, :].astype(dt))
            for t0 in range(0, T, chunk):
                t1 = min(T, t0 + chunk)
                qg = torch.from_numpy(q[t0:t1, g * group:(g + 1) * group, :])  # [C,16,D]
                m = torch.full((t1 - t0, group), -math.inf, dtype=qg.dtype)
                l = torch.zeros((t1 - t0, group), dtype=qg.dtype)
                acc = torch.zeros((t1 - t0, group, hd), dtype=qg.dtype)
                tpos = torch.arange(t0, t1)
                for b in range(nb):
                    use = torch.from_numpy(sel[g, t0:t1, b])
                    if not bool(use.any()):
                        continue
                    lo, hi = b * self.bs, min(T, (b + 1) * self.bs)
                    s = torch.einsum("chd,nd->chn", qg, kg[lo:hi]) * scale
                    ok = use[:, None] & (torch.arange(lo, hi)[None, :] <= tpos[:, None])
                    s = s.masked_fill(~ok[:, None, :], -math.inf)
                    m_new = torch.maximum(m, s.amax(-1))
                    safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
                    p = torch.exp(s - safe[..., None])
                    corr = torch.exp(torch.where(torch.isinf(m), torch.full_like(m, -math.inf), m - safe))
                    l = l * corr + p.sum(-1)
                    acc = acc * corr[..., None] + torch.einsum("chn,nd->chd", p, vg[lo:hi])
                    m = m_new
                out[t0:t1, g * group:(g + 1) * group, :] = (acc / l[..., None]).numpy()
        o = self.lin(lin[a + "o_proj"], out.reshape(T, nh * hd)).astype(dt)
        return (x.astype(dt) + o), {"sel": sel, "block_scores": bscores}

    def ffn(self, L, x, lin):
        c, dt = self.cfg, self.dt
        h = gemma_rmsnorm(x, self.pack.plain(f"{PFX}layers.{L}.post_attention_layernorm.weight"),
                          c["rms_norm_eps"], dt)
        alpha, limit = dt(c["swiglu_alpha"]), dt(c["swiglu_limit"])
        if not c["moe_layer_freq"][L]:
            f = f"{PFX}layers.{L}.mlp."
            act = swiglu_oai(self.lin(lin[f + "gate_proj"], h).astype(dt),
                             self.lin(lin[f + "up_proj"], h).astype(dt), alpha, limit)
            return x.astype(dt) + self.lin(lin[f + "down_proj"], act).astype(dt), {}
        b = f"{PFX}layers.{L}.block_sparse_moe."
        logits = self.plain_mm(h, self.pack.plain(b + "gate.weight"))
        scores = dt(1.0) / (dt(1.0) + np.exp(-logits))
        choice = scores + self.pack.plain(b + "e_score_correction_bias").astype(dt)[None, :]
        # Declared selection: score desc, expert id asc on ties.
        order = np.lexsort((np.broadcast_to(np.arange(choice.shape[1]), choice.shape), -choice), axis=1)
        ids = order[:, : c["num_experts_per_tok"]]
        wts = np.take_along_axis(scores, ids, axis=1)
        wts = wts / (wts.sum(-1, keepdims=True, dtype=dt) + dt(1e-20)) * dt(c["routed_scaling_factor"])
        y = np.zeros_like(h)
        for e in np.unique(ids):
            rows, slot = np.nonzero(ids == e)
            pre = f"{b}experts.{e}."
            act = swiglu_oai(self.lin(lin[pre + "w1"], h[rows]).astype(dt),
                             self.lin(lin[pre + "w3"], h[rows]).astype(dt), alpha, limit)
            y[rows] += wts[rows, slot][:, None] * self.lin(lin[pre + "w2"], act).astype(dt)
        s = b + "shared_experts."
        act = swiglu_oai(self.lin(lin[s + "gate_proj"], h).astype(dt),
                         self.lin(lin[s + "up_proj"], h).astype(dt), alpha, limit)
        y += self.lin(lin[s + "down_proj"], act).astype(dt)
        return x.astype(dt) + y, {"ids": ids, "weights": wts}


# --------------------------------------------------------------------------- comparison
def rel_err(a, ref):
    a, ref = np.asarray(a, np.float64), np.asarray(ref, np.float64)
    d = a - ref
    per_tok = np.linalg.norm(d.reshape(len(d), -1), axis=1) / np.maximum(
        np.linalg.norm(ref.reshape(len(ref), -1), axis=1), 1e-300)
    return {"max_abs": float(np.abs(d).max()), "max_abs_ref": float(np.abs(ref).max()),
            "rel_l2_global": float(np.linalg.norm(d) / np.linalg.norm(ref)),
            "rel_l2_token_max": float(per_tok.max()), "rel_l2_token_p50": float(np.median(per_tok))}


def selection_diff(sel_a, sel_b, trivial_upto):
    diff = (sel_a != sel_b).any(-1)  # [heads, T]
    return {"rows_total": int(diff.size), "rows_differ": int(diff.sum()),
            "rows_nontrivial": int(diff[:, trivial_upto:].size),
            "differ_positions_sample": [int(x) for x in np.nonzero(diff.any(0))[0][:20]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--trellis-python", required=True)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--text", nargs="+", required=True, help="text files concatenated as the prompt")
    ap.add_argument("--layers", default="0,1,2,3")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gold-dir", help="golden mode: oracle + CPU prod chain only; dump hidden states per layer")
    ap.add_argument("--full-dir", help="full-model mode: exact (float64, rotated basis) and CPU prod chains over all "
                    "layers + final norm + lm_head; dumps logits of the scored positions (G4 oracle)")
    ap.add_argument("--full-layers", type=int, default=0, help="full mode: first N layers only (smoke), 0 = all")
    args = ap.parse_args()
    gold = Path(args.gold_dir) if args.gold_dir else None
    torch.set_num_threads(max(1, (torch.get_num_threads())))
    pack = Pack(args.model, args.trellis_python)
    cfg = pack.cfg
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(args.model) / "dspark" / "tokenizer.json"))
    text = "\n\n".join(Path(p).read_text(encoding="utf-8") for p in args.text)
    ids = tok.encode(text).ids[: args.tokens]
    if len(ids) < args.tokens:
        raise SystemExit(f"prompt too short: {len(ids)} tokens")
    T = len(ids)
    positions = np.arange(T)
    x = pack.embed_rows(ids)
    if gold:
        gold.mkdir(parents=True, exist_ok=True)
        np.asarray(ids, dtype="<u4").tofile(gold / "tokens.u32")
    if args.full_dir:
        return run_full(pack, cfg, ids, x, args)
    report = {"tokens": T, "sglang_revision": "299a75ac5d962087f8f09e19b23ff43548fc64c8", "layers": {}}
    oracle, exact, prod = Oracle(pack, cfg), Ours(pack, cfg, "exact"), Ours(pack, cfg, "prod")
    x_prod_chain = x.astype(np.float32)
    trivial = cfg["sparse_attention_config"]["sparse_topk_blocks"] * cfg["sparse_attention_config"]["sparse_block_size"]
    for L in [int(s) for s in args.layers.split(",")]:
        t0 = time.time()
        a = f"{PFX}layers.{L}.self_attn."
        names = [a + n for n in ("q_proj", "k_proj", "v_proj", "o_proj")]
        lin = load_linears(pack, names, args.workers)
        for n in names:
            lin[n].W = lin[n].original_weight()
        xa, ia = oracle.attention(L, x, positions, lin)
        xpc, ipc = prod.attention(L, x_prod_chain, positions, lin)
        if gold:
            xe, ie, xp, ip = xa, ia, xpc, ipc
        else:
            xe, ie = exact.attention(L, x, positions, lin)
            xp, ip = prod.attention(L, x, positions, lin)
        rep = {"attn_out_exact": rel_err(xe - x, xa - x), "attn_out_prod": rel_err(xp - x, xa - x),
               "sel_exact_vs_oracle": selection_diff(ia["sel"], ie["sel"], trivial),
               "sel_prod_vs_oracle": selection_diff(ia["sel"], ip["sel"], trivial)}
        if gold:
            np.packbits(ia["sel"], axis=-1).tofile(gold / f"sel{L}.u8")  # [4][T][ceil(blocks/8)]
        # Near-tie diagnostics for selection: gap between 16th and 17th block score (oracle).
        bsc = ia["block_scores"]
        gaps = []
        for h_ in range(bsc.shape[0]):
            for t in range(trivial, T, 7):
                valid = t // 128 + 1
                s = np.sort(bsc[h_, t, : valid - 1])[::-1]
                if len(s) >= 16:
                    gaps.append(s[14] - s[15])
        rep["oracle_topk_margin_15_16"] = {"min": float(np.min(gaps)), "p01": float(np.percentile(gaps, 1)),
                                            "median": float(np.median(gaps))} if gaps else None
        del lin
        mlp_names = ([f"{PFX}layers.{L}.mlp.{n}" for n in ("gate_proj", "up_proj", "down_proj")]
                     if not cfg["moe_layer_freq"][L] else
                     [f"{PFX}layers.{L}.block_sparse_moe.shared_experts.{n}" for n in ("gate_proj", "up_proj", "down_proj")])
        if cfg["moe_layer_freq"][L]:
            # Experts actually used by the oracle or by either variant of ours.
            b = f"{PFX}layers.{L}.block_sparse_moe."
            hh = gemma_rmsnorm(xa, pack.plain(f"{PFX}layers.{L}.post_attention_layernorm.weight"),
                               cfg["rms_norm_eps"], np.float64)
            used = set()
            for hx in (hh, gemma_rmsnorm(xp, pack.plain(f"{PFX}layers.{L}.post_attention_layernorm.weight"),
                                         cfg["rms_norm_eps"], np.float32),
                       gemma_rmsnorm(xpc, pack.plain(f"{PFX}layers.{L}.post_attention_layernorm.weight"),
                                     cfg["rms_norm_eps"], np.float32)):
                lg = np.asarray(hx, np.float64) @ pack.plain(b + "gate.weight").T
                ch = 1 / (1 + np.exp(-lg)) + pack.plain(b + "e_score_correction_bias")
                used |= set(np.argsort(-ch, axis=1)[:, :8].ravel().tolist())
            mlp_names += [f"{b}experts.{e}.{w}" for e in sorted(used) for w in ("w1", "w3", "w2")]
            rep["experts_decoded"] = len(used)
        lin = load_linears(pack, mlp_names, args.workers)
        for n in mlp_names:
            if "experts." not in n or "shared_experts" in n:
                lin[n].W = lin[n].original_weight()
        ya, fa = oracle.ffn(L, xa, lin)
        ypc, fpc = prod.ffn(L, xpc, lin)
        if gold:
            ye, fe, yp, fp = ya, fa, ypc, fpc
            np.ascontiguousarray(ya, dtype="<f4").tofile(gold / f"h{L}.f32")
            np.ascontiguousarray(ypc, dtype="<f4").tofile(gold / f"h{L}_cpuprod.f32")
            if "ids" in fa:
                np.ascontiguousarray(np.sort(fa["ids"], 1), dtype="<u1").tofile(gold / f"router{L}.u8")
        else:
            ye, fe = exact.ffn(L, xa, lin)
            yp, fp = prod.ffn(L, xa.astype(np.float32), lin)
        rep["ffn_out_exact"] = rel_err(ye - xa, ya - xa)
        rep["ffn_out_prod"] = rel_err(yp - xa, ya - xa)
        if "ids" in fa:
            sa_ = np.sort(fa["ids"], 1)
            rep["router_rows_differ_exact"] = int((sa_ != np.sort(fe["ids"], 1)).any(1).sum())
            rep["router_rows_differ_prod"] = int((sa_ != np.sort(fp["ids"], 1)).any(1).sum())
        rep["layer_out_exact"] = rel_err(ye, ya)
        rep["layer_out_prod"] = rel_err(yp, ya)
        rep["prod_chain_vs_oracle"] = rel_err(ypc, ya)
        rep["seconds"] = round(time.time() - t0, 1)
        report["layers"][str(L)] = rep
        print(json.dumps({"layer": L, **{k: v for k, v in rep.items() if k.startswith(("layer_out", "prod_chain", "sel_", "router", "seconds"))}}), flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1))
        x, x_prod_chain = ya, ypc
        del lin


def scored_positions(T):
    """G4 scoring rows: every 8th position of the dense-attention part, then every position past 16 blocks."""
    return [p for p in range(7, min(T, 2048), 8)] + list(range(2048, T))


def run_full(pack, cfg, ids, x, args):
    """Exact chain (G4 oracle) and CPU production chain through every layer, then norm + lm_head logits."""
    out = Path(args.full_dir)
    out.mkdir(parents=True, exist_ok=True)
    T = len(ids)
    positions = np.arange(T)
    np.asarray(ids, dtype="<u4").tofile(out / "tokens.u32")
    rows = np.array(scored_positions(T), dtype=np.int64)
    rows.astype("<u4").tofile(out / "rows.u32")
    exact, prod = Ours(pack, cfg, "exact"), Ours(pack, cfg, "prod")
    xe, xp = x.astype(np.float64), x.astype(np.float32)
    log = {"tokens": T, "rows": len(rows), "layers": {}}
    for L in range(args.full_layers or cfg["num_hidden_layers"]):
        t0 = time.time()
        a = f"{PFX}layers.{L}.self_attn."
        names = [a + n for n in ("q_proj", "k_proj", "v_proj", "o_proj")]
        lin = load_linears(pack, names, args.workers)
        xe2, _ = exact.attention(L, xe, positions, lin)
        xp2, _ = prod.attention(L, xp, positions, lin)
        del lin
        if cfg["moe_layer_freq"][L]:
            b = f"{PFX}layers.{L}.block_sparse_moe."
            w = pack.plain(f"{PFX}layers.{L}.post_attention_layernorm.weight")
            used = set()
            for hx in (gemma_rmsnorm(xe2, w, cfg["rms_norm_eps"], np.float64),
                       gemma_rmsnorm(xp2, w, cfg["rms_norm_eps"], np.float32)):
                lg = np.asarray(hx, np.float64) @ pack.plain(b + "gate.weight").T
                ch = 1 / (1 + np.exp(-lg)) + pack.plain(b + "e_score_correction_bias")
                used |= set(np.argsort(-ch, axis=1)[:, :8].ravel().tolist())
            names = [f"{b}shared_experts.{n}" for n in ("gate_proj", "up_proj", "down_proj")]
            names += [f"{b}experts.{e}.{w_}" for e in sorted(used) for w_ in ("w1", "w3", "w2")]
        else:
            names = [f"{PFX}layers.{L}.mlp.{n}" for n in ("gate_proj", "up_proj", "down_proj")]
        lin = load_linears(pack, names, args.workers)
        xe, _ = exact.ffn(L, xe2, lin)
        xp, _ = prod.ffn(L, xp2, lin)
        del lin
        e = rel_err(xp, xe)
        log["layers"][str(L)] = {"prod_vs_exact": e, "seconds": round(time.time() - t0, 1)}
        print(json.dumps({"layer": L, "rel_l2": e["rel_l2_global"], "tok_max": e["rel_l2_token_max"],
                          "seconds": round(time.time() - t0, 1)}), flush=True)
        Path(args.out).write_text(json.dumps(log, indent=1))
    np.ascontiguousarray(xe, dtype="<f4").tofile(out / "h_last_exact.f32")
    np.ascontiguousarray(xp, dtype="<f4").tofile(out / "h_last_cpuprod.f32")
    # Final norm + lm_head (K8), output columns in chunks (H128 is block diagonal along the output).
    wn = pack.plain(f"{PFX}norm.weight")
    he = gemma_rmsnorm(xe[rows], wn, cfg["rms_norm_eps"], np.float64)
    hp = gemma_rmsnorm(xp[rows], wn, cfg["rms_norm_eps"], np.float32)
    head = load_linears(pack, ["language_model.lm_head"], 1)["language_model.lm_head"]
    st = pack.st
    ae = st.hadamard(he * head.suh.astype(np.float64))
    ap_ = st.input_stages_fp32(hp, head.suh)["core_input_fp16"].astype(np.float32)
    V = head.rot.shape[1]
    le = np.empty((len(rows), V), dtype=np.float32)
    lp = np.empty((len(rows), V), dtype=np.float32)
    for c0 in range(0, V, 16384):
        c1 = min(V, c0 + 16384)
        r64 = torch.from_numpy(head.rot[:, c0:c1].astype(np.float64))
        core = (torch.from_numpy(ae) @ r64).numpy()
        le[:, c0:c1] = st.hadamard(core) * head.svh[c0:c1].astype(np.float64)
        corep = (torch.from_numpy(ap_) @ r64.float()).numpy()
        lp[:, c0:c1] = st.output_stages_fp32(corep, head.svh[c0:c1])["output_fp32"]
    le.tofile(out / "logits_exact.f32")
    lp.tofile(out / "logits_cpuprod.f32")
    # KL(exact || cpuprod) and top-1 agreement over the scored rows: the numerics floor for the G4 rule.
    def logsm(z):
        z = z.astype(np.float64)
        z = z - z.max(1, keepdims=True)
        return z - np.log(np.exp(z).sum(1, keepdims=True))
    kl, top = [], []
    for r0 in range(0, len(rows), 256):
        a_, b_ = logsm(le[r0:r0 + 256]), logsm(lp[r0:r0 + 256])
        kl.append((np.exp(a_) * (a_ - b_)).sum(1))
        top.append(le[r0:r0 + 256].argmax(1) == lp[r0:r0 + 256].argmax(1))
    kl, top = np.concatenate(kl), np.concatenate(top)
    log["cpuprod_vs_exact_logits"] = {"kl_mean": float(kl.mean()), "kl_p99": float(np.percentile(kl, 99)),
                                      "kl_max": float(kl.max()), "top1_agree": float(top.mean())}
    Path(args.out).write_text(json.dumps(log, indent=1))
    print(json.dumps(log["cpuprod_vs_exact_logits"]), flush=True)


if __name__ == "__main__":
    main()
