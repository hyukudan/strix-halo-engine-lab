"""PyTorch float64 reference of the MiniMax M3.1 Flash DSpark draft (DSparkMiniMaxDraftModel).

Independent transcription of the reference semantics (ROCm/ATOM atom/models/minimax_m3_dspark.py + dspark_proposer.py,
lightseekorg/tokenspeed minimax_m3_dspark.py; see qualification notes section 1), NOT of the declared order of
the C++ host reference: float64 everywhere, torch matmuls, dense softmax. The only rounding it mirrors from the
declared identity is the BF16 storage of K/V (--kv-bf16, default on), so C++ vs this script differs by FP32
accumulation only.

  gen   writes a case directory: taps.f32 [n][30720] (seeded normal), embed.f32 [7][6144] (the target's real embedding
        rows of [anchor, 200058 x 6]), base.f32 [7][vocab] (seeded normal x 3), meta.json.
  run   reads a case, writes ref_y.f64 [7][6144], ref.json (drafts, confidences, per-step Markov top-2 margin).
"""
import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch

P = "language_model.model.dspark."
H, L, NH, NKV, D, ROT, I, T, WIN = 6144, 5, 64, 4, 128, 64, 12288, 7, 4096
V, R, NOISE, EPS, THETA, ALPHA, LIMIT = 200064, 256, 200058, 1e-6, 1e7, 1.702, 7.0
TAPS = 5


def read_safetensors(path, names=None):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        out = {}
        for k, v in head.items():
            if k == "__metadata__" or (names is not None and k not in names):
                continue
            s, e = v["data_offsets"]
            f.seek(8 + n + s)
            out[k] = (v["dtype"], v["shape"], f.read(e - s))
    return out


def bf16(raw, shape):
    u = np.frombuffer(raw, np.uint16).astype(np.uint32) << 16
    return torch.from_numpy(u.view(np.float32).astype(np.float64).reshape(shape))


def mxfp8(codes, scales, shape):
    w = torch.from_numpy(np.frombuffer(codes, np.uint8).copy()).view(torch.float8_e4m3fn).to(torch.float64).reshape(shape)
    s = torch.from_numpy(np.frombuffer(scales, np.uint8).astype(np.float64) - 127.0).reshape(shape[0], shape[1] // 32)
    return w * torch.repeat_interleave(torch.pow(2.0, s), 32, dim=1)


def to_bf16(x):
    return x.to(torch.float32).to(torch.bfloat16).to(x.dtype)


class Draft:
    def __init__(self, draft_dir):
        files = sorted(Path(draft_dir).glob("*.safetensors"))
        t = {}
        for f in files:
            t.update(read_safetensors(f))
        self.t = t
        self.cast = torch.float64

    def plain(self, name):
        dt, shape, raw = self.t[P + name]
        assert dt == "BF16", name
        return bf16(raw, shape).to(self.cast)

    def linear(self, name):
        dt, shape, raw = self.t[P + name + ".weight"]
        assert dt == "F8_E4M3", name
        _, _, sraw = self.t[P + name + ".weight_scale_inv"]
        return mxfp8(raw, sraw, shape).to(self.cast)


def gemma_norm(x, w):
    var = (x * x).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + EPS) * (1.0 + w)


def rope(x, pos):
    # x [..., n, heads, D]; NeoX halves on the first 64 lanes (gpt_oss rotary), theta 1e7.
    inv = 1.0 / (THETA ** (torch.arange(0, ROT, 2, dtype=torch.float64) / ROT))
    ang = pos.to(torch.float64)[:, None] * inv[None, :]
    c, s = torch.cos(ang)[:, None, :].to(x.dtype), torch.sin(ang)[:, None, :].to(x.dtype)
    x1, x2 = x[..., : ROT // 2], x[..., ROT // 2 : ROT]
    out = x.clone()
    out[..., : ROT // 2] = x1 * c - x2 * s
    out[..., ROT // 2 : ROT] = x2 * c + x1 * s
    return out


def swiglu_oai(g, u):
    g = torch.clamp(g, max=LIMIT)
    u = torch.clamp(u, -LIMIT, LIMIT)
    return g * torch.sigmoid(ALPHA * g) * (u + 1.0)


def forward(dr, taps, embed, P_anchor, kv_bf16=True):
    """taps [n][30720] for positions 0..n-1 (n == P_anchor), embed [7][6144] -> y [7][6144]."""
    n = taps.shape[0]
    first = max(0, P_anchor - (WIN - 1))
    ctx_pos = torch.arange(first, P_anchor)
    c = gemma_norm(taps[first:P_anchor] @ dr.plain("fc.weight").T, dr.plain("hidden_norm.weight"))
    blk_pos = torch.arange(P_anchor, P_anchor + T)
    r = embed.clone()
    for li in range(L):
        p = f"layers.{li}.decoder_layer."
        qn, kn = dr.plain(p + "self_attn.q_norm.weight"), dr.plain(p + "self_attn.k_norm.weight")
        Wq, Wk, Wv, Wo = (dr.linear(p + "self_attn." + k) for k in ("q_proj", "k_proj", "v_proj", "o_proj"))
        # context K/V: straight from c (no input_layernorm), k_norm + RoPE at the context positions
        kc = rope(gemma_norm((c @ Wk.T).view(-1, NKV, D), kn), ctx_pos)
        vc = (c @ Wv.T).view(-1, NKV, D)
        x = gemma_norm(r, dr.plain(p + "input_layernorm.weight"))
        q = rope(gemma_norm((x @ Wq.T).view(T, NH, D), qn), blk_pos)
        kb = rope(gemma_norm((x @ Wk.T).view(T, NKV, D), kn), blk_pos)
        vb = (x @ Wv.T).view(T, NKV, D)
        K, Vv = torch.cat([kc, kb]), torch.cat([vc, vb])
        if kv_bf16:
            K, Vv = to_bf16(K), to_bf16(Vv)
        K = K.repeat_interleave(NH // NKV, dim=1)  # [keys, 64, D]
        Vv = Vv.repeat_interleave(NH // NKV, dim=1)
        s = torch.einsum("thd,khd->htk", q, K) * (D ** -0.5)  # non-causal: every block row sees every key
        a = torch.softmax(s, dim=-1)
        o = torch.einsum("htk,khd->thd", a, Vv).reshape(T, NH * D)
        r = r + o @ Wo.T
        x = gemma_norm(r, dr.plain(p + "post_attention_layernorm.weight"))
        g, u = x @ dr.linear(p + "mlp.gate_proj").T, x @ dr.linear(p + "mlp.up_proj").T
        r = r + swiglu_oai(g, u) @ dr.linear(p + "mlp.down_proj").T
    return gemma_norm(r, dr.plain("final_norm.weight"))


def sample(dr, y, base, anchor):
    y = y.to(dr.cast)
    w1, w2 = dr.plain("markov_head.markov_w1.weight"), dr.plain("markov_head.markov_w2.weight")
    cw, cb = dr.plain("confidence_head.proj.weight")[0], dr.plain("confidence_head.proj.bias")[0]
    toks, conf, margin = [], [], []
    prev = anchor
    for k in range(T):
        m = w1[prev]
        conf.append(float(torch.sigmoid(torch.dot(torch.cat([y[k], m]), cw) + cb)))
        z = base[k] + w2 @ m
        top = torch.topk(z, 2)
        margin.append(float(top.values[0] - top.values[1]))
        prev = int(top.indices[0])
        toks.append(prev)
    return toks, conf, margin


def embed_rows(pack_dir, ids):
    idx = json.loads((Path(pack_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    name = "language_model.model.embed_tokens.weight"
    path = Path(pack_dir) / idx[name]
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        s, _ = head[name]["data_offsets"]
        width = head[name]["shape"][1]
        rows = []
        for i in ids:
            f.seek(8 + n + s + i * width * 2)
            rows.append(np.frombuffer(f.read(width * 2), np.uint16))
    u = np.stack(rows).astype(np.uint32) << 16
    return u.view(np.float32)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--pack", required=True, help="target EXL3 dir (embedding rows)")
    g.add_argument("--out", required=True)
    g.add_argument("--context", type=int, required=True, help="committed rows = anchor position P")
    g.add_argument("--anchor", type=int, default=1000)
    g.add_argument("--seed", type=int, default=1)
    r = sub.add_parser("run")
    r.add_argument("--draft", required=True)
    r.add_argument("--case", required=True)
    r.add_argument("--no-kv-bf16", action="store_true")
    r.add_argument("--fp32", action="store_true", help="all arithmetic in float32 (sensitivity of the draft to FP32)")
    a = ap.parse_args()
    if a.cmd == "gen":
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(a.seed)
        rng.standard_normal((a.context, TAPS * H), dtype=np.float32).tofile(out / "taps.f32")
        embed_rows(a.pack, [a.anchor] + [NOISE] * (T - 1)).astype("<f4").tofile(out / "embed.f32")
        (rng.standard_normal((T, V), dtype=np.float32) * 3.0).tofile(out / "base.f32")
        (out / "meta.json").write_text(json.dumps({"context": a.context, "position": a.context, "anchor": a.anchor, "seed": a.seed}))
        return
    torch.set_num_threads(max(1, torch.get_num_threads()))
    case = Path(a.case)
    meta = json.loads((case / "meta.json").read_text())
    n = meta["context"]
    taps = torch.from_numpy(np.fromfile(case / "taps.f32", np.float32).reshape(n, TAPS * H).astype(np.float64))
    embed = torch.from_numpy(np.fromfile(case / "embed.f32", np.float32).reshape(T, H).astype(np.float64))
    base = torch.from_numpy(np.fromfile(case / "base.f32", np.float32).reshape(T, V).astype(np.float64))
    dr = Draft(a.draft)
    if a.fp32:
        torch.set_default_dtype(torch.float32)
        dr.cast = torch.float32
        taps, embed, base = taps.float(), embed.float(), base.float()
    y = forward(dr, taps, embed, meta["position"], kv_bf16=not a.no_kv_bf16).double()
    toks, conf, margin = sample(dr, y, base, meta["anchor"])
    tag = ("nokvbf16" if a.no_kv_bf16 else "ref") + ("_fp32" if a.fp32 else "")
    y.numpy().astype("<f8").tofile(case / f"{tag}_y.f64")
    (case / f"{tag}.json").write_text(json.dumps({"drafts": toks, "confidence": conf, "markov_margin": margin,
                                                   "y_rms": float(y.pow(2).mean().sqrt())}, indent=1))
    print(json.dumps({"case": str(case), "drafts": toks, "confidence": [round(c, 5) for c in conf],
                      "min_margin": min(margin)}))


if __name__ == "__main__":
    main()
