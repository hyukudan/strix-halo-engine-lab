"""Offline DSpark acceptance on real target states (target_taps.py output), float32 PyTorch draft + the target's EXL3 K8
head (prod numerics), plus optional per-site Hessians of the draft linears for the EXL3 requant.

Teacher-forced estimate. For an anchor position P with text[P] == greedy[P-1] (the anchor is the target's own token),
the block drafts d_1..d_7 for P+1..P+7 from the context rows [P-4095, P-1]. Depth i is accepted when d_i == greedy[P+i-1];
the chain continues to depth i+1 only while text[P+i] == greedy[P+i-1] (the teacher-forced prefix is still the target's
greedy path), else the anchor is censored after depth i. So `accepted` is a lower bound of the real greedy acceptance and
is meant for RANKING drafts (checkpoints, formats), not as the engine's number.

usage: python accept_eval.py --draft DIR --taps DIR [--segments name:start:end ...] [--stride 3] [--hessians OUT]
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dspark_reference as D  # noqa: E402

ORACLE_DIR = os.environ.get("MINIMAX_ORACLE_DIR", "minimax-oracle")
H, T, WIN, NKV, NH, HD = D.H, D.T, D.WIN, D.NKV, D.NH, D.D


class Head:
    """The target's lm_head (EXL3 K8) with the CPU production stages of the oracle."""

    def __init__(self, model, trellis_python):
        sys.path.insert(0, ORACLE_DIR)
        import cpu_layer_reference_fast as F
        self.R = F.R
        pack = F.R.Pack(model, trellis_python)
        F.install_fast_hadamards(pack.st)
        self.st = pack.st
        lin = F.Loader(pack).get("language_model.lm_head", casts=False)
        self.rot = torch.from_numpy(lin.rot.astype(np.float32))  # [6144, vocab]
        self.suh, self.svh = lin.suh, lin.svh

    def __call__(self, y):  # y [n, 6144] float32 numpy -> logits [n, vocab]
        a = self.st.input_stages_fp32(np.ascontiguousarray(y, np.float32), self.suh)["core_input_fp16"].astype(np.float32)
        core = (torch.from_numpy(a) @ self.rot).numpy()
        return self.st.output_stages_fp32(core, self.svh)["output_fp32"]


class Exl3Linears:
    """Draft linears from an EXL3 requant (quant_draft.py output), decoded with the oracle's EXL3 path (exact float64
    reconstruction in the original basis, as R.Linear.original_weight), returned [out, in] float32."""

    def __init__(self, exl3_dir, trellis_python):
        sys.path.insert(0, ORACLE_DIR)
        sys.path.insert(0, trellis_python)
        import cpu_layer_reference_fast as F
        import strix_trellis as st
        F.install_fast_hadamards(st)
        self.F, self.st = F, st
        files = sorted(Path(exl3_dir).glob("*.safetensors"))
        self.file = st.SafetensorsFile(files[0])

    def has(self, rel):
        return D.P + rel + ".trellis" in self.file.header

    def linear(self, rel):
        w = self.st.load_exl3(self.file, D.P + rel)
        r16, r32, r64 = self.F.dequant_fast(w.trellis, w.bits, w.codebook, True)
        H = self.st.hadamard  # H128 blocks along the last axis (fast oracle version); axis 0 through a transpose
        t = H(np.ascontiguousarray(H(r64).T)).T
        W = np.asarray(w.suh, np.float64)[:, None] * t * np.asarray(w.svh, np.float64)[None, :]  # [in, out]
        return torch.from_numpy(np.ascontiguousarray(W.T, dtype=np.float32))


class AltHead:
    """A draft-only head [vocab, 6144] float32 (exact float64 reconstruction of an EXL3 lm_head, applied as y @ W^T in
    float32): `K8` = the target pack's own head through this float path (numerics control against Head's prod stages),
    `rtn<bits>:<group>` = that K8 head RTN-requantized (a pessimistic CPU proxy for an EXL3 requant), else a
    quant_head.py output dir (language_model.lm_head.* EXL3 tensors)."""

    NAME = "language_model.lm_head"

    def __init__(self, src, model, trellis_python):
        rtn = src.startswith("rtn")
        d = Path(model) if src == "K8" or rtn else Path(src)
        idx = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
        lin = Exl3Linears(d, trellis_python)
        lin.file = lin.st.SafetensorsFile(d / idx[self.NAME + ".trellis"])
        w = lin.st.load_exl3(lin.file, self.NAME)
        self.bits = float(w.bits)
        # Exl3Linears.linear's reconstruction in chunks of 8192 outputs (every step is per element or per 128-block of
        # one axis, so the chunks give the same bits) -> host peak ~6 GB instead of ~35 GB for the 200064 x 6144 head
        tr, suh, svh = np.asarray(w.trellis), np.asarray(w.suh, np.float64), np.asarray(w.svh, np.float64)
        Hd = lin.st.hadamard
        self.W = torch.empty(svh.shape[0], suh.shape[0], dtype=torch.float32)
        for o0 in range(0, svh.shape[0], 8192):
            o1 = min(o0 + 8192, svh.shape[0])
            _, _, r64 = lin.F.dequant_fast(tr[:, o0 // 16:o1 // 16], w.bits, w.codebook, True)
            t = Hd(np.ascontiguousarray(Hd(r64).T)).T
            Wc = suh[:, None] * t * svh[None, o0:o1]  # [in, out chunk]
            self.W[o0:o1] = torch.from_numpy(np.ascontiguousarray(Wc.T, dtype=np.float32))
        if rtn:
            b, g = src[3:].split(":")
            self.bits = float(b)
            self.W = torch.cat([int4_rtn(self.W[r:r + 8192], int(g), int(b)) for r in range(0, self.W.shape[0], 8192)])

    def __call__(self, y):  # y [n, 6144] float32 numpy -> logits [n, vocab]
        return (torch.from_numpy(np.ascontiguousarray(y, np.float32)) @ self.W.T).numpy()


def int4_rtn(W, group, bits=4):
    """Round-to-nearest asymmetric INT<bits> (default 4) per group of `group` inputs (scale = (max - min) / (2^bits - 1),
    zero = min), dequantized to float32 [out, in]: the candidate format for a bandwidth-bound draft gemv (report §14)."""
    out, inp = W.shape
    top = float(2 ** bits - 1)
    g = W.reshape(out, inp // group, group)
    lo, hi = g.amin(dim=2, keepdim=True), g.amax(dim=2, keepdim=True)
    scale = torch.clamp((hi - lo) / top, min=1e-12)
    q = torch.clamp(torch.round((g - lo) / scale), 0, top)
    return (q * scale + lo).reshape(out, inp).to(torch.float32)


class Draft32:
    def __init__(self, draft_dir, exl3=None, int4_group=0):
        torch.set_default_dtype(torch.float32)
        dr = D.Draft(draft_dir)
        dr.cast = torch.float32
        if exl3 is not None:
            dr.linear = exl3.linear  # linears from the requant, everything else as shipped
        elif int4_group:
            shipped = dr.linear
            dr.linear = lambda rel: int4_rtn(shipped(rel), int4_group)  # block linears only (fc and heads as shipped)
        self.dr = dr
        self.fc = exl3.linear("fc") if exl3 is not None and exl3.has("fc") else dr.plain("fc.weight")
        self.hn = dr.plain("hidden_norm.weight")
        self.fn = dr.plain("final_norm.weight")
        self.L = []
        for li in range(D.L):
            p = f"layers.{li}.decoder_layer."
            self.L.append({k: (dr.linear(p + n) if n.endswith("proj") else dr.plain(p + n + ".weight"))
                           for k, n in [("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"), ("v", "self_attn.v_proj"),
                                        ("o", "self_attn.o_proj"), ("g", "mlp.gate_proj"), ("u", "mlp.up_proj"),
                                        ("d", "mlp.down_proj"), ("qn", "self_attn.q_norm"), ("kn", "self_attn.k_norm"),
                                        ("ln1", "input_layernorm"), ("ln2", "post_attention_layernorm")]})
        self.w1 = dr.plain("markov_head.markov_w1.weight")
        self.w2 = dr.plain("markov_head.markov_w2.weight")
        self.cw = dr.plain("confidence_head.proj.weight")[0]
        self.cb = dr.plain("confidence_head.proj.bias")[0]
        self.hess = None

    def track(self, x, site):
        if self.hess is not None:
            x64 = x.reshape(-1, x.shape[-1]).double()
            h = self.hess.setdefault(site, [torch.zeros(x64.shape[1], x64.shape[1], dtype=torch.float64), 0])
            h[0] += x64.T @ x64
            h[1] += x64.shape[0]

    def context(self, taps):
        """K/V of every context row, per layer: [n, 4, 128] BF16-rounded (as the engine stores them)."""
        n = taps.shape[0]
        ks, vs = [[] for _ in range(D.L)], [[] for _ in range(D.L)]
        for r0 in range(0, n, 512):
            a = torch.from_numpy(taps[r0:r0 + 512].reshape(-1, D.TAPS * H).astype(np.float32))
            self.track(a, "fc")
            c = D.gemma_norm(a @ self.fc.T, self.hn)
            pos = torch.arange(r0, r0 + a.shape[0])
            for li, w in enumerate(self.L):
                self.track(c, f"L{li}.kv")
                k = D.rope(D.gemma_norm((c @ w["k"].T).view(-1, NKV, HD), w["kn"]), pos)
                ks[li].append(D.to_bf16(k))
                vs[li].append(D.to_bf16((c @ w["v"].T).view(-1, NKV, HD)))
        return [torch.cat(k) for k in ks], [torch.cat(v) for v in vs]

    def blocks(self, embed_rows, anchors, K, V):
        """Batched block passes: embed_rows [B, 7, 6144], anchors (positions) [B] -> y [B, 7, 6144]."""
        B = len(anchors)
        r = torch.from_numpy(embed_rows.astype(np.float32))
        pos = torch.tensor([[p + k for k in range(T)] for p in anchors]).reshape(-1)
        for li, w in enumerate(self.L):
            x = D.gemma_norm(r, w["ln1"]).reshape(B * T, H)
            self.track(x, f"L{li}.qkv")
            q = D.rope(D.gemma_norm((x @ w["q"].T).view(B * T, NH, HD), w["qn"]), pos).view(B, T, NH, HD)
            kb = D.to_bf16(D.rope(D.gemma_norm((x @ w["k"].T).view(B * T, NKV, HD), w["kn"]), pos)).view(B, T, NKV, HD)
            vb = D.to_bf16((x @ w["v"].T).view(B * T, NKV, HD)).view(B, T, NKV, HD)
            att = torch.empty(B, T, NH, HD)
            for b, P in enumerate(anchors):
                f = max(0, P - (WIN - 1))
                kk = torch.cat([K[li][f:P], kb[b]]).repeat_interleave(NH // NKV, dim=1)
                vv = torch.cat([V[li][f:P], vb[b]]).repeat_interleave(NH // NKV, dim=1)
                s = torch.einsum("thd,khd->htk", q[b], kk) * (HD ** -0.5)
                att[b] = torch.einsum("htk,khd->thd", torch.softmax(s, -1), vv)
            a2 = att.reshape(B * T, NH * HD)
            self.track(a2, f"L{li}.o")
            r = r + (a2 @ w["o"].T).view(B, T, H)
            x = D.gemma_norm(r, w["ln2"]).reshape(B * T, H)
            self.track(x, f"L{li}.gu")
            act = D.swiglu_oai(x @ w["g"].T, x @ w["u"].T)
            self.track(act, f"L{li}.down")
            r = r + (act @ w["d"].T).view(B, T, H)
        return D.gemma_norm(r, self.fn)

    def sample(self, base, y, anchor):  # base [7, vocab] numpy, y [7, 6144] torch
        toks, conf, prev = [], [], anchor
        for k in range(T):
            m = self.w1[prev]
            conf.append(float(torch.sigmoid(torch.dot(torch.cat([y[k], m]), self.cw) + self.cb)))
            z = torch.from_numpy(base[k]) + self.w2 @ m
            prev = int(torch.argmax(z))
            toks.append(prev)
        return toks, conf


class Copy:
    """Python twin of minimax_ngram.cpp (CopyDrafter): anchor-window index, most recent occurrence, unanimous agreement."""

    def __init__(self, anchor=8, max_occ=64):
        self.a, self.max_occ, self.h, self.ends = anchor, max_occ, [], {}

    def append(self, toks):
        for t in toks:
            if len(self.h) >= self.a:
                v = self.ends.setdefault(tuple(self.h[-self.a:]), [])
                v.append(len(self.h))
                if len(v) > self.max_occ:
                    v.pop(0)
            self.h.append(int(t))

    def propose(self, k):
        n = len(self.h)
        if n < self.a:
            return []
        occ = list(reversed(self.ends.get(tuple(self.h[-self.a:]), [])))
        out = []
        for i in range(k):
            if not occ or occ[0] + i >= n:
                break
            t = self.h[occ[0] + i]
            if any(o + i < n and self.h[o + i] != t for o in occ[1:]):
                break
            out.append(t)
        return out


def score(d, P, toks, greedy):
    """Teacher-forced accepted depth of drafts d at anchor P (see the module docstring); returns (acc, tried, ok, censored)."""
    acc, tried, ok = 0, [], []
    for i in range(1, len(d) + 1):
        tried.append(i)
        if d[i - 1] != greedy[P + i - 1]:
            return acc, tried, ok, False
        ok.append(i)
        acc = i
        if toks[P + i] != greedy[P + i - 1]:
            return acc, tried, ok, True
    return acc, tried, ok, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--taps", required=True)
    ap.add_argument("--model", default="minimax-m3.1-flash-exl3-3.06bpw")
    ap.add_argument("--trellis-python", default=os.environ.get("STRIX_TRELLIS_PYTHON", ""))
    ap.add_argument("--segments", nargs="*", default=[], help="name:start:end token ranges of the anchors")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--min-context", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hessians", help="write per-site Hessians (sum x x^T, count) here as .npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-filter", action="store_true", help="plumbing check: keep anchors off the greedy path")
    ap.add_argument("--copy-only", action="store_true", help="score only the copy drafter (no draft model, no head)")
    ap.add_argument("--exl3-linears", help="EXL3 requant dir (quant_draft.py): draft linears from it")
    ap.add_argument("--int4", type=int, default=0, help="block linears RTN INT4 with this group size (0 = as shipped)")
    ap.add_argument("--head-alt", help="draft-only head arm: 'K8' (float-path control) or a quant_head.py dir; scored on "
                                       "the same anchors and blocks, plus per-depth agreement of its drafts with the K8 head's")
    ap.add_argument("--head-hessian", help="write the head-input Hessian (sum y y^T over draft rows, count) here as .npz")
    a = ap.parse_args()
    td = Path(a.taps)
    meta = json.loads((td / "meta.json").read_text())
    n = meta["tokens"]
    toks = np.fromfile(td / "tokens.u32", np.uint32)
    greedy = np.fromfile(td / "greedy.u32", np.uint32)
    taps = np.memmap(td / "taps.f32", np.float32, "r", shape=(n, D.TAPS, H))
    segs = [(s.split(":")[0], int(s.split(":")[1]), int(s.split(":")[2])) for s in a.segments] or [("all", 0, n)]
    t0 = time.time()
    alt, head_h = None, None
    if not a.copy_only:
        head = Head(a.model, a.trellis_python)
        dr = Draft32(a.draft, Exl3Linears(a.exl3_linears, a.trellis_python) if a.exl3_linears else None, a.int4)
        if a.hessians:
            dr.hess = {}
        alt = AltHead(a.head_alt, a.model, a.trellis_python) if a.head_alt else None
        head_h = [torch.zeros(H, H, dtype=torch.float64), 0] if a.head_hessian else None
        K, V = dr.context(taps)
    embed_cache = {}

    def embed(ids):
        need = [i for i in set(ids) if i not in embed_cache]
        if need:
            rows = D.embed_rows(a.model, need)
            for i, r in zip(need, rows):
                embed_cache[i] = r
        return np.stack([embed_cache[i] for i in ids])

    results = {}
    calib = []
    for name, s0, s1 in segs:
        anchors = [P for P in range(max(s0, a.min_context), min(s1, n - T - 1), a.stride) if a.no_filter or toks[P] == greedy[P - 1]]
        # copy drafter on the same anchors: history = tokens[0..P] (the anchor included), drafts up to 7
        cp, cp_pos = Copy(8), 0
        c_hist, c_prop, c_len = np.zeros(T + 1, np.int64), 0, 0
        for P in anchors:
            cp.append(toks[cp_pos:P + 1])
            cp_pos = P + 1
            d = cp.propose(T)
            if d:
                c_prop += 1
                c_len += len(d)
            acc, _, _, _ = score(d, P, toks, greedy)
            c_hist[acc] += 1
        copy_res = {"anchors_with_proposal": c_prop, "mean_proposal": c_len / max(1, c_prop),
                    "tau_copy": float((c_hist * np.arange(T + 1)).sum() / max(1, c_hist.sum()) + 1),
                    "accepted_hist": c_hist.tolist()}
        if a.copy_only:
            results[name] = {"anchors": len(anchors), "copy": copy_res}
            continue
        acc_hist = np.zeros(T + 1, np.int64)
        depth_try, depth_ok = np.zeros(T, np.int64), np.zeros(T, np.int64)
        censored = 0
        alt_hist = np.zeros(T + 1, np.int64)
        agree_base, agree_draft, agree_n = np.zeros(T, np.int64), np.zeros(T, np.int64), 0
        for b0 in range(0, len(anchors), a.batch):
            batch = anchors[b0:b0 + a.batch]
            rows = np.stack([embed([int(toks[P])] + [D.NOISE] * (T - 1)) for P in batch])
            y = dr.blocks(rows, batch, K, V)
            yf = y.reshape(-1, H)
            if head_h is not None:
                y64 = yf.double()
                head_h[0] += y64.T @ y64
                head_h[1] += y64.shape[0]
            base = head(yf.numpy()).reshape(len(batch), T, -1)
            base_alt = alt(yf.numpy()).reshape(len(batch), T, -1) if alt is not None else None
            for bi, P in enumerate(batch):
                d, conf = dr.sample(base[bi], y[bi], int(toks[P]))
                if base_alt is not None:
                    # alt arm: its own Markov chain from its own draws; agreement = same draft token at depth i (and
                    # same base argmax), over every anchor (no censoring)
                    da, _ = dr.sample(base_alt[bi], y[bi], int(toks[P]))
                    agree_n += 1
                    agree_draft += np.array([da[i] == d[i] for i in range(T)], np.int64)
                    agree_base += (base_alt[bi].argmax(-1) == base[bi].argmax(-1)).astype(np.int64)
                    alt_hist[score(da, P, toks, greedy)[0]] += 1
                acc = 0
                for i in range(1, T + 1):
                    depth_try[i - 1] += 1
                    ok = d[i - 1] == greedy[P + i - 1]
                    calib.append((name, i, conf[i - 1], int(ok)))
                    if not ok:
                        break
                    depth_ok[i - 1] += 1
                    acc = i
                    if toks[P + i] != greedy[P + i - 1]:
                        censored += 1
                        break
                acc_hist[acc] += 1
            print(json.dumps({"segment": name, "done": b0 + len(batch), "of": len(anchors),
                              "tau_so_far": float((acc_hist * np.arange(T + 1)).sum() / max(1, acc_hist.sum()) + 1),
                              "s": round(time.time() - t0)}), flush=True)
        cnt = int(acc_hist.sum())
        results[name] = {"anchors": cnt, "censored": censored,
                         "tau_fixed7": float((acc_hist * np.arange(T + 1)).sum() / max(1, cnt) + 1),
                         "accepted_hist": acc_hist.tolist(),
                         "depth_accept": [float(depth_ok[i] / max(1, depth_try[i])) for i in range(T)], "copy": copy_res}
        if alt is not None:
            results[name]["head_alt"] = {"src": a.head_alt, "bits": alt.bits, "accepted_hist": alt_hist.tolist(),
                                         "tau_fixed7": float((alt_hist * np.arange(T + 1)).sum() / max(1, alt_hist.sum()) + 1),
                                         "agree_draft": (agree_draft / max(1, agree_n)).tolist(),
                                         "agree_base_argmax": (agree_base / max(1, agree_n)).tolist()}
    out = {"draft": a.draft, "taps": a.taps, "stride": a.stride, "segments": results, "seconds": round(time.time() - t0)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    if calib:
        np.save(Path(a.out).with_suffix(".calib.npy"), np.array([(i, c, o) for _, i, c, o in calib], dtype=np.float64))
    if a.hessians and not a.copy_only:
        np.savez(a.hessians, **{k: v[0].numpy() for k, v in dr.hess.items()}, **{"n." + k: v[1] for k, v in dr.hess.items()})
    if head_h is not None:
        np.savez(a.head_hessian, head=head_h[0].numpy(), **{"n.head": head_h[1]})
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
