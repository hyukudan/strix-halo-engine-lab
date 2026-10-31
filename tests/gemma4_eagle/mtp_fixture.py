"""Fixture for the MTP assistant drafter (tests/gemma4_eagle/mtp_tests.cpp).

Runs HF Gemma4AssistantForCausalLM (FP32, real weights) step by step exactly like assist_core.ref_chain_hf with the
CLEAN KV extent: q_len 1, position p for every depth, shared_kv_states = target layer K/V of positions < p, the
sliding layers seeing the last window+1 = 513 of them (HF mask). The target side is synthetic but shaped like E4B:
  - K/V of the last sliding (head 256) and full (head 512) target layers: unit-RMS random rows, [T][2][D]
  - post-final-norm hidden rows h [T][2560]
  - target token_embd: a procedural BF16 table that C++ regenerates bit-exactly (splitmix64 of t*2560+i), scaled like
    the engine: bf16(bf16(E[t]) * 50.5)
Writes <out>/meta.json and <out>/data.bin (float32, little endian: Ks, Vs, Kf, Vf, hidden, then the per-depth HF input
hidden rows [cases][K][2560]). Every depth records its HF input (token, h) so the C++ test can compare teacher-forced
single steps: free-running chains on random synthetic KV are chaotic (even HF fp32 vs fp64 diverge after ~3 depths;
HF itself runs RMSNorm/softmax in fp32 and builds RoPE angles in fp32).

usage: python mtp_fixture.py <assistant_dir> <out_dir> [T=700] [K=7]
"""
import json
import os
import sys

import numpy as np
import torch
from transformers.models.gemma4_assistant.modeling_gemma4_assistant import Gemma4AssistantForCausalLM

MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix(x):
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def table_row(t, width=2560):
    """Procedural target embedding row (BF16 values as float32): ((h>>40 & 0xffff)/65536 - 0.5) * 0.125."""
    x = np.uint64(t) * np.uint64(width) + np.arange(width, dtype=np.uint64)
    h = splitmix(x)
    v = (((h >> np.uint64(40)) & np.uint64(0xFFFF)).astype(np.float64) / 65536.0 - 0.5) * 0.125
    return torch.from_numpy(v.astype(np.float32)).to(torch.bfloat16)


def embed(t):
    return (table_row(t).float() * 50.5).to(torch.bfloat16).float()  # engine embedding_value (scale 50.5, BF16)


def unit_rows(g, *shape):
    x = torch.randn(*shape, generator=g, dtype=torch.float64)
    return (x / x.pow(2).mean(-1, keepdim=True).sqrt()).float()


@torch.no_grad()
def main():
    src, out = sys.argv[1], sys.argv[2]
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 700
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 7
    torch.set_num_threads(4)
    asst = Gemma4AssistantForCausalLM.from_pretrained(src, dtype=torch.float32, attn_implementation="eager").eval()
    g = torch.Generator().manual_seed(1234)
    Ks, Vs = unit_rows(g, T, 2, 256), unit_rows(g, T, 2, 256)
    Kf, Vf = unit_rows(g, T, 2, 512), unit_rows(g, T, 2, 512)
    hidden = (torch.randn(T, 2560, generator=g) * 2.0).float()
    cases, step_h = [], []
    for p in [1, 2, 7, 100, 511, 512, 513, 514, 600, T - 1]:
        anchor = int(torch.randint(0, 262144, (1,), generator=g))
        kv = {"sliding_attention": (Ks[:p].permute(1, 0, 2)[None], Vs[:p].permute(1, 0, 2)[None]),
              "full_attention": (Kf[:p].permute(1, 0, 2)[None], Vf[:p].permute(1, 0, 2)[None])}
        am = torch.ones(1, p, dtype=torch.long)
        h = hidden[p - 1][None, None]
        tok = anchor
        toks, margins, tin, hin = [], [], [], []
        for _ in range(K):
            tin.append(tok)
            hin.append(h[0, 0].clone())
            x = torch.cat([embed(tok)[None, None], h], -1)
            o = asst(inputs_embeds=x, attention_mask=am, position_ids=torch.tensor([[p]]), shared_kv_states=kv,
                     use_cache=False)
            z = o.logits[0, 0]
            top = z.topk(2)
            tok = int(z.argmax())
            toks.append(tok)
            margins.append(float(top.values[0] - top.values[1]))
            h = o.last_hidden_state
        cases.append({"p": p, "anchor": anchor, "tokens": toks, "margins": margins, "inputs": tin})
        step_h.append(torch.stack(hin))
        print(p, anchor, toks, ["%.3g" % m for m in margins], flush=True)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "data.bin"), "wb") as f:
        for a in (Ks, Vs, Kf, Vf, hidden, torch.stack(step_h)):
            f.write(a.contiguous().numpy().astype("<f4").tobytes())
    meta = {"T": T, "K": K, "sliding_keys": 513, "table": "splitmix64(t*2560+i)>>40&0xffff /65536-0.5 *0.125, bf16",
            "embedding_scale": 50.5, "assistant": os.path.abspath(src), "cases": cases}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
