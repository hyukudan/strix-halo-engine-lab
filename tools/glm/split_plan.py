"""Two-node TP split of the GLM-5.3-Flash EXL3 pack (design notes section 4).

Per tensor: replicate | out (output columns, EXL3 slice_outputs) | in (input rows, EXL3 slice_inputs) |
rows (plain tensor, leading-dim rows).  Ranges are per rank, H128-aligned for EXL3, so no requantization:
  KDA: qkv_proj out in three ranges (q, k, v heads 32 per rank), conv1d/f_b/g_b/dt_bias/b_proj/A_log rows by head,
       o_proj in; f_a/g_a/o_norm replicated.
  MLA: q_b out (32 heads), kv_b rows (heads x 512), o_proj in; q_a, kv_a, norms, indexer replicated.
  MoE: experts + shared I 2048 -> 1024 per rank (gate/up out, down in), every expert on both ranks; router replicated.
  dense MLP I 12288 -> 6144; lm_head vocab 154880 -> 77440 (605 H128 blocks); embed, eh_proj, norms, mHC replicated.
Writes JSON {tensor: {mode, ranges: [[[begin, end], ...] rank 0, [...] rank 1]}} and per-rank byte totals.
usage: split_plan.py --headers tensors.json --out split.json   (or --model DIR)
"""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).parent))
from validate_exl3_pack import HeaderSource  # noqa: E402

QKV, KDA_H, MLA_H = 8192, 64, 64


def rule(group, shape):
    """(mode, [rank0 ranges, rank1 ranges]) for an EXL3 group or plain tensor name."""
    half = lambda n: [[[0, n // 2]], [[n // 2, n]]]
    if re.search(r"\.self_attn\.qkv_proj$", group):
        return "out", [[[c * QKV + r * QKV // 2, c * QKV + (r + 1) * QKV // 2] for c in range(3)] for r in (0, 1)]
    if re.search(r"\.self_attn\.conv1d\.weight$", group):
        return "rows", [[[c * QKV + r * QKV // 2, c * QKV + (r + 1) * QKV // 2] for c in range(3)] for r in (0, 1)]
    if re.search(r"(\.mlp\.(experts\.\d+\.|shared_experts\.)?(gate|up)_proj|\.self_attn\.q_b_proj|^lm_head)$", group):
        return "out", half(shape[1] * 16)
    if re.search(r"(\.mlp\.(experts\.\d+\.|shared_experts\.)?down_proj|\.self_attn\.o_proj)$", group):
        return "in", half(shape[0] * 16)
    if re.search(r"\.self_attn\.(f_b_proj|g_b_proj|b_proj|kv_b_proj)\.weight$|\.self_attn\.(A_log|dt_bias)$", group):
        return "rows", half(shape[0])
    return "replicate", [[[0, -1]], [[0, -1]]]


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model")
    src.add_argument("--headers")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    hs = HeaderSource(a.model, a.headers).t
    raw = json.loads(Path(a.headers).read_text())["tensors"] if a.headers else None
    groups = {}
    for k in hs:
        if k.startswith("model.visual."):
            continue
        g = k[:-len(".trellis")] if k.endswith(".trellis") else None
        if g:
            groups[g] = ("exl3", hs[k][1])
        elif not k.endswith((".suh", ".svh", ".mul1")):
            groups[k] = ("plain", hs[k][1])
    plan, total = {}, [0, 0]
    nbytes = lambda n: raw[n]["bytes"] if raw else 0
    for g, (kind, shape) in sorted(groups.items()):
        mode, ranges = rule(g, shape)
        plan[g] = {"kind": kind, "mode": mode, "ranges": ranges}
        for r in (0, 1):
            if kind == "plain":
                full = nbytes(g)
                frac = 1.0 if mode == "replicate" else sum(e - b for b, e in ranges[r]) / shape[0]
                total[r] += full * frac
                continue
            tr, suh, svh, mul = (nbytes(g + s) for s in (".trellis", ".suh", ".svh", ".mul1"))
            if mode == "replicate":
                total[r] += tr + suh + svh + mul
            elif mode == "out":
                n = sum(e - b for b, e in ranges[r]); N = shape[1] * 16
                total[r] += tr * n / N + suh + svh * n / N + mul
            else:
                n = sum(e - b for b, e in ranges[r]); K = shape[0] * 16
                total[r] += tr * n / K + suh * n / K + svh + mul
    for g, p in plan.items():  # every split range must be H128-aligned for EXL3
        if p["kind"] == "exl3" and p["mode"] != "replicate":
            assert all(b % 128 == 0 and e % 128 == 0 for rr in p["ranges"] for b, e in rr), g
    out = {"tensors": plan, "rank_bytes": total}
    Path(a.out).write_text(json.dumps(out))
    modes = {}
    for p in plan.values():
        modes[p["mode"]] = modes.get(p["mode"], 0) + 1
    print(json.dumps({"groups": len(plan), "modes": modes,
                      "rank_bytes_GB": [round(t / 1e9, 3) for t in total],
                      "rank_GiB": [round(t / 2 ** 30, 2) for t in total]}))


if __name__ == "__main__":
    main()
