"""Two-node TP split of the MiniMax-M3.1-Flash EXL3 pack from the real tensor byte sizes.

Layout (same family as DS41 mix-B + expert TP):
  attention TP: q heads 32/rank, kv heads 2/rank (q/k/v output halves, o_proj input half);
  indexer: index heads 2/rank (index head g serves kv head g), index_k replicated;
  dense MLP (0-2) and shared expert: intermediate halves (gate/up outputs, down input);
  routed experts: expert TP on the intermediate dim (I 3072 -> 1536 per rank), every expert on both nodes;
  router + score bias replicated; embed replicated (BF16); lm_head split by vocab at H128 blocks.
Every split is H128-aligned, so EXL3 slices need no requantization.
"""
import argparse
import collections
import json
from pathlib import Path
import re

GB = 1e9
GIB = 2 ** 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--contexts", default="8192,32768,131072,524288,1048576")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.model)
    cfg = json.loads((root / "config.json").read_text())["text_config"]
    q = json.loads((root / "quantization_config.json").read_text())["tensor_storage"]
    per_role = collections.Counter()
    for mod, v in q.items():
        role = re.sub(r"\.\d+\.", ".N.", mod).replace("language_model.model.", "").replace("language_model.", "")
        per_role[role] += sum(s["n_bytes"] for s in v["stored_tensors"].values())

    def b(role):
        return per_role[role]

    roles_split = {  # role -> fraction stored per rank
        "layers.N.self_attn.q_proj": 0.5, "layers.N.self_attn.k_proj": 0.5, "layers.N.self_attn.v_proj": 0.5,
        "layers.N.self_attn.o_proj": 0.5, "layers.N.self_attn.index_q_proj": 0.5,
        "layers.N.self_attn.index_k_proj": 1.0,
        "layers.N.mlp.gate_proj": 0.5, "layers.N.mlp.up_proj": 0.5, "layers.N.mlp.down_proj": 0.5,
        "layers.N.block_sparse_moe.shared_experts.gate_proj": 0.5,
        "layers.N.block_sparse_moe.shared_experts.up_proj": 0.5,
        "layers.N.block_sparse_moe.shared_experts.down_proj": 0.5,
        "layers.N.block_sparse_moe.experts.N.w1": 0.5, "layers.N.block_sparse_moe.experts.N.w3": 0.5,
        "layers.N.block_sparse_moe.experts.N.w2": 0.5,
        "layers.N.block_sparse_moe.gate": 1.0,
        "embed_tokens": 1.0, "lm_head": 782 / 1563,  # larger rank of the 782/781 H128 vocab split
    }
    resident, total = {}, {}
    for role, frac in roles_split.items():
        total[role] = b(role)
        resident[role] = b(role) * frac
    small = sum(v for r, v in per_role.items() if r not in roles_split)  # norms, biases
    resident["norms_and_biases"] = small
    total["norms_and_biases"] = small
    node_bytes = sum(resident.values())
    # Read bytes per decoded token (R1) per node.
    L, moe = cfg["num_hidden_layers"], sum(cfg["moe_layer_freq"])
    E, topk = cfg["num_local_experts"], cfg["num_experts_per_tok"]
    expert_one = (b("layers.N.block_sparse_moe.experts.N.w1") + b("layers.N.block_sparse_moe.experts.N.w3") +
                  b("layers.N.block_sparse_moe.experts.N.w2")) / (moe * E)
    reads = {
        "experts (top-4, half each)": expert_one * topk * moe * 0.5,
        "shared expert (half)": resident["layers.N.block_sparse_moe.shared_experts.gate_proj"] +
        resident["layers.N.block_sparse_moe.shared_experts.up_proj"] +
        resident["layers.N.block_sparse_moe.shared_experts.down_proj"],
        "attention q/k/v/o (half)": sum(resident[f"layers.N.self_attn.{p}"] for p in ("q_proj", "k_proj", "v_proj", "o_proj")),
        "dense MLP 0-2 (half)": sum(resident[f"layers.N.mlp.{p}"] for p in ("gate_proj", "up_proj", "down_proj")),
        "indexer projections": resident["layers.N.self_attn.index_q_proj"] + resident["layers.N.self_attn.index_k_proj"],
        "router (replicated)": resident["layers.N.block_sparse_moe.gate"],
        "lm_head (half)": resident["lm_head"],
        "embed row": cfg["hidden_size"] * 2,
    }
    weights_read = sum(reads.values())
    hd, nkv, idim = cfg["head_dim"], cfg["num_key_value_heads"], cfg["sparse_attention_config"]["sparse_index_dim"]
    topk_blocks = cfg["sparse_attention_config"]["sparse_topk_blocks"]
    bs = cfg["sparse_attention_config"]["sparse_block_size"]
    ctx_rows = []
    for ctx in (int(c) for c in args.contexts.split(",")):
        for name, kv_b, ik_b in (("BF16", 2.0, 2.0), ("FP8 E4M3 + E8M0/32", 1 + 1 / 32, 1 + 1 / 32)):
            kv_store = L * (nkv // 2) * hd * 2 * kv_b * ctx       # K and V, 2 kv heads per rank
            ik_store = L * idim * ik_b * ctx                       # index key, replicated
            sel_keys = min(ctx, topk_blocks * bs)
            kv_read = L * (nkv // 2) * hd * 2 * kv_b * sel_keys    # selected blocks, both groups
            ik_read = L * idim * ik_b * ctx                        # full index scan
            ctx_rows.append({"context": ctx, "cache": name,
                             "store_GB_per_node": round((kv_store + ik_store) / GB, 2),
                             "decode_read_GB_per_token_node": round((kv_read + ik_read) / GB, 3),
                             "of_which_index_scan_GB": round(ik_read / GB, 3)})
    hidden = cfg["hidden_size"]
    exchanges = L + L + 1  # o_proj all-reduce, FFN half-sum per layer, head argmax
    plan = {
        "pack_text_bytes": sum(per_role.values()),
        "bytes_by_role_total_GB": {k: round(v / GB, 3) for k, v in sorted(total.items())},
        "resident_by_role_per_node_GB": {k: round(v / GB, 3) for k, v in sorted(resident.items())},
        "resident_per_node_GB": round(node_bytes / GB, 2), "resident_per_node_GiB": round(node_bytes / GIB, 2),
        "decode_reads_per_token_node_GB": {k: round(v / GB, 3) for k, v in reads.items()},
        "decode_weight_reads_total_GB": round(weights_read / GB, 3),
        "kv_and_index": ctx_rows,
        "usb4": {"exchanges_per_token": exchanges, "row_bytes_fp32": hidden * 4,
                 "bytes_per_token_each_direction": (exchanges - 1) * hidden * 4 + 8,
                 "indexer_exchanges": 0},
    }
    Path(args.out).write_text(json.dumps(plan, indent=1))
    print(json.dumps(plan, indent=1))


if __name__ == "__main__":
    main()
