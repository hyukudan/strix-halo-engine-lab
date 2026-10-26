"""Validate every text tensor of the MiniMax-M3.1-Flash EXL3 pack against the config.

Reads safetensors headers plus, for each EXL3 group, the suh/svh scale vectors and
the codebook marker through strix-trellis `load_exl3` (trellis payloads are only
memory-mapped, not read). Unquantized tensors are checked by dtype and shape.
Writes a JSON report; exits nonzero on any mismatch.
"""
import argparse
import collections
import json
from pathlib import Path
import re
import sys
import time

LAYER_INDEX = re.compile(r"\.\d+\.")


def expected_manifest(cfg):
    t = cfg["text_config"]
    h, nh, nkv, hd = t["hidden_size"], t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    sa = t["sparse_attention_config"]
    nih, idim = sa["sparse_num_index_heads"], sa["sparse_index_dim"]
    exp, shared, dense = t["intermediate_size"], t["shared_intermediate_size"], t["dense_intermediate_size"]
    vocab = t["vocab_size"]
    m = {}
    p = "language_model.model."
    m[p + "embed_tokens.weight"] = ("plain", "BF16", [vocab, h])
    m[p + "norm.weight"] = ("plain", "BF16", [h])
    m["language_model.lm_head"] = ("exl3", 8, h, vocab)
    for i in range(t["num_hidden_layers"]):
        L = f"{p}layers.{i}."
        m[L + "input_layernorm.weight"] = ("plain", "BF16", [h])
        m[L + "post_attention_layernorm.weight"] = ("plain", "BF16", [h])
        a = L + "self_attn."
        m[a + "q_proj"] = ("exl3", 6, h, nh * hd)
        m[a + "k_proj"] = ("exl3", 6, h, nkv * hd)
        m[a + "v_proj"] = ("exl3", 6, h, nkv * hd)
        m[a + "o_proj"] = ("exl3", 6, nh * hd, h)
        m[a + "q_norm.weight"] = ("plain", "BF16", [hd])
        m[a + "k_norm.weight"] = ("plain", "BF16", [hd])
        if sa["sparse_attention_freq"][i]:
            m[a + "index_q_proj.weight"] = ("plain", "F16", [nih * idim, h])
            m[a + "index_k_proj.weight"] = ("plain", "F16", [idim, h])
            m[a + "index_q_norm.weight"] = ("plain", "BF16", [idim])
            m[a + "index_k_norm.weight"] = ("plain", "BF16", [idim])
        if t["moe_layer_freq"][i]:
            b = L + "block_sparse_moe."
            m[b + "gate.weight"] = ("plain", "F32", [t["num_local_experts"], h])
            m[b + "e_score_correction_bias"] = ("plain", "F32", [t["num_local_experts"]])
            for e in range(t["num_local_experts"]):
                m[f"{b}experts.{e}.w1"] = ("exl3", 3, h, exp)
                m[f"{b}experts.{e}.w3"] = ("exl3", 3, h, exp)
                m[f"{b}experts.{e}.w2"] = ("exl3", 3, exp, h)
            m[b + "shared_experts.gate_proj"] = ("exl3", 5, h, shared)
            m[b + "shared_experts.up_proj"] = ("exl3", 5, h, shared)
            m[b + "shared_experts.down_proj"] = ("exl3", 5, shared, h)
        else:
            f = L + "mlp."
            m[f + "gate_proj"] = ("exl3", 5, h, dense)
            m[f + "up_proj"] = ("exl3", 5, h, dense)
            m[f + "down_proj"] = ("exl3", 5, dense, h)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--trellis-python", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--headers-only", action="store_true", help="skip suh/svh reads")
    args = ap.parse_args()
    sys.path.insert(0, args.trellis_python)
    from strix_trellis import SafetensorsFile, load_exl3
    from strix_trellis.container import _DTYPES

    root = Path(args.model)
    cfg = json.loads((root / "config.json").read_text())
    qcfg = json.loads((root / "quantization_config.json").read_text())
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    manifest = expected_manifest(cfg)
    errors, stats, bytes_by = [], collections.Counter(), collections.Counter()
    files = {}

    def shard(name):
        fn = index[name]
        if fn not in files:
            files[fn] = SafetensorsFile(root / fn)
        return files[fn]

    text_names = {k for k in index if k.startswith("language_model.")}
    covered = set()
    started = time.time()
    for key, spec in manifest.items():
        try:
            if spec[0] == "plain":
                _, dtype, shape = spec
                f = shard(key)
                info = f.header[key]
                covered.add(key)
                if info["dtype"] != dtype or info["shape"] != shape:
                    errors.append(f"{key}: {info['dtype']} {info['shape']} != {dtype} {shape}")
                    continue
                f.tensor(key, raw_bf16=True)  # range/size validation
                stats[f"plain:{dtype}"] += 1
                b0, b1 = info["data_offsets"]
                bytes_by["plain:" + dtype] += b1 - b0
                continue
            _, bits, inputs, outputs = spec
            names = [key + s for s in (".trellis", ".suh", ".svh", ".mul1")]
            f = shard(names[0])
            if any(index.get(n) != index[names[0]] for n in names):
                errors.append(f"{key}: group missing or split across shards")
                continue
            covered.update(names)
            q = qcfg["tensor_storage"][key]
            if q.get("quant_format") != "exl3" or q.get("bits_per_weight") != bits or \
                    q.get("mul1_multiplier") != 0x83DCD12D:
                errors.append(f"{key}: quantization_config {q.get('bits_per_weight')} {q.get('mul1_multiplier')}")
            if args.headers_only:
                tr = f.header[names[0]]
                if tr["dtype"] != "I16" or tr["shape"] != [inputs // 16, outputs // 16, 16 * bits]:
                    errors.append(f"{key}: trellis {tr['dtype']} {tr['shape']}")
            else:
                w = load_exl3(f, key)
                if (w.input_size, w.output_size, w.bits, w.codebook, w.bias is None) != \
                        (inputs, outputs, bits, "mul1", True):
                    errors.append(f"{key}: {w.input_size}x{w.output_size} K{w.bits} {w.codebook}")
                    continue
            for n in names:
                b0, b1 = f.header[n]["data_offsets"]
                bytes_by[f"exl3:K{bits}"] += b1 - b0
            role = LAYER_INDEX.sub(".N.", key).replace("language_model.model.", "")
            stats[f"exl3:K{bits}:{role}"] += 1
        except Exception as exc:  # report every failure, keep going
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    extra = sorted(text_names - covered)
    unsupported = collections.Counter(
        f.header[k]["dtype"] for f in files.values() for k in f.header
        if k.startswith("language_model.") and f.header[k]["dtype"] not in _DTYPES)
    report = {
        "model": str(root), "tensors_in_index_text": len(text_names), "covered": len(covered),
        "n_extra": len(extra), "extra_text_tensors": extra[:50],
        "n_errors": len(errors), "errors": errors[:200],
        "groups": dict(sorted(stats.items())), "bytes": dict(sorted(bytes_by.items())),
        "total_text_bytes": sum(bytes_by.values()),
        "dtypes_outside_strix_trellis_base_DTYPES": dict(unsupported),
        "seconds": round(time.time() - started, 1), "headers_only": args.headers_only,
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("tensors_in_index_text", "covered", "n_extra", "n_errors",
                                              "total_text_bytes", "seconds")}))
    return 1 if errors or extra else 0


if __name__ == "__main__":
    sys.exit(main())
