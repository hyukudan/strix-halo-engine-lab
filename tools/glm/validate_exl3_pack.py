"""Validate every text tensor of the GLM-5.3-Flash EXL3 pack against the config (G0 admission).

Expected names and shapes come from config.json (text_config); EXL3 groups are checked for the full
[trellis I16 [K/16, N/16, 16*bits], suh F16 [K], svh F16 [N], mul1 I32 []] set in one shard, plain tensors by
dtype and shape.  The bits per role are reported and must match the pinned identity (--identity JSON, written by
--write-identity on the first admitted copy).  Sources: a model directory (headers read from the files) or
--headers JSON ({"tensors": {name: {shape, dtype, bytes}}}, from saved model headers) for CPU-only checks.
Exits nonzero on any mismatch, missing or extra text tensor.
"""
import argparse
import collections
import json
from pathlib import Path
import re
import struct
import sys

LAYER_INDEX = re.compile(r"\.\d+\.")
P = "model.language_model."


def expected_manifest(cfg):
    t = cfg["text_config"]
    h, vocab, nl = t["hidden_size"], t["vocab_size"], t["num_hidden_layers"]
    la = t["linear_attn_config"]
    nh, hd = la["num_heads"], la["head_dim"]
    qkv = nh * hd
    q_lora, kv_lora, mla_h, mla_hd = t["q_lora_rank"], t["kv_lora_rank"], t["num_attention_heads"], t["qk_nope_head_dim"]
    ih, idim, kpool = t["index_n_heads"], t["index_head_dim"], t["index_kpool"]
    ne, ei, dense_i = t["n_routed_experts"], t["moe_intermediate_size"], t["intermediate_size"]
    mix = (2 + t["hc_mult"]) * t["hc_mult"]
    kda_layers = set(la["kda_layers"])
    m = {P + "embed_tokens.weight": ("plain", {"BF16"}, [vocab, h]),
         P + "norm.weight": ("plain", {"BF16", "F16"}, [h]),
         "lm_head": ("exl3", h, vocab)}

    def mla(a):
        m[a + "q_a_proj"] = ("exl3", h, q_lora)
        m[a + "q_a_layernorm.weight"] = ("plain", {"BF16", "F16"}, [q_lora])
        m[a + "q_b_proj"] = ("exl3", q_lora, mla_h * mla_hd)
        m[a + "kv_a_proj_with_mqa"] = ("exl3", h, kv_lora)
        m[a + "kv_a_layernorm.weight"] = ("plain", {"BF16", "F16"}, [kv_lora])
        m[a + "kv_b_proj.weight"] = ("plain", {"F16", "BF16"}, [mla_h * 2 * mla_hd, kv_lora])
        m[a + "o_proj"] = ("exl3", mla_h * t["v_head_dim"], h)
        i = a + "indexer."
        m[i + "wq_b"] = ("exl3", q_lora, ih * idim)
        m[i + "wk.weight"] = ("plain", {"F16", "BF16"}, [idim, h])
        m[i + "k_norm.weight"] = ("plain", {"F16", "BF16"}, [idim])
        m[i + "k_norm.bias"] = ("plain", {"F16", "BF16"}, [idim])
        m[i + "weights_proj.weight"] = ("plain", {"F16", "BF16"}, [ih, h])
        m[i + "index_kpool_compress_gate"] = ("plain", {"F16", "BF16"}, [idim, h])
        m[i + "index_kpool_compress_ape"] = ("plain", {"F32", "F16", "BF16"}, [kpool, idim])

    def moe(L):
        b = L + "mlp."
        m[b + "gate.weight"] = ("plain", {"F16", "BF16", "F32"}, [ne, h])
        m[b + "gate.e_score_correction_bias"] = ("plain", {"F32"}, [ne])
        for e in range(ne):
            m[f"{b}experts.{e}.gate_proj"] = ("exl3", h, ei)
            m[f"{b}experts.{e}.up_proj"] = ("exl3", h, ei)
            m[f"{b}experts.{e}.down_proj"] = ("exl3", ei, h)
        for n, (i, o) in (("gate", (h, ei)), ("up", (h, ei)), ("down", (ei, h))):
            m[f"{b}shared_experts.{n}_proj"] = ("exl3", i, o)

    for li in range(nl):
        L = f"{P}layers.{li}."
        m[L + "input_layernorm.weight"] = ("plain", {"BF16", "F16"}, [h])
        m[L + "post_attention_layernorm.weight"] = ("plain", {"BF16", "F16"}, [h])
        for s in ("attn", "ffn"):
            m[f"{L}hc_{s}_fn"] = ("plain", {"F32", "BF16"}, [mix, t["hc_mult"] * h])
            m[f"{L}hc_{s}_base"] = ("plain", {"F32"}, [mix])
            m[f"{L}hc_{s}_scale"] = ("plain", {"F32"}, [3])
        a = L + "self_attn."
        if li in kda_layers:
            m[a + "qkv_proj"] = ("exl3", h, 3 * qkv)
            m[a + "o_proj"] = ("exl3", qkv, h)
            m[a + "conv1d.weight"] = ("plain", {"BF16", "F32"}, [3 * qkv, 1, la["short_conv_kernel_size"]])
            for n, shape in (("f_a_proj", [hd, h]), ("f_b_proj", [qkv, hd]), ("g_a_proj", [hd, h]),
                             ("g_b_proj", [qkv, hd]), ("b_proj", [nh, h])):
                m[a + n + ".weight"] = ("plain", {"F16", "BF16"}, shape)
            m[a + "A_log"] = ("plain", {"F32"}, [nh])
            m[a + "dt_bias"] = ("plain", {"F32"}, [qkv])
            m[a + "o_norm.weight"] = ("plain", {"BF16", "F16", "F32"}, [hd])
        else:
            mla(a)
        if t["mlp_layer_types"][li] == "sparse":
            moe(L)
        else:
            for n, (i, o) in (("gate", (h, dense_i)), ("up", (h, dense_i)), ("down", (dense_i, h))):
                m[f"{L}mlp.{n}_proj"] = ("exl3", i, o)
    for k in range(t["num_nextn_predict_layers"]):  # MTP layer(s): no hc tensors, own norms, eh_proj, DSA + MoE
        L = f"{P}layers.{nl + k}."
        for n in ("input_layernorm", "post_attention_layernorm", "enorm", "hnorm", "shared_head.norm"):
            m[f"{L}{n}.weight"] = ("plain", {"BF16", "F16"}, [h])
        m[L + "eh_proj"] = ("exl3", 2 * h, h)
        mla(L + "self_attn.")
        moe(L)
    return m


class HeaderSource:
    def __init__(self, model=None, headers=None):
        self.t = {}
        if headers:
            for k, v in json.loads(Path(headers).read_text())["tensors"].items():
                self.t[k] = (v["dtype"], v["shape"], "?")
        else:
            root = Path(model)
            for f in sorted(root.glob("*.safetensors")):
                with open(f, "rb") as fh:
                    n = struct.unpack("<Q", fh.read(8))[0]
                    h = json.loads(fh.read(n))
                size = f.stat().st_size
                for k, v in h.items():
                    if k == "__metadata__":
                        continue
                    a, b = v["data_offsets"]
                    if 8 + n + b > size:
                        raise ValueError(f"{f.name}: {k} past end of file")
                    self.t[k] = (v["dtype"], v["shape"], f.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model")
    src.add_argument("--headers")
    ap.add_argument("--identity", help="pinned bits per role (JSON); mismatch = error")
    ap.add_argument("--write-identity")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    hs = HeaderSource(a.model, a.headers).t
    manifest = expected_manifest(cfg)
    errors, covered, bits_by_role, split_groups = [], set(), collections.defaultdict(set), []
    for key, spec in manifest.items():
        if spec[0] == "plain":
            _, dtypes, shape = spec
            if key not in hs:
                errors.append(f"missing {key}")
                continue
            covered.add(key)
            dt, sh, _ = hs[key]
            if dt not in dtypes or sh != shape:
                errors.append(f"{key}: {dt} {sh} != {sorted(dtypes)} {shape}")
            continue
        _, k_in, n_out = spec
        names = [key + s for s in (".trellis", ".suh", ".svh", ".mul1")]
        if any(n not in hs for n in names):
            errors.append(f"{key}: EXL3 group incomplete")
            continue
        covered.update(names)
        if len({hs[n][2] for n in names}) != 1:  # the upstream writer fills shards by size: a group may straddle a
            split_groups.append(key)                # boundary (one per boundary); every part is read by its own name
        dt, sh, _ = hs[names[0]]
        if dt != "I16" or len(sh) != 3 or sh[:2] != [k_in // 16, n_out // 16] or sh[2] % 16:
            errors.append(f"{key}: trellis {dt} {sh} for {k_in}x{n_out}")
            continue
        if hs[names[1]][:2] != ("F16", [k_in]) or hs[names[2]][:2] != ("F16", [n_out]) or hs[names[3]][:2] != ("I32", []):
            errors.append(f"{key}: suh/svh/mul1 {hs[names[1]][:2]} {hs[names[2]][:2]} {hs[names[3]][:2]}")
        role = LAYER_INDEX.sub(".N.", key)
        role = re.sub(r"experts\.N\.", "experts.E.", re.sub(r"experts\.\d+\.", "experts.E.", role))
        bits_by_role[role].add(sh[2] // 16)
    text = {k for k in hs if not k.startswith("model.visual.")}
    extra = sorted(text - covered)
    bits = {r: sorted(b) for r, b in sorted(bits_by_role.items())}
    if a.identity:
        pinned = json.loads(Path(a.identity).read_text())
        if pinned != bits:
            errors.append("bits per role differ from the pinned identity")
    if a.write_identity:
        Path(a.write_identity).write_text(json.dumps(bits, indent=1))
    report = {"expected": len(manifest), "text_tensors": len(text), "covered": len(covered), "n_extra": len(extra),
              "extra": extra[:50], "n_errors": len(errors), "errors": errors[:200], "bits_by_role": bits,
              "n_split_groups": len(split_groups), "split_groups": split_groups[:50]}
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("expected", "text_tensors", "covered", "n_extra", "n_errors", "n_split_groups")}))
    return 1 if errors or extra else 0


if __name__ == "__main__":
    sys.exit(main())
