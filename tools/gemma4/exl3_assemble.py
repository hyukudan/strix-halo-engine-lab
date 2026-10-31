#!/usr/bin/env python3
"""Stream verified per-class EXL3 tensors into relocatable Gemma arm packs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct


ARMS = {
    "M4": dict(ffn=4, attn_qo=5, attn_kv=6, ple_lin=16, head=6),
    "M5": dict(ffn=5, attn_qo=6, attn_kv=6, ple_lin=16, head=6),
    "M6": dict(ffn=6, attn_qo=6, attn_kv=8, ple_lin=16, head=8),
    "U5": dict(ffn=5, attn_qo=5, attn_kv=5, ple_lin=5, head=5),
    "M4-h5": dict(ffn=4, attn_qo=5, attn_kv=6, ple_lin=16, head=5),
    "M4-p8": dict(ffn=4, attn_qo=5, attn_kv=6, ple_lin=8, head=6),
}
CHUNK = 4 * 1024 * 1024
PREFIX = "model.language_model."


def plan():
    result = {}
    def add(tail, shape, kind):
        result[PREFIX + tail] = (shape, kind)
    for layer in range(42):
        p = f"layers.{layer}."
        dim = 512 if layer % 6 == 5 else 256
        add(p + "self_attn.q_proj.weight", [8 * dim, 2560], "attn_qo")
        add(p + "self_attn.o_proj.weight", [2560, 8 * dim], "attn_qo")
        if layer < 24:
            for name in ("k", "v"):
                add(p + f"self_attn.{name}_proj.weight", [2 * dim, 2560], "attn_kv")
        for name in ("gate", "up"):
            add(p + f"mlp.{name}_proj.weight", [10240, 2560], "ffn")
        add(p + "mlp.down_proj.weight", [2560, 10240], "ffn")
        add(p + "per_layer_input_gate.weight", [256, 2560], "ple_lin")
        add(p + "per_layer_projection.weight", [2560, 256], "ple_lin")
    add("per_layer_model_projection.weight", [10752, 2560], "ple_lin")
    result["lm_head.weight"] = ([262144, 2560], "head")
    return result


def safe_file(root, name):
    if not isinstance(name, str) or not name or Path(name).name != name or any(c in name for c in "/\\:"):
        raise ValueError("unsafe shard name")
    return root / name


def header(path):
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors header")
        size, = struct.unpack("<Q", prefix)
        if size > 16 * 1024 * 1024:
            raise ValueError("excessive safetensors header")
        raw = f.read(size)
        if len(raw) != size:
            raise ValueError("truncated safetensors header")
    h = json.loads(raw)
    end = path.stat().st_size - 8 - size
    intervals = []
    for name, t in h.items():
        if name == "__metadata__":
            continue
        a, b = t["data_offsets"]
        if not isinstance(a, int) or not isinstance(b, int) or not 0 <= a <= b <= end:
            raise ValueError("safetensors offset outside file")
        intervals.append((a, b))
    intervals.sort()
    if any(a < previous for (_, previous), (a, _) in zip(intervals, intervals[1:])):
        raise ValueError("overlapping safetensors tensors")
    return h, size + 8


def copy_range(source, offset, count, output, digest):
    source.seek(offset)
    while count:
        data = source.read(min(CHUNK, count))
        if not data:
            raise ValueError("truncated tensor data")
        output.write(data)
        digest.update(data)
        count -= len(data)


def sources(root):
    result, recipes = {}, []
    for k in (4, 5, 6, 8):
        receipt = json.loads((root / f"verify-K{k}.json").read_text())
        if receipt.get("K") != k or receipt.get("errors") != []:
            raise ValueError("quant verification receipt missing or failed")
        for part in ("a", "b"):
            directory = root / f"K{k}-{part}"
            index = json.loads((directory / "index.json").read_text())
            r = index["recipe"]
            if (r["codebook"] != "mul1" or r["backend"] != "exl3" or r["scaled_hessian"] is not True
                    or r["fake_hess"] is not False or r["crop"] is not None
                    or r["seed"] != 787 or r["sigma_reg"] != 0.025):
                raise ValueError("unqualified quant recipe")
            recipe = {key: value for key, value in r.items() if key not in ("hess", "model")}
            recipes.append(recipe)
            for stem, tensor in index["tensors"].items():
                key = (k, tensor["hf"] if tensor["class"] != "head" else "lm_head.weight")
                if key in result or tensor["K"] != k or not math.isfinite(tensor["proxy_err"]):
                    raise ValueError("duplicate or invalid quant tensor")
                files = {index["weight_map"][stem + "." + suffix] for suffix in ("suh", "svh", "trellis", "mul1")}
                if len(files) != 1:
                    raise ValueError("split EXL3 tensor components")
                result[key] = (safe_file(directory, files.pop()), stem, tensor)
    if any(recipe != recipes[0] for recipe in recipes):
        raise ValueError("quant recipes do not match")
    return result, recipes[0]


def install_notices(output, model_card):
    license_path = Path(__file__).resolve().parents[2] / "src/gemma4/serve/notices/OPENMDW-LICENSE.txt"
    result = {"model_license": "OpenMDW-1.1"}
    for name, path, filename in (("model_card", model_card, "MODEL-CARD.md"),
                                 ("model_license", license_path, "MODEL-LICENSE.txt")):
        digest = hashlib.sha256()
        with path.open("rb") as source, (output / filename).open("wb") as target:
            copy_range(source, 0, path.stat().st_size, target, digest)
        result[name + "_file"] = filename
        result[name + "_sha256"] = digest.hexdigest()
    return result


def assemble(root, output, arm, classes=None, model_card=None):
    if model_card is None or not model_card.is_file():
        raise ValueError("original model card required to retain notices of origin")
    source, recipe = sources(root)
    classes = dict(ARMS[arm] if classes is None else classes)
    if set(classes) != set(ARMS["M4"]) or any(v not in (4, 5, 6, 8, 16) for v in classes.values()):
        raise ValueError("invalid class map")
    if any(classes[kind] == 16 for kind in ("head", "ffn", "attn_qo", "attn_kv")):
        raise ValueError("only PLE linears may use the BF16 exemption")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() or (output / "weights.bin").exists():
        raise ValueError("output already contains an arm")
    records, total_params, streamed = [], 0, 0
    temporary = output / "weights.bin.partial"
    with temporary.open("xb") as out:
        for name, (shape, kind) in sorted(plan().items()):
            k = classes[kind]
            count = math.prod(shape)
            total_params += count
            if k == 16:
                streamed += count * 2
                records.append(dict(name=name, shape=shape, **{"class": kind}, format="BF16", bytes=count * 2))
                continue
            path, stem, meta = source[(k, name)]
            if meta["shape_out_in"] != shape or meta["class"] != kind:
                raise ValueError("quant tensor shape/class mismatch")
            h, base = header(path)
            offset, digest = out.tell(), hashlib.sha256()
            expected = {"suh": ("F16", [shape[1]], 2 * shape[1]),
                        "svh": ("F16", [shape[0]], 2 * shape[0]),
                        "trellis": ("I16", [shape[1] // 16, shape[0] // 16, 16 * k], count * k // 8),
                        "mul1": ("I32", [], 4)}
            if set(h) - {"__metadata__"} != {stem + "." + x for x in expected}:
                raise ValueError("unexpected EXL3 component")
            with path.open("rb") as f:
                for suffix, (dtype, dims, size) in expected.items():
                    t = h[stem + "." + suffix]
                    a, b = t["data_offsets"]
                    if t["dtype"] != dtype or t["shape"] != dims or b - a != size:
                        raise ValueError("EXL3 component geometry")
                    if suffix == "mul1":
                        f.seek(base + a)
                        if f.read(4) != struct.pack("<I", 0x83DCD12D):
                            raise ValueError("EXL3 mul1 codebook marker mismatch")
                    else:
                        copy_range(f, base + a, size, out, digest)
            size = out.tell() - offset
            streamed += size
            records.append(dict(name=name, shape=shape, **{"class": kind}, format="EXL3_MUL1", K=k,
                                offset=offset, bytes=size, sha256=digest.hexdigest()))
        out.flush()
        os.fsync(out.fileno())
    manifest = dict(schema="g4-exl3-pack-v1", identity="g4-exl3-v1", arm=arm,
                    classes=classes, recipe=recipe, tensors=records, weights="weights.bin",
                    quantized_file_bytes=temporary.stat().st_size, linear_parameters=total_params,
                    linear_bytes_per_token=streamed, effective_linear_bpw=8 * streamed / total_params,
                    codebook_bytes=0, codebook="algorithmic-mul1", scales_included=True,
                    embedding_format="BF16", ple_table_format="BF16", qualification="draft")
    # Resident gathers are separate from streamed linear weights. Norms/KV and
    # activation traffic depend on context and are not claimed as weight traffic.
    manifest["embedding_gather_bytes_per_token"] = 2 * (2560 + 10752)
    manifest["weight_bytes_per_token"] = streamed + manifest["embedding_gather_bytes_per_token"]
    manifest["resident_bf16_gather_bytes"] = 2 * 262144 * (2560 + 10752)
    manifest.update(install_notices(output, model_card))
    os.replace(temporary, output / "weights.bin")
    (output / "manifest.json.partial").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(output / "manifest.json.partial", output / "manifest.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--model-card", type=Path, required=True, help="original checkpoint README; copied unchanged")
    parser.add_argument("--class-k", action="append", default=[], metavar="CLASS=K")
    args = parser.parse_args()
    classes = dict(ARMS[args.arm])
    for item in args.class_k:
        key, value = item.split("=", 1)
        classes[key] = int(value)
    result = assemble(args.source, args.output, args.arm, classes, args.model_card)
    print(json.dumps({key: result[key] for key in ("arm", "classes", "effective_linear_bpw", "weight_bytes_per_token")}))


if __name__ == "__main__":
    main()
