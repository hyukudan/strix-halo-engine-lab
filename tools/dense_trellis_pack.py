#!/usr/bin/env python3
"""Package the selected dense K5 assets, with source pins; no quantization.

Writes an experimental decode-only pack. Its recipe is not a model-quality or
performance qualification. Generated assets and receipts belong outside Git.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from make_dense_trellis_fixture import read_container
from q6p_quantize import write_pack


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--target-content", required=True)
    ap.add_argument("--target-sha256", required=True)
    ap.add_argument("--mix", choices=("G0", "G1"), default="G1")
    args = ap.parse_args()
    for digest in (args.target_content, args.target_sha256):
        if not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("target pins must be lowercase SHA-256")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise ValueError("pack output must be new")
    index = json.loads((args.source / "index.json").read_text())["weight_map"]
    rx = re.compile(r"blk\.(\d+)\.(attn_gate|attn_q)\.trellis$")
    selected = sorted(key[:-len(".trellis")] for key in index if rx.fullmatch(key)
                      and (args.mix == "G1" or ".attn_gate." in key))
    expected = {f"blk.{layer}.attn_gate" for layer in range(48) if layer % 4 != 3}
    if args.mix == "G1":
        expected |= {f"blk.{layer}.attn_q" for layer in range(3, 48, 4)}
    if set(selected) != expected:
        raise ValueError("selected mix does not contain every expected layer")
    entries, sources = [], {}
    compact_bytes, q8_bytes = 0, 0
    for base in selected:
        data, tiles, su, sv, inputs, outputs, bits = read_container(args.source / index[base + ".trellis"], base)
        if inputs != 2560 or outputs != (6144 if base.endswith("attn_gate") else 12288) or bits != 5:
            raise ValueError("selected tensor is outside the fixed G0/G1 geometry")
        name = base + ".weight"
        packed, u, v = tiles.astype("<u2").tobytes(), su.astype("<f2").tobytes(), sv.astype("<f2").tobytes()
        entries += [(name + ".trellis", 24, [len(packed)], packed),
                    (name + ".suh", 1, [inputs], u), (name + ".svh", 1, [outputs], v)]
        sources[name] = hashlib.sha256(data).hexdigest()
        compact_bytes += len(packed) + len(u) + len(v)
        q8_bytes += inputs * outputs // 32 * 34
    meta = {
        "strix.dense_trellis.format": "TRELLIS128-mul1-K5-v1",
        "strix.dense_trellis.mix": args.mix,
        "strix.dense_trellis.recipe": "fp32-transform-fma-split4-v1",
        "strix.dense_trellis.scope": "decode-r1-r2-r4-r8",
        "strix.dense_trellis.marker": "83dcd12d",
        "strix.dense_trellis.target_content": args.target_content,
        "strix.dense_trellis.target_file_sha256": args.target_sha256,
        "strix.dense_trellis.source_sha256": json.dumps(sources, sort_keys=True, separators=(",", ":")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_pack(str(args.output), entries, meta)
    h = hashlib.sha256()
    with args.output.open("rb") as pack:
        while chunk := pack.read(8 << 20):
            h.update(chunk)
    receipt = dict(format=meta["strix.dense_trellis.format"], recipe=meta["strix.dense_trellis.recipe"], mix=args.mix,
                   target_content=args.target_content, target_file_sha256=args.target_sha256, source_sha256=sources,
                   tensors=len(selected), packed_bytes=compact_bytes, source_q8_bytes=q8_bytes, pack_sha256=h.hexdigest(),
                   model_quality_qualified=False, performance_qualified=False)
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
