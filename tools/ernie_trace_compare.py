#!/usr/bin/env python3
"""Compare the captured first token of a two-rank ERNIE session."""

import argparse
import csv
import json
import math
from pathlib import Path
import struct
import sys
import tomllib


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hidden_rows(path, positions, layers):
    expected = {(position, layer) for position in range(positions) for layer in layers}
    rows = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            position, layer = int(row["position"]), int(row["layer"])
            if position >= positions:
                continue
            key = position, layer
            require(key in expected and key not in rows, f"{path} unexpected or duplicate layer row {key}")
            rows[key] = row["attention_sha256"], row["sha256"]
    require(rows.keys() == expected, f"{path} incomplete layer trace")
    return rows


def seam_rows(path, positions):
    rows = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            position = int(row["position"])
            require(position not in rows, f"{path} duplicate seam position {position}")
            rows[position] = row["sha256"]
    require(rows.keys() == set(range(positions)), f"{path} incomplete seam trace")
    return rows


def compare(reference, rank0, rank1, layers, vocab, cut):
    require(0 < cut < layers, "invalid layer cut")
    receipt = json.loads((reference / "run.json").read_text(encoding="utf-8"))
    prompt = (reference / "prompt.u32").read_bytes()
    require(bool(prompt) and len(prompt) % 4 == 0, "invalid reference prompt bytes")
    positions = len(prompt) // 4
    for rank in (rank0, rank1):
        require((rank / "prompt.u32").read_bytes() == prompt, f"{rank} prompt IDs differ")
    host_id = (reference / "generated.u32").read_bytes()[:4]
    require(len(host_id) == 4, "missing reference generated token")
    for rank in (rank0, rank1):
        require((rank / "generated.u32").read_bytes() == host_id, f"{rank} greedy token differs")

    with (reference / "raw-logits.f32").open("rb") as handle:
        host_logits = handle.read(vocab * 4)
    rank_logits = (rank1 / "raw-logits.f32").read_bytes()
    require(len(host_logits) == len(rank_logits) == vocab * 4, "invalid full-vocabulary logit row byte count")
    require(host_logits == rank_logits, "raw logits differ")
    logits = struct.unpack(f"<{vocab}f", host_logits)
    require(all(map(math.isfinite, logits)), "nonfinite logits")
    top = sorted(range(vocab), key=lambda index: (-logits[index], index))[:20]
    require(top[0] == struct.unpack("<I", host_id)[0], "reference logit argmax differs from selected ID")

    host_hidden = hidden_rows(reference / "hidden.tsv", positions, range(layers))
    for rank, owned in ((rank0, range(cut)), (rank1, range(cut, layers))):
        native = hidden_rows(rank / "hidden.tsv", positions, owned)
        for key, digest in native.items():
            require(host_hidden[key] == digest, f"{rank} layer digests differ at position/layer {key}")
        seam = seam_rows(rank / "seam.tsv", positions)
        for position, digest in seam.items():
            require(host_hidden[position, cut - 1][1] == digest, f"{rank} seam digest differs at position {position}")
    return {"reference_identity": receipt["identity"], "matched_prompt_rows": positions,
            "matched_layer_rows": positions * layers, "matched_seam_rows_per_rank": positions,
            "id": top[0], "top20": top, "raw_logit_bytes": len(rank_logits)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle_prompt", type=Path)
    parser.add_argument("rank0", type=Path)
    parser.add_argument("rank1", type=Path)
    parser.add_argument("--cut", type=int, required=True)
    args = parser.parse_args()
    model = Path(__file__).resolve().parents[1] / "models" / "ernie-5.5-flash-preview.toml"
    with model.open("rb") as handle:
        dims = tomllib.load(handle)["dims"]
    print(json.dumps(compare(args.oracle_prompt, args.rank0, args.rank1, dims["layers"], dims["vocab"], args.cut)))


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
