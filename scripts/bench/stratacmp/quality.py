#!/usr/bin/env python3
"""Full-vocabulary diagnostic bridge; statistical TURBO qualification remains in turbo_bar."""
import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.qualification.turbo_bar import losses, spool
from tools.qualification.turbo_bar import manifest as mf
from tools.qualification.turbo_bar.common import sha256_file


def inputs(path):
    values = [int(x) for x in Path(path).read_text().split()]
    if len(values) < 2 or any(x < 0 or x >= 248320 for x in values):
        raise ValueError("expected at least two Qwen vocabulary IDs")
    return values


def selections(count, stride):
    if stride < 1:
        raise ValueError("stride must be positive")
    # Dumped final prompt row predicts an unknown generated token; exclude it from labeled losses.
    # CUDA verify needs a nonempty prefilled state; its API cannot observe row zero.
    return [p for p in range(1, count - 1) if p % stride == 0]


def convert(raw, ids_path, stride, out):
    ids = inputs(ids_path)
    with open(raw, "rb") as f:
        header = f.read(8)
    if len(header) != 8:
        raise ValueError("incomplete Strata logits header")
    vocab, rows = struct.unpack("<ii", header)
    positions = selections(len(ids), stride)
    # Strata max-new=1 stores multiples of stride plus the final prompt position once.
    expected_rows = len(positions) + 2  # unsupported row zero and unlabeled final row
    if vocab != 248320 or rows != expected_rows or Path(raw).stat().st_size != 8 + rows * vocab * 4:
        raise ValueError("Strata logits dimensions/size do not match IDS, stride and max-new=1")
    data = np.memmap(raw, dtype="<f4", mode="r", offset=8, shape=(rows, vocab))
    if not np.isfinite(data).all():
        raise ValueError("nonfinite Strata logits")
    if Path(out).exists():
        raise ValueError("output exists")
    if not positions:
        raise ValueError("no supported selected positions")
    spool.write(out, ids, positions, data[1:1+len(positions)], version=2)
    return {"source_sha256": sha256_file(raw), "ids_sha256": sha256_file(ids_path),
            "stride": stride, "dropped_unlabeled_final_row": True, "dropped_unsupported_row_zero": True,
            "rows": len(positions)}


def compare(reference, arms):
    files = {"E": reference}
    for item in arms:
        name, path = item.split("=", 1)
        if not name or name in files:
            raise ValueError("arm names must be unique; E is reserved")
        files[name] = path
    if len(files) < 2:
        raise ValueError("at least one candidate arm is required")
    pairs = [("E", a) for a in files if a != "E"]
    sums = losses.case_sums({"case": "diagnostic", "files": files}, pairs)
    n = sums["n"]
    if not n:
        raise ValueError("no comparable rows")
    return {"scope": "diagnostic; no TURBO pass verdict", "rows": n,
            "payload_sha256": {a: sha256_file(p) for a, p in files.items()},
            "arms": {a: {"kl_ref_run": sums["sums"][losses.pair_name("E", a)]["kl"] / n,
                         "top1_agreement": 1 - sums["sums"][losses.pair_name("E", a)]["dis"] / n,
                         "dnll_run_minus_ref": sums["sums"][losses.pair_name("E", a)]["dnll"] / n,
                         "bitwise_equal": sums["bitwise"][losses.pair_name("E", a)]}
                     for a in files if a != "E"}}


def convert_first(prefix, ids_path, positions_path, out):
    ids = inputs(ids_path)
    positions = [int(x) for x in Path(positions_path).read_text().split()]
    if not positions or positions != sorted(set(positions)) or positions[0] < 0 or positions[-1] >= len(ids) - 1:
        raise ValueError("invalid selected positions")
    if Path(out).exists():
        raise ValueError("output exists")
    paths = [Path(str(prefix) + "." + str(i)) for i in range(len(positions))]
    rows = []
    for path in paths:
        if path.stat().st_size != 248320 * 4:
            raise ValueError("first-logits row has wrong extent")
        row = np.memmap(path, dtype="<f4", mode="r", shape=(248320,))
        if not np.isfinite(row).all():
            raise ValueError("nonfinite first logits")
        rows.append(row)
    spool.write(out, ids, positions, rows, version=2)
    return {"source_sha256": [sha256_file(p) for p in paths], "ids_sha256": sha256_file(ids_path),
            "positions_sha256": sha256_file(positions_path), "rows": len(rows),
            "conditioning_contract": "request i consumed IDS[:positions[i]+1]; externally recorded request order required"}


def receipt(args):
    frozen = mf.load(args.manifest)
    spec = frozen["arms"]["E" if args.arm == "Erep" else args.arm]
    binary_sha = sha256_file(args.binary)
    if binary_sha != spec["binary"]["sha256"]:
        raise ValueError("binary differs from frozen arm")
    for weight in spec["weights"]:
        if Path(weight["path"]).stat().st_size != weight["bytes"] or sha256_file(weight["path"]) != weight["sha256"]:
            raise ValueError("weight differs from frozen arm")
    header = spool.header(args.spool)
    # Full stream validation includes each row's label, position and finite-logit checks.
    for _ in spool.read(args.spool):
        pass
    engine_log = Path(args.engine_log).read_text(encoding="utf-8", errors="replace")
    match = re.search(r"teacher batching: step_rows=(\d+) prefix_rows=(\d+) context_capacity=(\d+) block_rows=(\d+)", engine_log)
    if not match:
        raise ValueError("missing actual teacher batch/config receipt in engine log")
    effective = dict(zip(("step_rows", "prefix_rows", "context_capacity", "block_rows"), map(int, match.groups())))
    return {"schema": "stratacmp-diagnostic-receipt/v1", "manifest_core_sha256": frozen["frozen"]["core_sha256"],
            "manifest_version": frozen["version"], "arm": args.arm, "identity": spec["identity"],
            "arithmetic": spec["arithmetic"], "binary_sha256": binary_sha,
            "weights_sha256": [w["sha256"] for w in spec["weights"]],
            "payload_sha256": sha256_file(args.spool), "engine_log_sha256": sha256_file(args.engine_log),
            "engine_log": engine_log, "effective_teacher": effective,
            "inputs_sha256": hashlib.sha256(header["inputs"].tobytes()).hexdigest(),
            "positions": header["positions"].tolist() if header["positions"] is not None else list(range(header["rows"])),
            "rows": header["rows"], "vocab": header["vocab"],
            "effective_scope": "teacher-forced decode after optional prefix; no drafts; full-vocabulary selected rows",
            "context_input_tokens": len(header["inputs"]),
            "scope": "diagnostic; not a TURBO qualification receipt"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("positions")
    p.add_argument("--ids", required=True); p.add_argument("--stride", type=int, default=256)
    p.add_argument("--out", required=True)
    p = sub.add_parser("strata")
    p.add_argument("--raw", required=True); p.add_argument("--ids", required=True)
    p.add_argument("--stride", type=int, default=256); p.add_argument("--out", required=True)
    p = sub.add_parser("strata-first")
    for option in ("prefix", "ids", "positions", "out"):
        p.add_argument("--" + option, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--reference", required=True); p.add_argument("--arm", action="append", required=True)
    p.add_argument("--receipt", action="append", default=[], help="ARM=JSON; same CUDA teacher shape required")
    p.add_argument("--out", required=True)
    p = sub.add_parser("receipt")
    for option in ("manifest", "arm", "spool", "binary", "engine-log", "out"):
        p.add_argument("--" + option, required=True)
    args = parser.parse_args()
    if args.command == "positions":
        Path(args.out).write_text("\n".join(map(str, selections(len(inputs(args.ids)), args.stride))) + "\n")
        return
    if args.command in ("strata", "strata-first"):
        result = (convert(args.raw, args.ids, args.stride, args.out) if args.command == "strata"
                  else convert_first(args.prefix, args.ids, args.positions, args.out))
        Path(str(args.out) + ".source.json").write_text(json.dumps(result, indent=2) + "\n")
    elif args.command == "compare":
        result = compare(args.reference, args.arm)
        receipts = {name: json.loads(Path(path).read_text()) for name, path in (x.split("=", 1) for x in args.receipt)}
        configs = []
        for name, record in receipts.items():
            if result["payload_sha256"].get(name) != record.get("payload_sha256"):
                raise ValueError("teacher receipt does not bind compared payload")
            config = record["effective_teacher"]
            configs.append(tuple(config[k] for k in ("step_rows", "prefix_rows", "context_capacity", "block_rows")))
        if len(set(configs)) > 1:
            raise ValueError("CUDA teacher batch/prefix/context/block settings differ across arms")
        result["teacher_batch_validation"] = "matched supplied receipts" if receipts else "unverified diagnostic only; no E/R/T gate"
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    else:
        result = receipt(args)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
