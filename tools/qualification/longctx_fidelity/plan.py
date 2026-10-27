"""CPU-only deterministic matrix expansion and conservative turn packing."""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
try:
    from .common import canonical, atomic, case_id
except ImportError:
    from common import canonical, atomic, case_id

PREFILL = {8192: 4, 32768: 25, 65536: 50, 131072: 105, 196608: 160, 262144: 210}


def predicted(case, table=None, load=40, decode=.03):
    table = PREFILL if table is None else table
    if not table or any(not math.isfinite(v) or v <= 0 for v in table.values()):
        raise ValueError("prediction table requires positive finite upper bounds")
    depth = case["depth"]
    buckets = sorted(k for k in table if k >= depth)
    if not buckets:
        raise ValueError("no prediction upper bound for depth")
    value = load + table[buckets[0]] + case["generate"] * decode
    if not math.isfinite(value) or value <= 0:
        raise ValueError("invalid predicted seconds")
    return value


def matrix(modes, depths, families, seeds, positions, generate=32, stratum="prose", variant="normal"):
    cells = []
    for m, d, f, s, p in sorted(itertools.product(modes, depths, families, seeds, positions)):
        if m != "native" or not 1 <= d or not 1 <= generate or d + generate > 262144:
            raise ValueError("matrix cell violates native D+G budget")
        if d == 262143 and (generate != 1 or variant != "symbol"):
            raise ValueError("edge matrix requires G=1 and symbol answers")
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("invalid matrix position")
        cells.append({"mode": m, "depth": d, "family": f, "seed": s, "position": p,
                      "generate": generate, "stratum": stratum, "variant": variant,
                      "case_id": case_id(f, d, p, s, stratum, variant, m), "status": "PLANNED"})
    if len({c["case_id"] for c in cells}) != len(cells):
        raise ValueError("duplicate matrix cells")
    return cells


def key(case):
    return tuple(case[k] for k in ("mode", "depth", "family", "seed", "position")) + (case["case_id"],)


def pack(cases, soft=500, table=None):
    if not math.isfinite(soft) or soft <= 0:
        raise ValueError("invalid soft budget")
    turns, current, seconds, group = [], [], 0.0, None
    for case in sorted(cases, key=key):
        cost = predicted(case, table)
        if cost > soft:
            raise ValueError("one case exceeds soft turn budget: " + case["case_id"])
        next_group = (case["mode"], case["depth"], case["family"])
        # Keep concrete turn classes homogeneous. Deep cells are singleton turns,
        # even if a short arm could fit an arithmetic sum of timing estimates.
        if current and (seconds + cost > soft or next_group != group or case["depth"] >= 196608):
            turns.append({"cases": current, "predicted_seconds": round(seconds, 6)})
            current, seconds = [], 0.0
        current.append(case["case_id"])
        seconds += cost
        group = next_group
    if current:
        turns.append({"cases": current, "predicted_seconds": round(seconds, 6)})
    return turns


def write_plan(cases, directory, soft=500, table=None, case_files=None):
    directory = Path(directory)
    turns = pack(cases, soft, table)
    lines = ["PLANNED; CPU only; estimates are not measured timing; model load charged per case."]
    for i, turn in enumerate(turns, 1):
        name = f"turn-{i:03d}.txt"
        paths = [os.path.relpath(case_files[c], directory).replace(os.sep, "/")
                 if case_files else c + ".json" for c in turn["cases"]]
        atomic(directory / name, ("\n".join(paths) + "\n").encode())
        lines.append(f"{name} predicted_upper_seconds={turn['predicted_seconds']:.2f}")
        lines.extend("  " + c for c in turn["cases"])
    result = "\n".join(lines) + "\n"
    atomic(directory / "plan.txt", result.encode())
    atomic(directory / "turns.json", canonical({"status": "PLANNED", "soft_seconds": soft, "turns": turns}))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifests", nargs="*")
    p.add_argument("--matrix", help="dimension JSON: modes,depths,families,seeds,positions; optional generate,stratum,variant")
    p.add_argument("--out", required=True)
    p.add_argument("--soft", type=float, default=500)
    p.add_argument("--prediction-table", help="JSON object mapping depth upper bounds to seconds")
    a = p.parse_args()
    table = {int(k): float(v) for k, v in json.loads(Path(a.prediction_table).read_text()).items()} if a.prediction_table else None
    if bool(a.manifests) == bool(a.matrix):
        p.error("provide either explicit case manifests or --matrix dimension JSON")
    cases = (matrix(**json.loads(Path(a.matrix).read_text(encoding="utf-8"))) if a.matrix else
             [json.loads(Path(f).read_text(encoding="utf-8")) for f in a.manifests])
    if len({c["case_id"] for c in cases}) != len(cases):
        p.error("duplicate case IDs")
    files = {case["case_id"]: str(Path(path).resolve()) for case, path in zip(cases, a.manifests)} if a.manifests else None
    if a.matrix:
        atomic(Path(a.out) / "matrix.json", canonical({"schema": 1, "status": "PLANNED", "cells": cases}))
    print(write_plan(cases, a.out, a.soft, table, files), end="")


if __name__ == "__main__":
    main()
