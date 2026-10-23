#!/usr/bin/env python3
"""Compare exactly 64 full-logit rows: exact reference, fast control, replay.

This local numerical gate does not qualify AMD continuation or performance.
Exit 0 = local KL/top1 pass; exit 1 = quality fail; exit 2 = invalid evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path, vocab):
    data = path.read_bytes()
    if len(data) != 64 * vocab * 4:
        raise ValueError(f"{path}: expected exactly 64 x {vocab} float32 logits")
    rows = np.frombuffer(data, dtype="<f4").reshape(64, vocab).astype(np.float64)
    if not np.isfinite(rows).all():
        raise ValueError(f"{path}: non-finite logits")
    return rows, hashlib.sha256(data).hexdigest()


def metrics(ref, candidate):
    def logsm(x):
        y = x - x.max(axis=1, keepdims=True)
        return y - np.log(np.exp(y).sum(axis=1, keepdims=True))
    a, b = logsm(ref), logsm(candidate)
    kl = (np.exp(a) * (a - b)).sum(axis=1)
    return {"kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
            "top1": float((ref.argmax(axis=1) == candidate.argmax(axis=1)).mean()),
            "kl_positions": kl.tolist()}


def evaluate(ref, fast, replay):
    control, candidate = metrics(ref, fast), metrics(ref, replay)
    # No absolute quality allowance is added when the control KL is zero.
    checks = {key: candidate[key] <= control[key] * 1.10 for key in ("kl_mean", "kl_max")}
    checks["top1"] = candidate["top1"] >= control["top1"] - 0.01
    return {"positions": 64, "control": control, "replay": candidate, "checks": checks,
            "local_quality_pass": all(checks.values()), "qualified": False,
            "remaining_gates": ["AMD continuation 64-256 tokens", "complete-workload performance"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exact", type=Path)
    parser.add_argument("fast", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--vocab", type=int, default=129280)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.vocab < 1:
            raise ValueError("vocab must be positive")
        files = [load(p, args.vocab) for p in (args.exact, args.fast, args.replay)]
        result = evaluate(*(f[0] for f in files))
        result["inputs"] = [{"path": str(p), "sha256": f[1]} for p, f in
                            zip((args.exact, args.fast, args.replay), files)]
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 0 if result["local_quality_pass"] else 1
    except (ValueError, OSError) as error:
        parser.exit(2, f"invalid evidence: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
