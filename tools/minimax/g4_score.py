"""Score a G4 pair run against the exact-path oracle with the pre-registered rule (qualification notes).

Inputs per prompt: ORACLE_DIR (cpu_layer_reference.py --full-dir: logits_exact.f32, logits_cpuprod.f32, rows.u32) and
the pair's logits_pair.f32 (strix-hip-minimax-g4, rank 0), all [rows x 200064] float32. Prints one JSON line per prompt
and the verdict; exit code 0 only if every prompt passes rules 1-5 (6-7 come from the run logs).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

VOCAB = 200064


def log_softmax(z):
    z = z.astype(np.float64)
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def compare(ref, cand, chunk=128):
    kl, top = [], []
    for r0 in range(0, len(ref), chunk):
        a, b = log_softmax(ref[r0:r0 + chunk]), log_softmax(cand[r0:r0 + chunk])
        kl.append((np.exp(a) * (a - b)).sum(1))
        top.append(ref[r0:r0 + chunk].argmax(1) == cand[r0:r0 + chunk].argmax(1))
    kl, top = np.concatenate(kl), np.concatenate(top)
    return {"kl_mean": float(kl.mean()), "kl_p99": float(np.percentile(kl, 99)), "kl_max": float(kl.max()),
            "top1_agree": float(top.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", nargs=3, action="append", metavar=("NAME", "ORACLE_DIR", "PAIR_LOGITS"), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    results, ok_all = {}, True
    for name, oracle, pair in args.prompt:
        rows = np.fromfile(Path(oracle) / "rows.u32", dtype="<u4")
        shape = (len(rows), VOCAB)
        exact = np.fromfile(Path(oracle) / "logits_exact.f32", dtype="<f4").reshape(shape)
        cpu = np.fromfile(Path(oracle) / "logits_cpuprod.f32", dtype="<f4").reshape(shape)
        dev = np.fromfile(pair, dtype="<f4").reshape(shape)
        base, cand = compare(exact, cpu), compare(exact, dev)
        rules = {
            "1_kl_mean": cand["kl_mean"] <= 2 * base["kl_mean"] + 0.002,
            "2_kl_p99": cand["kl_p99"] <= 2 * base["kl_p99"] + 0.02,
            "3_top1": cand["top1_agree"] >= base["top1_agree"] - 0.01,
            "4_abs_cap": cand["kl_mean"] <= 0.05,
            "5_finite": bool(np.isfinite(dev).all()),
        }
        ok = all(rules.values())
        ok_all &= ok
        results[name] = {"cpuprod_vs_exact": base, "pair_vs_exact": cand, "rules": rules, "pass": ok}
        print(json.dumps({name: results[name]}), flush=True)
    results["verdict_rules_1_5"] = "PASS" if ok_all else "FAIL"
    Path(args.out).write_text(json.dumps(results, indent=1))
    print("G4 rules 1-5:", results["verdict_rules_1_5"])
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
