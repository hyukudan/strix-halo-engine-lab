#!/usr/bin/env python3
"""Compare a free reference capture with a variant forced on its emitted tokens.

Records are little-endian <6I followed by vocab float32 values. Use fresh paths
per request (published only after successful finalization). --tokens-out exports A's emissions for forcing.
"""

import argparse
import json
import struct
from itertools import zip_longest
from pathlib import Path

import numpy as np


HEADER = struct.Struct("<6I")
MAGIC = 0x42434C47


def records(path):
    if Path(path).name.endswith(".partial"):
        raise ValueError(f"{path}: .partial captures are incomplete and cannot be compared")
    with open(path, "rb") as source:
        size = source.seek(0, 2)
        source.seek(0)
        expected_step = 0
        previous_position = None
        while header := source.read(HEADER.size):
            if len(header) != HEADER.size:
                raise ValueError(f"{path}: truncated header at step {expected_step}")
            magic, step, position, argmax, emitted, vocab = HEADER.unpack(header)
            if magic != MAGIC or step != expected_step or not vocab or max(argmax, emitted) >= vocab:
                raise ValueError(f"{path}: invalid header at step {expected_step}")
            if previous_position is not None and position != previous_position + 1:
                raise ValueError(f"{path}: nonconsecutive position at step {step}")
            if vocab * 4 > size - source.tell():
                raise ValueError(f"{path}: truncated logits at step {step}")
            logits = np.frombuffer(source.read(vocab * 4), dtype="<f4").astype(np.float64)
            if not np.isfinite(logits).all() or int(np.argmax(logits)) != argmax:
                raise ValueError(f"{path}: nonfinite logits or inconsistent firstmax at step {step}")
            yield step, position, argmax, emitted, logits
            expected_step += 1
            previous_position = position


def log_softmax(values):
    shifted = values - values.max()
    return shifted - np.log(np.exp(shifted).sum())


def relative_l2(diff_squared, reference_squared):
    if reference_squared == 0:
        return 0.0 if diff_squared == 0 else None
    return float(np.sqrt(diff_squared / reference_squared))


def compare(reference, variant):
    n = agreement = 0
    divergence = None
    kl, overlaps, tokens = [], [], []
    max_delta = diff_squared = reference_squared = 0.0
    step0 = None
    for a, b in zip_longest(records(reference), records(variant)):
        if a is None or b is None:
            raise ValueError("captures have different step counts")
        step, position, argmax_a, emitted_a, logits_a = a
        _, position_b, argmax_b, emitted_b, logits_b = b
        if position != position_b or logits_a.shape != logits_b.shape:
            raise ValueError(f"unaligned position or vocabulary at step {step}")
        if emitted_a != argmax_a or emitted_b != emitted_a:
            raise ValueError(f"step {step}: A must be free and B must emit A's tokens")
        agreement += argmax_a == argmax_b
        if divergence is None and argmax_b != emitted_a:
            divergence = step
        tokens.append(emitted_a)
        log_a, log_b = log_softmax(logits_a), log_softmax(logits_b)
        kl.append(max(0.0, float(np.dot(np.exp(log_a), log_a - log_b))))
        k = min(5, logits_a.size)
        top_a = np.argsort(-logits_a, kind="stable")[:k]
        top_b = np.argsort(-logits_b, kind="stable")[:k]
        overlaps.append(len(set(top_a) & set(top_b)) / k)
        delta = logits_b - logits_a
        maximum = float(np.abs(delta).max())
        squared = float(np.dot(delta, delta))
        norm = float(np.dot(logits_a, logits_a))
        if step == 0:
            step0 = {"max_abs_dlogit": maximum, "rel_l2": relative_l2(squared, norm)}
        max_delta = max(max_delta, maximum)
        diff_squared += squared
        reference_squared += norm
        n += 1
    if not n:
        raise ValueError("captures contain no steps")
    result = {
        "n_steps": n,
        "top1_agreement": agreement / n,
        "first_greedy_divergence_step": divergence,
        "greedy_divergence_length": n if divergence is None else divergence,
        "kl_pa_pb": {"mean": float(np.mean(kl)), "p50": float(np.percentile(kl, 50)),
                     "p99": float(np.percentile(kl, 99)), "max": max(kl)},
        "step0": step0,
        "all_steps": {"max_abs_dlogit": max_delta,
                      "rel_l2": relative_l2(diff_squared, reference_squared)},
        "top5_overlap_mean": float(np.mean(overlaps)),
    }
    return result, tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="A: reference free run")
    parser.add_argument("variant", help="B: variant forced on A's emitted tokens")
    parser.add_argument("--tokens-out", type=Path, help="write A's tokens for STRIX_BITCALC_FORCE_TOKENS (compare A A first)")
    args = parser.parse_args()
    try:
        result, tokens = compare(args.reference, args.variant)
        if args.tokens_out:
            args.tokens_out.write_text(" ".join(map(str, tokens)) + "\n", encoding="ascii")
    except (OSError, ValueError) as error:
        parser.exit(2, f"bitcalc_cmp: {error}\n")
    print(json.dumps(result, allow_nan=False))
    print(f"{result['n_steps']} steps; top-1 {result['top1_agreement']:.2%}; "
          f"first divergence {result['first_greedy_divergence_step']}; "
          f"KL mean {result['kl_pa_pb']['mean']:.6g}; "
          f"max |dlogit| {result['all_steps']['max_abs_dlogit']:.6g}; "
          f"top-5 overlap {result['top5_overlap_mean']:.2%}")


if __name__ == "__main__":
    main()
