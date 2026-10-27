"""Exact symbolic scoring. Only trim outer whitespace and ONE final period.

No case folding, substring acceptance, reasoning stripping, or punctuation repair.
Multi-key JSON pairs preserve key order and require unique keys. Missing outputs
always score as unmeasured INCOMPLETE, never a success.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
try:
    from .common import canonical, atomic, file_sha
    from .tok import StrixTokenize
    from .run_cli import NATIVE_EOS_ID
except ImportError:
    from common import canonical, atomic, file_sha
    from tok import StrixTokenize
    from run_cli import NATIVE_EOS_ID

# <|im_end|> in the Qwen3.8-Flash-Next vocabulary (strix-tokenize encode); the protocol
# EOS that stops the plain CLI is <|endoftext|> = NATIVE_EOS_ID.
IM_END_ID = 248046


def normalize(text):
    text = text.strip()
    return text[:-1].rstrip() if text.endswith(".") else text


def exact(answer, output):
    if output is None:
        return {"exact": False, "per_value": [], "all_values": False}
    output = normalize(output)
    if not isinstance(answer, list):
        okay = output == str(answer)
        return {"exact": okay, "per_value": [okay], "all_values": okay}
    try:
        parsed = json.loads(output)
        if not isinstance(parsed, list) or any(not isinstance(p, list) or len(p) != 2
                                              or not all(isinstance(x, str) for x in p) for p in parsed):
            raise ValueError("not ordered string pairs")
        if len({p[0] for p in parsed}) != len(parsed):
            raise ValueError("duplicate keys")
        values = dict(parsed)
        per_value = [values.get(k) == v for k, v in answer]
        okay = parsed == answer
    except (ValueError, TypeError):
        per_value, okay = [False] * len(answer), False
    return {"exact": okay, "per_value": per_value, "all_values": okay}


def wilson(successes, total, z=1.959963984540054):
    if not 0 <= successes <= total or total < 0:
        raise ValueError("invalid counts")
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z*z / total
    middle = (p + z*z / (2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0.0, middle-half), min(1.0, middle+half)]


def mutation_self_test():
    answer = [["key-a", "value-a"], ["key-b", "value-b"]]
    swapped = json.dumps([["key-a", "value-b"], ["key-b", "value-a"]])
    return exact(answer, json.dumps(answer))["exact"] and not exact(answer, swapped)["exact"]


def decode_receipt(tokenizer, receipt):
    """Separate a declared terminal protocol EOS and end answer content at the first <|im_end|>.

    serial_request.cpp includes the EOS ID in generated_tokens; strix-tokenize decode
    preserves control tokens. raw_text keeps every emitted ID, including both controls.
    """
    ids = receipt["generated_tokens"]
    terminal = None
    if receipt.get("stop_reason") == "eos":
        if receipt.get("eos_token_id") != NATIVE_EOS_ID or ids[-1:] != [NATIVE_EOS_ID]:
            raise ValueError("invalid native terminal EOS receipt")
        terminal = ids[-1]
    raw = tokenizer.decode(ids)
    content = ids[:-1] if terminal is not None else ids
    # The assistant turn ends at <|im_end|>; the model may emit "\n" before the
    # protocol EOS. The answer is the content before the first end-of-turn token.
    end_of_turn = content.index(IM_END_ID) if IM_END_ID in content else None
    if end_of_turn is not None:
        content = content[:end_of_turn]
    return {"raw_text": raw, "text": tokenizer.decode(content), "terminal_eos_id": terminal,
            "end_of_turn_index": end_of_turn}


def score_case(case, receipt=None, output=None, manifest_sha256=None):
    status = "PLANNED" if receipt is None else "INCOMPLETE"
    if (receipt is not None and receipt.get("status") == "COMPLETE" and output is not None
            and receipt.get("case_id") == case["case_id"]
            and receipt.get("input_ids_sha256") == case["ids_sha256"]
            and receipt.get("ground_truth_sha256") == case["ground_truth_sha256"]
            and receipt.get("generated_tokens")
            and (manifest_sha256 is None or receipt.get("manifest_sha256") == manifest_sha256)):
        status = "COMPLETE"
    result = exact(case["ground_truth"], output if status == "COMPLETE" else None)
    return {"case_id": case["case_id"], "family": case["family"], "depth": case["depth"],
            "position": case["position"], "status": status, **result}


def heatmap(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["family"], row["depth"], row["position"]].append(row)
    lines = ["family | depth | position | complete/planned | exact | Wilson95 | per-value EM | all-values EM"]
    for (family, depth, position), cells in sorted(buckets.items()):
        done = [r for r in cells if r["status"] == "COMPLETE"]
        wins = sum(r["exact"] for r in done)
        values = [v for r in done for v in r["per_value"]]
        interval = wilson(wins, len(done))
        ci = "NA" if interval is None else f"{interval[0]:.4f}..{interval[1]:.4f}"
        value_em = f"{sum(values)}/{len(values)}" if values else "NA"
        all_em = f"{sum(r['all_values'] for r in done)}/{len(done)}" if done else "NA"
        lines.append(f"{family} | {depth} | {position:.6f} | {len(done)}/{len(cells)} | {wins} | {ci} | {value_em} | {all_em}")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("decode", help="CPU decode complete receipts, after the GPU turn")
    d.add_argument("--tokenizer", required=True)
    d.add_argument("--model", required=True)
    d.add_argument("--model-receipt", required=True)
    d.add_argument("receipts", nargs="+")
    s = sub.add_parser("score")
    s.add_argument("--receipts", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("manifests", nargs="+")
    sub.add_parser("self-test")
    a = p.parse_args()
    if not mutation_self_test():
        raise RuntimeError("scorer mutation self-test failed")
    if a.command == "self-test":
        print("mutation self-test: PASS (swapped values rejected)")
    elif a.command == "decode":
        tokenizer = StrixTokenize(a.tokenizer, a.model, a.model_receipt)
        for path in map(Path, a.receipts):
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if receipt.get("status") != "COMPLETE":
                continue
            if receipt["provenance"]["model_receipt_sha256"] != file_sha(a.model_receipt):
                raise ValueError("decode model receipt mismatch")
            if receipt.get("tokenizer") != tokenizer.identity:
                raise ValueError("decode tokenizer identity mismatch")
            decoded = decode_receipt(tokenizer, receipt)
            # Bind decoded text to the exact raw receipt; never mutate that receipt.
            atomic(str(path) + ".decoded.json", canonical({"receipt_sha256": file_sha(path),
                   **decoded, "tokenizer": tokenizer.identity}))
    else:
        rows = []
        for path in map(Path, a.manifests):
            case = json.loads(path.read_text(encoding="utf-8"))
            receipt_path = Path(a.receipts) / (case["case_id"] + ".receipt.json")
            decoded_path = Path(str(receipt_path) + ".decoded.json")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else None
            output = None
            if decoded_path.exists() and receipt_path.exists():
                decoded = json.loads(decoded_path.read_text(encoding="utf-8"))
                if (decoded.get("receipt_sha256") == file_sha(receipt_path)
                        and decoded.get("tokenizer") == case["tokenizer"]):
                    output = decoded["text"]
            rows.append(score_case(case, receipt, output, file_sha(path)))
        atomic(a.out, canonical({"mutation_self_test": True, "cells": rows}))
        print(heatmap(rows), end="")


if __name__ == "__main__":
    main()
