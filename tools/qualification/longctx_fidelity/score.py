"""Exact symbolic scoring with explicit reasoning and ordered JSON protocols.

Missing or unbound outputs are unmeasured; malformed measured answers are wrong.
"""
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
try:
    from .tasks import reasoning_protocol
    from .common import canonical, atomic, file_sha
    from .tok import StrixTokenize
    from .run_cli import NATIVE_EOS_ID
except ImportError:
    from tasks import reasoning_protocol
    from common import canonical, atomic, file_sha
    from tok import StrixTokenize
    from run_cli import NATIVE_EOS_ID

# <|im_end|> in the Qwen3.8-Flash-Next vocabulary (strix-tokenize encode); the protocol
# EOS that stops the plain CLI is <|endoftext|> = NATIVE_EOS_ID.
IM_END_ID = 248046


def normalize(text):
    text = text.strip()
    return text[:-1].rstrip() if text.endswith(".") else text


def extract_answer(text):
    """Only a literal line prefix declares an answer; the last such line wins."""
    lines = [line[len("Answer:"):] for line in text.splitlines() if line.startswith("Answer:")]
    if not lines:
        return None, False
    value = lines[-1].strip()
    outer_period = value.endswith(".")
    if outer_period:
        value = value[:-1].rstrip()
    # Permit a final period either outside or inside a quoted/backticked value.
    value = value.strip("`\"'").strip()
    return (value if outer_period else normalize(value)), True


def ordered_pairs(output):
    """Decode one complete outer array; never rescue an inner/truncated array."""
    if "```" in output:
        # Reasoning text before the answer may use inline `code`; only triple-backtick
        # fences delimit blocks. The answer is the last complete fenced block.
        if output.count("```") % 2:
            raise ValueError("invalid or truncated fence")
        fences = re.findall(r"```(?:json)?[ \t]*\n?(.*?)```", output, re.DOTALL)
        if not fences:
            raise ValueError("invalid or truncated fence")
        output = fences[-1].strip()
    # Find complete top-level arrays while skipping nested arrays and quoted brackets.
    # Balanced non-JSON bracketed prose can surround the one JSON answer.
    candidates, stack, quoted, escaped, start = [], [], False, False, None
    for i, char in enumerate(output):
        if not stack:
            if char in "[{":
                stack.append(char)
                start = i
            continue
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            opening = stack.pop()
            if (opening, char) not in (("[", "]"), ("{", "}")):
                raise ValueError("mismatched JSON delimiter")
            if not stack and opening == "[":
                candidates.append(output[start:i + 1])
    if stack:
        raise ValueError("truncated JSON")
    def unique_object(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate object field")
            result[key] = value
        return result
    arrays = []
    for candidate in candidates:
        try:
            arrays.append(json.loads(candidate, object_pairs_hook=unique_object))
        except json.JSONDecodeError:
            continue  # e.g. a bracketed prose label before the answer
    if len(arrays) != 1:
        raise ValueError("expected one complete JSON array")
    parsed = arrays[0]
    if all(isinstance(p, list) and len(p) == 2 for p in parsed):
        pairs = parsed
    elif all(isinstance(p, dict) and set(p) == {"key", "value"} for p in parsed):
        pairs = [[p["key"], p["value"]] for p in parsed]
    else:
        raise ValueError("not a homogeneous ordered pair/object array")
    if any(not all(isinstance(x, str) for x in p) for p in pairs):
        raise ValueError("not string keys/values")
    if len({p[0] for p in pairs}) != len(pairs):
        raise ValueError("duplicate keys")
    return pairs


def exact(answer, output, normalized=False):
    if output is None:
        return {"exact": False, "per_value": [], "all_values": False}
    output = output if normalized else normalize(output)
    if not isinstance(answer, list):
        okay = output == str(answer)
        return {"exact": okay, "per_value": [okay], "all_values": okay}
    try:
        parsed = ordered_pairs(output)
        values = dict(parsed)
        per_value = [values.get(k) == v for k, v in answer]
        okay = parsed == answer
        error = None if okay else "values_or_order"
    except (ValueError, TypeError):
        per_value, okay, error = [False] * len(answer), False, "invalid_json_or_shape"
    return {"exact": okay, "per_value": per_value, "all_values": okay,
            "answer_error": error}


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
    measured = output if status == "COMPLETE" else None
    extracted, found = None, False
    protocol = reasoning_protocol(case["family"], case.get("variant", "normal"))
    if protocol and measured is not None:
        extracted, found = extract_answer(measured)
        measured = extracted if found else ""
    result = exact(case["ground_truth"], measured, normalized=protocol)
    return {"case_id": case["case_id"], "family": case["family"], "depth": case["depth"],
            "position": case["position"], "seed": case.get("seed"),
            "layout": {"position": case["position"], "stratum": case.get("stratum", "prose"),
                       "variant": case.get("variant", "normal")},
            "short": case.get("short", False), "status": status,
            "extracted_answer": extracted, "answer_line_found": found, **result}


def paired_controls(rows):
    """Pair measured short/long cases by family, seed and complete task layout."""
    groups = defaultdict(list)
    for row in rows:
        layout = json.dumps(row["layout"], sort_keys=True)
        groups[row["family"], row["seed"], layout].append(row)
    pairs = []
    for (family, seed, layout), group in sorted(groups.items()):
        shorts = [r for r in group if r["short"] and r["status"] == "COMPLETE"]
        longs = [r for r in group if not r["short"] and r["status"] == "COMPLETE"]
        # Ambiguous controls must not silently select a baseline.
        if len(shorts) != 1:
            continue
        short = shorts[0]
        for long in sorted(longs, key=lambda r: (r["depth"], r["case_id"])):
            pairs.append({"family": family, "seed": seed, "layout": json.loads(layout),
                          "short_case_id": short["case_id"], "long_case_id": long["case_id"],
                          "short_depth": short["depth"], "long_depth": long["depth"],
                          "long_minus_short_exact": int(long["exact"]) - int(short["exact"]),
                          "long_minus_short_all_values": int(long["all_values"]) - int(short["all_values"]),
                          "long_minus_short_per_value": [int(a) - int(b) for a, b in
                              zip(long["per_value"], short["per_value"])]})
    return pairs


def heatmap(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["family"] + (".short" if row.get("short") else ""), row["depth"], row["position"]].append(row)
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
        atomic(a.out, canonical({"mutation_self_test": True, "cells": rows, "short_long_pairs": paired_controls(rows)}))
        print(heatmap(rows), end="")
        for pair in paired_controls(rows):
            print(f"{pair['family']} seed={pair['seed']} layout={pair['layout']} "
                  f"D={pair['long_depth']} long-minus-short EM={pair['long_minus_short_exact']}")


if __name__ == "__main__":
    main()
