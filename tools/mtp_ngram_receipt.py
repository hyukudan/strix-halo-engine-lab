#!/usr/bin/env python3
"""Validate exact-token arms and summarize context-draft receipts stored outside Git.

Input JSON: {"cases": [{"name": "code1k", "reference": "base-serial.json",
  "arms": {"default": "candidate-default.json", "off": "candidate-off.json",
           "ctx": "candidate-ctx.json", "copy": "candidate-copy.json"}}]}
Paths are relative to the manifest. References must be outputs of the pinned base
serial greedy binary, with the same prompt, assets and arithmetic profile.
This tool does not execute GPU jobs or establish provenance from a filename.
"""
import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path


def receipt(path):
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    for pos, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text, pos)
        except ValueError:
            continue
        if isinstance(data, dict) and isinstance(data.get("generated_tokens"), list):
            tokens = data["generated_tokens"]
            if not tokens or any(type(t) is not int or t < 0 for t in tokens):
                raise ValueError(f"{path}: invalid generated tokens")
            return data
    raise ValueError(f"{path}: no native generation receipt")


def metrics(data):
    n = len(data["generated_tokens"])
    cycles = data.get("mtp_cycles")
    ms = data.get("mtp_generation_cycle_elapsed_ms")
    if type(cycles) is not int or cycles <= 0 or not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms <= 0:
        raise ValueError("missing/invalid completed MTP cycle timing")
    hist = data.get("mtp_chosen_window_rows_histogram", {})
    if sum(hist.values()) != cycles:
        raise ValueError("window histogram does not cover all cycles")
    ng = data.get("mtp_ngram_draft") or {}
    delivered = ng.get("extension_rows", 0)
    accepted = ng.get("accepted_rows", 0)
    selected = ng.get("selected_rows", 0)
    if any(type(x) is not int or x < 0 for x in (delivered, accepted, selected)) or not accepted <= delivered <= selected:
        raise ValueError("copy counters are not accepted <= delivered <= selected")
    # First output is selected during prompt processing, outside the cycle timer.
    emitted = n - 1
    if not emitted:
        raise ValueError("receipt contains no cycle output")
    return {"cycle_tokens": emitted, "cycles": cycles, "rows_selected": selected,
            "rows_delivered": delivered, "rows_accepted": accepted,
            "copy_first_cycles": ng.get("copy_first_cycles", 0),
            "copy_acceptance": accepted / delivered if delivered else None,
            "tokens_per_cycle": emitted / cycles,
            "short_run_ms_per_token": ms / emitted if emitted else None,
            "mean_cycle_ms": ms / cycles, "window_histogram": hist,
            "shadow_proposals": ng.get("proposal_cycles", 0),
            "shadow_scored": ng.get("shadow_scored_cycles", 0),
            "shadow_right": ng.get("shadow_right_rows", 0)}


def summarize(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    for case in manifest["cases"]:
        ref_path = manifest_path.parent / case["reference"]
        ref = receipt(ref_path)
        if "mtp_cycles" in ref or ref.get("mtp_draft_chain"):
            raise ValueError(f"{case['name']}: reference is MTP, not serial greedy")
        ref_tokens = ref["generated_tokens"]
        arms = {}
        for name, path in case["arms"].items():
            data = receipt(manifest_path.parent / path)
            if data["generated_tokens"] != ref_tokens:
                raise ValueError(f"{case['name']}/{name}: tokens differ from complete base reference")
            for key in ("mtp_replay_maximum_hidden_abs_error", "mtp_replay_maximum_logit_abs_error"):
                if key not in data or data[key] != 0:
                    raise ValueError(f"{case['name']}/{name}: missing/nonzero {key}")
            arms[name] = metrics(data)
            if name in ("off", "default") and data.get("mtp_ngram_draft") is not None:
                raise ValueError(f"{case['name']}/{name}: context proposals were enabled")
        # A deliberately limited schedule-only estimate, separate from the observed short run.
        # New windows change expert unions/cost; this estimate cannot certify throughput.
        if "off" in arms:
            for value in arms.values():
                value["expected_ms_per_token_fixed_off_cycle_cost"] = arms["off"]["mean_cycle_ms"] / value["tokens_per_cycle"]
        results.append({"case": case["name"], "exact_tokens": len(ref_tokens),
                        "reference_sha256": hashlib.sha256(ref_path.read_bytes()).hexdigest(), "arms": arms})
    return {"scope": "short acceptance gate, no sustained speedup qualification",
            "estimate_assumption": "fixed off-arm mean cycle cost; ignores changed owner/union/draft costs",
            "cases": results}


def self_test():
    data = {"generated_tokens": [10, 11, 12, 13, 14], "mtp_cycles": 2,
            "mtp_generation_cycle_elapsed_ms": 40, "mtp_chosen_window_rows_histogram": {"2": 2},
            "mtp_ngram_draft": {"selected_rows": 5, "extension_rows": 3, "accepted_rows": 2}}
    m = metrics(data)
    assert m["rows_delivered"] == 3 and m["rows_accepted"] == 2
    assert m["tokens_per_cycle"] == 2 and m["short_run_ms_per_token"] == 10
    bad = dict(data, mtp_ngram_draft={"selected_rows": 1, "extension_rows": 2})
    try:
        metrics(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("undelivered rows passed validation")
    # Verify complete arrays (including a differing last token), actual replay fields,
    # and serial-reference admission through the same public summarization path.
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        reference = {"generated_tokens": data["generated_tokens"]}
        arm = dict(data, mtp_replay_maximum_hidden_abs_error=0,
                   mtp_replay_maximum_logit_abs_error=0)
        manifest = {"cases": [{"name": "fixture", "reference": "ref.json", "arms": {"copy": "copy.json"}}]}
        for name, value in (("ref.json", reference), ("copy.json", arm), ("manifest.json", manifest)):
            (directory / name).write_text(json.dumps(value), encoding="utf-8")
        assert summarize(directory / "manifest.json")["cases"][0]["arms"]["copy"]["rows_accepted"] == 2
        mutations = [dict(arm, generated_tokens=[10, 11, 12, 13, 99]),
                     dict(arm, mtp_replay_maximum_logit_abs_error=0.01)]
        missing = dict(arm)
        del missing["mtp_replay_maximum_hidden_abs_error"]
        mutations.append(missing)
        for broken in mutations:
            (directory / "copy.json").write_text(json.dumps(broken), encoding="utf-8")
            try:
                summarize(directory / "manifest.json")
            except ValueError:
                pass
            else:
                raise AssertionError("invalid native receipt passed exactness gate")
    print("context receipt tests: PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.manifest:
        print(json.dumps(summarize(args.manifest), indent=2, allow_nan=False))
    else:
        parser.error("manifest or --self-test required")


if __name__ == "__main__":
    main()
