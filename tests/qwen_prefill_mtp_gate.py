#!/usr/bin/env python3
"""Native MTP full-logits/seven-component gate and cross-build payload comparison.

The command file is a pinned JSON argv array using strix-hip-mtp-model-qualification,
with model, token IDs and all required MTP assets/budgets. No qualification flags.
Capture public main first; compare OFF and ordinary-profile ON against that output.
Fast HC ON is explicitly excluded from MTP admission and must fail at startup.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess


def signature(value):
    if (value.get("schema") != "mtp-prefix-gate-v1" or not value.get("passed")
            or not value.get("full_logits_bits_equal")
            or not value.get("all_seven_committed_state_components_equal")):
        raise ValueError("NO-REFERENCE: incomplete MTP gate")
    keys = ("initial_position", "numerical_profile", "target_content", "state_scope",
            "physical_state_scope", "initial_committed_state", "initial_physical_state", "cases", "case_count")
    result = {key: value[key] for key in keys}
    if len(result["initial_committed_state"]) != 7 or not result["cases"]:
        raise ValueError("NO-REFERENCE: requires seven components and window cases")
    for case in result["cases"]:
        if (len(case["committed_state"]) != 7 or len(case["full_physical_state"]) != 7
                or len(case["all_physical_logits"]) != case["rows"]):
            raise ValueError("NO-REFERENCE: incomplete logits/state case")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("on", "off", "public-main"))
    parser.add_argument("--reference", type=Path, help="prior output/native/result.json")
    args = parser.parse_args()
    command = json.loads(args.command.read_text())
    if (not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command)
            or any(x.startswith(("--qualification-", "--prefill-disable", "--prefill-exact-fp16", "--config")) for x in command)):
        parser.error("requires pinned base argv without qualification or prefill control flags")
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("outputs must stay outside the checkout")
    env = os.environ.copy()
    for key in ("STRIX_PREFILL_HC_OUTPUT_WMMA", "STRIX_PREFILL_EXACT_FP16"):
        env.pop(key, None)
    if args.arm == "off":
        env["STRIX_PREFILL_HC_OUTPUT_WMMA"] = "0"
    env["STRIX_HCW_OUTPUT"] = "1"
    args.output.mkdir(parents=False, exist_ok=False)
    with (args.output / "receipt.json").open("wb") as out, (args.output / "stderr.log").open("wb") as err:
        subprocess.run(command + ["--qualification-prefix-gate", str(args.output.resolve() / "native")],
                       env=env, stdout=out, stderr=err, check=True)
    actual = signature(json.loads((args.output / "native" / "result.json").read_text()))
    if args.reference:
        if actual != signature(json.loads(args.reference.read_text())):
            raise ValueError("MISMATCH: MTP numerical payload differs from reference")
        print("EXACT: MTP IDs, full logits and all seven committed/physical state components")
    else:
        print("CAPTURED: native serial/window gate passed; no cross-build verdict")


if __name__ == "__main__":
    main()
