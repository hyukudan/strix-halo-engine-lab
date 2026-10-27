"""Run real, CPU-only verifier smoke checks on handwritten synthetic PTX.

The caller supplies an executable and its SHA-256. No tool is installed and no
CUDA code is compiled or launched. These fixtures are not engine compiler
output, and their results cannot qualify engine kernels or floating-point bits.
All generated fixtures, logs and receipts stay in a new directory outside the
checkout. Every numeric mutation must be refuted; execution errors do not pass.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


DIRECTORY = Path(__file__).resolve().parent
CHECKOUT = DIRECTORY.parents[1]
QUALIFICATION_FLAGS = ("native_qualified", "model_qualified", "bitwise_verified")


@dataclass(frozen=True)
class Check:
    name: str
    case: str
    fixture: str
    kernel: str
    expected: str
    before: str = ""
    after: str = ""


CHECKS = (
    Check("carry_row0", "carry_even_row0", "synthetic_carry.ptx", "pilot_carry_fixture", "ACCEPT"),
    Check("carry_row1", "carry_even_row1", "synthetic_carry.ptx", "pilot_carry_fixture", "ACCEPT"),
    Check("hc_h32", "hc_expand_rows1_h32", "synthetic_hc.ptx", "pilot_hc_fixture", "ACCEPT"),
    Check("carry_wrong_valid", "carry_even_row0", "synthetic_carry.ptx", "pilot_carry_fixture",
          "REFUTED", "xor.b32 %valid_value, %parity, 1;", "mov.u32 %valid_value, 0;"),
    Check("carry_wrong_row", "carry_even_row1", "synthetic_carry.ptx", "pilot_carry_fixture",
          "REFUTED", "mul.wide.u32 %row_offset, %row, 4096;", "mov.u64 %row_offset, 0;"),
    Check("hc_wrong_coefficient", "hc_expand_rows1_h32", "synthetic_hc.ptx", "pilot_hc_fixture",
          "REFUTED", "add.u64 %coefficient_ptr, %mix_ptr, 32;", "add.u64 %coefficient_ptr, %mix_ptr, 28;"),
    Check("carry_missing_valid", "carry_even_row0", "synthetic_carry.ptx", "pilot_carry_fixture",
          "MISSING_OUTPUT", "st.global.u32 [%valid_ptr], %valid_value;", "// mutation: omit valid[0] store"),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(check, receipt, runner_code):
    """Require the intended proof/refutation, never an unrelated tool failure."""
    if any(receipt.get(flag) is not False for flag in QUALIFICATION_FLAGS):
        return False, "receipt must keep every qualification flag false"
    decision = receipt.get("decision", {})
    status = receipt.get("status")
    stdout = receipt.get("stdout", "")
    tool_verdict = decision.get("tool_verdict")
    tool_code = decision.get("return_code")
    accepted = receipt.get("accepted_under_model")
    if check.expected == "ACCEPT":
        expected = receipt.get("case", {}).get("expected_elements")
        passed = (runner_code == 0 and status == "ACCEPT" and tool_verdict == "ACCEPT"
                  and tool_code == 0 and accepted is True
                  and type(expected) is int and expected > 0
                  and decision.get("elements_checked") == expected
                  and decision.get("elements_total") == expected)
        return passed, "requires aggregate ACCEPT and complete declared coverage"
    refuted = (runner_code == 1 and status == "REJECT" and tool_verdict == "REJECT"
               and tool_code == 1 and accepted is False)
    if check.expected == "REFUTED":
        passed = (refuted and "NOT EQUIVALENT:" in stdout
                  and re.search(r"^  equivalence: [1-9][0-9]* element\(s\) refuted by a numeric counterexample",
                                stdout, re.MULTILINE) is not None)
        return passed, "requires an actual numeric refutation and aggregate REJECT"
    if check.expected == "MISSING_OUTPUT":
        witness = ("  equivalence: valid does not have the same set of indices "
                   "(1 never written, first valid[0], 0 written outside it)")
        passed = refuted and witness in stdout.splitlines()
        return passed, "requires verifier REJECT identifying the missing valid[0] output"
    return False, "unknown smoke expectation"


def run_check(check, verifier, verifier_hash, output_dir, timeout):
    fixture = DIRECTORY / "fixtures" / check.fixture
    source = fixture.read_text(encoding="utf-8")
    if check.before:
        if source.count(check.before) != 1:
            raise ValueError(f"{check.name}: mutation anchor must occur exactly once")
        source = source.replace(check.before, check.after, 1)
    input_path = output_dir / f"{check.name}.ptx"
    with input_path.open("x", encoding="utf-8") as stream:
        stream.write(source)
    receipt_path = output_dir / f"{check.name}.receipt.json"
    command = [sys.executable, str(DIRECTORY / "runner.py"), "run", "--case", check.case,
               "--ptx", str(input_path), "--kernel", check.kernel,
               "--verifier", str(verifier), "--verifier-sha256", verifier_hash,
               "--output", str(receipt_path), "--timeout-seconds", str(timeout)]
    process = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                             text=True, encoding="utf-8", errors="replace", cwd=output_dir,
                             check=False)
    # Keep even prelaunch or wrapper failures reviewable; these never count as passes.
    (output_dir / f"{check.name}.runner.stdout").write_text(process.stdout, encoding="utf-8")
    (output_dir / f"{check.name}.runner.stderr").write_text(process.stderr, encoding="utf-8")
    result = {"name": check.name, "case": check.case, "expected": check.expected,
              "fixture_only": True, "fixture": check.fixture,
              "fixture_sha256": sha256(fixture), "input_sha256": sha256(input_path),
              "mutation": {"before": check.before, "after": check.after} if check.before else None,
              "command": command, "runner_return_code": process.returncode,
              "receipt": str(receipt_path), "passed": False}
    if not receipt_path.is_file():
        result["reason"] = "runner did not produce a receipt"
        return result
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        result["passed"], result["reason"] = evaluate(check, receipt, process.returncode)
        result.update(status=receipt.get("status"), decision=receipt.get("decision"),
                      accepted_under_model=receipt.get("accepted_under_model"))
    except (ValueError, TypeError, AttributeError) as exc:
        result["reason"] = f"unreadable receipt: {exc}"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--verifier-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory outside this checkout; parent must exist")
    parser.add_argument("--timeout-seconds", type=float, default=60.0,
                        help="per verifier invocation, between 0.1 and 600")
    args = parser.parse_args(argv)
    report = {"schema": "strix/ptx-verifier-integration-v1", "status": "ERROR",
              "fixture_only": True, "ptx_matches_engine_source": False,
              "verifier_binary_matches_source": "not_attested",
              "native_qualified": False, "model_qualified": False, "bitwise_verified": False,
              "created_utc": datetime.now(timezone.utc).isoformat(), "checks": []}
    output_dir = None
    try:
        if not math.isfinite(args.timeout_seconds) or not 0.1 <= args.timeout_seconds <= 600:
            raise ValueError("timeout must be between 0.1 and 600 seconds")
        if not re.fullmatch(r"[0-9a-f]{64}", args.verifier_sha256):
            raise ValueError("verifier SHA-256 must be 64 lowercase hexadecimal characters")
        verifier = args.verifier.resolve(strict=True)
        if not verifier.is_file() or not os.access(verifier, os.X_OK):
            raise ValueError("verifier is not an executable file")
        if sha256(verifier) != args.verifier_sha256:
            raise ValueError("verifier SHA-256 does not match the supplied pin")
        destination = args.output_dir.resolve()
        if destination.is_relative_to(CHECKOUT):
            raise ValueError("generated smoke artifacts must be outside the checkout")
        if destination.exists() or not destination.parent.is_dir():
            raise ValueError("output directory must be new and its parent must exist")
        destination.mkdir()
        output_dir = destination
        report.update(verifier_binary_sha256=args.verifier_sha256,
                      integration_sha256=sha256(__file__))
        for check in CHECKS:
            report["checks"].append(run_check(check, verifier, args.verifier_sha256,
                                              output_dir, args.timeout_seconds))
        report["status"] = "PASS" if all(check["passed"] for check in report["checks"]) else "FAIL"
    except (OSError, ValueError, TypeError) as exc:
        report["error"] = str(exc)
    if output_dir is not None:
        with (output_dir / "summary.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    print(json.dumps(report, indent=2))
    return {"PASS": 0, "FAIL": 1}.get(report["status"], 3)


if __name__ == "__main__":
    raise SystemExit(main())
