"""Reproduce two module-compatibility limits using independent handwritten PTX.

The baseline must first be accepted under the verifier's model. Each probe then
requires its exact pre-verdict diagnostic. LIMITATIONS_REPRODUCED means the
incompatibilities were observed; it never means an engine is qualified. This
does not edit engine PTX, increase dynamic shared memory, or install a tool.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re


DIRECTORY = Path(__file__).resolve().parent
CHECKOUT = DIRECTORY.parents[1]
_spec = importlib.util.spec_from_file_location("_compatibility_runner", DIRECTORY / "runner.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)
PROBES = (
    ("unselected_match", "compatibility_unselected_match.ptx",
     "Unexpected identifier at statement level: match.any.sync.b32"),
    ("unused_shared", "compatibility_unused_shared.ptx",
     "error: kernel kernel: configuration error: kernel uses extern shared memory; set dynamic_shared_bytes"),
)


def limitation_reproduced(receipt, diagnostic):
    decision = receipt.get("decision", {})
    return (receipt.get("status") == "ERROR"
            and receipt.get("accepted_under_model") is False
            and decision.get("return_code") == 1 and decision.get("tool_verdict") is None
            and receipt.get("timed_out") is False and receipt.get("logs_truncated") is False
            and receipt.get("input_identities_stable") is True
            and receipt.get("stdout", "").strip() == ""
            and diagnostic in receipt.get("stderr", "").splitlines()
            and all(receipt.get(flag) is False for flag in
                    ("native_qualified", "model_qualified", "bitwise_verified")))


def run_probe(name, filename, verifier, verifier_hash, output_dir, spec_path, timeout):
    ptx_path = output_dir / filename
    with ptx_path.open("xb") as stream:
        stream.write(runner.read_text(DIRECTORY / "fixtures" / filename))
    identities = {"verifier": runner.file_hash(verifier), "ptx": runner.file_hash(ptx_path),
                  "spec": runner.file_hash(spec_path)}
    if identities["verifier"] != verifier_hash:
        raise ValueError("verifier SHA-256 changed before invocation")
    command = [str(verifier), "verify", str(ptx_path), str(spec_path), "-k", "pilot_selected_entry",
               "-b", "1,1,1", "-g", "1,1,1", "--dyn-shared", "0", "--no-log-file", "--no-profile",
               "--array", "out:65536:4:1:out", "--param", "ptr:out", "--dim", "ONE=1",
               "--sample", "0", "--verify-numeric"]
    returncode, logs, timed_out, truncated, elapsed = runner._execute(command, timeout)
    stable = identities == {"verifier": runner.file_hash(verifier), "ptx": runner.file_hash(ptx_path),
                            "spec": runner.file_hash(spec_path)}
    decision = runner.parse_verdict(logs[0], returncode, 1)
    if timed_out or truncated or not stable:
        decision.update(outcome="ERROR", accepted_under_model=False,
                        reason="timeout, truncated log or changed input invalidates compatibility evidence")
    receipt = {"schema": "strix/ptx-verifier-compatibility-receipt-v1", "probe": name,
               "fixture_only": True, "ptx_matches_engine_source": False,
               "verifier_binary_matches_source": "not_attested",
               "native_qualified": False, "model_qualified": False, "bitwise_verified": False,
               "status": decision["outcome"], "accepted_under_model": decision["accepted_under_model"],
               "decision": decision, "command": command, "sha256": identities,
               "elapsed_seconds": elapsed, "stdout": logs[0], "stderr": logs[1],
               "timed_out": timed_out, "logs_truncated": truncated, "input_identities_stable": stable}
    with (output_dir / f"{name}.receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier", required=True, type=Path)
    parser.add_argument("--verifier-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="new directory outside this checkout; parent must exist")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    report = {"schema": "strix/ptx-verifier-compatibility-v1", "status": "ERROR",
              "fixture_only": True, "ptx_matches_engine_source": False, "accepted_under_model": False,
              "native_qualified": False, "model_qualified": False, "bitwise_verified": False,
              "created_utc": datetime.now(timezone.utc).isoformat(), "probes": []}
    output_dir = None
    try:
        if not math.isfinite(args.timeout_seconds) or not 0.1 <= args.timeout_seconds <= 600:
            raise ValueError("timeout must be between 0.1 and 600 seconds")
        if not re.fullmatch(r"[0-9a-f]{64}", args.verifier_sha256):
            raise ValueError("verifier SHA-256 must be 64 lowercase hexadecimal characters")
        verifier = args.verifier.resolve(strict=True)
        if not verifier.is_file() or not os.access(verifier, os.X_OK):
            raise ValueError("verifier is not an executable file")
        if runner.file_hash(verifier) != args.verifier_sha256:
            raise ValueError("verifier SHA-256 does not match the supplied pin")
        destination = args.output_dir.resolve()
        if destination.is_relative_to(CHECKOUT):
            raise ValueError("compatibility artifacts must stay outside the checkout")
        if destination.exists() or not destination.parent.is_dir():
            raise ValueError("output directory must be new and its parent must exist")
        destination.mkdir()
        output_dir = destination
        spec_path = output_dir / "compatibility.spec"
        with spec_path.open("xb") as stream:
            stream.write(runner.read_text(DIRECTORY / "fixtures" / "compatibility.spec"))
        report.update(verifier_binary_sha256=args.verifier_sha256,
                      compatibility_sha256=runner.file_hash(__file__), runner_sha256=runner.file_hash(runner.__file__))
        baseline = run_probe("baseline", "compatibility_baseline.ptx", verifier, args.verifier_sha256,
                             output_dir, spec_path, args.timeout_seconds)
        report["baseline_receipt"] = str(output_dir / "baseline.receipt.json")
        report["baseline_accepted_under_model"] = baseline["accepted_under_model"]
        if not baseline["accepted_under_model"]:
            report["status"] = "BASELINE_FAILED"
        else:
            for name, filename, diagnostic in PROBES:
                receipt = run_probe(name, filename, verifier, args.verifier_sha256,
                                    output_dir, spec_path, args.timeout_seconds)
                report["probes"].append({"name": name, "status": receipt["status"],
                                         "limitation_reproduced": limitation_reproduced(receipt, diagnostic),
                                         "expected_diagnostic": diagnostic,
                                         "decision": receipt["decision"],
                                         "receipt": str(output_dir / f"{name}.receipt.json")})
            report["status"] = ("LIMITATIONS_REPRODUCED" if all(p["limitation_reproduced"] for p in report["probes"])
                                else "NOT_REPRODUCED")
    except (OSError, ValueError, TypeError) as exc:
        report["error"] = str(exc)
    if output_dir is not None:
        with (output_dir / "summary.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    print(json.dumps(report, indent=2))
    return {"LIMITATIONS_REPRODUCED": 0, "BASELINE_FAILED": 1, "NOT_REPRODUCED": 1}.get(report["status"], 3)


if __name__ == "__main__":
    raise SystemExit(main())
