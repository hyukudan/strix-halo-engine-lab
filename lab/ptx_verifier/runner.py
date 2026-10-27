"""Optional offline PTX checks. Acceptance is evidence under a model, not qualification.

The caller supplies both PTX and a hash-pinned local executable. This module
never downloads tools, generates kernels, launches CUDA, or changes serving.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time


DEFAULT_MANIFEST = Path(__file__).with_name("cases.json")
VERIFIER_REVISION = "0c7e372f27e5599ee3ad28215f487115b1883585"
VERIFIER_REPOSITORY = "https://github.com/francois141/ptx_verifier"
MAX_TEXT_BYTES = 16 * 1024 * 1024
MAX_LOG_BYTES = 1024 * 1024
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
KERNEL = re.compile(r"[A-Za-z_$][A-Za-z0-9_$.]*\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SHA1 = re.compile(r"[0-9a-f]{40}\Z")
VERDICT = re.compile(r"^VERDICT: (ACCEPT|REJECT|UNKNOWN)$", re.MULTILINE)
COVERAGE = re.compile(
    r"^VC check: [0-9.]+s \(decision procedure [0-9.]+s\)"
    r"\s+elements: ([0-9]+)(?:/([0-9]+))?\s*$", re.MULTILINE
)


def _keys(value, expected, what):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"invalid {what} fields")


def _integer(value, low, high, what):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"invalid {what}")
    return value


def _identifier(value, what):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid {what}")
    return value


def _relative(value, what):
    if (not isinstance(value, str) or not value or "\\" in value
            or Path(value).is_absolute() or ".." in Path(value).parts):
        raise ValueError(f"invalid relative {what}")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_text(path, maximum=MAX_TEXT_BYTES):
    with Path(path).open("rb") as stream:
        data = stream.read(maximum + 1)
    if not data or len(data) > maximum:
        raise ValueError("empty or oversized input file")
    text = data.decode("utf-8")
    if "\0" in text:
        raise ValueError("NUL in text input")
    return data


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path=DEFAULT_MANIFEST):
    manifest = json.loads(read_text(path, 256 * 1024), object_pairs_hook=_unique_object)
    _keys(manifest, {"schema", "verifier", "cases"}, "manifest")
    if manifest["schema"] != "strix/ptx-verifier-cases-v1":
        raise ValueError("unsupported manifest schema")
    if manifest["verifier"] != {"repository": VERIFIER_REPOSITORY,
                                "revision": VERIFIER_REVISION, "license": "MIT"}:
        raise ValueError("unsupported verifier contract revision")
    cases = manifest["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 32:
        raise ValueError("cases must be a bounded nonempty list")
    names = set()
    for case in cases:
        _keys(case, {"id", "kernel", "source_path", "source_kernel", "source_revision",
                     "spec", "block", "grid", "arrays", "params", "dims",
                     "expected_elements"}, "case")
        name = _identifier(case["id"], "case id")
        if name in names:
            raise ValueError("duplicate case id")
        names.add(name)
        for key in ("kernel", "source_kernel"):
            _identifier(case[key], key)
        if not isinstance(case["source_revision"], str) or not SHA1.fullmatch(case["source_revision"]):
            raise ValueError("source revision must be an immutable commit")
        _relative(case["source_path"], "source path")
        _relative(case["spec"], "spec path")
        if case["grid"] != [1, 1, 1] or any(type(x) is not int for x in case["grid"]):
            raise ValueError("this pilot supports exactly one CTA")
        if not isinstance(case["block"], list) or len(case["block"]) != 3:
            raise ValueError("invalid block dimensions")
        threads = math.prod(_integer(x, 1, 1024, "block dimension") for x in case["block"])
        if threads > 1024:
            raise ValueError("block exceeds 1024 threads")
        arrays = case["arrays"]
        if not isinstance(arrays, list) or not 1 <= len(arrays) <= 16:
            raise ValueError("invalid arrays")
        array_names, ranges, outputs = set(), [], 0
        for array in arrays:
            _keys(array, {"name", "base", "elem_width", "length", "kind"}, "array")
            array_name = _identifier(array["name"], "array name")
            if array_name in array_names:
                raise ValueError("duplicate array name")
            array_names.add(array_name)
            width = _integer(array["elem_width"], 1, 8, "element width")
            if width not in (1, 2, 4, 8):
                raise ValueError("unsupported element width")
            length = _integer(array["length"], 1, 1000000, "array length")
            base = _integer(array["base"], 1, 2**63 - 1, "array base")
            end = base + width * length
            if base % width or end > 2**63 or any(base < hi and lo < end for lo, hi in ranges):
                raise ValueError("unaligned, overlapping or overflowing arrays")
            ranges.append((base, end))
            if array["kind"] not in ("in", "out", "inout"):
                raise ValueError("unsupported array kind")
            if array["kind"] in ("out", "inout"):
                outputs += length
        if _integer(case["expected_elements"], 1, 1000000, "expected elements") != outputs:
            raise ValueError("expected elements must cover every declared output")
        if not isinstance(case["params"], list) or not 1 <= len(case["params"]) <= 32:
            raise ValueError("invalid parameters")
        for param in case["params"]:
            _keys(param, {"kind", "value"}, "parameter")
            if param["kind"] == "ptr":
                if not isinstance(param["value"], str) or param["value"] not in array_names:
                    raise ValueError("unknown pointer array")
            elif param["kind"] == "int":
                _integer(param["value"], -(2**63), 2**63 - 1, "integer parameter")
            else:
                raise ValueError("unsupported parameter kind")
        if not isinstance(case["dims"], dict) or not 1 <= len(case["dims"]) <= 16:
            raise ValueError("invalid specification dimensions")
        for key, value in case["dims"].items():
            _identifier(key, "dimension name")
            _integer(value, 1, 1000000, "dimension value")
    return manifest


def parse_verdict(stdout, returncode, expected_elements):
    """Read the aggregate verdict, never the earlier EQUIVALENT message."""
    _integer(expected_elements, 1, 1000000, "expected elements")
    result = {"outcome": "ERROR", "tool_verdict": None, "return_code": returncode,
              "elements_checked": None, "elements_total": None,
              "expected_elements": expected_elements, "accepted_under_model": False,
              "reason": "missing, malformed or ambiguous aggregate verdict"}
    verdict_lines = [line for line in stdout.splitlines() if line.startswith("VERDICT:")]
    verdicts = VERDICT.findall(stdout)
    if len(verdict_lines) != 1 or len(verdicts) != 1:
        return result
    verdict = result["tool_verdict"] = verdicts[0]
    coverages = COVERAGE.findall(stdout)
    if len(coverages) == 1:
        checked, total = coverages[0]
        result["elements_checked"] = int(checked)
        result["elements_total"] = int(total or checked)
    if type(returncode) is not int or returncode != {"ACCEPT": 0, "REJECT": 1, "UNKNOWN": 2}[verdict]:
        result["reason"] = "aggregate verdict and process exit code disagree"
        return result
    if verdict != "ACCEPT":
        result.update(outcome=verdict, reason="tool rejected or could not establish the full contract")
        return result
    if (len(coverages) != 1 or result["elements_checked"] != expected_elements
            or result["elements_total"] != expected_elements):
        result["reason"] = "ACCEPT lacks complete coverage of the declared outputs"
        return result
    result.update(outcome="ACCEPT", accepted_under_model=True,
                  reason="aggregate ACCEPT with full declared coverage under the tool's model")
    return result


def build_command(case, ptx_path, spec_path, verifier, kernel=None):
    symbol = kernel if kernel is not None else case["kernel"]
    if not isinstance(symbol, str) or not KERNEL.fullmatch(symbol):
        raise ValueError("invalid PTX entry symbol")
    command = [str(verifier), "verify", str(ptx_path), str(spec_path), "-k", symbol,
               "-b", ",".join(map(str, case["block"])),
               "-g", ",".join(map(str, case["grid"])),
               "--dyn-shared", "0", "--no-log-file", "--no-profile"]
    for array in case["arrays"]:
        command += ["--array", f"{array['name']}:{array['base']}:{array['elem_width']}:"
                    f"{array['length']}:{array['kind']}"]
    for param in case["params"]:
        command += ["--param", f"{param['kind']}:{param['value']}"]
    for name, value in sorted(case["dims"].items()):
        command += ["--dim", f"{name}={value}"]
    return command + ["--sample", "0", "--verify-numeric"]


def _execute(command, timeout_seconds):
    """Bound execution time and retained logs; no shell or GPU launcher."""
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        start = time.monotonic()
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                   stdin=subprocess.DEVNULL, start_new_session=(os.name == "posix"))
        timed_out = False
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
        logs, truncated = [], False
        for stream in (stdout, stderr):
            stream.seek(0)
            raw = stream.read(MAX_LOG_BYTES + 1)
            truncated |= len(raw) > MAX_LOG_BYTES
            logs.append(raw[:MAX_LOG_BYTES].decode("utf-8", errors="replace"))
        return process.returncode, logs, timed_out, truncated, time.monotonic() - start


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("list", help="list fixed pilot contracts")
    for action in ("plan", "run"):
        command = commands.add_parser(action)
        command.add_argument("--case", required=True)
        command.add_argument("--ptx", required=True, type=Path)
        command.add_argument("--kernel", help="entry symbol in the actual PTX, including mangling")
        if action == "run":
            command.add_argument("--verifier", required=True, type=Path)
            command.add_argument("--verifier-sha256", required=True)
            command.add_argument("--output", required=True, type=Path)
            command.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    receipt = {"schema": "strix/ptx-verifier-receipt-v1", "status": "ERROR",
               "accepted_under_model": False, "native_qualified": False,
               "model_qualified": False, "bitwise_verified": False,
               "created_utc": datetime.now(timezone.utc).isoformat()}
    try:
        manifest = load_manifest(args.manifest)
        if args.action == "list":
            print(json.dumps({"schema": manifest["schema"], "verifier": manifest["verifier"],
                              "cases": manifest["cases"]}, indent=2))
            return 0
        cases = [case for case in manifest["cases"] if case["id"] == args.case]
        if len(cases) != 1:
            raise ValueError("unknown case")
        case = cases[0]
        manifest_root = args.manifest.resolve().parent
        spec_path = (manifest_root / case["spec"]).resolve()
        if not spec_path.is_relative_to(manifest_root):
            raise ValueError("spec escapes manifest directory")
        ptx_path = args.ptx.resolve()
        ptx, spec = read_text(ptx_path), read_text(spec_path)
        receipt.update(case=case,
                       case_sha256=hashlib.sha256(json.dumps(case, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                       runner_sha256=file_hash(__file__), verifier_contract=manifest["verifier"],
                       manifest_sha256=file_hash(args.manifest),
                       ptx={"path": str(ptx_path), "sha256": hashlib.sha256(ptx).hexdigest()},
                       spec={"path": str(spec_path), "sha256": hashlib.sha256(spec).hexdigest()},
                       ptx_matches_engine_source="not_attested",
                       verifier_binary_matches_source="not_attested")
        if args.action == "plan":
            receipt.update(status="NOT_RUN", command=build_command(case, ptx_path, spec_path,
                                                                   "<local-verifier>", args.kernel))
            print(json.dumps(receipt, indent=2))
            return 0
        if not math.isfinite(args.timeout_seconds) or not 0.1 <= args.timeout_seconds <= 600:
            raise ValueError("timeout must be between 0.1 and 600 seconds")
        if not SHA256.fullmatch(args.verifier_sha256):
            raise ValueError("expected verifier hash must be a lowercase SHA-256")
        verifier_path = args.verifier.resolve(strict=True)
        if not verifier_path.is_file() or not os.access(verifier_path, os.X_OK):
            raise ValueError("verifier is not an executable file")
        binary_hash = file_hash(verifier_path)
        if binary_hash != args.verifier_sha256:
            raise ValueError("verifier SHA-256 does not match the supplied pin")
        if args.output.exists() or not args.output.parent.is_dir():
            raise ValueError("receipt path must be new and its parent directory must exist")
        output_path = args.output.resolve()
        if output_path in (ptx_path, spec_path, args.manifest.resolve(), verifier_path):
            raise ValueError("receipt must not replace an input")
        command = build_command(case, ptx_path, spec_path, str(verifier_path), args.kernel)
        receipt.update(command=command, verifier_binary_sha256=binary_hash)
        returncode, logs, timed_out, truncated, elapsed = _execute(command, args.timeout_seconds)
        decision = parse_verdict(logs[0], returncode, case["expected_elements"])
        if timed_out or truncated:
            decision.update(outcome="TIMEOUT" if timed_out else "ERROR", accepted_under_model=False,
                            reason="verifier timed out" if timed_out else "verifier output exceeded log limit")
        # The file identities must describe what was supplied for the entire run.
        if (file_hash(verifier_path) != binary_hash or file_hash(ptx_path) != receipt["ptx"]["sha256"]
                or file_hash(spec_path) != receipt["spec"]["sha256"]):
            decision.update(outcome="ERROR", accepted_under_model=False, reason="an input changed during execution")
        receipt.update(status=decision["outcome"], accepted_under_model=decision["accepted_under_model"],
                       decision=decision, elapsed_seconds=elapsed, stdout=logs[0], stderr=logs[1],
                       timed_out=timed_out, logs_truncated=truncated)
        with output_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
        print(json.dumps(receipt, indent=2))
        return {"ACCEPT": 0, "REJECT": 1, "UNKNOWN": 2}.get(receipt["status"], 3)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        receipt.update(status="ERROR", accepted_under_model=False, error=str(exc))
        print(json.dumps(receipt, indent=2))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
