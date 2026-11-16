#!/usr/bin/env python3
"""Run a test-only capture adapter, retaining provenance and its raw records."""
import argparse
import hashlib
import json
import pathlib
import platform
import shlex
import subprocess
import sys
from records import read_matrix, validate_inventory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=pathlib.Path)
    parser.add_argument("--matrix", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--source", type=pathlib.Path)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--flags", default="")
    parser.add_argument("--device", default="host")
    parser.add_argument("--runner", default="")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if list(args.out.rglob("*.wfmt")):
        parser.error("output already contains records; use a fresh capture directory")
    summary = {"schema": "wfmt.capture@2", "pass": False,
               "compiler": args.compiler, "architecture": args.architecture,
               "flags": args.flags, "device": args.device, "platform": platform.platform()}
    try:
        exe = args.exe.resolve(strict=True)
        matrix = args.matrix.resolve(strict=True)
        contract = read_matrix(matrix)
        matrix_bytes = matrix.read_bytes()
        summary["executable_sha256"] = hashlib.sha256(exe.read_bytes()).hexdigest()
        summary["matrix_sha256"] = hashlib.sha256(matrix_bytes).hexdigest()
        if args.source:
            summary["source"] = subprocess.check_output(
                ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
        extra = args.args[1:] if args.args[:1] == ["--"] else args.args
        summary["adapter_args"] = extra
        command = shlex.split(args.runner) + [str(exe), "--wfmt-capture", str(args.out.resolve()),
                   "--wfmt-matrix", str(matrix)] + extra
        with (args.out / "capture.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=args.timeout, check=False)
        if result.returncode:
            raise ValueError(f"capture adapter exit {result.returncode}")
        (args.out / "matrix.json").write_bytes(matrix_bytes)
        records = validate_inventory(args.out, contract)
        summary.update({"pass": True, "records": len(records), "record_sha256": records})
    except (ValueError, OSError, KeyError, TypeError, AttributeError, subprocess.SubprocessError) as error:
        summary["error"] = str(error)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
