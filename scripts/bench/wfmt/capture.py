#!/usr/bin/env python3
"""Run a test-only capture adapter, retaining provenance and its raw records."""
import argparse
import hashlib
import json
import pathlib
import platform
import re
import shlex
import subprocess
import sys
from records import read_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=pathlib.Path)
    parser.add_argument("--matrix", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=pathlib.Path)
    source.add_argument("--source-ref")
    parser.add_argument("--adapter", required=True, choices=("protocol", "reference", "native", "backend"))
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--flags", required=True)
    parser.add_argument("--device", default="host")
    parser.add_argument("--runner", default="")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if list(args.out.rglob("*.wfmt")):
        parser.error("output already contains records; use a fresh capture directory")
    summary = {"schema": "wfmt.capture@1", "pass": False,
               "compiler": args.compiler, "architecture": args.architecture,
               "flags": args.flags, "device": args.device, "platform": platform.platform(),
               "adapter": args.adapter, "fixture_only": args.adapter == "protocol"}
    try:
        exe = args.exe.resolve(strict=True)
        matrix = args.matrix.resolve(strict=True)
        summary["executable_sha256"] = hashlib.sha256(exe.read_bytes()).hexdigest()
        summary["matrix_sha256"] = hashlib.sha256(matrix.read_bytes()).hexdigest()
        if args.source:
            summary["source"] = subprocess.check_output(
                ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
        else:
            summary["source"] = args.source_ref
        if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", summary["source"]) or not args.flags:
            raise ValueError("source commit and exact compiler flags are mandatory")
        extra = args.args[1:] if args.args[:1] == ["--"] else args.args
        if args.adapter == "reference":
            adapter_args = ["--wfmt-reference-capture", str(args.out.resolve())]
        elif args.adapter == "backend":
            adapter_args = ["--wfmt-backend-capture", str(args.out.resolve())]
        else:
            adapter_args = ["--wfmt-capture", str(args.out.resolve()), "--wfmt-matrix", str(matrix)]
        command = shlex.split(args.runner) + [str(exe)] + adapter_args + extra
        with (args.out / "capture.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=args.timeout, check=False)
        if result.returncode:
            raise ValueError(f"capture adapter exit {result.returncode}")
        records = list(args.out.rglob("*.wfmt"))
        if not records:
            raise ValueError("capture adapter produced no byte records")
        for record in records:
            header, _ = read_record(record)
            if args.adapter != "protocol" and header["case"] == "protocol-fixture":
                raise ValueError("protocol fixture cannot qualify a numerical capture")
        for child in ("baseline", "candidate"):
            directory = args.out / child
            if directory.is_dir():
                arm = dict(summary)
                arm.update({"pass": True, "arm": child, "records": len(list(directory.glob("*.wfmt")))})
                (directory / "summary.json").write_text(json.dumps(arm, indent=2) + "\n")
        summary.update({"pass": True, "records": len(records)})
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        summary["error"] = str(error)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
