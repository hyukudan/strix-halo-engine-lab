#!/usr/bin/env python3
"""Run one bounded mandatory CTest group, then compare supplied byte captures."""
import argparse
import json
import pathlib
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET

REGEX = {
    "P01": r"^wfmt\.(contracts|census|compare)$",
    "P02": r"^wfmt\.(reference|geometry|iq3s)$|strix-ds41-(formats|native-fp8|trellis-identity)-tests|nvfp4_(host|tolerance_host)",
    "P03": r"^wfmt\.(readers|admission)$",
    "P04": r"^wfmt\.(resolve|cache|effective-config)$",
    "P05": r"^wfmt\.(cpu|intel)$|strix-intel-(backend-host-tests|dense-boundaries|memory-boundaries)",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True, type=pathlib.Path)
    parser.add_argument("--baseline", required=True, type=pathlib.Path)
    parser.add_argument("--candidate", required=True, type=pathlib.Path)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--pr", required=True, choices=REGEX)
    parser.add_argument("--runner", default="")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if args.timeout < 1 or args.timeout > 120:
        parser.error("partition cases into tickets of at most 120 seconds")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "wfmt.run@1", "pr": args.pr, "backend": args.backend, "pass": False}
    junit = args.output.resolve() / "ctest.xml"
    try:
        command = shlex.split(args.runner) + ["ctest", "--test-dir", str(args.build.resolve()),
            "-C", "Release", "--output-on-failure", "--no-tests=error", "--output-junit", str(junit),
            "-R", REGEX[args.pr]]
        with (args.output / "ctest.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
        if result.returncode:
            raise ValueError(f"mandatory CTest group failed: {result.returncode}")
        root = ET.parse(junit).getroot()
        cases = list(root.iter("testcase"))
        if not cases or any(case.find("skipped") is not None for case in cases):
            raise ValueError("empty or skipped mandatory cases")
        command = [sys.executable, str(pathlib.Path(__file__).with_name("compare.py")),
                   "--baseline", str(args.baseline), "--candidate", str(args.candidate),
                   "--require-bytes", "--out", str(args.output / "compare")]
        with (args.output / "compare.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise ValueError("byte comparison failed")
        summary.update({"pass": True, "tests": len(cases)})
    except (ValueError, OSError, ET.ParseError, subprocess.SubprocessError) as error:
        summary["error"] = str(error)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
