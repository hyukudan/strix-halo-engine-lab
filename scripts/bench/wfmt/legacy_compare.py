#!/usr/bin/env python3
"""Build the identical legacy API capture adapter against two supplied source refs.
Run only inside the deployment's authorized CPU build lane. No device work.
"""
import argparse
import hashlib
import json
import pathlib
import platform
import subprocess
import sys


def logged(command, path, required=True):
    with path.open("w") as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    if required and code:
        raise ValueError(f"command exited {code}; see {path.name}")
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-source", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--build-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-build", type=pathlib.Path)
    parser.add_argument("--candidate-build", type=pathlib.Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--verify-session", action="store_true")
    args = parser.parse_args()
    kit = pathlib.Path(__file__).resolve().parent
    matrix = args.candidate_source / "tests/wfmt/bindings.json"
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "wfmt.legacy-wrappers@1", "pass": False,
               "baseline_revision": args.baseline_revision, "candidate_revision": args.candidate_revision,
               "architecture": platform.machine(), "flags": "Release; strict reference FP", "sources": {}}
    try:
        for mode, source, revision in (("baseline", args.baseline_source, args.baseline_revision),
                                       ("candidate", args.candidate_source, args.candidate_revision)):
            build = (args.baseline_build if mode == "baseline" else args.candidate_build) or args.build_root / mode
            logged(["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                    "-DBUILD_TESTING=ON", "-DSTRIX_ENABLE_HIP_RUNTIME=OFF", "-DSTRIX_FAMILY_MINIMAX=ON",
                    "-DSTRIX_FAMILY_GLM5=ON", "-DSTRIX_FAMILY_GEMMA4=ON", "-DSTRIX_FAMILY_H3=ON", "-DSTRIX_WFMT_TESTS=ON",
                    "-DCMAKE_PROJECT_INCLUDE=" + str(kit / "legacy_capture.cmake")], args.output / f"{mode}-configure.log")
            targets = ["strix-wfmt-legacy-capture"]
            if args.verify_session:
                targets.append("strix-minimax-session-tests")
            logged(["cmake", "--build", str(build), "--config", "Release", "-j", str(args.jobs), "--target", *targets],
                   args.output / f"{mode}-build.log")
            candidates = [build / "strix-wfmt-legacy-capture", build / "Release/strix-wfmt-legacy-capture.exe"]
            executable = next((p for p in candidates if p.exists()), None)
            if executable is None:
                raise ValueError("legacy capture executable missing")
            cache = (build / "CMakeCache.txt").read_text()
            compiler_line = next(line for line in cache.splitlines() if line.startswith("CMAKE_CXX_COMPILER:"))
            summary["sources"][mode] = {"revision": revision, "compiler": compiler_line.split("=", 1)[1],
                                        "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest()}
            logged([sys.executable, str(kit / "capture.py"), "--exe", str(executable), "--matrix", str(matrix),
                    "--out", str(args.output / mode), "--compiler", compiler_line.split("=", 1)[1],
                    "--architecture", platform.machine(), "--flags", "Release; strict reference FP"],
                   args.output / f"{mode}-capture.log")
            if args.verify_session:
                code = logged(["ctest", "--test-dir", str(build), "-C", "Release", "--output-on-failure",
                               "-R", "^strix-minimax-session-tests$"], args.output / f"{mode}-session.log", False)
                summary[mode + "_session_exit"] = code
                if mode == "candidate" and code:
                    raise ValueError("candidate session regression test failed")
                if mode == "baseline" and code and "duplicate tensor across safetensors shards: local" not in (args.output / "baseline-session.log").read_text():
                    raise ValueError("baseline session test has an unexpected failure")
        logged([sys.executable, str(kit / "compare.py"), "--baseline", str(args.output / "baseline"),
                "--candidate", str(args.output / "candidate"), "--require-bytes", "--out", str(args.output / "comparison")],
               args.output / "compare.log")
        comparison = json.loads((args.output / "comparison/summary.json").read_text())
        summary.update({"pass": True, "records": comparison["records"]})
    except (ValueError, OSError, StopIteration) as error:
        summary["error"] = str(error)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "sources"}, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
