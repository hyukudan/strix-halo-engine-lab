"""CPU test of scripts/bench/cuda/qwen_fast/gate.sh --arms (step selection; no GPU, no model).

The stub --tf records the profile it was launched with and fails, so run_arm stops right after the launch: the test sees
which arms a selection starts, in which order, and that bad selections / a compare without sealed arms refuse with rc 2.
Run: python tests/qwen_fast_gate_arms_tests.py   (needs bash on PATH; Git Bash on Windows)
"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "bench" / "cuda" / "qwen_fast" / "gate.sh"
BASH = shutil.which("bash") or r"C:\Program Files\Git\bin\bash.exe"


def posix(p):
    s = str(p).replace("\\", "/")
    return "/" + s[0].lower() + s[2:] if len(s) > 1 and s[1] == ":" else s


def run(tmp, *args):
    stub = tmp / "tf_stub.sh"
    stub.write_text('#!/bin/bash\nfor ((i=1;i<=$#;i++)); do [ "${!i}" = --profile ] && { j=$((i+1)); '
                    'echo "${!j}" >> "$(dirname "$0")/launched"; }; done\nexit 1\n', newline="\n")
    stub.chmod(0o755)
    cmd = [BASH, posix(GATE), "--tf", posix(stub), "--out", posix(tmp / "out"), "--prompt", "p=x.u32", *args]
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PYTHON=sys.executable))
    launched = (tmp / "launched").read_text().split() if (tmp / "launched").exists() else []
    return r.returncode, launched, r.stdout + r.stderr


def case(name, args, want_rc, want_launched, want_text=None, prep=None):
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        if prep:
            prep(tmp)
        rc, launched, text = run(tmp, *args)
        ok = rc == want_rc and launched == want_launched and (want_text is None or want_text in text)
        print(f"{'PASS' if ok else 'FAIL'} {name}: rc {rc} launched {launched}" + ("" if ok else f"\n{text[-600:]}"))
        return ok


def main():
    results = [
        # default = previous behaviour: the exact arm is launched first (the stub fails it, so the gate stops there)
        case("default starts exact", [], 2, ["exact"]),
        case("--arms exact", ["--arms", "exact"], 2, ["exact"]),
        case("--arms fast,compare starts only fast", ["--arms", "fast,compare"], 2, ["fast"]),
        case("unknown step refused", ["--arms", "exact,bogus"], 2, [], "unknown step 'bogus'"),
        case("empty selection refused", ["--arms", ","], 2, [], "no step selected"),
        case("compare without sealed arms refused", ["--arms", "compare"], 2, [], "is missing"),
        case("compare with only exact sealed refused", ["--arms", "compare"], 2, [], "out/fast is missing",
             prep=lambda t: (t / "out" / "exact").mkdir(parents=True)),
    ]
    print(f"qwen_fast_gate_arms: {sum(results)}/{len(results)} PASS")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
