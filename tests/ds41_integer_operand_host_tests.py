"""Run CPU decode/Contract tests and the edited-source byte check. No GPU access."""
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def run(args, **kw):
    return subprocess.run(args, text=True, capture_output=True, cwd=ROOT, **kw)

def byte_check():
    tracked = run(["git", "diff", "--name-only", "--diff-filter=ACMR"]).stdout.splitlines()
    new = run(["git", "ls-files", "--others", "--exclude-standard"]).stdout.splitlines()
    # Skip comments and C++ raw strings before inspecting ordinary string literals.
    tokens = re.compile(r'//[^\n]*|/\*.*?\*/|R"([^ ()\\\t\r\n]{0,16})\(.*?\)\1"|\'(?:\\.|[^\'\\\r\n]){1,8}\'|"(?:\\.|[^"\\])*"', re.S)
    checked = 0
    for name in sorted(set(tracked + new)):
        p = ROOT / name
        if p.suffix not in {".cpp", ".hpp", ".h", ".hip", ".inc", ".inl"} or not p.is_file():
            continue
        data = p.read_bytes()
        if b"\0" in data:
            raise RuntimeError(f"raw NUL in {name}")
        source = data.decode("utf-8-sig")
        for match in tokens.finditer(source):
            token = match.group()
            if token.startswith('"') and ("\n" in token or "\r" in token):
                raise RuntimeError(f"raw newline inside string literal: {name}:{source.count(chr(10), 0, match.start()) + 1}")
        checked += 1
    print(f"PASS edited-source byte check ({checked} C++/HIP headers and sources)")

def main():
    with tempfile.TemporaryDirectory(prefix="ds41-integer-host-") as temp:
        build = Path(temp)
        source = ROOT / "tests/ds41_integer_operand_host_tests.cpp"
        inc = [ROOT / p for p in ("tests/ds41_integer_host", "src", "include")]
        exe = build / ("host-tests.exe" if os.name == "nt" else "host-tests")
        if os.name == "nt":
            vs = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"
            if not vs.exists():
                raise RuntimeError("MSVC vcvars64.bat not found; use the CMake host target")
            log = build / "build.log"
            command = ["cl", "/nologo", "/std:c++20", "/EHsc", "/fp:strict", "/O2", str(source),
                       *["/I" + str(p) for p in inc], "/Fe:" + str(exe), "/Fo:" + str(build / "test.obj")]
            batch = build / "build.cmd"
            batch.write_text('@echo off\ncall "' + str(vs) + '" > "' + str(build / "env.log") + '" 2>&1\n'
                             'if errorlevel 1 exit /b 1\n' + subprocess.list2cmdline(command) + ' > "' + str(log) + '" 2>&1\n'
                             'exit /b %errorlevel%\n')
            result = run(["cmd", "/d", "/c", str(batch)], timeout=120)
            output = log.read_text(errors="replace") if log.exists() else result.stdout + result.stderr
        else:
            compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
            if not compiler:
                raise RuntimeError("C++20 compiler not found")
            result = run([compiler, "-std=c++20", "-O2", "-ffp-contract=off", str(source), *["-I" + str(p) for p in inc], "-o", str(exe)], timeout=120)
            output = result.stdout + result.stderr
        if result.returncode:
            print("\n".join(output.splitlines()[-20:]))
            raise RuntimeError("host compilation failed")
        result = run([str(exe)], timeout=120)
        print("\n".join((result.stdout + result.stderr).splitlines()[-20:]))
        if result.returncode:
            raise RuntimeError("host execution failed")
    byte_check()

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("FAIL:", error, file=sys.stderr)
        sys.exit(1)
