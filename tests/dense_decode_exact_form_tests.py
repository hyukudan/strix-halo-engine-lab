#!/usr/bin/env python3
"""Host check of the EXACT arithmetic form of the shared dense decode attention (src/attn_core/dense_decode.hpp).

The engine compiles that header with STRIX_HIP_EXACT_FP_OPTIONS (-ffp-contract=off); a test or another includer may
use the HIP default (contraction that honours pragmas). The bits are the same under both only while:
  - the per-position update exists once (online_step, acc_step), each under `#pragma clang fp contract(off)`, both
    attend kernels use it and have no per-position arithmetic of their own past the dot; the wave merge is main's
    text, byte-identical in both kernels, each kernel body under the same pragma;
  - the only fused operations are the dots' explicit __builtin_fmaf (main's form);
  - the engine and the rows tests' frozen v0 take their options from the one CMake variable, and nothing in the build
    asks for -ffp-contract=fast (which ignores the pragma).
The device rows tests prove the bits on a GPU; this test catches a drift of the source form on a host-only CI.
"""
import pathlib
import re
import sys

root = pathlib.Path(__file__).resolve().parents[1]
failures = []


def check(ok, why):
    if not ok:
        failures.append(why)


def body(text, signature):
    """Text of the brace block that follows the first occurrence of `signature`."""
    at = text.find(signature)
    if at < 0:
        return None
    open_at = text.find("{", at)
    depth = 0
    for i in range(open_at, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[open_at + 1 : i]
    return None


header = (root / "src/attn_core/dense_decode.hpp").read_text(encoding="utf-8")
code = re.sub(r"//[^\n]*", "", header)

fused = re.findall(r"__builtin_fmaf?\s*\(|\bfmaf?\s*\(|__fmaf?_r[nduz]\s*\(", code)
check(len(fused) == 2, f"dense_decode.hpp: {len(fused)} fused operations, expected the 2 dots")
check(len(re.findall(r"s = __builtin_fmaf\(q\[", code)) == 2, "the 2 fused operations are the dots s = fmaf(q, k, s)")

for name in ("online_step(float s", "acc_step(float acc"):
    b = body(code, name)
    check(b is not None, f"{name}: shared function present")
    if b is not None:
        check(b.lstrip().startswith("#pragma clang fp contract(off)"), f"{name}: body starts with the contract(off) pragma")

merges = []
for kernel in ("attend_main_kernel(Params p", "attend_kernel(Params p"):
    b = body(code, kernel)
    check(b is not None, f"{kernel}: kernel present")
    if b is None:
        continue
    check(b.lstrip().startswith("#pragma clang fp contract(off)"), f"{kernel}: body starts with the contract(off) pragma")
    check(b.count("online_step(") == 1 and b.count("acc_step(") == 1, f"{kernel}: one online_step and one acc_step")
    check("* corr" not in b and "* p.scale" not in b and "fmaxf(m" not in b,
          f"{kernel}: no per-position softmax arithmetic outside the shared functions")
    at = b.find("for (unsigned d = threadIdx.x;")
    check(at >= 0, f"{kernel}: wave merge present")
    if at >= 0:
        lines = b[at - (len(b[:at]) - len(b[:at].rstrip(" "))) :].split("\n")
        indent = len(b[:at]) - len(b[:at].rstrip(" "))  # the for line's own indentation
        block = []
        for line in lines:
            block.append(line[indent:] if line.strip() else "")
            if len(block) > 1 and line.strip() == "}" and len(line) - len(line.lstrip()) == indent:
                break
        merges.append("\n".join(block))
check(len(merges) == 2 and merges[0] == merges[1], "the wave merge text is byte-identical in both kernels")

cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
check(len(re.findall(r'set\(STRIX_HIP_EXACT_FP_OPTIONS "-ffp-contract=off"\)', cmake)) == 1,
      "CMake: STRIX_HIP_EXACT_FP_OPTIONS = -ffp-contract=off, set once")
for src in ("src/qwen35/hip/engine.hip", "tests/hip_dense_decode_rows_v0.hip"):
    check(re.search(re.escape(src) + r' PROPERTIES COMPILE_OPTIONS "\$\{STRIX_HIP_EXACT_FP_OPTIONS\}"', cmake) is not None,
          f"CMake: {src} compiles with STRIX_HIP_EXACT_FP_OPTIONS")
check("strix-hip-dense-decode-rows-default-tests" in cmake, "CMake: the rows test is also built with the HIP default")
# Permit fast contraction only on dedicated standalone host tests that exercise the
# *unfused* route contract under an FMA-enabled caller. These are plain C++ test
# executables, not flags for the HIP engine, dense decode, or its frozen v0 test.
# All other CMake uses (including global compile flags) stay forbidden.
HOST_FMA_TEST_OPTIONS = re.compile(
    r"(?m)^[ \t]*target_compile_options\(strix-route-(?:lane-dot|unfused)-contract-test PRIVATE "
    r"(?:-O3 )?-ffp-contract=fast\)[ \t]*$"
)


def forbidden_fast_math(text):
    guarded = HOST_FMA_TEST_OPTIONS.sub("", text)
    return "-ffp-contract=fast" in guarded or "-ffast-math" in guarded


check(not forbidden_fast_math(cmake), "CMake: no unscoped -ffp-contract=fast / -ffast-math")
check(not forbidden_fast_math("target_compile_options(strix-route-lane-dot-contract-test PRIVATE -ffp-contract=fast)"),
      "CMake guard: allow explicit host lane-dot contraction regression")
check(not forbidden_fast_math("target_compile_options(strix-route-unfused-contract-test PRIVATE -O3 -ffp-contract=fast)"),
      "CMake guard: allow explicit host unfused contraction regression")
check(forbidden_fast_math("add_compile_options(-ffp-contract=fast)"),
      "CMake guard: reject global fast contraction")
check(forbidden_fast_math("target_compile_options(strix-route-lane-dot-contract-test PRIVATE -ffast-math)"),
      "CMake guard: reject fast-math even in host contract tests")
check(forbidden_fast_math("target_compile_options(strix-hip-dense-decode-rows PRIVATE -ffp-contract=fast)"),
      "CMake guard: reject fast contraction on a HIP target")

for f in failures:
    print("FAIL:", f)
print(("FAIL" if failures else "PASS") + f": dense_decode exact form ({len(failures)} failure(s))")
sys.exit(1 if failures else 0)
