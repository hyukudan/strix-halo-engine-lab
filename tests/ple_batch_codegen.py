#!/usr/bin/env python3
"""Compare existing PLE GPU instruction bodies in two amdclang++ device assembly files.

Create both with -O3 -std=gnu++20 --offload-arch=gfx1151 --cuda-device-only -S.
Only local compiler label numbers, comments, and source location directives are ignored.
"""
import re
import sys
from pathlib import Path


def bodies(path):
    source = Path(path).read_text()
    return {m[1]: m[2] for m in re.finditer(
        r"^(_ZN[^\n:]+):[^\n]*\n(.*?)^\.Lfunc_end\d+:", source, re.M | re.S)}


def normalize(body):
    body = re.sub(r"\.L(?:BB|CPI|func_end)\d+", ".LID", body)
    return "\n".join(line.split(";")[0].rstrip() for line in body.splitlines()
                     if not line.lstrip().startswith((".loc ", ".file ")))


if __name__ == "__main__":
    baseline, candidate = map(bodies, sys.argv[1:3])
    names = [name for name in baseline if "PleRowsSite" in name]
    assert names, "no baseline PLE sites found"
    for name in names:
        assert name in candidate, f"missing {name}"
        assert normalize(baseline[name]) == normalize(candidate[name]), f"changed {name}"
    print(f"PASS: {len(names)} existing PleRowsSite kernel instruction bodies unchanged")

    def default_projection(kind):
        return next(body for name, body in candidate.items()
                    if kind in name and "project_wide_rows_lane_kernel" in name
                    and "Lj16ELj8ELj2E" in name)

    original = default_projection("PleRowsSite")
    grouped = default_projection("PleRowsGroupSite")
    for opcode in ("v_dual_fmac_f32", "v_dual_add_f32", "v_add_f32"):
        def instructions(body):
            return [line.strip().split(";")[0].strip() for line in body.splitlines()
                    if line.strip().startswith(opcode)]
        assert instructions(original) == instructions(grouped), f"changed {opcode} sequence"
        print(f"PASS: grouped projection keeps {len(instructions(original))} {opcode} instructions and operands")
