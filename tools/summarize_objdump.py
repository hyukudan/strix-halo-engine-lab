#!/usr/bin/env python3
"""Summarize instruction mix from llvm-objdump output.

This intentionally keeps aggregate counts only. It is suitable for clean-room
comparison because it does not emit disassembled kernel bodies.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter


GROUPS = {
    "matrix": ("v_wmma", "v_mfma"),
    "global_load": ("global_load", "buffer_load", "flat_load"),
    "global_store": ("global_store", "buffer_store", "flat_store"),
    "lds_read": ("ds_read", "ds_load"),
    "lds_write": ("ds_write", "ds_store"),
    "valu_fp_fma_mul_add": ("v_fma", "v_fmac", "v_mul_f", "v_add_f"),
    "valu_integer_address": ("v_add", "v_sub", "v_mul_u", "v_mul_i", "v_mad_i"),
    "valu_convert": ("v_cvt",),
    "valu_packed_dot": ("v_pk_", "v_dot"),
    "vopd_dual_issue": ("v_dual",),
    "cross_lane": ("ds_bpermute", "ds_permute", "ds_swizzle"),
    "wait": ("s_wait",),
    "barrier": ("s_barrier",),
    "branch": ("s_branch", "s_cbranch"),
}

HEADER = re.compile(r"^[0-9a-f]+\s+<(.+)>:$")
INSTRUCTION = re.compile(
    r"^\s*([._A-Za-z][.A-Za-z0-9_]*)\b.*//\s+[0-9A-F]+:"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match", required=True, help="substring in demangled symbol")
    args = parser.parse_args()

    current: str | None = None
    per_function: dict[str, Counter[str]] = {}
    for raw in sys.stdin:
        header = HEADER.match(raw.strip())
        if header:
            name = header.group(1)
            current = name if args.match in name else None
            if current is not None:
                per_function.setdefault(current, Counter())
            continue
        if current is None:
            continue
        instruction = INSTRUCTION.match(raw)
        if instruction:
            opcode = instruction.group(1)
            # AMDGPU code objects may pad the remainder of a symbol with many
            # s_code_end markers. They are not executed instructions and would
            # otherwise dominate small-kernel summaries.
            if opcode != "s_code_end":
                per_function[current][opcode] += 1

    aggregate: Counter[str] = Counter()
    for counts in per_function.values():
        aggregate.update(counts)
    total = sum(aggregate.values())
    grouped: dict[str, int] = {}
    for group, prefixes in GROUPS.items():
        grouped[group] = sum(
            count
            for opcode, count in aggregate.items()
            if opcode.startswith(prefixes)
        )
    payload = {
        "match": args.match,
        "functions": len(per_function),
        "instructions": total,
        "groups": grouped,
        "top_opcodes": aggregate.most_common(20),
        "largest_function_instruction_counts": sorted(
            (sum(counts.values()) for counts in per_function.values()),
            reverse=True,
        )[:10],
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
