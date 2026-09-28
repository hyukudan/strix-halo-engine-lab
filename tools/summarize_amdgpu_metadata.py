#!/usr/bin/env python3
"""Summarize AMDGPU kernel metadata emitted by llvm-readelf --notes.

The script records only structural facts (kernel names and resource usage). It
does not disassemble or reconstruct source code.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter


FIELDS = (
    "group_segment_fixed_size",
    "private_segment_fixed_size",
    "sgpr_count",
    "sgpr_spill_count",
    "vgpr_count",
    "vgpr_spill_count",
    "wavefront_size",
    "max_flat_workgroup_size",
)


def kernel_family(mangled: str) -> str:
    """Return a broad, non-template family without preserving signatures."""
    if "q4moe" in mangled:
        return "q4moe"
    match = re.search(r"(k_[a-z0-9_]+)", mangled)
    if not match:
        return "other"
    name = match.group(1)
    prefixes = (
        "k_attn",
        "k_ix_",
        "k_dnc_",
        "k_dn_",
        "k_lq4",
        "k_lq6",
        "k_lq8",
        "k_linear",
        "k_expert",
        "k_moe",
        "k_rmsnorm",
        "k_rope",
        "k_conv",
        "k_sample",
        "k_smp",
        "k_vis_",
    )
    return next((prefix.rstrip("_") for prefix in prefixes if name.startswith(prefix)), "other")


def parse(text: str) -> list[dict[str, int | str]]:
    kernels: list[dict[str, int | str]] = []
    for block in text.split("  - .args:")[1:]:
        name_match = re.search(r"^    \.name:\s+(.+)$", block, re.MULTILINE)
        if not name_match:
            continue
        item: dict[str, int | str] = {"name": name_match.group(1).strip()}
        item["family"] = kernel_family(str(item["name"]))
        for field in FIELDS:
            match = re.search(
                rf"^    \.{re.escape(field)}:\s+(\d+)$", block, re.MULTILINE
            )
            if match:
                item[field] = int(match.group(1))
        kernels.append(item)
    return kernels


def main() -> None:
    kernels = parse(sys.stdin.read())
    waves = Counter(str(k.get("wavefront_size", "unknown")) for k in kernels)
    families = Counter(str(k["family"]) for k in kernels)
    spilling = [
        k
        for k in kernels
        if int(k.get("sgpr_spill_count", 0)) or int(k.get("vgpr_spill_count", 0))
    ]
    compact = "--compact" in sys.argv[1:]
    summary = {
        "kernel_count": len(kernels),
        "wavefront_counts": dict(sorted(waves.items())),
        "max_vgpr": max((int(k.get("vgpr_count", 0)) for k in kernels), default=0),
        "max_sgpr": max((int(k.get("sgpr_count", 0)) for k in kernels), default=0),
        "max_lds_bytes": max(
            (int(k.get("group_segment_fixed_size", 0)) for k in kernels), default=0
        ),
        "kernels_with_spills": len(spilling),
        "top_families": families.most_common(30),
        "largest_lds": [
            {key: value for key, value in item.items() if key != "name"}
            for item in sorted(
            kernels,
            key=lambda k: int(k.get("group_segment_fixed_size", 0)),
            reverse=True,
            )[: (3 if compact else 10)]
        ],
        "largest_vgpr": [
            {key: value for key, value in item.items() if key != "name"}
            for item in sorted(
            kernels, key=lambda k: int(k.get("vgpr_count", 0)), reverse=True
            )[: (3 if compact else 10)]
        ],
        "spilling": [
            {key: value for key, value in item.items() if key != "name"}
            for item in spilling[: (3 if compact else 30)]
        ],
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
