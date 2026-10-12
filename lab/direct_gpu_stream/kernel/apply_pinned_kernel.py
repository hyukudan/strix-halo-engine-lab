#!/usr/bin/env python3
"""Prepare the exact private upstream source tree; never build/install/load it."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess

PIN = "72d3fcf802c45d00b300f25b848a93c3a2bd7c7e"
STOCK = {
    "drivers/thunderbolt/stream.c": "94598012d67f3225af3b572fc7f1ee5471daa00e1c0a4752cac908b1b0d7251d",
    "drivers/thunderbolt/Kconfig": "e9217e080dd7e3ec3f33a0225d6134581de9a2c2fd4799527c7cba6dc4dc7004",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="private freshly extracted pinned Linux tree, outside Git")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    if not source.is_dir() or source == repo or repo in source.parents:
        parser.error("upstream source must be a private directory outside the project")
    if any((parent / ".git").exists() for parent in (source, *source.parents)):
        parser.error("the full upstream/build artifacts must remain outside Git")
    for relative, expected in STOCK.items():
        target = (source / relative).resolve(strict=True)
        if source not in target.parents or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            parser.error(f"source is not the unmodified pinned {PIN}: {relative}")
    patch = shutil.which("patch")
    if patch is None:
        parser.error("standard patch tool is required")
    for filename in ("stream.patch", "Kconfig.patch"):
        subprocess.run([patch, "--batch", "--forward", "--dry-run", "-p1",
                        "--input", str(here / filename)], cwd=source, check=True, timeout=30)
    for filename in ("stream.patch", "Kconfig.patch"):
        subprocess.run([patch, "--batch", "--forward", "-p1", "--input", str(here / filename)],
                       cwd=source, check=True, timeout=30)
    for filename in ("gpu_stream.c", "gpu_stream.h", "gpu_stream_logic.h"):
        target = (source / "drivers/thunderbolt" / filename).resolve()
        if source not in target.parents:
            parser.error("extension destination escaped the private source tree")
        shutil.copyfile(here / filename, target)
    target = (source / "include/uapi/linux/strix_gpu_stream.h").resolve()
    if source not in target.parents:
        parser.error("UAPI destination escaped the private source tree")
    shutil.copyfile(repo / "include/strix/linux_direct_gpu_stream_uapi.h", target)
    print(f"Prepared project extension against Linux {PIN}; no build/install/load performed")


if __name__ == "__main__":
    main()
