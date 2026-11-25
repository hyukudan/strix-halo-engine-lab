"""Check the B4 candidate object against the base translation unit offline."""

import argparse
from pathlib import Path
import subprocess

from tall_sass_test import read_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--cuobjdump", default="cuobjdump")
    args = parser.parse_args()
    try:
        tables = []
        for option, heading in (
            ("--dump-sass", r"^\s*Function\s*:\s*(\S+)\s*$"),
            ("--dump-resource-usage", r"^\s*Function\s+(\S+):\s*$"),
        ):
            baseline = read_records(args.cuobjdump, args.baseline, option, heading)
            candidate = read_records(args.cuobjdump, args.candidate, option, heading)
            changed = [name for name, record in baseline.items() if candidate.get(name) != record]
            if changed:
                raise ValueError(f"{option}: changed or missing incumbent functions: {changed}")
            added = candidate.keys() - baseline.keys()
            if len(added) != 1 or "big_kernelILb0ELi3ELi2ELi0ELi1ELb0ELi2ELi2E" not in next(iter(added)):
                raise ValueError(f"{option}: expected only the 64x64 three-plane tile, found {sorted(added)}")
            tables.append((baseline.keys(), next(iter(added)), candidate))
        (incumbent, new_tile, sass), (resource_keys, resource_tile, resources) = tables
        if incumbent != resource_keys or new_tile != resource_tile:
            raise ValueError("SASS and resource function sets differ")
        if not any("HMMA" in instruction and "BF16" in instruction for instruction in sass[new_tile]):
            raise ValueError("candidate has no BF16 tensor instruction")
        print(f"PASS B4 untouched CUDA SASS and resources ({len(incumbent)} functions)")
        print(f"PASS B4 candidate CUDA BF16 tile ({resources[new_tile][0]})")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"FAIL B4 SASS: {error}\n")


if __name__ == "__main__":
    main()
