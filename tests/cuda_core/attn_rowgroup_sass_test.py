"""Compare the shared QSA kernels with the frozen baseline offline."""

import argparse
import difflib
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "bench" / "cuda"))
from sass_compare import sass


def find_kernel(functions, name):
    token = f"{len(name)}{name}"
    matches = [(arch, body) for (arch, symbol), body in functions.items() if token in symbol]
    if len(matches) != 1 or matches[0][0] != "sm_120a":
        raise ValueError(f"expected one sm_120a instance of {name}")
    # cuobjdump's column widths vary with other functions in the object.
    return [re.sub(r"\s+", " ", line) for line in matches[0][1].splitlines()]


def compare_kernel(expected_functions, actual_functions, old, new):
    expected = find_kernel(expected_functions, old)
    actual = find_kernel(actual_functions, new)
    if expected != actual:
        delta = difflib.unified_diff(expected, actual, old, new, lineterm="")
        print("\n".join(line for _, line in zip(range(24), delta)), file=sys.stderr)
        raise ValueError(f"{old} differs from {new}")
    print(f"QSA sm_120a SASS PASS: {old} == {new}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_object", help="fixture object containing frozen and shared kernels")
    parser.add_argument("--production-object", help="optional qwen_qsa CUDA object from 5b220d036")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    args = parser.parse_args()
    try:
        fixture = sass(args.cuobjdump, args.fixture_object)
        compare_kernel(fixture, fixture, "frozen_union", "rowgroup_union_kernel")
        compare_kernel(fixture, fixture, "frozen_flash", "rowgroup_flash_kernel")
        if args.production_object:
            production = sass(args.cuobjdump, args.production_object)
            compare_kernel(production, fixture, "qsa_union_kernel", "frozen_union")
            compare_kernel(production, fixture, "qsa_rowgroup_kernel", "frozen_flash")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"QSA sm_120a SASS FAIL: {error}\n")


if __name__ == "__main__":
    main()
