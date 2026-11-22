"""Compare sm_120a instructions, encoded words, and resource records offline."""

import argparse
from pathlib import Path
import re
import subprocess


# nvcc adds path-dependent discriminators to anonymous namespace symbols.
# Keep the basename and all parameter/instantiation information in the key.
ANONYMOUS_PATH = re.compile(r"(_GLOBAL__N__)[0-9a-fA-F]{8}_")


def read_records(tool, artifact, option, heading):
    dump = subprocess.check_output([tool, option, str(artifact)], text=True)
    if set(re.findall(r"^arch = (\w+)$", dump, re.M)) != {"sm_120a"}:
        raise ValueError(f"{artifact}: expected only sm_120a device code")
    parts = re.split(heading, dump, flags=re.M)
    records = {}
    for raw_name, text in zip(parts[1::2], parts[2::2], strict=True):
        name = ANONYMOUS_PATH.sub(r"\g<1>00000000_", raw_name)
        lines = []
        for line in text.splitlines():
            if line.startswith("Fatbin"):
                break
            if line.strip():
                lines.append(ANONYMOUS_PATH.sub(r"\g<1>00000000_", line.strip()))
        if name in records or not lines:
            raise ValueError(f"{artifact}: duplicate or empty record for {name}")
        records[name] = tuple(lines)
    if not records:
        raise ValueError(f"{artifact}: no function records")
    return records


def group(name):
    if "big_kernelILb1" in name:
        return "expanded FP16"
    if "big_kernelILb0" in name:
        return "BF16 / Q8"
    if "big_kernel_f8I" in name:
        return "FP8"
    if "5nvfp44tile5sm12011gemm_kernelI" in name:
        return "FP4"
    return "support kernels"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("incumbent", type=Path)
    parser.add_argument("family", type=Path)
    parser.add_argument("--cuobjdump", default="cuobjdump")
    args = parser.parse_args()
    try:
        comparisons = (
            ("--dump-sass", r"^\s*Function\s*:\s*(\S+)\s*$"),
            ("--dump-resource-usage", r"^\s*Function\s+(\S+):\s*$"),
        )
        tables = []
        for option, heading in comparisons:
            old = read_records(args.cuobjdump, args.incumbent, option, heading)
            new = read_records(args.cuobjdump, args.family, option, heading)
            if old.keys() != new.keys():
                raise ValueError(f"{option}: function sets differ: {sorted(old.keys() ^ new.keys())}")
            different = [name for name in old if old[name] != new[name]]
            if different:
                raise ValueError(f"{option}: differing functions: {different}")
            tables.append(new)
        sass, resources = tables
        if sass.keys() != resources.keys():
            raise ValueError("disassembly and resource function sets differ")
        for kind in ("BF16 / Q8", "expanded FP16", "FP8", "FP4", "support kernels"):
            entries = [resources[name][0] for name in sass if group(name) == kind]
            if not entries:
                raise ValueError(f"no functions for {kind}")
            registers = [int(re.search(r"\bREG:(\d+)", entry)[1]) for entry in entries]
            stack = [int(re.search(r"\bSTACK:(\d+)", entry)[1]) for entry in entries]
            local = [int(re.search(r"\bLOCAL:(\d+)", entry)[1]) for entry in entries]
            print(f"PASS {kind}: {len(entries)} functions, registers {min(registers)}-{max(registers)}, "
                  f"max stack {max(stack)} B, max local {max(local)} B")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"FAIL K2c SASS: {error}\n")


if __name__ == "__main__":
    main()
