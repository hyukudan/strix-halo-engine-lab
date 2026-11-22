"""D4 bootstrap: emit host registrations for declared existing launchers.

Input is data/catalog/instances.toml, schema strix-kernel-instances-v1.
Each weight is [FormatId.name, FormatId.revision]. Native bodies and model
planning are handled by the later family generator.
"""

import argparse
import json
from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/catalog/instances.toml"
SOURCE = ROOT / "src/exec/catalog.cpp"
BEGIN = "// BEGIN D4 INSTANCES\n"
END = "// END D4 INSTANCES\n"
REQUIRED = {"id", "backend", "arch", "op", "form", "weights", "act", "max_rows",
            "identity", "source", "launcher", "mode"}
ARCH = {"cuda": "sm120", "hip": "gfx1151"}
ROWS = ((1, "r1"), (2, "r2_8"), (9, "r9_32"), (33, "tall"))


def render(records):
    declarations = []
    tables = {backend: [] for backend in ARCH}
    keys = set()
    names = set()
    for row in records:
        name = row["id"]
        if (REQUIRED - row.keys() or row.keys() - (REQUIRED | {"capturable"})
                or not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in names):
            raise ValueError(f"{name}: invalid or duplicate instance declaration")
        names.add(name)
        backend, arch = row["backend"], row["arch"]
        if ARCH.get(backend) != arch or not re.fullmatch(fr"{backend}-{arch}-[a-z0-9-]+-v[1-9][0-9]*", row["identity"]):
            raise ValueError(f"{name}: invalid backend, architecture or identity")
        op, form, act = row["op"], row["form"], row["act"]
        if op not in {"linear", "norm", "combine", "elementwise"} or act not in {"f32_fma", "f16_mma", "bf16_mma"}:
            raise ValueError(f"{name}: unsupported bootstrap operation or operand")
        limit = row["max_rows"]
        if type(limit) is not int or not 1 <= limit <= 0xFFFFFFFF or not row["weights"]:
            raise ValueError(f"{name}: invalid rows or weight list")
        symbol, source = row["launcher"], row["source"]
        if not re.fullmatch(r"strix(?:::[a-zA-Z_]\w*)+", symbol):
            raise ValueError(f"{name}: invalid launcher name")
        source_path = (ROOT / source).resolve()
        if (not source_path.is_relative_to(ROOT / "src")
                or not re.search(rf"\b{symbol.split('::')[-1]}\s*\(", source_path.read_text(encoding="utf-8"))):
            raise ValueError(f"{name}: launcher missing from {source}")
        capturable = row.get("capturable", True)
        if type(capturable) is not bool:
            raise ValueError(f"{name}: capturable must be boolean")
        declarations.append(
            f"constexpr LauncherRef {name}{{{json.dumps(symbol)},\n"
            f"    {json.dumps(source)}, {json.dumps(row['mode'])}, {limit}U}};"
        )
        for weight, revision in row["weights"]:
            if type(revision) is not int or not 0 <= revision <= 0xFFFFFFFF or bool(weight) != bool(revision):
                raise ValueError(f"{name}: invalid format ID")
            for minimum, rows in ROWS:
                if limit < minimum:
                    break
                key = (backend, op, form, weight, revision, act, rows)
                if key in keys:
                    raise ValueError(f"{name}: duplicate key {key}")
                keys.add(key)
                tables[backend].append(
                    f"    instance<OperandClass::{act}, RowsClass::{rows}, {name}>(\n"
                    f"        graph::Op::{op}, {json.dumps(form)}, {{{json.dumps(weight)}, {revision}U}}, "
                    f"device::ArchMask::{arch}, {json.dumps(row['identity'])}, {str(capturable).lower()}),"
                )
    blocks = ["\n".join(declarations)]
    for backend, table in tables.items():
        blocks.append(f"constexpr KernelEntry {backend}_entries[] = {{\n" + "\n".join(table) + "\n};")
    return "\n\n".join(blocks) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check the generated block without writing")
    args = parser.parse_args()
    with MANIFEST.open("rb") as stream:
        manifest = tomllib.load(stream)
    if manifest.keys() != {"schema", "instance"} or manifest["schema"] != "strix-kernel-instances-v1":
        parser.error("invalid instance manifest schema")
    source = SOURCE.read_text(encoding="utf-8")
    head, generated = source.split(BEGIN)
    _, tail = generated.split(END)
    expected = head + BEGIN + render(manifest["instance"]) + END + tail
    if args.check:
        if source != expected:
            parser.error("stale D4 instance block. Run tools/strix-kernelgen/kernelgen.py")
    else:
        SOURCE.write_text(expected, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
