"""D4 bootstrap: emit host registrations for declared existing launchers.

Input is data/catalog/instances.toml and its ordered list of manifest fragments,
schema strix-kernel-instances-v1.
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
OPTIONAL = {"capturable", "cost", "workspace", "exact"}
OPS = {"linear", "norm", "combine", "elementwise", "mix", "select", "gather", "store", "attend", "argmax"}
OPERANDS = {"f32_fma", "f16_mma", "bf16_mma", "fp8_mma", "i8_dp4a"}
ARCH = {"cuda": "sm120", "hip": "gfx1151"}
ROWS = ((1, "r1"), (2, "r2_8"), (9, "r9_32"), (33, "tall"))


def read_manifest(path, *, root=False):
    with path.open("rb") as stream:
        manifest = tomllib.load(stream)
    keys = {"schema", "instance", "fragments"} if root else {"schema", "instance"}
    if manifest.keys() != keys or manifest["schema"] != "strix-kernel-instances-v1":
        raise ValueError(f"{path.relative_to(ROOT)}: invalid instance manifest schema")
    return manifest


def load_records():
    manifest = read_manifest(MANIFEST, root=True)
    records = list(manifest["instance"])
    fragments = manifest["fragments"]
    if not isinstance(fragments, list) or not all(isinstance(path, str) for path in fragments):
        raise ValueError("fragments must be a list of repository-relative TOML paths")
    for fragment in fragments:
        path = (ROOT / fragment).resolve()
        if not path.is_relative_to(ROOT) or Path(fragment).is_absolute() or path.suffix != ".toml":
            raise ValueError(f"{fragment}: expected a repository-relative TOML fragment")
        records.extend(read_manifest(path)["instance"])
    return records


def render(records):
    declarations = []
    tables = {backend: [] for backend in ARCH}
    keys = set()
    names = set()
    for row in records:
        name = row["id"]
        if (REQUIRED - row.keys() or row.keys() - (REQUIRED | OPTIONAL)
                or not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in names):
            raise ValueError(f"{name}: invalid or duplicate instance declaration")
        names.add(name)
        backend, arch = row["backend"], row["arch"]
        if ARCH.get(backend) != arch or not re.fullmatch(fr"{backend}-{arch}-[a-z0-9-]+-v[1-9][0-9]*", row["identity"]):
            raise ValueError(f"{name}: invalid backend, architecture or identity")
        op, form, act = row["op"], row["form"], row["act"]
        if op not in OPS or act not in OPERANDS:
            raise ValueError(f"{name}: unsupported bootstrap operation or operand")
        limit = row["max_rows"]
        if type(limit) is not int or not 1 <= limit <= 0xFFFFFFFF or not row["weights"]:
            raise ValueError(f"{name}: invalid rows or weight list")
        symbol, source = row["launcher"], row["source"]
        if not re.fullmatch(r"strix(?:::[a-zA-Z_]\w*)+", symbol):
            raise ValueError(f"{name}: invalid launcher name")
        source_path = (ROOT / source).resolve()
        leaf = symbol.split("::")[-1]
        launcher_pattern = rf"\b{leaf}\s*\(|\busing\s+(?:\w+::)+{leaf}\s*;"
        if (not source_path.is_relative_to(ROOT / "src")
                or not re.search(launcher_pattern, source_path.read_text(encoding="utf-8"))):
            raise ValueError(f"{name}: launcher missing from {source}")
        capturable = row.get("capturable", True)
        if type(capturable) is not bool:
            raise ValueError(f"{name}: capturable must be boolean")
        cost = row.get("cost", "roofline")
        workspace = row.get("workspace", "site")
        if cost not in ("roofline", "unpriced_single_launch") or workspace not in ("site", "none"):
            raise ValueError(f"{name}: invalid cost or workspace policy")
        exact = row.get("exact", "exact")
        if exact not in ("exact", "bit_changing"):
            raise ValueError(f"{name}: invalid exactness")
        policies = ""
        if cost != "roofline" or workspace != "site":
            cost_function = f"estimate<OperandClass::{act}>" if cost == "roofline" else "unpriced_single_launch"
            workspace_function = "workspace" if workspace == "site" else "no_workspace"
            policies = f", {cost_function}, {workspace_function}"
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
                text = (
                    f"instance<OperandClass::{act}, RowsClass::{rows}, {name}{policies}>(\n"
                    f"        graph::Op::{op}, {json.dumps(form)}, {{{json.dumps(weight)}, {revision}U}}, "
                    f"device::ArchMask::{arch}, {json.dumps(row['identity'])}, {str(capturable).lower()})"
                )
                if exact == "bit_changing":  # FAST schedule: the bits differ from the exact reference
                    tables[backend].append(
                        f"    [] {{\n        auto e = {text};\n"
                        "        e.exact = lever::Exactness::bit_changing;\n        return e;\n    }(),")
                else:
                    tables[backend].append(f"    {text},")
    blocks = ["\n".join(declarations)]
    for backend, table in tables.items():
        blocks.append(f"constexpr KernelEntry {backend}_entries[] = {{\n" + "\n".join(table) + "\n};")
    return "\n\n".join(blocks) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check the generated block without writing")
    args = parser.parse_args()
    try:
        instances = render(load_records())
    except (OSError, ValueError) as error:
        parser.error(str(error))
    source = SOURCE.read_text(encoding="utf-8")
    head, generated = source.split(BEGIN)
    _, tail = generated.split(END)
    expected = head + BEGIN + instances + END + tail
    if args.check:
        if source != expected:
            parser.error("stale D4 instance block. Run tools/strix-kernelgen/kernelgen.py")
    else:
        SOURCE.write_text(expected, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
