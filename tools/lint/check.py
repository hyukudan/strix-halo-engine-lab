#!/usr/bin/env python3
"""Report shared-core layering and laboratory constants in C++ sources."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import io
import ipaddress
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
import sys
from zipfile import ZipFile


ROOTS = ("src", "include/strix", "apps")
SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hip", ".hpp", ".inc", ".inl", ".ipp", ".mm"}
MODELS = r"deepseek(?:[._-]?v?4(?:[._-]?1)?)?|ds41|dspark|gemma(?:4)?|glm(?:5(?:[._-]?3)?)?|h3|hy4|inkling2|mmflash|minimax|nemotron|pflash|qwen(?:27|35|3(?:[._-]?[58])?)?"
MODEL_PATH = re.compile(rf"(?:{MODELS})(?:[._-]|$)", re.IGNORECASE)
MODEL_LITERAL = re.compile(rf"(?<![\w])(?:{MODELS})(?=$|[^a-zA-Z0-9])", re.IGNORECASE)
MODEL_NAMESPACE = re.compile(rf"\b(?:{MODELS})(?:::|_\w*::)")
BACKENDS = {"cpu", "cuda", "hip", "intel", "metal", "vulkan"}
HOMES = {
    0: {"device/runtime.hpp", "hip/arch.hpp", "cuda/arch/capability.hpp", "xport/backend.hpp"},
    1: {"src/core", "src/moe_core", "src/attn_core", "src/ops", "cuda/core", "cpu/core", "cuda_core",
        "cuda/arch", "hip/arch", "hip/weight", "cuda/weight", "cpu/weight"},
    2: {"include/strix/core", "attn", "moe", "hc", "ops", "weight", "state/decl.hpp", "device", "dist",
        "envelope.hpp", "lever.hpp", "engine_profile.hpp", "profiles", "session/engine.hpp", "session/backend.hpp"},
    3: {"graph", "model_graph", "pair/planner.hpp", "pair/split_plan.hpp"},
    4: {"exec"},
    5: {"exec/generic_engine.hpp", "exec/generic_engine.cpp"},
    6: {"serve", "session", "spec", "pair", "profile", "receipt", "deploy", "residency"},
    7: {"cli"},
}
SDK = re.compile(r"^(?:hip/|hipblas|roc(?:blas|wmma|m/)|(?:cuda|cu(?:blas|dnn|fft|rand|solver|sparse))(?:[_.]|/)|"
                 r"cub/|thrust/|vulkan/|level_zero/|sycl/|Metal/|driver_types\.h$|mma\.h$)")
INCLUDE = re.compile(r"^\s*#\s*include\s*[<\"]([^>\"\n]+)[>\"]", re.MULTILINE)
TOKENS = re.compile(r'R"(?P<delimiter>[^()\\\s]{0,16})\(.*?\)(?P=delimiter)"|'
                    r'"(?:\\.|[^"\\])*"|(?P<char>(?<!\w)(?:u8|u|U|L)?\'(?:\\.|[^\'\\\n]){1,4}\')|'
                    r'(?P<comment>//[^\n]*|/\*.*?\*/)', re.DOTALL)
SDK_TYPE = re.compile(r"\b(?:hip|cuda)(?:Stream|Event)_t\b|\bCU(?:stream|event)\b")
PRIVATE = re.compile(r"\b(?P<kernel>__global__)\b|\b(?P<name>(?:resolve|plan|load)(?:_[a-zA-Z]\w*)?)\s*"
                     r"\([^;{}]*\)\s*(?:const\s*)?(?:noexcept\s*)?(?:->[^;{}]+)?\{")
LAB_NUMBER = re.compile(
    r"\b(?P<name>(?:k_?(?:nodes|ranks|devices)|(?:k_?)?(?:(?:num|n)_?(?:nodes|ranks|devices)|"
    r"world(?:_?size)?|(?:world|nodes?|ranks?|devices?)_?count|"
    r"(?:[a-z0-9]*_?)?(?:link|pcie|usb4|fabric|xgmi)_?(?:[a-z0-9]+_)*(?:bw|bandwidth|gbps|gbs|rate|latency|bytes_per_second)"
    r"(?:_?(?:gbps|gbs|us|ns|ms))?)))\s*(?:==|!=|=(?!=)|\{)\s*"
    r"(?P<number>[0-9][0-9']*(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)(?![0-9'.])\s*[uUlLfF]*\s*(?=[,;})]|&&|\|\|)",
    re.IGNORECASE)
IP_LITERAL = re.compile(r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9.])|\[([0-9a-fA-F:]+:[0-9a-fA-F:]*)\]")
NODE_LITERAL = re.compile(r"(?<![\w-])(?:[\w-]*lab[-_]?(?:node[-_]?)?\d+|node[-_]?[a-z]*\d+|[\w.-]+\.local)\b", re.I)
HOST_PATH = re.compile(r"(?:/(?:home|Users|mnt|scratch)/|[a-z]:\\+(?:Users|tmp)\\+)", re.I)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    lint: str
    message: str
    value: str = ""

    @property
    def key(self):
        # Retain precise counts without recording machine identifiers or unstable line numbers.
        detail = json.dumps([self.message, self.value]).encode()
        return self.path, self.lint, sha256(detail).hexdigest()[:24]


def home(path):
    prefix = next((root for root in ROOTS if path.startswith(root + "/")), "")
    local = path.removeprefix(prefix + "/")
    parts = local.split("/")
    if parts[0] in BACKENDS and len(parts) > 1:
        model_part = parts[2] if parts[1] == "models" and len(parts) > 2 else parts[1]
        if MODEL_PATH.match(model_part):
            return "model"
    if MODEL_PATH.match(parts[0]) or parts[0] == "models":
        return "model"
    if prefix == "apps":
        return 7
    candidates = [(len(match), layer) for layer, matches in HOMES.items() for match in matches
                  if any(value == match or value.startswith(match + "/") for value in (path, local))]
    return max(candidates, default=(0, 2))[1]


def include_target(source, header, files):
    header = header.replace("\\", "/")
    if header.startswith("strix/"):
        return "include/" + header
    possibilities = (posixpath.normpath(posixpath.join(posixpath.dirname(source), header)), header, "src/" + header)
    return next((target for target in possibilities if target in files), None)


def without_comments(match):
    return re.sub(r"[^\n]", " ", match[0]) if match["comment"] else match[0]


def code_only(match):
    return re.sub(r"[^\n]", " ", match[0])


def inspect(path, text, files):
    layer = home(path)
    clean = TOKENS.sub(without_comments, text)
    code = TOKENS.sub(code_only, clean)
    line_breaks = [match.start() for match in re.finditer("\n", clean)]

    def line_number(offset):
        return bisect_right(line_breaks, offset) + 1

    # Process each finding in source order after collection, so line movement cannot affect keys.
    def issue(offset, lint, message, value=""):
        return Finding(path, line_number(offset), lint, message, value)

    for match in INCLUDE.finditer(clean):
        header = match[1]
        target = include_target(path, header, files)
        if layer == "model":
            continue
        if SDK.match(header) and 2 <= layer <= 6:
            yield issue(match.start(1), "layering", f"L{layer} includes SDK header {header}")
        if target is None:
            continue
        target_layer = home(target)
        if target_layer == "model":
            yield issue(match.start(1), "layering", f"L{layer} includes model header {target}")
        elif max(layer, 2) < target_layer and layer < 7:
            yield issue(match.start(1), "layering", f"L{layer} includes L{target_layer} header {target}")
        if (path.startswith("src/cuda_core/") and target.startswith("src/cuda/core/")) or (
                path.startswith("src/cuda/core/") and target.startswith("src/cuda_core/")):
            yield issue(match.start(1), "layering", f"CUDA kernel homes cross into {target}")

    if layer == "model":
        for match in PRIVATE.finditer(code):
            name = match["name"]
            if match["kernel"]:
                yield issue(match.start(), "layering", "model defines a private kernel")
            else:
                yield issue(match.start(), "layering", "model defines a private " + name.split("_", 1)[0], name)
        return

    if 2 <= layer <= 6:
        for match in SDK_TYPE.finditer(code):
            yield issue(match.start(), "layering", f"L{layer} exposes SDK handle {match[0]}")
    for match in LAB_NUMBER.finditer(code):
        yield issue(match.start(), "no_lab_constants", f"literal topology or link value for {match['name']}", match["number"])
    for match in MODEL_NAMESPACE.finditer(code):
        yield issue(match.start(), "no_lab_constants", "model namespace in shared code", match[0])
    include_lines = {line_number(match.start(1)) for match in INCLUDE.finditer(clean)}
    for match in TOKENS.finditer(clean):
        if match["char"] or line_number(match.start()) in include_lines:
            continue
        literal = match[0]
        for label, pattern in (("node name", NODE_LITERAL), ("host path", HOST_PATH), ("model name", MODEL_LITERAL)):
            found = pattern.search(literal)
            if found:
                yield issue(match.start() + found.start(), "no_lab_constants", f"literal {label} in shared code", literal)
        for ip_match in IP_LITERAL.finditer(literal):
            ip = ip_match[1] or ip_match[0]
            try:
                address = ipaddress.ip_address(ip)
            except ValueError:
                continue
            if not (address.is_unspecified or address.is_loopback or address.is_multicast or address.is_reserved):
                yield issue(match.start() + ip_match.start(), "no_lab_constants", "literal network address in shared code", ip)


def source_file(path):
    return PurePosixPath(path).suffix.lower() in SUFFIXES


def scan(read, files):
    return sorted((item for path in files if source_file(path) for item in inspect(path, read(path), files)),
                  key=lambda item: (item.path, item.line, item.lint, item.message, item.value))


def scan_tree(root):
    if not (root / "src").is_dir():
        raise ValueError("--root must contain src/")
    files = {item.relative_to(root).as_posix() for base in ROOTS for item in (root / base).rglob("*")
             if item.is_file() and not item.is_symlink()}
    return scan(lambda path: (root / path).read_text(encoding="utf-8", errors="replace"), files)


def baseline_rows(findings):
    grouped = defaultdict(dict)
    for (source, lint, key), count in sorted(Counter(item.key for item in findings).items()):
        grouped[source, lint][key] = count
    return [[source, lint, counts] for (source, lint), counts in grouped.items()]


def write_baseline(root, ref, path):
    git = ["git", "-C", str(root)]
    commit = subprocess.check_output([*git, "rev-parse", "--verify", ref + "^{commit}"], text=True).strip()
    blob = subprocess.check_output([*git, "archive", "--format=zip", commit, *ROOTS])
    with ZipFile(io.BytesIO(blob)) as archive:
        findings = scan(lambda name: archive.read(name).decode("utf-8", errors="replace").replace("\r\n", "\n"),
                        {item.filename for item in archive.infolist() if not item.is_dir()})
    rows = ",\n".join("    " + json.dumps(row) for row in baseline_rows(findings))
    path.write_text(f'{{\n  "baseline": "{commit}",\n  "findings": [\n{rows}\n  ]\n}}\n', encoding="utf-8", newline="\n")


def write_report(findings, baseline, path):
    remaining = Counter({(source, lint, key): count for source, lint, counts in baseline["findings"]
                         for key, count in counts.items()})
    new = 0
    with path.open("w", encoding="utf-8", newline="\n") as report:
        report.write(f"Baseline {baseline['baseline']}\n")
        for item in findings:
            allowed = remaining[item.key] > 0
            remaining[item.key] -= int(allowed)
            new += int(not allowed)
            status = "baseline" if allowed else "new"
            report.write(f"{item.path}:{item.line}: {item.lint} [{status}] {item.message}\n")
        summary = f"R12 lint report: {len(findings)} findings, {new} new, {sum(remaining.values())} baseline entries resolved"
        report.write(summary + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--allow-list", type=Path, default=Path(__file__).with_name("allow-list.json"))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--report", type=Path, help="Write an advisory report and exit successfully even with findings")
    action.add_argument("--baseline-ref", help="Regenerate the allow-list from a local Git commit")
    args = parser.parse_args()
    try:
        if args.baseline_ref:
            write_baseline(args.root, args.baseline_ref, args.allow_list)
        else:
            baseline = json.loads(args.allow_list.read_text(encoding="utf-8"))
            print(write_report(scan_tree(args.root), baseline, args.report))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"R12 lint: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
