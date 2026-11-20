#!/usr/bin/env python3
"""Common-core layering guard with a ratchet allow-list (AGENTS.md "Shared code first", "Adding a model").

Rules (counted per file over tracked sources under src/ and include/; apps/, tests/ and tools/ are composition roots
and test code, not checked):
  model-include  shared code includes a header that belongs to one model
  cross-model    a file of one model includes a header of another model
  tu-include     a source includes another translation unit (.cpp / .cc / .cu / .hip)
  model-kernel   kernels (__global__) defined in a model-owned file: a private kernel family to bind to the shared one
  raw-getenv     getenv() outside the environment homes (include/strix/env_flags.hpp, env_arms.hpp, lever.hpp)
  spin-wait      `while (` on an atomic load: a wait on another block, stream, thread or node. A lost flag hangs the
                 kernel (or the host thread behind it) with no error; new waits go through a bounded wait that
                 reports a timeout instead of adding a raw loop

A file belongs to a model when a directory on its path, or its file name (after an optional hip_ / cuda_ prefix),
names the model: ALIASES below. Everything else under src/ and include/ is shared code.

tools/ci/layering_allowlist.txt lists today's counts as `rule path count`. The check fails when a count rises or a
(rule, path) pair is new, and also when a count fell without the list being lowered, so the list stays the exact
to-do list of private families to bind and delete. Lower it with --write (review the diff: it must only shrink).

  check_layering.py            check against the allow-list (exit 1 on a violation)
  check_layering.py --write    rewrite the allow-list from the tree
  check_layering.py --report   print the per-rule totals and the per-model share
Standard library only.
"""
import argparse
import pathlib
import re
import subprocess
import sys

ROOT_DIRS = ("src/", "include/")
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".hip", ".inc", ".inl"}
TU_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".hip"}
ALIASES = {"ds41": "deepseek41", "deepseek41": "deepseek41", "glm": "glm", "minimax": "minimax", "mmx": "minimax",
           "qwen": "qwen", "qwen27": "qwen", "qwen35": "qwen", "gemma4": "gemma4", "h3": "h3"}
ENV_HOMES = {"include/strix/env_flags.hpp", "include/strix/env_arms.hpp", "include/strix/lever.hpp"}
RULES = ("model-include", "cross-model", "tu-include", "model-kernel", "raw-getenv", "spin-wait")
ALLOWLIST = "tools/ci/layering_allowlist.txt"

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)
KERNEL_RE = re.compile(r"\b__global__\b")
GETENV_RE = re.compile(r"(?<![\w.])(?:std::)?getenv\s*\(")
# `while (` + an atomic load of a flag + `)` closing the condition on the same line, without a counter in the condition.
SPIN_RE = re.compile(r"\bwhile\s*\(\s*(?:__hip_atomic_load|__atomic_load_n|__atomic_load|atomicAdd\s*\([^,]*,\s*0\s*\)|"
                     r"cuda::atomic_ref[^)]*\)\s*\.load|[\w.\->\[\]]+\.load\s*\()")


def model_of(path):
    """The model a repo-relative path belongs to, or None for shared code."""
    parts = path.split("/")
    for d in parts[:-1]:
        if d in ALIASES:
            return ALIASES[d]
    name = parts[-1]
    for prefix in ("hip_", "cuda_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    stem = re.split(r"[_.]", name, maxsplit=1)[0]
    return ALIASES.get(stem)


CHAR_RE = re.compile(r"'(?:\\.|[^\\'\n])'")


def strip_comments(text, keep_strings=False):
    """Blank out // and /* */ comments, and string / character literals unless keep_strings (keeps line structure)."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        m = CHAR_RE.match(text, i) if c == "'" else None
        if m:
            out.append(m.group(0) if keep_strings else "'" + " " * (len(m.group(0)) - 2) + "'")
            i = m.end()
        elif c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"' and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            end = j + 1 if j < n and text[j] == '"' else j
            out.append(text[i:end] if keep_strings else '"' + " " * (end - i - 2) + '"' if end - i >= 2 else text[i:end])
            i = end
        else:
            out.append(c)
            i += 1
    return "".join(out)


def resolve(includer, target, tracked):
    """Repo-relative path of a quoted include, or None when it is not a tracked file (system / generated)."""
    base = pathlib.PurePosixPath(includer).parent
    for root in (base, pathlib.PurePosixPath("include"), pathlib.PurePosixPath("src"), pathlib.PurePosixPath(".")):
        parts = []
        for p in (root / target).parts:
            if p == "..":
                if not parts:
                    break
                parts.pop()
            elif p != ".":
                parts.append(p)
        else:
            cand = "/".join(parts)
            if cand in tracked:
                return cand
    return None


def scan(files, read):
    """{(rule, path): count} over the given repo-relative paths; read(path) -> text."""
    tracked = set(files)
    counts = {}

    def add(rule, path, k=1):
        if k:
            counts[(rule, path)] = counts.get((rule, path), 0) + k

    for path in sorted(files):
        if not path.startswith(ROOT_DIRS) or pathlib.PurePosixPath(path).suffix not in SOURCE_SUFFIXES:
            continue
        raw = read(path)
        if raw is None:
            continue
        mine = model_of(path)
        for target in INCLUDE_RE.findall(strip_comments(raw, keep_strings=True)):
            dep = resolve(path, target, tracked)
            if dep is None:
                continue
            theirs = model_of(dep)
            if pathlib.PurePosixPath(dep).suffix in TU_SUFFIXES:
                add("tu-include", path)
            if theirs is not None and mine is None:
                add("model-include", path)
            elif theirs is not None and mine is not None and theirs != mine:
                add("cross-model", path)
        code = strip_comments(raw)
        if mine is not None:
            add("model-kernel", path, len(KERNEL_RE.findall(code)))
        if path not in ENV_HOMES:
            add("raw-getenv", path, len(GETENV_RE.findall(code)))
        add("spin-wait", path, len(SPIN_RE.findall(code)))
    return counts


def read_allowlist(path):
    allowed = {}
    for no, line in enumerate(pathlib.Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        f = line.split()
        if len(f) != 3 or f[0] not in RULES or not f[2].isdigit() or int(f[2]) <= 0:
            raise ValueError(f"{path}:{no}: expected `rule path count` (count > 0, rule in {', '.join(RULES)})")
        if (f[0], f[1]) in allowed:
            raise ValueError(f"{path}:{no}: duplicate entry {f[0]} {f[1]}")
        allowed[(f[0], f[1])] = int(f[2])
    return allowed


def compare(counts, allowed):
    """Messages for every (rule, path) whose count differs from the allow-list; empty = pass."""
    msgs = []
    for key in sorted(set(counts) | set(allowed)):
        have, may = counts.get(key, 0), allowed.get(key, 0)
        rule, path = key
        if have > may:
            what = "new" if may == 0 else f"rose from {may}"
            msgs.append(f"{path}: {rule} {have} ({what}); see tools/ci/check_layering.py for the rule and the shared home")
        elif have < may:
            msgs.append(f"{path}: {rule} fell to {have} from {may}: lower tools/ci/layering_allowlist.txt "
                        f"(run tools/ci/check_layering.py --write)")
    return msgs


def format_allowlist(counts):
    head = ["# Ratchet of tools/ci/check_layering.py: `rule path count`. Counts may only go down; this list is the",
            "# to-do list of private families, cross-model includes, raw getenv and unbounded spin waits.",
            "# Regenerate with tools/ci/check_layering.py --write and check that the diff only shrinks."]
    return "\n".join(head + [f"{r} {p} {c}" for (r, p), c in sorted(counts.items())]) + "\n"


def git_files():
    out = subprocess.run(["git", "ls-files", "-z", "--", "src", "include"], check=True, capture_output=True).stdout
    return [x.decode() for x in out.split(b"\0") if x]


def read_file(path):
    try:
        return pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--write", action="store_true", help="rewrite the allow-list from the tree")
    ap.add_argument("--report", action="store_true", help="print per-rule totals and the per-model share")
    ap.add_argument("--allowlist", default=ALLOWLIST)
    a = ap.parse_args(argv)
    counts = scan(git_files(), read_file)
    if a.write:
        pathlib.Path(a.allowlist).write_text(format_allowlist(counts), encoding="utf-8", newline="\n")
        print(f"wrote {a.allowlist}: {len(counts)} entries")
        return 0
    if a.report:
        for rule in RULES:
            rows = {p: c for (r, p), c in counts.items() if r == rule}
            by_model = {}
            for p, c in rows.items():
                m = model_of(p) or "shared"
                by_model[m] = by_model.get(m, 0) + c
            print(f"{rule}: {sum(rows.values())} in {len(rows)} files; "
                  + ", ".join(f"{m} {c}" for m, c in sorted(by_model.items(), key=lambda x: -x[1])))
        return 0
    try:
        allowed = read_allowlist(a.allowlist)
    except (OSError, ValueError) as e:
        print(f"check_layering: {e}")
        return 2
    msgs = compare(counts, allowed)
    for m in msgs:
        print(m)
    print(f"check_layering: {len(counts)} counted entries, {len(msgs)} differences from {a.allowlist}")
    return 1 if msgs else 0


if __name__ == "__main__":
    sys.exit(main())
