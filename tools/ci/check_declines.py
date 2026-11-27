"""Fail on an unrecorded lever decline.

A decline is a lever a run asked for (its tier column, an explicit setting or --all-on) and does not run. Every one is
written in the one strix-decline/v1 form (include/strix/profile/decline.hpp: record_decline, Resolution::decline,
GenerationStats::declined) or through the --all-on record (all_on::engaged), so the receipt states it. This check scans
the engine sources for diagnostics that announce a decline ("declined", "unmeasured regime") on the error stream and
requires each to sit next to one of those records. The sites that predate the record are listed in
tools/ci/decline_baseline.txt: a new unrecorded site fails, a listed site that is gone fails until it is removed from
the list, and with --base the list may only shrink.
"""

import argparse
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tools/ci/decline_baseline.txt"
SCAN = ("src", "apps", "include")
SUFFIXES = {".cpp", ".hpp", ".h", ".hip", ".cu", ".cuh", ".inc"}
EXEMPT = {"include/strix/profile/decline.hpp", "include/strix/profile/all_on_engaged.hpp"}
LITERAL = re.compile(r'"((?:[^"\\]|\\.)*)"')
ANNOUNCES = re.compile(r"declin|unmeasured regime", re.IGNORECASE)
# Counters and statistics ("declined=%llu", "declined=" followed by a value) report a count, not a lever decision.
COUNTER = re.compile(r"declined\s*=\s*(%|$)", re.IGNORECASE)
STREAM = re.compile(r"\bstd::cerr\b|\bcerr\s*<<|\bstderr\b|\bstd::clog\b")
RECORDED = re.compile(r"\brecord_decline\s*\(|[.>]decline\s*\(|\ball_on::engaged\s*\(|\bdeclined\.(push_back|emplace_back)\s*\("
                      r"|\bDecline\s*\{")
WINDOW = 8  # lines on either side of the diagnostic where its record must be
STATEMENT = 3  # lines before the literal that may open the stream statement


def site_key(path, literal):
    return path + "|" + " ".join(literal.split())


def scan_text(path, text):
    """(unrecorded keys, recorded count) for one source file's text."""
    lines = text.splitlines()
    unrecorded, recorded = [], 0
    for index, line in enumerate(lines):
        for match in LITERAL.finditer(line):
            literal = match.group(1)
            if not ANNOUNCES.search(literal) or COUNTER.search(literal):
                continue
            statement = "\n".join(lines[max(0, index - STATEMENT):index + 1])
            if not STREAM.search(statement):
                continue
            window = "\n".join(lines[max(0, index - WINDOW):index + WINDOW + 1])
            if RECORDED.search(window):
                recorded += 1
            else:
                unrecorded.append(site_key(path, literal))
            break
    return unrecorded, recorded


def scan_tree(root):
    unrecorded, recorded = [], 0
    for top in SCAN:
        base = root / top
        if not base.exists():
            continue
        for file in sorted(base.rglob("*")):
            if file.suffix not in SUFFIXES or not file.is_file():
                continue
            rel = file.relative_to(root).as_posix()
            if rel in EXEMPT:
                continue
            found, count = scan_text(rel, file.read_text(encoding="utf-8", errors="replace"))
            unrecorded += found
            recorded += count
    return unrecorded, recorded


def parse_baseline(text):
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def render_baseline(keys):
    head = ("# Error-stream decline diagnostics that predate the strix-decline/v1 record (tools/ci/check_declines.py).\n"
            "# Shrink-only: bind a site to profile::record_decline / Resolution::decline / GenerationStats::declined and\n"
            "# remove its line. One site per line: path|diagnostic text.\n")
    return head + "".join(key + "\n" for key in sorted(set(keys)))


def compare(unrecorded, baseline, base_baseline=None):
    """Failure messages (empty: pass)."""
    failures = []
    listed = set(baseline)
    found = set(unrecorded)
    for key in sorted(found - listed):
        failures.append("unrecorded decline (bind it to profile::record_decline or the request receipt): " + key)
    for key in sorted(listed - found):
        failures.append("baseline lists a site that is gone or now recorded (remove its line): " + key)
    if len(baseline) != len(listed):
        failures.append("baseline lists a site twice")
    if base_baseline is not None:
        for key in sorted(listed - set(base_baseline)):
            failures.append("baseline grew against the base revision (shrink-only): " + key)
    return failures


def base_text(ref):
    try:
        return subprocess.run(["git", "show", ref + ":tools/ci/decline_baseline.txt"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None  # the base predates the list: nothing to shrink from


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", help="git revision whose baseline this one may not exceed")
    parser.add_argument("--write", action="store_true", help="rewrite the baseline from the tree (review the diff)")
    args = parser.parse_args(argv)
    unrecorded, recorded = scan_tree(ROOT)
    if args.write:
        BASELINE.write_text(render_baseline(unrecorded), encoding="utf-8", newline="\n")
        print("decline baseline: %d unrecorded sites written" % len(set(unrecorded)))
        return 0
    baseline = parse_baseline(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else []
    base = None
    if args.base:
        text = base_text(args.base)
        base = parse_baseline(text) if text is not None else None
    failures = compare(unrecorded, baseline, base)
    for failure in failures:
        print("FAIL: " + failure, file=sys.stderr)
    print("declines: %d recorded diagnostics, %d listed unrecorded sites, %d failures"
          % (recorded, len(baseline), len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
