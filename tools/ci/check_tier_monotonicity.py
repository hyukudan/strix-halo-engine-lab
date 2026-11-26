#!/usr/bin/env python3
"""Check comparable tier throughput in the published results table."""

import argparse
from decimal import Decimal
from pathlib import Path
import re
from statistics import median
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
TIERS = ("exact", "fast", "turbo")
HEADER = "| Model | Hardware | Route | Decode mode | Profile | Prompt (tokens) | Decode (tok/s) | Prefill (tok/s) | Main |"
RATES = r"[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*(?:/|–)\s*[0-9][0-9,]*(?:\.[0-9]+)?)*"
LABEL = r"\d+K"


def rates(text):
    if not re.fullmatch(RATES, text):
        raise ValueError(f"invalid throughput: {text!r}")
    values = [Decimal(n.replace(",", "").strip()) for n in re.split(r"[/–]", text)]
    # A published range has no paired samples. It proves a loss only if
    # its entire interval lies below the other measurement.
    if "–" in text:
        return min(values), max(values)
    value = median(values)
    return value, value


def phase_cells(phase, text, prompts):
    if text == "—":
        return {}
    # Parenthesized observations have a different prompt or give the sample count.
    text = re.sub(r" \([^)]*\)", "", text)
    if phase == "decode" and not re.match(LABEL + r" ", text):
        return {prompts.partition("; decode after ")[2] or prompts: rates(text)}
    prefix = prompts.partition("; decode after ")[0]
    if re.match(LABEL + r" ", text):
        result = {}
        for part in text.split(", "):
            label, sep, rate = part.partition(" ")
            if not sep or not re.fullmatch(LABEL, label) or label in result:
                raise ValueError(f"invalid {phase} band: {part!r}")
            result[label] = rates(rate)
        return result
    return {prefix: rates(text)}


def observations(readme):
    lines = iter(readme.splitlines())
    if not any(line == HEADER for line in lines):
        raise ValueError("published throughput table is missing")
    if not next(lines, "").startswith("|---|"):
        raise ValueError("published throughput separator is missing")
    result = {}
    previous_prompts = {}
    for line in lines:
        if not line.startswith("|"):
            break
        model, hardware, route, mode, profile, prompts, decode, prefill, revision = (
            cell.strip() for cell in line.strip("|").split("|")
        )
        # Qwen's no-drafter annotation says why the observed run was plain.
        mode = re.sub(r" \(no MTP drafter bundle on our nodes\)$", "", mode)
        group = revision.strip("`"), model, hardware, route, mode
        if prompts == "same":
            prompts = previous_prompts[group]
        else:
            previous_prompts[group] = prompts
        tier = profile.split(" ", 1)[0].strip("`,").lower()
        if tier not in TIERS:
            raise ValueError(f"unknown tier: {profile!r}")
        # Combined/explicit-control rows are attribution runs. A Hy4 BASE
        # control with explicit rows is kept as a separately named comparison.
        if " / " in profile or ("set by hand" in profile and tier != "exact"):
            continue
        for phase, value in (("decode", decode), ("prefill", prefill)):
            for band, sample in phase_cells(phase, value, prompts).items():
                # A decode result spanning several contexts without individual
                # values is comparable only to the same listed contexts.
                context = band
                if re.fullmatch(LABEL, band):
                    width = int(band[:-1]) * 1000
                    counts = [int(n.replace(",", "")) for n in prompts.partition(";")[0].split(" / ")]
                    context = str(min(counts, key=lambda n: abs(n - width)))
                if model == "Hy4-flash" and tier == "exact" and "set by hand" in profile:
                    if phase == "decode":
                        # The range spans 8K and 32K. Its lower bound still
                        # provides an 8K BASE control for the default FAST row.
                        context = prompts.split(" / ", 1)[0].replace(",", "")
                    else:
                        continue
                elif " / " not in context:
                    context = context.replace(",", "")
                key = (*group, phase, context, tier)
                if key in result:
                    raise ValueError(f"duplicate throughput measurement: {key}")
                result[key] = sample
    if not result:
        raise ValueError("published throughput table is empty")
    return result


def violations(readme):
    measurements = observations(readme)
    found = set()
    for (*group, higher), (_, higher_hi) in measurements.items():
        index = TIERS.index(higher)
        if index == 0:
            continue
        lower = TIERS[index - 1]
        control = measurements.get((*group, lower))
        if control and higher_hi < control[0]:
            found.add("\t".join((*group, lower, higher, str(control[0]), str(higher_hi))))
    return found


def known_rows(content):
    rows = [row for row in content.splitlines() if row and not row.startswith("#")]
    known = set(rows)
    if len(known) != len(rows):
        raise ValueError("duplicate entries in known tier violations")
    return known


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--known", type=Path, default=Path(__file__).with_name("tier_monotonicity_known.tsv"))
    parser.add_argument("--base", help="base revision for the shrink-only exception list")
    args = parser.parse_args()
    try:
        found = violations(args.readme.read_text(encoding="utf-8"))
        known = known_rows(args.known.read_text(encoding="utf-8"))
        if args.base:
            relative = args.known.resolve().relative_to(ROOT).as_posix()
            git = ["git", "-C", str(ROOT)]
            previous = subprocess.check_output([*git, "ls-tree", args.base, "--", relative])
            if previous:
                content = subprocess.check_output([*git, "show", f"{args.base}:{relative}"], encoding="utf-8")
                added = known - known_rows(content)
                if added:
                    raise ValueError("known tier violations may only shrink:\n" + "\n".join(sorted(added)))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        return 1
    for row in sorted(found & known):
        print("known\t" + row)
    for row in sorted(found - known):
        print("new tier slowdown\t" + row, file=sys.stderr)
    for row in sorted(known - found):
        print("remove resolved tier slowdown\t" + row, file=sys.stderr)
    return int(found != known)


if __name__ == "__main__":
    sys.exit(main())
