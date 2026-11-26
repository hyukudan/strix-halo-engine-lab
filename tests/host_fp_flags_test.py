"""Check effective host-reference flags in a Ninja or Make compile database."""

import argparse
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent.parent.as_posix().lower()
REFERENCES = {
    "src/iq4_nl.cpp",
    "src/iq4_xs.cpp",
    "src/token_sampling.cpp",
    "src/speculation_sampling.cpp",
    "src/serial_decode_reference.cpp",
    "src/mtp_forward_semantics.cpp",
    "src/oracles.cpp",
    "src/cuda/core/nvfp4_reference.cpp",
}


def fp_flags(command):
    return re.findall(r"(?:^|\s)(/fp:[\w]+|-ffp-contract=[\w-]+|-f(?:no-)?fast-math)(?=\s|$)", command)


def strict(flags, logprob):
    msvc = [flag for flag in flags if flag.startswith("/fp:")]
    if msvc:
        if msvc[-1] == "/fp:strict":
            return True
        return not logprob and msvc[-1] == "/fp:precise" and "/fp:contract" not in msvc
    contraction = [flag for flag in flags if flag.startswith("-ffp-contract=")]
    math = [flag for flag in flags if flag.endswith("fast-math")]
    return contraction[-1:] == ["-ffp-contract=off"] and math[-1:] == ["-fno-fast-math"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compile_commands", type=Path)
    parser.add_argument("--uncontracted-logprob", action="store_true")
    args = parser.parse_args()
    required = REFERENCES.copy()
    if args.uncontracted_logprob:
        required.add("src/logprob_partials.cpp")

    seen = set()
    bad = []
    for entry in json.loads(args.compile_commands.read_text(encoding="utf-8")):
        source = entry["file"].replace("\\", "/").lower().removeprefix(ROOT + "/")
        if source not in required:
            continue
        seen.add(source)
        if not strict(fp_flags(entry["command"]), source == "src/logprob_partials.cpp"):
            bad.append(f"{entry['output']}: missing strict host floating-point flags")
    for source in sorted(required - seen):
        bad.append(f"{source}: missing host reference in the compile database")
    if bad:
        raise SystemExit("\n".join(sorted(bad)))


if __name__ == "__main__":
    main()
