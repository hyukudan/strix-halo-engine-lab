#!/usr/bin/env python3
"""Build disjoint plain-text gate windows from checked-in documentation."""

import argparse
from collections import deque
import json
from pathlib import Path
import re
import subprocess
import tempfile


REPO = Path(__file__).resolve().parents[3]
SOURCES = (
    "docs/adding-an-engine.md",
    "docs/serving.md",
    "docs/router.md",
    "docs/pair.md",
    "docs/xfer.md",
    "docs/install.md",
    "docs/getting-started/engine-reference.md",
)
GATE_ROWS = 64
OUTPUT_VOCAB = 200058


def prose(path):
    """Keep paragraph text with link labels, omitting Markdown blocks and HTML."""
    paragraphs = []
    paragraph = []
    fence = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        marker = re.match(r"^(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                fence = marker[1]
            elif marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = None
        excluded = (marker or fence or not line or raw.startswith(("    ", "\t")) or
                    re.match(r"^(?:[#|><*+-]|\d+[.)]\s)", line))
        if excluded:
            if paragraph and paragraph[0][0].isalpha():
                paragraphs.append(" ".join(paragraph))
            paragraph.clear()
            continue
        line = re.sub(r"!?\[([^\]]+)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"<[^>]+>", "", line).replace("`", "").replace("**", "")
        if line:
            paragraph.append(line)
    if paragraph and paragraph[0][0].isalpha():
        paragraphs.append(" ".join(paragraph))
    return "\n\n".join(paragraphs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="strix-inkling2-mini-hip or strix-inkling2-mini-cuda")
    parser.add_argument("model", type=Path, help="used only to locate tokenizer.json")
    parser.add_argument("out", type=Path, help="new output directory")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=256)
    args = parser.parse_args()
    if args.count < 16 or not GATE_ROWS < args.tokens <= (1 << 20):
        parser.error("need at least 16 windows and more than 64 tokens per decode window")
    command = [str(args.binary.resolve()), str(args.model.resolve())]
    tokenizer = ["--tokenizer", str(args.tokenizer.resolve())] if args.tokenizer else []
    args.out.mkdir(parents=True, exist_ok=False)

    sources = deque()
    with tempfile.TemporaryDirectory(dir=args.out) as temporary:
        text = Path(temporary) / "prose.txt"
        for source in SOURCES:
            text.write_text(prose(REPO / source), encoding="utf-8")
            result = subprocess.run(command + ["--tokenize", str(text)] + tokenizer,
                                    capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"{source}: tokenizer failed: {result.stderr.strip()}")
            ids = [int(value) for value in result.stdout.split()]
            if any(not 0 <= token < OUTPUT_VOCAB for token in ids):
                raise ValueError(f"{source}: token outside Inkling's effective vocabulary")
            if len(ids) >= args.tokens:
                sources.append((source, ids, 0))

    windows, classes, provenance = [], {}, {}
    for index in range(args.count):
        if not sources:
            raise ValueError(f"only {index} complete windows at {args.tokens} tokens from the fixed sources")
        source, ids, start = sources.popleft()
        stop = start + args.tokens
        if stop + args.tokens <= len(ids):
            sources.append((source, ids, stop))
        name = f"prose-{index + 1:02d}"
        path = f"{name}.ids"
        (args.out / path).write_text(" ".join(map(str, ids[start:stop])) + "\n", encoding="ascii")
        windows.append(f"{name} {args.tokens} {path}\n")
        classes[name] = "prose"
        provenance[name] = {"source": source, "first_token": start, "end_token_exclusive": stop}

    (args.out / "classes.json").write_text(json.dumps(classes, indent=2) + "\n", encoding="ascii")
    (args.out / "sources.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="ascii")
    (args.out / "windows.tsv").write_text("".join(windows), encoding="ascii")
    print(f"{args.out / 'windows.tsv'}: {args.count} windows, {args.count * (GATE_ROWS - 1)} scored rows")


if __name__ == "__main__":
    main()
