#!/usr/bin/env python3
"""Build disjoint plain-text gate windows from declared held-out sources."""

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
MIN_CLASS_ROWS = 800
CONTENT_CLASSES = ("prose", "code", "structured")
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


def tokenize(command, source, tokenizer):
    result = subprocess.run(command + ["--tokenize", str(source)] + tokenizer,
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{source}: tokenizer failed: {result.stderr.strip()}")
    ids = [int(value) for value in result.stdout.split()]
    if any(not 0 <= token < OUTPUT_VOCAB for token in ids):
        raise ValueError(f"{source}: token outside Inkling's effective vocabulary")
    return ids


def key_value(value):
    key, sep, item = value.partition("=")
    if not sep or not re.fullmatch(r"[a-zA-Z0-9_-]+", key) or not item:
        raise argparse.ArgumentTypeError("expected LABEL=VALUE")
    return key, item


def held_out_sources(command, tokenizer, inputs, bands, count):
    grouped = {name: deque() for name in CONTENT_CLASSES}
    seen_paths = set()
    for kind, name in inputs:
        if kind not in grouped:
            raise ValueError(f"unknown content class {kind}: use prose, code or structured")
        path = Path(name).resolve(strict=True)
        if path in seen_paths:
            raise ValueError(f"duplicate held-out source {path}")
        seen_paths.add(path)
        grouped[kind].append((str(path), tokenize(command, path, tokenizer), 0))

    result = []
    band_counts = {band: 0 for band, _ in bands}
    for offset, (kind, sources) in enumerate(grouped.items()):
        for index in range(count):
            band, length = bands[(offset + index) % len(bands)]
            for _ in range(len(sources)):
                source, ids, start = sources.popleft()
                stop = start + length
                if stop > len(ids):
                    sources.append((source, ids, start))
                    continue
                sources.append((source, ids, stop))
                break
            else:
                raise ValueError(f"need another disjoint {length}-token {kind} source window for {band}")
            band_counts[band] += 1
            result.append((f"{kind}-{index + 1:02d}", kind, band, source, start, ids[start:stop]))
    for band, windows in band_counts.items():
        if windows * (GATE_ROWS - 1) < MIN_CLASS_ROWS:
            raise ValueError(f"only {windows * (GATE_ROWS - 1)} scored rows in band {band}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="strix-inkling2-mini-hip or strix-inkling2-mini-cuda")
    parser.add_argument("model", type=Path, help="used only to locate tokenizer.json")
    parser.add_argument("out", type=Path, help="new output directory")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--source", action="append", type=key_value, default=[], metavar="CLASS=FILE",
                        help="declare a held-out plain-text source; repeat for prose, code and structured")
    parser.add_argument("--band", action="append", type=key_value, default=[], metavar="NAME=TOKENS",
                        help="declare the token length for a held-out band; repeat for each band")
    args = parser.parse_args()
    command = [str(args.binary.resolve()), str(args.model.resolve())]
    tokenizer = ["--tokenizer", str(args.tokenizer.resolve())] if args.tokenizer else []
    if args.source or args.band:
        if not args.source or not args.band:
            parser.error("--source and --band are both required for held-out windows")
        if args.count * (GATE_ROWS - 1) < MIN_CLASS_ROWS:
            parser.error("each content class needs at least 800 scored rows (13 windows)")
        try:
            bands = [(label, int(tokens)) for label, tokens in args.band]
            if len(dict(bands)) != len(bands) or any(not GATE_ROWS < n <= (1 << 20) for _, n in bands):
                raise ValueError("use distinct band labels and more than 64 tokens per window")
            windows = held_out_sources(command, tokenizer, args.source, bands, args.count)
        except ValueError as error:
            parser.error(str(error))
        args.out.mkdir(parents=True, exist_ok=False)
        write_windows(args.out, windows, with_bands=True)
        return

    if args.count < 16 or not GATE_ROWS < args.tokens <= (1 << 20):
        parser.error("need at least 16 windows and more than 64 tokens per decode window")
    args.out.mkdir(parents=True, exist_ok=False)
    sources = deque()
    with tempfile.TemporaryDirectory(dir=args.out) as temporary:
        text = Path(temporary) / "prose.txt"
        for source in SOURCES:
            text.write_text(prose(REPO / source), encoding="utf-8")
            ids = tokenize(command, text, tokenizer)
            if len(ids) >= args.tokens:
                sources.append((source, ids, 0))

    windows = []
    for index in range(args.count):
        if not sources:
            raise ValueError(f"only {index} complete windows at {args.tokens} tokens from the fixed sources")
        source, ids, start = sources.popleft()
        stop = start + args.tokens
        if stop + args.tokens <= len(ids):
            sources.append((source, ids, stop))
        windows.append((f"prose-{index + 1:02d}", f"prose-{args.tokens}-tokens", "", source, start, ids[start:stop]))
    write_windows(args.out, windows)


def write_windows(out, windows, with_bands=False):
    manifest, classes, bands, provenance = [], {}, {}, {}
    for name, kind, band, source, start, ids in windows:
        path = f"{name}.ids"
        (out / path).write_text(" ".join(map(str, ids)) + "\n", encoding="ascii")
        manifest.append(f"{name} {len(ids)} {path}\n")
        classes[name] = kind
        bands[name] = band
        provenance[name] = {"source": source, "first_token": start, "end_token_exclusive": start + len(ids)}
    (out / "classes.json").write_text(json.dumps(classes, indent=2) + "\n", encoding="ascii")
    if with_bands:
        (out / "bands.json").write_text(json.dumps(bands, indent=2) + "\n", encoding="ascii")
    (out / "sources.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="ascii")
    (out / "windows.tsv").write_text("".join(manifest), encoding="ascii")
    print(f"{out / 'windows.tsv'}: {len(windows)} windows, {len(windows) * (GATE_ROWS - 1)} scored rows")


if __name__ == "__main__":
    main()
