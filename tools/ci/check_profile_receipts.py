"""Check the public evidence bound by the profile receipt constants."""

import re
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
PROFILES = "include/strix/profiles/"
QWEN = "docs/models/qwen3.8-27b/README.md"
LOWBIT = "docs/engine/lowbit-calibration.md"
REGISTER = "docs/decisions/evidence-register.md"


def read(path):
    return (ROOT / path).read_text(encoding="utf-8", errors="surrogateescape")


def reject_unless(condition, message):
    if not condition:
        raise ValueError(message)


def section(path, anchor):
    source = read(path)
    if not anchor:
        return source
    headings = list(re.finditer(r"^#{1,6} (.+)$", source, re.M))
    for i, heading in enumerate(headings):
        slug = re.sub(r"[^\w -]", "", heading[1].lower()).replace(" ", "-")
        if slug == anchor:
            end = headings[i + 1].start() if i + 1 < len(headings) else len(source)
            return source[heading.end():end]
    raise ValueError(f"missing heading #{anchor} in {path}")


def fields(source, pattern, label):
    match = re.search(pattern, source, re.M)
    reject_unless(match, f"missing source or receipt field: {label}")
    return match.groups()


def matches_number(value, text, label):
    number = f"{int(value):,}"
    reject_unless(re.search(rf"(?<![\d,]){re.escape(number)}(?![\d,])", text), f"receipt mismatch: {label} = {value}")


def check_doc_links():
    for path in sorted((ROOT / PROFILES).glob("*.hpp")):
        # C++ strings, including compact rows that use positional initialization.
        for file, anchor in re.findall(r'"(docs/[\w./-]+\.md)(?:#([\w-]+))?', path.read_text(encoding="utf-8", errors="surrogateescape")):
            section(file, anchor)


def check_qwen():
    source = read(PROFILES + "qwen_dense_hip.hpp")
    receipt = section(QWEN, "fast-prefill-window-hip-decode-app")
    band_table = read("docs/band-edges.md")
    for suffix, tokens in (("2k", "2,048"), ("8k", "8,192"), ("24k", "24,576")):
        gate, = fields(source, rf'kGate{suffix}\s*=\s*"([^"]+)";', f"kGate{suffix}")
        timing, = fields(source, rf'kTime{suffix}\s*=\s*"([^"]+)";', f"kTime{suffix}")
        reject_unless(gate in band_table and timing in band_table, f"Qwen {tokens}: band receipt text drift")
        fields(receipt, rf"^\| {tokens}[^|]*\| 512-row window \|.*\| PASS \|", f"Qwen {tokens} PASS")
        left, right = fields(receipt, rf"^\| {tokens} \| ([\d. /]+) \| ([\d. /]+) \|", f"Qwen {tokens} timings")
        exact, wide = fields(source, rf"kWide{suffix} = (\d+\.\d+) / (\d+\.\d+);", f"kWide{suffix}")
        for arm, ms, rounded in (("EXACT", exact, left), ("wide", wide, right)):
            reject_unless(abs(float(ms) / 1000 - mean(map(float, rounded.split("/")))) <= 0.006,
                          f"Qwen {tokens} {arm}: time differs from recorded rounded runs")


def check_inkling():
    source = read(PROFILES + "inkling2_mini.hpp")
    q8 = section(REGISTER, "inkling-2-mini-q8_0-attention-and-head-on-gfx1151")
    date, rows, context = fields(source, r'gfx1151", "(\d{4}-\d\d-\d\d)", (\d+),\s*(\d+)\}', "Inkling Q8 receipt")
    reject_unless(date in q8, "Inkling Q8 receipt date drift")
    matches_number(rows, q8, "Inkling Q8 scored rows")
    matches_number(context, q8, "Inkling Q8 maximum context")

    q4 = section(LOWBIT, "inkling-2-mini-rows-gfx1151-hip-71")
    rows, = fields(source, r"\.rows = (\d+),", "Inkling q4 scored rows")
    context, lanes = fields(source, r"kMiniLowBitMeasuredExtent\{(\d+)U, (\d+)U\}", "Inkling q4 scope")
    pack, = fields(source, r'kMiniLowBitPackIdentity =\s*weight::digest_from_hex\("([0-9a-f]+)"\)', "Inkling q4 pack")
    for n, label in ((rows, "rows"), (context, "maximum context"), (lanes, "lanes")):
        reject_unless(f"{int(n):,}" in q4 or n in q4, f"Inkling q4 receipt mismatch: {label} = {n}")
    step_band, = fields(source, r"kMiniLowBitStepBand = (\d+)U", "Inkling q4 step band")
    widths, = fields(q4, r"decode widths ([\d, and]+) using", "Inkling q4 documented widths")
    reject_unless(step_band == str(max(map(int, re.findall(r"\d+", widths)))), "Inkling q4 max step width drift")
    reject_unless(pack[:16] in q4, "Inkling q4 published pack identity drift")
    multi, single = fields(source, r"\.speedup = (\d+\.\d+) / (\d+\.\d+),", "Inkling q4 served speed ratio")
    screened = q4.split("Against a build that screens it", 1)[1]
    reject_unless(all(re.search(rf"\b{re.escape(rate)}\b", screened) for rate in (multi, single)),
                  "Inkling q4 post-screen timing drift")
    for cpp, doc in (("kl", "KL "), ("top1", "top-1 "), ("dnll", "dNLL ")):
        value, = fields(source, rf"{cpp}=([+\-]?\d+\.\d+)", f"Inkling q4 {cpp}")
        reject_unless(doc + value in q4, f"Inkling q4 quality drift: {cpp}={value}")


def main():
    try:
        check_doc_links()
        check_qwen()
        check_inkling()
    except (OSError, ValueError, IndexError) as error:
        raise SystemExit(f"profile receipts: {error}") from error
    print("profile receipts: public links and recorded fields agree")


if __name__ == "__main__":
    main()
