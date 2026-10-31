"""Freeze Python str.lower's Unicode metadata for Jinja's case-insensitive dictsort.

Generation only: query case context through Python's final-sigma rule.
No Python implementation code is copied. Usage: python unicode_lower.py OUTPUT
"""
from pathlib import Path
import sys
import unicodedata

def cased(cp):
    ch = chr(cp)
    return ch.islower() or ch.isupper() or ch.istitle()

def ignorable(cp):
    ch = chr(cp)
    # A following cased character prevents final sigma unless ignored. For a
    # non-cased character, put A after it: only an ignored character lets the
    # sigma see that A. This also identifies cased-and-ignorable characters.
    is_cased = cased(cp)
    result = ("AΣ" + ch + ("" if is_cased else "A")).lower()[1]
    return result == ("ς" if is_cased else "σ")

def ranges(f):
    out = []
    for cp in range(0x110000):
        if f(cp):
            if out and out[-1][1] + 1 == cp:
                out[-1][1] = cp
            else:
                out.append([cp, cp])
    return out

lines = [f"// Generated Unicode {unicodedata.unidata_version} lowercase/context metadata; see tests/gemma4_serve/gen/unicode_lower.py.",
         "// Mathematical Unicode mappings, queried from CPython; see README.md for provenance.",
         "struct LowerMapping { std::uint32_t cp; std::string_view text; };",
         "constexpr LowerMapping lower_mappings[] = {"]
for cp in range(0x110000):
    s = chr(cp)
    if s.lower() != s:
        escaped = ''.join(f"\\x{b:02X}" for b in s.lower().encode('utf-8'))
        lines.append(f'    {{0x{cp:X}U, "{escaped}"}},')
lines.append("};")
lines.append("struct CaseRange { std::uint32_t first, last; };")
for name, prop in [("cased_ranges", cased), ("ignorable_ranges", ignorable)]:
    lines.append(f"constexpr CaseRange {name}[] = {{")
    for a, b in ranges(prop):
        lines.append(f"    {{0x{a:X}U, 0x{b:X}U}},")
    lines.append("};")
Path(sys.argv[1]).write_text('\n'.join(lines) + '\n', encoding='utf-8')
