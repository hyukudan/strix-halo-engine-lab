#!/usr/bin/env python3
"""Generate src/minimax/serve/minimax_unicode_tables.inc for the MiniMax tokenizer.

Two separate Unicode versions are needed to be token-identical with HuggingFace `tokenizers` 0.22
(the reference for MiniMax M3.1's tokenizer.json):

* Pre-tokenizer regex classes (\\p{L}, \\p{Lu}, ..., \\p{M}, \\p{N}, \\s): `tokenizers` runs the Split
  regex with Oniguruma (onig crate), whose tables are Unicode 16.0. Probing every code point through
  `tokenizers.pre_tokenizers.Split` agrees 100 % with UCD 16.0.0 UnicodeData.txt general categories,
  while Python 3.13's `unicodedata` (Unicode 15.1) misses the ~4300 code points added in 16.0.
  \\s is Unicode White_Space (NOT Python's str.isspace, which also accepts U+001C..U+001F).
* NFC normalizer: `tokenizers` uses the `unicode-normalization-alignments` crate, whose canonical
  combining classes / decompositions are Unicode 9.0 (e.g. U+07FD, U+08D3, U+0D3B have ccc 0 there,
  while Python 3.13 NFC reorders them). Tables are built from UCD 9.0.0 UnicodeData.txt +
  CompositionExclusions.txt.

The UCD files are read from --ucd-dir (default: ~/.cache/strix-ucd, downloaded from unicode.org
when missing). With --fallback-unicodedata the tables come from Python's `unicodedata` instead (then
NFC and categories are both the interpreter's Unicode version; NOT token-identical with HF).

With `tokenizers` installed, --check probes HF directly and reports any mismatch with the generated
tables (classes for every code point; NFC on several probe strings per code point).

usage: gen_mm_unicode.py [--ucd-dir DIR] [--check] [--fallback-unicodedata] OUT.inc
"""
import argparse
import os
import sys
import urllib.request
import unicodedata

CAT_VERSION = '16.0.0'
NFC_VERSION = '9.0.0'

WHITE_SPACE = {0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000} | set(range(0x2000, 0x200B))

# flags of the category table
F_UP = 1     # [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]
F_LO = 2     # [\p{Ll}\p{Lm}\p{Lo}\p{M}]
F_L = 4      # \p{L}
F_N = 8      # \p{N}


def fetch(ucd_dir, version, name):
    path = os.path.join(ucd_dir, '%s-%s.txt' % (name, version))
    if not os.path.exists(path):
        os.makedirs(ucd_dir, exist_ok=True)
        url = 'https://www.unicode.org/Public/%s/ucd/%s.txt' % (version, name)
        print('downloading', url, file=sys.stderr)
        with urllib.request.urlopen(url, timeout=60) as r, open(path, 'wb') as f:
            f.write(r.read())
    return path


def read_unicode_data(path):
    """-> (category dict, ccc dict, canonical decomposition dict) for every assigned code point."""
    cat, ccc, dec = {}, {}, {}
    first = None
    for line in open(path, encoding='utf-8'):
        f = line.rstrip('\n').split(';')
        cp = int(f[0], 16)
        if f[1].endswith(', First>'):
            first = cp
            continue
        span = range(first, cp + 1) if f[1].endswith(', Last>') else (cp,)
        for c in span:
            cat[c] = f[2]
            if int(f[3]):
                ccc[c] = int(f[3])
        if f[5] and not f[5].startswith('<'):
            dec[cp] = [int(x, 16) for x in f[5].split()]
    return cat, ccc, dec


def read_exclusions(path):
    out = set()
    for line in open(path, encoding='utf-8'):
        line = line.split('#')[0].strip()
        if line:
            out.add(int(line.split()[0], 16))
    return out


def from_unicodedata():
    cat, ccc, dec = {}, {}, {}
    for c in range(0x110000):
        ch = chr(c)
        k = unicodedata.category(ch)
        if k != 'Cn':
            cat[c] = k
        if unicodedata.combining(ch):
            ccc[c] = unicodedata.combining(ch)
        d = unicodedata.decomposition(ch)
        if d and not d.startswith('<'):
            dec[c] = [int(x, 16) for x in d.split()]
    # primary composites: canonical pairs that NFC recomposes
    excl = set()
    for c, d in dec.items():
        if len(d) == 2 and unicodedata.normalize('NFC', ''.join(map(chr, d))) != chr(c):
            excl.add(c)
    return cat, ccc, dec, excl


def class_flags(cat):
    def flags(c):
        k = cat.get(c, 'Cn')
        f = 0
        if k in ('Lu', 'Lt', 'Lm', 'Lo') or k[0] == 'M':
            f |= F_UP
        if k in ('Ll', 'Lm', 'Lo') or k[0] == 'M':
            f |= F_LO
        if k[0] == 'L':
            f |= F_L
        if k[0] == 'N':
            f |= F_N
        return f
    return flags


def ranges_of(value, limit=0x110000):
    out, cur = [], None
    for c in range(limit):
        v = value(c)
        if cur and cur[2] == v and cur[1] + 1 == c:
            cur[1] = c
            continue
        if cur and cur[2]:
            out.append(tuple(cur))
        cur = [c, c, v]
    if cur and cur[2]:
        out.append(tuple(cur))
    return out


class Nfc:
    """Reference NFC over the generated tables (same algorithm as the C++ code)."""
    SBASE, LBASE, VBASE, TBASE, LCOUNT, VCOUNT, TCOUNT = 0xAC00, 0x1100, 0x1161, 0x11A7, 19, 21, 28
    NCOUNT = VCOUNT * TCOUNT
    SCOUNT = LCOUNT * NCOUNT

    def __init__(self, ccc, dec, compose):
        self.ccc, self.dec, self.compose = ccc, dec, compose

    def decompose(self, c, out):
        if self.SBASE <= c < self.SBASE + self.SCOUNT:
            s = c - self.SBASE
            out.append(self.LBASE + s // self.NCOUNT)
            out.append(self.VBASE + (s % self.NCOUNT) // self.TCOUNT)
            if s % self.TCOUNT:
                out.append(self.TBASE + s % self.TCOUNT)
            return
        d = self.dec.get(c)
        if d is None:
            out.append(c)
            return
        for x in d:
            self.decompose(x, out)

    def __call__(self, text):
        cps = []
        for ch in text:
            self.decompose(ord(ch), cps)
        # canonical ordering (stable sort of each run of non-starters by ccc)
        i = 0
        while i < len(cps):
            if self.ccc.get(cps[i], 0) == 0:
                i += 1
                continue
            j = i
            while j < len(cps) and self.ccc.get(cps[j], 0):
                j += 1
            cps[i:j] = sorted(cps[i:j], key=lambda x: self.ccc[x])
            i = j
        # canonical composition
        out = []
        starter = -1
        last_ccc = -1
        for c in cps:
            cc = self.ccc.get(c, 0)
            if starter >= 0 and (last_ccc < cc or (last_ccc == 0 and len(out) - 1 == starter)):
                s = out[starter]
                comp = None
                if self.LBASE <= s < self.LBASE + self.LCOUNT and self.VBASE <= c < self.VBASE + self.VCOUNT:
                    comp = self.SBASE + ((s - self.LBASE) * self.VCOUNT + (c - self.VBASE)) * self.TCOUNT
                elif (self.SBASE <= s < self.SBASE + self.SCOUNT and (s - self.SBASE) % self.TCOUNT == 0
                      and self.TBASE < c < self.TBASE + self.TCOUNT):
                    comp = s + (c - self.TBASE)
                else:
                    comp = self.compose.get((s, c))
                if comp is not None:
                    out[starter] = comp
                    continue
            if cc == 0:
                starter = len(out)
                last_ccc = 0
            else:
                last_ccc = cc
            out.append(c)
        return ''.join(map(chr, out))


def check_against_hf(flags, nfc):
    try:
        from tokenizers import Regex
        from tokenizers.normalizers import NFC
        from tokenizers.pre_tokenizers import Split
    except ImportError:
        print('check: tokenizers not installed, skipped', file=sys.stderr)
        return True
    ok = True
    cps = [c for c in range(0x110000) if not (0xD800 <= c <= 0xDFFF)]
    classes = [
        ('UP', r'[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]', lambda c: bool(flags(c) & F_UP)),
        ('LO', r'[\p{Ll}\p{Lm}\p{Lo}\p{M}]', lambda c: bool(flags(c) & F_LO)),
        ('L', r'\p{L}', lambda c: bool(flags(c) & F_L)),
        ('N', r'\p{N}', lambda c: bool(flags(c) & F_N)),
        ('S', r'\s', lambda c: c in WHITE_SPACE),
    ]
    for name, rx, mine in classes:
        sp = Split(Regex(rx), 'removed', invert=True)
        hit = set()
        for k in range(0, len(cps), 20000):
            s = ''.join(chr(c) for c in cps[k:k + 20000])
            for piece, _ in sp.pre_tokenize_str(s):
                hit.update(ord(ch) for ch in piece)
        diff = [c for c in cps if (c in hit) != mine(c)]
        print('check class %-2s: %d HF code points, %d mismatches %s' % (name, len(hit), len(diff), [hex(c) for c in diff[:10]]), file=sys.stderr)
        ok &= not diff
    hf = NFC()
    probes = [lambda c: c, lambda c: 'a' + c, lambda c: c + '́', lambda c: 'a' + c + '̴', lambda c: 'a̴' + c,
              lambda c: 'a' + c + 'ͅ', lambda c: 'aͅ' + c, lambda c: 'a' + c + 'ְ', lambda c: 'aְ' + c,
              lambda c: 'e' + c, lambda c: 'A' + c, lambda c: 'o' + c, lambda c: 'ᄀ' + c, lambda c: c + 'ᅡ',
              lambda c: c + '̈', lambda c: c + '̀' + c, lambda c: c + c, lambda c: 'क' + c, lambda c: 'େ' + c,
              lambda c: 'ෙ' + c, lambda c: 'ဥ' + c, lambda c: 'う' + c, lambda c: 'ウ' + c, lambda c: c + '़',
              lambda c: c + '゙', lambda c: c + 'ା', lambda c: c + 'ා', lambda c: c + 'ီ', lambda c: '가' + c,
              lambda c: c + 'ᆨ', lambda c: 'ą' + c + '́', lambda c: c + '̧́']
    bad = 0
    for k in range(0, len(cps), 4000):
        chunk = [chr(c) for c in cps[k:k + 4000]]
        for pf in probes:
            s = '|'.join(pf(c) for c in chunk)
            if hf.normalize_str(s) != nfc(s):
                for c in chunk:
                    t = pf(c)
                    if hf.normalize_str(t) != nfc(t):
                        bad += 1
                        if bad <= 10:
                            print('check NFC mismatch', t.encode('unicode_escape'), file=sys.stderr)
    print('check NFC: %d probe strings x %d code points, %d mismatches' % (len(probes), len(cps), bad), file=sys.stderr)
    return ok and bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--ucd-dir', default=os.path.join(os.path.expanduser('~'), '.cache', 'strix-ucd'))
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--fallback-unicodedata', action='store_true')
    a = ap.parse_args()

    if a.fallback_unicodedata:
        cat, ccc, dec, excl = from_unicodedata()
        cat_version = nfc_version = 'unicodedata ' + unicodedata.unidata_version
    else:
        cat, _, _ = read_unicode_data(fetch(a.ucd_dir, CAT_VERSION, 'UnicodeData'))
        _, ccc, dec = read_unicode_data(fetch(a.ucd_dir, NFC_VERSION, 'UnicodeData'))
        excl = read_exclusions(fetch(a.ucd_dir, NFC_VERSION, 'CompositionExclusions'))
        cat_version, nfc_version = 'UCD ' + CAT_VERSION, 'UCD ' + NFC_VERSION

    flags = class_flags(cat)
    cat_ranges = ranges_of(flags)
    ccc_ranges = ranges_of(lambda c: ccc.get(c, 0))
    # composition: two-code-point canonical decompositions of starters, minus exclusions
    compose = {}
    for c, d in sorted(dec.items()):
        if len(d) == 2 and c not in excl and ccc.get(c, 0) == 0 and ccc.get(d[0], 0) == 0:
            compose[(d[0], d[1])] = c
    nfc = Nfc(ccc, dec, compose)
    # the C++ fast path leaves text below U+0300 untouched: check that it is NFC-stable
    assert all(y >= 0x300 for (_, y) in compose)
    assert all(ccc.get(c, 0) == 0 and nfc(chr(c)) == chr(c) for c in range(0x300))

    # stats vs the interpreter's unicodedata (documentation only)
    py_cat_diff = sum(1 for c in range(0x110000) if not (0xD800 <= c <= 0xDFFF) and
                      class_flags({c: unicodedata.category(chr(c))})(c) != flags(c))
    py_ccc_diff = sum(1 for c in range(0x110000) if unicodedata.combining(chr(c)) != ccc.get(c, 0))

    with open(a.out, 'w', encoding='utf-8', newline='\n') as out:
        out.write('// Generated by tools/minimax/gen_mm_unicode.py. Do not edit.\n')
        out.write('// Regex classes: %s (= Oniguruma tables in HF tokenizers 0.22). NFC: %s (= unicode-normalization-alignments).\n'
                  % (cat_version, nfc_version))
        out.write('// Python unicodedata %s differs on %d class flags and %d combining classes.\n'
                  % (unicodedata.unidata_version, py_cat_diff, py_ccc_diff))
        out.write('// kMmClassRanges flags: 1 [Lu Lt Lm Lo M], 2 [Ll Lm Lo M], 4 L, 8 N.\n')
        out.write('inline constexpr MmRange kMmClassRanges[] = {\n')
        for f, l, v in cat_ranges:
            out.write('    {0x%X, 0x%X, %d},\n' % (f, l, v))
        out.write('};\n')
        out.write('// Canonical combining class (non-zero ranges).\n')
        out.write('inline constexpr MmRange kMmCccRanges[] = {\n')
        for f, l, v in ccc_ranges:
            out.write('    {0x%X, 0x%X, %d},\n' % (f, l, v))
        out.write('};\n')
        out.write('// Canonical decompositions (one level; second == 0 for singletons), sorted by code point.\n')
        out.write('inline constexpr MmDecomp kMmDecomps[] = {\n')
        for c, d in sorted(dec.items()):
            out.write('    {0x%X, 0x%X, 0x%X},\n' % (c, d[0], d[1] if len(d) > 1 else 0))
        out.write('};\n')
        out.write('// Primary composites {first, second, composite}, sorted by (first, second).\n')
        out.write('inline constexpr MmCompose kMmCompositions[] = {\n')
        for (x, y), c in sorted(compose.items()):
            out.write('    {0x%X, 0x%X, 0x%X},\n' % (x, y, c))
        out.write('};\n')
        out.write('inline constexpr const char* kMmClassUnicode = "%s";\n' % cat_version)
        out.write('inline constexpr const char* kMmNfcUnicode = "%s";\n' % nfc_version)
    print('%d class ranges, %d ccc ranges, %d decompositions, %d compositions; classes %s, NFC %s; '
          'python unicodedata %s differs on %d class flags, %d ccc'
          % (len(cat_ranges), len(ccc_ranges), len(dec), len(compose), cat_version, nfc_version,
             unicodedata.unidata_version, py_cat_diff, py_ccc_diff))
    if a.check and not check_against_hf(flags, nfc):
        print('check FAILED', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
