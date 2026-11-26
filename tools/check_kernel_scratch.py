#!/usr/bin/env python3
"""Kernel scratch/spill audit and build check for the AMDGPU code objects in built binaries.

Reads every amdgcn code object embedded in the given files (host executables, shared
libraries, objects with clang offload bundles, or raw .hsaco/.co files), decodes the
NT_AMDGPU_METADATA note (msgpack) of each one and reports per kernel: private segment
(scratch) bytes, VGPR/SGPR spill counts, VGPR/SGPR counts, LDS bytes and dynamic stack.

With --allowance FILE it fails (exit 1) when a kernel exceeds its stored allowance.
Allowance lines: `<kernel-name-or-fnmatch-pattern> <max_private_bytes> [vgpr=N] [sgpr=N] [lds=N] [# reason]`.
Kernels without a matching line are allowed 0 bytes, 0 spills and no dynamic stack.
The first matching line wins. The optional fields are the kernel's static launch budget (a contract checked
without a GPU): VGPRs / SGPRs as allocated and static LDS bytes must not exceed them.

With --known FILE (tools/kernel_scratch_known.txt) the kernels that already exceeded the allowance when the baseline
was recorded are KNOWN: such a kernel passes while every value at fault stays at or below its recorded value. A kernel
over the allowance that the file does not list (a new kernel or a new fault) or a value above its record fails.
Known lines: `<kernel-name> private=N vgpr_spill=N sgpr_spill=N [dynamic_stack=1] [vgpr=N] [sgpr=N] [lds=N]`
(the maximum observed over the toolchains the CI and the nodes use). The list may only shrink: with
--known-base BASE (the same file at the base commit) a kernel absent from BASE or a value above BASE fails.
--write-known FILE writes the kernels at fault, merged (maximum per value) with --known when given.
--summary FILE appends the verdict and the failure lines as Markdown (the CI passes $GITHUB_STEP_SUMMARY).

With --isa-base BASE (repeatable) it compares the machine code of every kernel in the given files against BASE
(binaries of the base tree built with the same flags): per (target, kernel) the disassembly with addresses, encodings, symbolic
targets and comments removed must be identical. It prints IDENTICAL / CHANGED / NEW / GONE per kernel matching
--isa-kernels (fnmatch, default every kernel) and fails when a matching kernel changed or disappeared: the proof
that a move into a shared home did not change code. Needs llvm-objdump.

--self-test runs the allowance parser, the budget check and the ISA normalisation on built-in samples with
planted controls (each must fail as planted); it needs neither a GPU nor a binary.

With --nt it also disassembles each code object (llvm-objdump) and counts global/buffer
loads carrying the slc or nt cache-policy bit per kernel (__builtin_nontemporal_load).

Only python3 is needed for the audit; --nt needs llvm-objdump (--objdump PATH).
"""

import argparse
import csv
import fnmatch
import os
import re
import struct
import subprocess
import sys
import tempfile

BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
EM_AMDGPU = 224
NT_AMDGPU_METADATA = 32


class MsgpackReader:
    def __init__(self, data):
        self.d = data
        self.i = 0

    def take(self, n):
        v = self.d[self.i:self.i + n]
        self.i += n
        return v

    def u(self, fmt):
        size = struct.calcsize(fmt)
        return struct.unpack(">" + fmt, self.take(size))[0]

    def read(self):
        b = self.d[self.i]
        self.i += 1
        if b <= 0x7F:
            return b
        if 0x80 <= b <= 0x8F:
            return self.map(b & 0x0F)
        if 0x90 <= b <= 0x9F:
            return [self.read() for _ in range(b & 0x0F)]
        if 0xA0 <= b <= 0xBF:
            return self.take(b & 0x1F).decode("utf-8", "replace")
        if b >= 0xE0:
            return b - 0x100
        if b == 0xC0:
            return None
        if b == 0xC2:
            return False
        if b == 0xC3:
            return True
        if b in (0xC4, 0xC5, 0xC6):
            return bytes(self.take(self.u({0xC4: "B", 0xC5: "H", 0xC6: "I"}[b])))
        if b == 0xCA:
            return self.u("f")
        if b == 0xCB:
            return self.u("d")
        if b in (0xCC, 0xCD, 0xCE, 0xCF):
            return self.u({0xCC: "B", 0xCD: "H", 0xCE: "I", 0xCF: "Q"}[b])
        if b in (0xD0, 0xD1, 0xD2, 0xD3):
            return self.u({0xD0: "b", 0xD1: "h", 0xD2: "i", 0xD3: "q"}[b])
        if b in (0xD9, 0xDA, 0xDB):
            return self.take(self.u({0xD9: "B", 0xDA: "H", 0xDB: "I"}[b])).decode("utf-8", "replace")
        if b in (0xDC, 0xDD):
            return [self.read() for _ in range(self.u("H" if b == 0xDC else "I"))]
        if b in (0xDE, 0xDF):
            return self.map(self.u("H" if b == 0xDE else "I"))
        raise ValueError("unsupported msgpack byte 0x%02x" % b)

    def map(self, n):
        out = {}
        for _ in range(n):
            k = self.read()
            out[k] = self.read()
        return out


def code_objects(path):
    """Yield (label, bytes) for every amdgcn code object found in the file."""
    data = open(path, "rb").read()
    if data[:4] == b"\x7fELF" and struct.unpack_from("<H", data, 18)[0] == EM_AMDGPU:
        yield os.path.basename(path), data
        return
    if b"CCOB" in data and BUNDLE_MAGIC not in data:
        raise SystemExit("%s: compressed offload bundles (CCOB) are not supported" % path)
    pos = data.find(BUNDLE_MAGIC)
    index = 0
    while pos >= 0:
        count = struct.unpack_from("<Q", data, pos + 24)[0]
        q = pos + 32
        for _ in range(count):
            off, size, tlen = struct.unpack_from("<QQQ", data, q)
            triple = data[q + 24:q + 24 + tlen].decode("ascii", "replace")
            q += 24 + tlen
            if "amdgcn" in triple and size:
                yield "%s#%d:%s" % (os.path.basename(path), index, triple), data[pos + off:pos + off + size]
        index += 1
        pos = data.find(BUNDLE_MAGIC, pos + 1)


def metadata(co):
    """Return the decoded NT_AMDGPU_METADATA map of an ELF code object."""
    shoff, = struct.unpack_from("<Q", co, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", co, 0x3A)
    for s in range(shnum):
        base = shoff + s * shentsize
        stype, = struct.unpack_from("<I", co, base + 4)
        if stype != 7:  # SHT_NOTE
            continue
        off, size = struct.unpack_from("<QQ", co, base + 0x18)
        p = off
        while p < off + size:
            namesz, descsz, ntype = struct.unpack_from("<III", co, p)
            name = co[p + 12:p + 12 + namesz]
            dpos = p + 12 + ((namesz + 3) & ~3)
            if ntype == NT_AMDGPU_METADATA and name.startswith(b"AMDGPU"):
                return MsgpackReader(co[dpos:dpos + descsz]).read()
            p = dpos + ((descsz + 3) & ~3)
    return {}


def kernel_rows(path):
    rows = []
    for label, co in code_objects(path):
        md = metadata(co)
        for k in md.get("amdhsa.kernels", []):
            rows.append({
                "file": path,
                "object": label,
                "code": co,
                "name": k.get(".name", "?"),
                "symbol": k.get(".symbol", "?"),
                "private": int(k.get(".private_segment_fixed_size", 0)),
                "dynamic_stack": bool(k.get(".uses_dynamic_stack", False)),
                "vgpr_spill": int(k.get(".vgpr_spill_count", 0)),
                "sgpr_spill": int(k.get(".sgpr_spill_count", 0)),
                "vgpr": int(k.get(".vgpr_count", 0)),
                "sgpr": int(k.get(".sgpr_count", 0)),
                "lds": int(k.get(".group_segment_fixed_size", 0)),
                "wave": int(k.get(".wavefront_size", 0)),
                "max_wg": int(k.get(".max_flat_workgroup_size", 0)),
            })
    return rows


BUDGET_KEYS = ("vgpr", "sgpr", "lds")


def parse_allowance(lines):
    """(pattern, max_private_bytes, {budget key: max}) per non-comment line."""
    rules = []
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 2 or not fields[1].isdigit():
            raise SystemExit("allowance line needs a pattern and a byte count: %r" % line)
        budget = {}
        for field in fields[2:]:
            key, sep, value = field.partition("=")
            if not sep or key not in BUDGET_KEYS or not value.isdigit():
                raise SystemExit("allowance field %r: expected one of %s=N" % (field, "/".join(BUDGET_KEYS)))
            budget[key] = int(value)
        rules.append((fields[0], int(fields[1]), budget))
    return rules


def load_allowance(path):
    with open(path, encoding="utf-8") as f:
        return parse_allowance(f)


def rule_for(rules, name):
    for rule in rules:
        if rule[0] == name or fnmatch.fnmatchcase(name, rule[0]):
            return rule
    return None


def allowed(rules, name):
    rule = rule_for(rules, name)
    return rule[1] if rule else 0


def over_budget(rules, row):
    """[(key, value, limit)] of the row's launch-budget fields above the first matching line's budget."""
    rule = rule_for(rules, row["name"])
    if not rule:
        return []
    return [(k, row[k], v) for k, v in sorted(rule[2].items()) if row[k] > v]


KNOWN_KEYS = ("private", "vgpr_spill", "sgpr_spill", "dynamic_stack") + BUDGET_KEYS


def parse_known(lines):
    """{kernel name: {key: recorded maximum}} of the kernels over the allowance when the baseline was recorded."""
    known = {}
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 2 or fields[0] in known:
            raise SystemExit("known line needs a kernel name and its values, one line per kernel: %r" % line)
        values = {}
        for field in fields[1:]:
            key, sep, value = field.partition("=")
            if not sep or key not in KNOWN_KEYS or not value.isdigit() or key in values:
                raise SystemExit("known field %r: expected one of %s=N, once" % (field, "/".join(KNOWN_KEYS)))
            values[key] = int(value)
        known[fields[0]] = values
    return known


def load_known(path):
    with open(path, encoding="utf-8") as f:
        return parse_known(f)


def faults(rules, row):
    """[(key, value, limit)] of the row above its allowance: scratch, spills without scratch, dynamic stack, budget."""
    limit = allowed(rules, row["name"])
    out = []
    if row["private"] > limit:
        out.append(("private", row["private"], limit))
    if limit == 0:
        out.extend((k, row[k], 0) for k in ("vgpr_spill", "sgpr_spill") if row[k])
    if row["dynamic_stack"]:
        out.append(("dynamic_stack", 1, 0))
    out.extend(over_budget(rules, row))
    return out


def uncovered(known, row, found):
    """The faults the baseline does not cover: all of them for an unlisted kernel, else those above the record."""
    record = known.get(row["name"])
    if record is None:
        return list(found)
    return [(k, v, limit) for k, v, limit in found if v > record.get(k, 0)]


def known_growth(known, base):
    """['name: reason'] of the entries a change added to the known list or raised above the base commit's."""
    out = []
    for name, values in sorted(known.items()):
        if name not in base:
            out.append("%s: not in the base list" % name)
            continue
        out.extend("%s: %s %d above the base %d" % (name, k, v, base[name].get(k, 0))
                   for k, v in sorted(values.items()) if v > base[name].get(k, 0))
    return out


def known_line(name, values):
    keys = [k for k in KNOWN_KEYS if k in ("private", "vgpr_spill", "sgpr_spill") or values.get(k)]
    return "%s %s\n" % (name, " ".join("%s=%d" % (k, values.get(k, 0)) for k in keys))


def write_summary(path, title, verdict, lines):
    if not path:
        return
    with open(path, "a", encoding="utf-8") as out:
        out.write("### %s\n\n%s\n\n" % (title, verdict))
        if lines:
            out.write("```\n%s\n```\n\n" % "\n".join(lines[:200]))
            if len(lines) > 200:
                out.write("(%d more lines in the step log)\n\n" % (len(lines) - 200))


ADDRESS_RE = re.compile(r"^\s*[0-9a-f]+:\s*")
SYMBOL_RE = re.compile(r"^[0-9a-f]+ <(.+)>:$")


def normalized_functions(text):
    """{symbol: tuple of instructions} from `llvm-objdump -d --no-show-raw-insn` text: addresses, comments,
    symbolic targets and spacing removed, so that only the instruction stream is compared."""
    out, current = {}, None
    for line in text.splitlines():
        m = SYMBOL_RE.match(line)
        if m:
            current = m.group(1)
            out[current] = []
            continue
        if current is None:
            continue
        line = line.split("//", 1)[0]
        line = ADDRESS_RE.sub("", line)
        line = re.sub(r"<[^>]*>", "", line)
        line = " ".join(line.split())
        if line:
            out[current].append(line)
    return {k: tuple(v) for k, v in out.items()}


def disassemble(objdump, code):
    with tempfile.NamedTemporaryFile(suffix=".co", delete=False) as t:
        t.write(code)
        tmp = t.name
    try:
        return subprocess.run([objdump, "-d", "--no-show-raw-insn", tmp], capture_output=True,
                              text=True, check=True).stdout
    finally:
        os.unlink(tmp)


def isa_map(objdump, paths):
    """{(target triple, symbol): set of normalized instruction streams} over every code object of the files."""
    out = {}
    for path in paths:
        for label, co in code_objects(path):
            target = label.rsplit(":", 1)[-1] if ":" in label else "elf"
            for symbol, body in normalized_functions(disassemble(objdump, co)).items():
                out.setdefault((target, symbol), set()).add(body)
    return out


def isa_compare(base, head, pattern):
    """[(verdict, target, symbol)] for every function matching pattern: IDENTICAL, CHANGED, NEW or GONE."""
    rows = []
    for key in sorted(set(base) | set(head)):
        if not fnmatch.fnmatchcase(key[1], pattern):
            continue
        if key not in head:
            verdict = "GONE"
        elif key not in base:
            verdict = "NEW"
        else:
            verdict = "IDENTICAL" if base[key] == head[key] else "CHANGED"
        rows.append((verdict, key[0], key[1]))
    return rows


def self_test():
    """Planted controls: every check must accept its good sample and reject its planted fault."""
    failures = []

    def expect(ok, what):
        if not ok:
            failures.append(what)

    rules = parse_allowance(["# comment", "k_spill 64 vgpr=128 lds=32768  # reason", "k_wide* 0 vgpr=192", "k_plain 0"])
    expect([r[0] for r in rules] == ["k_spill", "k_wide*", "k_plain"], "allowance patterns")
    expect(rules[0][2] == {"vgpr": 128, "lds": 32768} and rules[2][2] == {}, "budget fields")
    row = {"name": "k_spill", "vgpr": 128, "sgpr": 90, "lds": 32768}
    expect(over_budget(rules, row) == [], "a kernel at its budget passes")
    expect(over_budget(rules, dict(row, vgpr=129)) == [("vgpr", 129, 128)], "planted control: one VGPR over fails")
    expect(over_budget(rules, dict(row, name="k_wide_x", vgpr=200)) == [("vgpr", 200, 192)], "pattern budget applies")
    expect(over_budget(rules, dict(row, name="unlisted", vgpr=255)) == [], "unlisted kernels carry no budget")
    for bad in (["k 1 vgpr=x"], ["k 1 regs=3"], ["k"], ["k vgpr=3"]):
        try:
            parse_allowance(bad)
            failures.append("planted control: malformed line %r accepted" % bad[0])
        except SystemExit:
            pass
    known = parse_known(["# comment", "k_old private=64 vgpr_spill=3 sgpr_spill=0", "k_stack private=0 vgpr_spill=0 "
                         "sgpr_spill=0 dynamic_stack=1 vgpr=200"])
    expect(known["k_stack"] == {"private": 0, "vgpr_spill": 0, "sgpr_spill": 0, "dynamic_stack": 1, "vgpr": 200},
           "known fields")
    base_row = {"name": "k_old", "private": 64, "vgpr_spill": 3, "sgpr_spill": 0, "dynamic_stack": False,
                "vgpr": 100, "sgpr": 90, "lds": 0}
    found = faults(rules, base_row)
    expect(found == [("private", 64, 0), ("vgpr_spill", 3, 0)], "faults of an unlisted kernel")
    expect(uncovered(known, base_row, found) == [], "a known kernel at its record passes")
    grown = dict(base_row, private=80)
    expect(uncovered(known, grown, faults(rules, grown)) == [("private", 80, 0)], "planted control: growth fails")
    spilled = dict(base_row, sgpr_spill=2)
    expect(uncovered(known, spilled, faults(rules, spilled)) == [("sgpr_spill", 2, 0)],
           "planted control: a new spill kind fails")
    new = dict(base_row, name="k_new")
    expect(uncovered(known, new, faults(rules, new)) == faults(rules, new), "planted control: a new kernel fails")
    stack = dict(base_row, name="k_stack", private=0, vgpr_spill=0, dynamic_stack=True, vgpr=200)
    expect(uncovered(known, stack, faults(rules, stack)) == [], "a known dynamic stack passes")
    expect(known_growth(known, known) == [], "an unchanged list passes")
    expect(known_growth(known, {"k_old": known["k_old"]}) == ["k_stack: not in the base list"],
           "planted control: an added kernel fails")
    expect(known_growth({"k_old": dict(known["k_old"], private=65)}, known) == ["k_old: private 65 above the base 64"],
           "planted control: a raised record fails")
    expect(known_growth({"k_old": dict(known["k_old"], private=32)}, known) == [], "a lowered record passes")
    expect(parse_known([known_line("k_stack", known["k_stack"])]) == {"k_stack": known["k_stack"]}, "known line round trip")
    for bad in (["k"], ["k private=x"], ["k regs=1"], ["k private=1 private=2"], ["k private=1", "k private=2"]):
        try:
            parse_known(bad)
            failures.append("planted control: malformed known list %r accepted" % bad)
        except SystemExit:
            pass
    a = ("\n0000000000001900 <kern>:\n    1900: s_load_dwordx4 s[0:3], s[4:5], 0x0 // 000000001900: F4080002\n"
         "    1908: s_cbranch_scc1 3 // 000000001914 <kern+0x14>\n    190c: v_add_f32_e32 v0, v1, v2\n"
         "\n0000000000001a00 <other>:\n    1a00: s_endpgm\n")
    moved = a.replace("1900", "2900").replace("1908", "2908").replace("190c", "290c").replace("1914", "2914")
    changed = a.replace("v_add_f32_e32 v0, v1, v2", "v_mul_f32_e32 v0, v1, v2")
    expect(len(normalized_functions(a)["kern"]) == 3, "three instructions parsed")
    expect(normalized_functions(a)["kern"] == normalized_functions(moved)["kern"], "relocated code is identical")
    base = {("gfx1151", k): {v} for k, v in normalized_functions(a).items()}
    head = {("gfx1151", k): {v} for k, v in normalized_functions(changed).items()}
    expect(isa_compare(base, head, "*") == [("CHANGED", "gfx1151", "kern"), ("IDENTICAL", "gfx1151", "other")],
           "planted control: one changed instruction is CHANGED")
    expect(isa_compare(base, {k: v for k, v in base.items() if k[1] != "kern"}, "k*") == [("GONE", "gfx1151", "kern")],
           "planted control: a vanished kernel is GONE")
    for f in failures:
        print("SELF-TEST FAIL: %s" % f, file=sys.stderr)
    if not failures:
        print("kernel contract self-test PASS (allowance and budget parser, budget check, known baseline, ISA normalisation, "
              "planted controls)")
    return 1 if failures else 0


LOAD_RE = re.compile(r"^\s*(global_load|buffer_load|global_load_tr|flat_load)\S*\s.*\b(slc|nt)\b")


def nontemporal_loads(objdump, rows):
    """Count slc/nt global/buffer/flat loads per kernel symbol (one disassembly per code object)."""
    counts = {}
    seen = {}
    for r in rows:
        key = id(r["code"])
        if key in seen:
            continue
        seen[key] = True
        text = disassemble(objdump, r["code"])
        current = None
        for line in text.splitlines():
            m = SYMBOL_RE.match(line)
            if m:
                current = m.group(1)
                continue
            if current and LOAD_RE.match(line):
                counts[current] = counts.get(current, 0) + 1
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--allowance", help="fail if a kernel exceeds this stored allowance")
    ap.add_argument("--csv", help="write every kernel row to this CSV file")
    ap.add_argument("--all", action="store_true", help="print every kernel, not only those with scratch or spills")
    ap.add_argument("--nt", action="store_true", help="count slc/nt loads per kernel (needs llvm-objdump)")
    ap.add_argument("--objdump", default="llvm-objdump")
    ap.add_argument("--write-allowance", help="write the current per-kernel scratch of every flagged kernel as an allowance file")
    ap.add_argument("--quiet", action="store_true", help="print only the summary and check failures")
    ap.add_argument("--isa-base", action="append", help="base-tree binary (repeatable): fail when a matching kernel's machine code changed")
    ap.add_argument("--isa-kernels", default="*", help="fnmatch pattern of the functions --isa-base compares")
    ap.add_argument("--known", help="known over-allowance kernels: pass while at or below their recorded values")
    ap.add_argument("--known-base", help="the known list at the base commit: fail when --known adds or raises an entry")
    ap.add_argument("--write-known", help="write the kernels at fault (merged with --known, maximum per value)")
    ap.add_argument("--summary", help="append the verdict and failure lines as Markdown (e.g. $GITHUB_STEP_SUMMARY)")
    ap.add_argument("--self-test", action="store_true", help="run the built-in checks with planted controls")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.files:
        ap.error("no files")
    if args.isa_base:
        verdicts = isa_compare(isa_map(args.objdump, args.isa_base), isa_map(args.objdump, args.files), args.isa_kernels)
        count = {}
        for verdict, target, symbol in verdicts:
            count[verdict] = count.get(verdict, 0) + 1
            if verdict != "IDENTICAL" or not args.quiet:
                print("%-9s %s %s" % (verdict, target, symbol))
        bad = count.get("CHANGED", 0) + count.get("GONE", 0)
        print("isa identity %s (%s)" % ("FAIL" if bad or not verdicts else "PASS",
                                       ", ".join("%d %s" % (n, v) for v, n in sorted(count.items())) or "no matching kernel"))
        return 1 if bad or not verdicts else 0

    rows = []
    for f in args.files:
        rows.extend(kernel_rows(f))
    # A TU may be linked several times. Keep each distinct allowance or launch-budget observation.
    unique = {}
    for r in rows:
        key = tuple(r[field] for field in ("name", "private", "dynamic_stack", "vgpr_spill", "sgpr_spill",
                                          "vgpr", "sgpr", "lds"))
        unique.setdefault(key, r)
    rows = sorted(unique.values(), key=lambda r: (-r["private"], -r["vgpr_spill"], r["name"]))

    nt = nontemporal_loads(args.objdump, rows) if args.nt else {}
    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as out:
            w = csv.writer(out)
            w.writerow(["name", "private", "dynamic_stack", "vgpr_spill", "sgpr_spill", "vgpr", "sgpr", "lds", "wave",
                        "max_wg", "nt_loads", "object"])
            for r in rows:
                w.writerow([r["name"], r["private"], int(r["dynamic_stack"]), r["vgpr_spill"], r["sgpr_spill"], r["vgpr"],
                            r["sgpr"], r["lds"], r["wave"], r["max_wg"], nt.get(r["symbol"], 0), r["object"]])
    if args.allowance and not rows:  # after the CSV: the artifact records that the inputs had no metadata
        message = "SCRATCH CHECK FAIL: no AMDGPU kernel metadata found"
        print(message, file=sys.stderr)
        write_summary(args.summary, "Kernel scratch budget",
                      "**FAIL**: no AMDGPU kernel metadata in %s" % ", ".join(args.files), [message])
        return 2

    flagged = [r for r in rows if r["private"] or r["vgpr_spill"] or r["sgpr_spill"] or r["dynamic_stack"]]
    print("kernels: %d unique, %d with scratch/spills/dynamic stack" % (len(rows), len(flagged)))
    for r in ([] if args.quiet else rows if args.all else flagged):
        print("%7d B scratch  vspill %4d  sspill %4d  vgpr %3d  lds %6d  w%d%s  %s" % (
            r["private"], r["vgpr_spill"], r["sgpr_spill"], r["vgpr"], r["lds"], r["wave"],
            "  DYNSTACK" if r["dynamic_stack"] else "", r["name"]))
    if args.nt:
        for sym, c in sorted(nt.items(), key=lambda kv: -kv[1]):
            print("nt/slc loads %4d  %s" % (c, sym))

    if args.write_allowance:
        with open(args.write_allowance, "w", encoding="utf-8") as out:
            out.write("# Stored scratch allowance per kernel (bytes of private segment), read by tools/check_kernel_scratch.py.\n"
                      "# Kernels not listed are allowed 0 bytes, 0 spills and no dynamic stack. Lower or delete a line when a\n"
                      "# kernel stops needing scratch; never raise one without a measured reason in the coord log.\n")
            worst = {}
            for r in flagged:  # a spill-only kernel still needs a nonzero line to keep its spills
                worst[r["name"]] = max(worst.get(r["name"], 1), r["private"])
            for name in sorted(worst):
                out.write("%s %d\n" % (name, worst[name]))

    if args.allowance:
        rules = load_allowance(args.allowance)
        known = load_known(args.known) if args.known else None
        failed = []    # (row, faults the baseline does not cover, its record or None)
        covered = {}   # known kernel name -> rows at fault within their record
        at_fault = {}  # kernel name -> maximum per value over its rows at fault (for --write-known)
        for r in rows:
            found = faults(rules, r)
            if not found:
                continue
            record = at_fault.setdefault(r["name"], {})
            for k in ("private", "vgpr_spill", "sgpr_spill"):
                record[k] = max(record.get(k, 0), r[k])
            for k, v, _ in found:
                record[k] = max(record.get(k, 0), v)
            rest = found if known is None else uncovered(known, r, found)
            if rest:
                failed.append((r, rest, None if known is None else known.get(r["name"])))
            else:
                covered.setdefault(r["name"], []).append(r)
        lines = []
        for r, rest, record in failed:
            scratch = [f for f in rest if f[0] not in BUDGET_KEYS]
            if known is None:
                tag = ""
            elif record is None:
                tag = " [new kernel over the allowance]"
            else:
                tag = " [above its known record: %s]" % ", ".join("%s %d" % (k, record.get(k, 0)) for k, _, _ in rest)
            if scratch:
                lines.append("SCRATCH CHECK FAIL: %s private %d B (allowance %d), vgpr_spill %d, sgpr_spill %d%s%s" % (
                    r["name"], r["private"], allowed(rules, r["name"]), r["vgpr_spill"], r["sgpr_spill"],
                    ", dynamic stack" if r["dynamic_stack"] else "", tag))
            for key, value, limit in (f for f in rest if f[0] in BUDGET_KEYS):
                lines.append("BUDGET CHECK FAIL: %s %s %d (budget %d)%s" % (r["name"], key, value, limit, tag))
        if known is not None and args.known_base:
            lines.extend("KNOWN LIST FAIL: %s (the list may only shrink)" % g
                         for g in known_growth(known, load_known(args.known_base)))
        for line in lines:
            print(line, file=sys.stderr)
        if args.write_known:
            merged = {name: dict(values) for name, values in (known or {}).items()}
            for name, values in at_fault.items():
                entry = merged.setdefault(name, {})
                for k, v in values.items():
                    entry[k] = max(entry.get(k, 0), v)
            with open(args.write_known, "w", encoding="utf-8", newline="\n") as out:
                out.write("# Kernels over tools/kernel_scratch_allowance.txt when the baseline was recorded (maximum per value\n"
                          "# over the toolchains measured), read by tools/check_kernel_scratch.py --known. A listed kernel\n"
                          "# passes while it stays at or below these values; the list may only shrink.\n")
                for name in sorted(merged):
                    out.write(known_line(name, merged[name]))
        if lines:
            verdict = "**FAIL**: %d line(s); %d known kernel(s) within their record" % (len(lines), len(covered))
            write_summary(args.summary, "Kernel scratch budget", verdict, lines)
            return 1
        verdict = "scratch check PASS (%d kernels; %d below their allowance, %d allowance lines match no kernel" % (
            len(rows), sum(1 for r in flagged if r["private"] < allowed(rules, r["name"])),
            sum(1 for pattern, _, _ in rules
                if not any(pattern == r["name"] or fnmatch.fnmatchcase(r["name"], pattern) for r in rows)))
        if known is not None:
            verdict += "; %d known kernels within their record, %d known lines not at fault here" % (
                len(covered), sum(1 for name in known if name not in covered))
        verdict += ")"
        print(verdict)
        write_summary(args.summary, "Kernel scratch budget", verdict, [])
    return 0


if __name__ == "__main__":
    sys.exit(main())
