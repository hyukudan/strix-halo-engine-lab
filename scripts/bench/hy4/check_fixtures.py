#!/usr/bin/env python3
"""check_fixtures.py: Hy4-flash gate fixtures (tests/hy4_fixtures) and the identity comparisons of the gate tickets.

  check     [--fixtures DIR] [--tokenize-cmd 'BIN PACK encode' | --tokenizer BIN --pack PART1.gguf]
            every prompt of prompts.json has <id>.ids (1..max_prompt ids, each < vocab), its sha256 equals the manifest row,
            every manifest row names an existing rung and a known status; with --tokenize-cmd the ids are re-encoded from the
            text by the shared tokenizer and must be equal.
  export    --out DIR [--fixtures DIR]   writes DIR/<id>.u32 (little-endian, the fast_logit_gate token format),
            DIR/<id>.csv and DIR/<id>.ids for every prompt, and DIR/prompts.sha256.
  identity  --ref REF_LINE_FILE --got ENGINE_STDOUT [--rung G2] [--class exact|fast]
            compares the engine's "hy4 identity" lines with the oracle reference lines (format in gates-plan.md);
            prints one "<rung> <prompt> ..." verdict line per prompt and "<rung> summary pass=N fail=M missing=K".
  self-test runs the invariants above on synthetic data (no files outside a temp dir).

No engine, model or host path is built in; exit status 0 = every checked row passed.
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIXTURES = os.path.normpath(os.path.join(HERE, "..", "..", "..", "tests", "hy4_fixtures"))
VOCAB = 120832            # tokenizer.ggml.tokens of the pack (hy4_host)
MAX_PROMPT = 32           # G2 prompt window: <= 32 prompt ids + 32 greedy ids
GREEDY = 32
RUNGS = ("G0", "G1", "G2", "G3", "G4", "G5")
STATUSES = ("committed", "pending")
MANIFEST_COLS = ("rung", "fixture", "file", "producer", "format", "class", "status", "sha256")
# Line contract (gates-plan.md section 3). Oracle and engine print the same form; class is exact or fast:<name>.
IDENTITY = re.compile(
    r"^hy4 identity prompt=(?P<prompt>[A-Za-z0-9_-]+) n_prompt=(?P<n>\d+) greedy=(?P<ids>[0-9,]+) "
    r"first_logits_fnv64=(?P<fnv>[0-9a-f]{16}) class=(?P<cls>exact|fast:[A-Za-z0-9_.-]+)"
    r"(?: top5=(?P<top5>\S+))?\s*$", re.M)


class FixtureError(Exception):
    pass


def sha256_ids(ids):
    return hashlib.sha256(("\n".join(str(i) for i in ids) + "\n").encode()).hexdigest()


def read_ids(path):
    with open(path, encoding="ascii") as f:
        text = f.read()
    ids = [int(t) for t in text.split()]
    if text != "".join(f"{i}\n" for i in ids):
        raise FixtureError(f"{path}: not one decimal id per line with a final newline")
    return ids


def load(fixtures):
    with open(os.path.join(fixtures, "prompts.json"), encoding="utf-8") as f:
        prompts = json.load(f)["prompts"]
    rows = []
    with open(os.path.join(fixtures, "manifest.tsv"), encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if tuple(header) != MANIFEST_COLS:
            raise FixtureError(f"manifest.tsv header {header} != {list(MANIFEST_COLS)}")
        for n, line in enumerate(f, 2):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cells = line.split("\t")
            if len(cells) != len(MANIFEST_COLS):
                raise FixtureError(f"manifest.tsv:{n}: {len(cells)} columns")
            rows.append(dict(zip(MANIFEST_COLS, cells), line=n))
    return prompts, rows


def check(fixtures, tokenize_cmd=None, out=sys.stdout):
    if isinstance(tokenize_cmd, str):
        tokenize_cmd = shlex.split(tokenize_cmd)
    prompts, rows = load(fixtures)
    by_file = {r["file"]: r for r in rows}
    seen, fails = set(), []
    for r in rows:
        if r["rung"] not in RUNGS or r["status"] not in STATUSES:
            fails.append(f"manifest.tsv:{r['line']}: rung {r['rung']} / status {r['status']}")
        if r["status"] == "committed" and not os.path.isfile(os.path.join(fixtures, r["file"])):
            fails.append(f"manifest.tsv:{r['line']}: committed file {r['file']} missing")
    for p in prompts:
        pid = p["id"]
        if pid in seen or not re.fullmatch(r"[A-Za-z0-9_-]+", pid):
            fails.append(f"prompt id {pid!r} duplicate or not [A-Za-z0-9_-]")
        seen.add(pid)
        name = f"{pid}.ids"
        path = os.path.join(fixtures, name)
        if not os.path.isfile(path):
            fails.append(f"{name}: missing")
            continue
        try:
            ids = read_ids(path)
        except (FixtureError, ValueError) as e:
            fails.append(str(e))
            continue
        if not 1 <= len(ids) <= MAX_PROMPT:
            fails.append(f"{name}: {len(ids)} ids outside 1..{MAX_PROMPT}")
        if any(i < 0 or i >= VOCAB for i in ids):
            fails.append(f"{name}: id outside the vocabulary [0, {VOCAB})")
        row = by_file.get(name)
        if row is None or row["sha256"] != sha256_ids(ids):
            fails.append(f"{name}: manifest sha256 {row and row['sha256']} != {sha256_ids(ids)}")
        if tokenize_cmd:
            with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as t:
                t.write(p["text"].encode("utf-8"))
            try:
                argv = list(tokenize_cmd)
                if os.path.isfile(argv[0]):  # a relative binary path is not resolved by CreateProcess on Windows
                    argv[0] = os.path.abspath(argv[0])
                res = subprocess.run(argv + [t.name], capture_output=True, text=True, check=False)
            finally:
                os.unlink(t.name)
            got = [int(x) for x in res.stdout.replace(",", " ").split()] if res.returncode == 0 else None
            if got != ids:
                fails.append(f"{name}: re-encoded ids differ (rc {res.returncode}): {got[:8] if got else res.stderr[-200:]}")
        print(f"G0 fixture {pid} ids={len(ids)} sha256={sha256_ids(ids)[:16]}", file=out)
    for f in fails:
        print(f"G0 fixture FAIL {f}", file=out)
    print(f"G0 fixtures {'PASS' if not fails else 'FAIL'} prompts={len(prompts)} manifest_rows={len(rows)} "
          f"retokenized={'yes' if tokenize_cmd else 'no'}", file=out)
    return not fails


def export(fixtures, out_dir):
    prompts, _ = load(fixtures)
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    for p in prompts:
        ids = read_ids(os.path.join(fixtures, f"{p['id']}.ids"))
        with open(os.path.join(out_dir, f"{p['id']}.u32"), "wb") as f:
            f.write(struct.pack(f"<{len(ids)}I", *ids))
        with open(os.path.join(out_dir, f"{p['id']}.csv"), "w", encoding="ascii", newline="\n") as f:
            f.write(",".join(map(str, ids)) + "\n")
        with open(os.path.join(out_dir, f"{p['id']}.ids"), "w", encoding="ascii", newline="\n") as f:
            f.write("".join(f"{i}\n" for i in ids))
        lines.append(f"{sha256_ids(ids)}  {p['id']}.ids")
    with open(os.path.join(out_dir, "prompts.sha256"), "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return len(prompts)


def parse_identity(text):
    rows = {}
    for m in IDENTITY.finditer(text):
        ids = [int(x) for x in m["ids"].split(",") if x]
        rows[m["prompt"]] = dict(n=int(m["n"]), ids=ids, fnv=m["fnv"], cls=m["cls"])
    return rows


def identity(ref_text, got_text, rung="G2", want_class="exact", out=sys.stdout):
    ref, got = parse_identity(ref_text), parse_identity(got_text)
    npass = nfail = nmiss = 0
    for prompt, r in sorted(ref.items()):
        g = got.get(prompt)
        if g is None:
            nmiss += 1
            print(f"{rung} {prompt} MISSING (no 'hy4 identity' line from the engine)", file=out)
            continue
        problems = []
        if g["n"] != r["n"]:
            problems.append(f"n_prompt {g['n']} != {r['n']}")
        if len(g["ids"]) != GREEDY or len(r["ids"]) != GREEDY:
            problems.append(f"greedy lengths {len(g['ids'])}/{len(r['ids'])} != {GREEDY}")
        same = 0
        for a, b in zip(r["ids"], g["ids"]):
            if a != b:
                break
            same += 1
        if same < GREEDY:
            k = same
            problems.append(f"first diff at {k}: ref {r['ids'][k] if k < len(r['ids']) else '-'} "
                            f"got {g['ids'][k] if k < len(g['ids']) else '-'}")
        cls = g["cls"]
        if want_class == "exact":
            if cls != "exact":
                problems.append(f"class {cls} where exact was required")
            elif g["fnv"] != r["fnv"]:
                problems.append(f"first_logits_fnv64 {g['fnv']} != {r['fnv']} (exact class must be bitwise)")
        elif not cls.startswith("fast:"):
            problems.append(f"class {cls} where a declared fast class was required")
        if problems:
            nfail += 1
            print(f"{rung} {prompt} FAIL ids {same}/{GREEDY} class={cls}: " + "; ".join(problems), file=out)
        else:
            npass += 1
            print(f"{rung} {prompt} PASS ids {GREEDY}/{GREEDY} class={cls} fnv={g['fnv']}", file=out)
    print(f"{rung} summary pass={npass} fail={nfail} missing={nmiss}", file=out)
    return nfail == 0 and nmiss == 0 and npass > 0


def self_test():
    import io
    fails = []

    def expect(cond, what):
        if not cond:
            fails.append(what)

    with tempfile.TemporaryDirectory() as d:
        ids = [100, 2000, 3]
        with open(os.path.join(d, "prompts.json"), "w", encoding="utf-8") as f:
            json.dump({"prompts": [{"id": "p1", "text": "x"}]}, f)
        with open(os.path.join(d, "p1.ids"), "w", encoding="ascii", newline="\n") as f:
            f.write("".join(f"{i}\n" for i in ids))

        def manifest(sha, status="committed", rung="G2"):
            with open(os.path.join(d, "manifest.tsv"), "w", encoding="utf-8", newline="\n") as f:
                f.write("\t".join(MANIFEST_COLS) + "\n")
                f.write("\t".join([rung, "p1 ids", "p1.ids", "tok", "ids", "-", status, sha]) + "\n")

        sink = io.StringIO()
        manifest(sha256_ids(ids))
        expect(check(d, out=sink), "valid fixture rejected")
        manifest("0" * 64)
        expect(not check(d, out=sink), "sha mismatch accepted")
        manifest(sha256_ids(ids), rung="G9")
        expect(not check(d, out=sink), "unknown rung accepted")
        with open(os.path.join(d, "p1.ids"), "w", encoding="ascii", newline="\n") as f:
            f.write(f"{VOCAB}\n")
        manifest(sha256_ids([VOCAB]))
        expect(not check(d, out=sink), "out-of-vocabulary id accepted")
        with open(os.path.join(d, "p1.ids"), "w", encoding="ascii", newline="\n") as f:
            f.write("".join(f"{i}\n" for i in range(MAX_PROMPT + 1)))
        manifest(sha256_ids(list(range(MAX_PROMPT + 1))))
        expect(not check(d, out=sink), "prompt longer than the G2 window accepted")
        with open(os.path.join(d, "p1.ids"), "w", encoding="ascii", newline="\n") as f:
            f.write("1 2\n")
        manifest(sha256_ids([1, 2]))
        expect(not check(d, out=sink), "malformed ids file accepted")
        with open(os.path.join(d, "p1.ids"), "w", encoding="ascii", newline="\n") as f:
            f.write("".join(f"{i}\n" for i in ids))
        manifest(sha256_ids(ids))
        n = export(d, os.path.join(d, "x"))
        with open(os.path.join(d, "x", "p1.u32"), "rb") as f:
            expect(n == 1 and list(struct.unpack("<3I", f.read())) == ids, "export u32 differs")

    g = ",".join(str(i) for i in range(GREEDY))
    line = "hy4 identity prompt=en n_prompt=5 greedy={} first_logits_fnv64=0123456789abcdef class={}\n"
    ref = line.format(g, "exact")
    sink = io.StringIO()
    expect(identity(ref, ref, out=sink), "identical exact lines rejected")
    expect(not identity(ref, "", out=sink), "missing engine line accepted")
    expect(not identity(ref, ref.replace("cdef", "cdee"), out=sink), "exact class with a different fnv accepted")
    bad = ",".join(str(i) for i in range(GREEDY - 1)) + ",999"
    sink = io.StringIO()
    expect(not identity(ref, line.format(bad, "exact"), out=sink), "diverging ids accepted")
    expect(f"first diff at {GREEDY - 1}" in sink.getvalue(), "first divergence index not reported")
    fast = line.format(g, "fast:a8").replace("cdef", "0000")
    expect(identity(ref, fast, want_class="fast", out=io.StringIO()), "declared fast class with equal ids rejected")
    expect(not identity(ref, fast, want_class="exact", out=io.StringIO()), "fast class accepted as exact")
    expect(not identity(ref, line.format(g, "exact"), want_class="fast", out=io.StringIO()),
           "undeclared class accepted on a fast gate")
    for f in fails:
        print(f"self-test FAIL {f}")
    print(f"check_fixtures self-test {'PASS' if not fails else 'FAIL'} ({len(fails)} failures)")
    return not fails


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--fixtures", default=DEFAULT_FIXTURES)
    c.add_argument("--tokenize-cmd", help="'BIN PACK encode' (one string)")
    c.add_argument("--tokenizer", help="strix-tokenize binary; with --pack equals --tokenize-cmd 'BIN PACK encode'")
    c.add_argument("--pack")
    e = sub.add_parser("export")
    e.add_argument("--fixtures", default=DEFAULT_FIXTURES)
    e.add_argument("--out", required=True)
    i = sub.add_parser("identity")
    i.add_argument("--ref", required=True)
    i.add_argument("--got", required=True)
    i.add_argument("--rung", default="G2", choices=RUNGS)
    i.add_argument("--class", dest="cls", default="exact", choices=("exact", "fast"))
    sub.add_parser("self-test")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "check":
            cmd = [a.tokenizer, a.pack, "encode"] if a.tokenizer and a.pack else a.tokenize_cmd
            return 0 if check(a.fixtures, cmd) else 1
        if a.cmd == "export":
            print(f"export {export(a.fixtures, a.out)} prompts -> {a.out}")
            return 0
        if a.cmd == "identity":
            with open(a.ref, encoding="utf-8") as f:
                ref = f.read()
            with open(a.got, encoding="utf-8", errors="replace") as f:
                got = f.read()
            return 0 if identity(ref, got, a.rung, a.cls) else 1
        return 0 if self_test() else 1
    except (FixtureError, OSError, ValueError, KeyError) as err:
        print(f"check_fixtures: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
