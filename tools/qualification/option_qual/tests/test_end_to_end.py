import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.qualification.option_qual import analyze, common, evidence, manifest, plan, run
from tools.qualification.longctx_fidelity.tok import FakeTokenizer
from tools.qualification.option_qual.tests.test_core import _sources

# A stand-in for the decode command: deterministic logits that depend on the recent history, plus
# optional noise (FAKE_NOISE) for an "option" arm; honours the measurement hook variables.
FAKE = r'''
import json, math, os, random, struct, sys
args = sys.argv[1:]
if "--print-effective-config" in args:
    print(json.dumps({"fake": True, "noise": os.environ.get("FAKE_NOISE", "0")})); sys.exit(0)
g = args.index("--generate")
ids = [int(x) for x in args[1:g]]
n = int(args[g + 1])
forced = [int(x) for x in open(os.environ["STRIX_BITCALC_FORCE_TOKENS"]).read().replace(",", " ").split()]
if os.environ.get("FAKE_TRUNCATE"):
    n = n // 2
if os.environ.get("FAKE_ONE_BYTE"):
    open(os.environ["STRIX_BITCALC_LOGITS_OUT"], "wb").write(b"x")
    print(json.dumps({"generated_tokens": [], "bitcalc_forced_steps": n}))
    sys.exit(0)
if os.environ.get("FAKE_IGNORE_FORCE"):
    forced = []
noise = float(os.environ.get("FAKE_NOISE", "0"))
if os.environ.get("FAKE_NOISE_FILE") and os.path.exists(os.environ["FAKE_NOISE_FILE"]):
    noise = float(open(os.environ["FAKE_NOISE_FILE"]).read())
vocab = 256
hist = list(ids)
final = os.environ["STRIX_BITCALC_LOGITS_OUT"]
if os.path.lexists(final) or os.path.lexists(final + ".partial"):
    sys.stderr.write("capture destination is not fresh\n"); sys.exit(5)
out = open(final + ".partial", "xb")
emitted = []
for step in range(n):
    key = (hist[-1] * 31 + hist[-2] * 7 + len(hist)) % 100003
    r = random.Random(key)
    z = [2.0 * r.gauss(0, 1) for _ in range(vocab)]
    z[(hist[-1] + 1) % vocab] += 4.0
    if noise:
        r2 = random.Random(key + 7919)
        z = [v + noise * r2.gauss(0, 1) for v in z]
    if os.environ.get("FAKE_NAN") or (os.environ.get("FAKE_NAN_REF") and step == 0):
        z[3] = math.nan
    top = max(range(vocab), key=lambda i: (z[i], -i))
    tok = forced[step] if step < len(forced) else top
    out.write(struct.pack("<6I", 0x42434C47, step, len(hist) - 1, top, tok, vocab) + struct.pack("<%df" % vocab, *z))
    emitted.append(tok)
    hist.append(tok)
out.close()
os.rename(final + ".partial", final)
sys.stderr.write("Qwen auto-prefill: automatic T8 context=%d; reason=test%s\n" % (len(ids), os.environ.get("FAKE_ROUTE", "")))
steps = n if os.environ.get("FAKE_CLAIM_FORCED") else min(n, len(forced))
if os.environ.get("FAKE_FORCED_TEXT_FILE") and os.path.exists(os.environ["FAKE_FORCED_TEXT_FILE"]):
    steps = str(steps)
print(json.dumps({"generated_tokens": emitted, "bitcalc_forced_steps": steps}))
'''


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.dir = Path(self.tmp.name)
        src = _sources(d / "src", parents=6)
        manifest.build(src, d / "m", FakeTokenizer(), [256, 512], cont=24, salt="e2e", search_per_stratum=2)
        (d / "fake.py").write_text(FAKE)
        (d / "model.gguf").write_bytes(b"model")
        (d / "receipt").write_text("receipt")
        arms = {"REF": {}, "CTRL": {"env": {"FAKE_NOISE": "0.02"}}, "A": {"env": {"FAKE_NOISE": "0.005"}},
                "B": {"env": {"FAKE_NOISE": "1.5"}}, "AB": {"env": {"FAKE_NOISE": "1.5"}}}
        (d / "arms.json").write_text(json.dumps({"arms": arms}))

    def tearDown(self):
        self.tmp.cleanup()

    def _plan(self, split, frozen=True, name="plan.json"):
        pl = plan.init(self.dir / "m", split, "REF", "CTRL", ["A", "B"], [{"name": "AB", "members": ["A", "B"]}],
                       [256, 512], bootstrap=500, min_parents=2,
                       repeat_cases=["en-00.L256"] if split == "search" else ["en-02.L256", "en-05.L256"])
        pl["endpoints"]["E2"] = {"fixed": 0.01}
        pl["endpoints"]["E3"] = {"fixed": 0.01}
        pl["endpoints"]["E4"] = {"fixed": 0.05}
        if frozen:
            pl = plan.freeze(pl)
        plan.save(self.dir / name, pl)
        return self.dir / name

    def _run(self, plan_path, out, extra=()):
        d = self.dir
        return run.main(["--manifest", str(d / "m"), "--plan", str(plan_path), "--arms", str(d / "arms.json"),
                         "--binary", str(d / "fake.py"), "--model", str(d / "model.gguf"), "--model-receipt",
                         str(d / "receipt"), "--out", str(out), "--soft-seconds", "600", "--allow-fake-for-tests",
                         "--python-binary-for-tests", *extra])

    def test_holdout_verdicts(self):
        p = self._plan("holdout")
        out = self.dir / "out"
        self.assertEqual(self._run(p, out), 0)
        self.assertFalse(list((out / "logits").glob("*.bin")))          # logits removed after reduction
        self.assertTrue((out / "holdout-ledger.jsonl").exists())
        ctx, results, _, means = analyze.analyze(self.dir / "m", p, [out])
        self.assertTrue(ctx["qualification"])
        self.assertEqual(results["A"]["verdict"], "PASS", results["A"])
        self.assertEqual(results["B"]["verdict"], "FAIL")
        self.assertEqual(results["AB"]["verdict"], "FAIL")
        self.assertEqual(ctx["k"], 3)
        s = analyze.write(self.dir / "rc", ctx, results, None, means)
        self.assertEqual(json.loads((self.dir / "rc" / "A.receipt.json").read_text())["verdict"], "PASS")
        self.assertIn("A: PASS", (self.dir / "rc" / "A.receipt.md").read_text())
        self.assertEqual(s["verdicts"][0]["option"], "A")
        # rerun is a no-op (resumable)
        self.assertEqual(self._run(p, out), 0)

        # a run stamped with another plan sha makes everything INVALID
        runs = out / "runs.jsonl"
        lines = runs.read_text().splitlines()
        rec = json.loads(lines[0])
        rec["plan_sha256"] = "0" * 64
        runs.write_text("\n".join([json.dumps(rec)] + lines[1:]) + "\n")
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "INVALID")
        runs.write_text("\n".join(lines) + "\n")

        # a missing reduction makes the option INCOMPLETE
        victim = next((out / "done").glob("*.A"))
        victim.unlink()
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "INCOMPLETE")

    def test_provenance_control_and_forcing_gates(self):
        p = self._plan("holdout")
        out = self.dir / "g"
        self.assertEqual(self._run(p, out), 0)
        # control results missing for one stratum: no option may pass on the remaining parents
        for marker in (out / "done").glob("code-*.CTRL"):
            marker.unlink()
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertNotEqual(results["A"]["verdict"], "PASS")
        # results without provenance are invalid
        out2 = self.dir / "g2"
        self.assertEqual(self._run(p, out2), 0)
        (out2 / "runs.jsonl").unlink()
        (out2 / "run-meta.json").unlink()
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out2])
        self.assertEqual(results["A"]["verdict"], "INVALID")
        # an arm that ignores teacher forcing (even when it claims otherwise) never completes
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["A"]["env"].update(FAKE_IGNORE_FORCE="1", FAKE_CLAIM_FORCED="1")
        (self.dir / "arms.json").write_text(json.dumps(arms))
        out3 = self.dir / "g3"
        self.assertEqual(self._run(p, out3), 1)
        self.assertEqual(self._run(p, out3, ["--max-attempts", "2"]), 1)
        rec = [json.loads(l) for l in (out3 / "runs.jsonl").read_text().splitlines() if '"arm": "A"' in l]
        self.assertTrue(rec and all(r["problem"] and not r["complete"] for r in rec))
        self.assertFalse(list((out3 / "logits").glob("*.A.bin")))
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out3])
        self.assertEqual(results["A"]["verdict"], "FAIL")
        self.assertEqual(results["B"]["verdict"], "FAIL")       # other options were still run and judged
        # resuming with a changed arms file is refused
        self.assertEqual(self._run(p, out), 1)

    def test_declared_routes(self):
        pl = plan.init(self.dir / "m", "search", "REF", "CTRL", ["A", "B"], [], [256], bootstrap=200, min_parents=2,
                       routes={"A": "T8 context", "B": "native-route"}, endpoints=["E1"])
        plan.save(self.dir / "r.json", pl)
        out = self.dir / "r"
        self.assertEqual(self._run(self.dir / "r.json", out), 0)
        _, results, _, _ = analyze.analyze(self.dir / "m", self.dir / "r.json", [out])
        self.assertFalse(any("route" in n for n in results["A"]["notes"]))
        self.assertTrue(any("declared route" in n for n in results["B"]["notes"]))
        self.assertEqual(results["B"]["verdict"], "FAIL")

    def test_merging_and_borrowed_provenance_are_invalid(self):
        p = self._plan("holdout")
        a, b = self.dir / "ma", self.dir / "mb"
        self.assertEqual(self._run(p, a), 0)
        self.assertEqual(self._run(p, b), 0)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [a])
        self.assertEqual(results["A"]["verdict"], "PASS")
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [a, b])     # the same cases twice
        self.assertEqual(results["A"]["verdict"], "INVALID")
        # provenance of one directory cannot be supplied by another
        (a / "run-meta.json").unlink()
        meta_only = self.dir / "meta-only"
        meta_only.mkdir()
        (meta_only / "run-meta.json").write_text((b / "run-meta.json").read_text())
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [a, meta_only])
        self.assertEqual(results["A"]["verdict"], "INVALID")
        # a reduction not bound to the candidate file of a validated run is invalid
        red = next((b / "red").glob("*.A.json"))
        info = json.loads(red.read_text())
        info["cand_sha256"] = "0" * 64
        red.write_text(json.dumps(info))
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [b])
        self.assertEqual(results["A"]["verdict"], "INVALID")

    def test_swapped_arrays_short_files_and_repeat_resume(self):
        p = self._plan("holdout")
        out = self.dir / "sw"
        self.assertEqual(self._run(p, out), 0)
        # metric arrays of one arm swapped in for another: never a valid result
        for a_npz in (out / "red").glob("*.A.npz"):
            target = Path(str(a_npz).replace(".A.npz", ".B.npz"))
            target.write_bytes(a_npz.read_bytes())
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["B"]["verdict"], "INVALID")
        # an INVALID reference repeat keeps the exit code at 1 on resume
        rep = next((out / "done").glob("*.REF@repeat"))
        mark = json.loads(rep.read_text())
        rep.write_text(json.dumps(dict(mark, status="INVALID", error="injected")))
        self.assertEqual(self._run(p, out), 1)
        self.assertEqual(len((out / "run-meta.json").read_text().splitlines()), 2)   # the resume really ran
        # a one-byte logits file is a recorded failure of that pair only
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["B"]["env"]["FAKE_ONE_BYTE"] = "1"
        (self.dir / "arms.json").write_text(json.dumps(arms))
        out2 = self.dir / "ob"
        self.assertEqual(self._run(p, out2), 1)
        marks = {q.name: json.loads(q.read_text()) for q in (out2 / "done").iterdir()}
        self.assertTrue(marks and all(v["status"] == "FAILED" for k, v in marks.items() if k.endswith(".B")))
        self.assertTrue(all(v["status"] == "COMPLETE" for k, v in marks.items() if k.endswith(".AB")))

    def test_route_from_run_records_and_retained_reference_repair(self):
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["A"]["env"]["FAKE_ROUTE"] = "other"
        (self.dir / "arms.json").write_text(json.dumps(arms))
        p = self._plan("holdout")
        out = self.dir / "rt"
        self.assertEqual(self._run(p, out, ["--keep-logits"]), 0)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "FAIL")
        # editing the route of a done marker, or of the run record, or dropping it: never a PASS
        ref_route = json.loads(next((out / "done").glob("*.REF")).read_text())["route"]
        saved = {m: m.read_text() for m in (out / "done").glob("*.A")}
        for marker in saved:
            marker.write_text(json.dumps(dict(json.loads(marker.read_text()), route=ref_route)))
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertIn(results["A"]["verdict"], ("FAIL", "INVALID"))
        for marker, text in saved.items():
            marker.write_text(text)
        runs = out / "runs.jsonl"
        original = runs.read_text()
        recs = [json.loads(l) for l in original.splitlines()]
        runs.write_text("".join(json.dumps(dict(r, route=ref_route) if r["arm"] == "A" else r) + "\n" for r in recs))
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertIn(results["A"]["verdict"], ("FAIL", "INVALID"))
        runs.write_text("".join(json.dumps({k: v for k, v in r.items() if k != "route"}) + "\n" for r in recs))
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "INVALID")
        # an interrupted append: analysis reports it, the runner quarantines it and the campaign resumes
        runs.write_text(original + '{"case_id": "x", "ar')
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "INVALID")
        self._run(p, out, ["--keep-logits"])
        self.assertIn("terminated_tail_bytes", (out / "recovery.jsonl").read_text())   # kept, never cut
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "FAIL")
        # a retained reference file that contradicts its record makes the case run again
        runs_before = len((out / "runs.jsonl").read_text().splitlines())
        victim = sorted((out / "logits").glob("*.REF.bin"))[0]
        data = bytearray(victim.read_bytes())
        data[-1] ^= 1
        victim.write_bytes(bytes(data))
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["B"]["verdict"], "INVALID")
        self._run(p, out, ["--keep-logits"])
        self.assertGreater(len((out / "runs.jsonl").read_text().splitlines()), runs_before)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["B"]["verdict"], "FAIL")      # valid again, same verdict as before

    def test_metadata_recovery_keeps_ids_unique(self):
        p = self._plan("holdout")
        out = self.dir / "md"
        self.assertEqual(self._run(p, out), 0)
        # an interrupted first metadata append: recovered, resumable, no crash
        meta = out / "run-meta.json"
        whole = meta.read_text()
        meta.write_text(whole[:len(whole) // 2])
        self.assertEqual(self._run(p, out), 0)
        # a lost metadata append while run records survive; another arm must be produced again
        meta.write_text(meta.read_text()[:-5])
        victim = next((out / "done").glob("*.A"))
        victim.unlink()
        self.assertEqual(self._run(p, out), 0)
        ids = [json.loads(l)["run_id"] for l in (out / "runs.jsonl").read_text().splitlines()]
        self.assertEqual(len(ids), len(set(ids)))
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "PASS", results["A"]["notes"][:3])

    def test_damaged_or_foreign_markers_recover(self):
        p = self._plan("holdout")
        out = self.dir / "mk"
        self.assertEqual(self._run(p, out), 0)
        markers = sorted((out / "done").glob("*.A"))
        # a half-transferred marker: analysis reports it, resume quarantines it and redoes the pair
        text = markers[0].read_text()
        markers[0].write_text(text[:len(text) // 2])
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "INVALID")
        self.assertEqual(self._run(p, out), 0)
        self.assertTrue(list((out / "quarantine").iterdir()))
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "PASS")
        # a failure marker copied from another case: never honoured
        foreign = dict(json.loads(markers[1].read_text()), status="FAILED", attempts=2, max_attempts=2)
        markers[0].write_text(json.dumps(foreign))
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "INVALID")
        self.assertEqual(self._run(p, out), 0)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "PASS")

    def test_resume_applies_the_analysis_rules(self):
        p = self._plan("holdout")
        out = self.dir / "rr"
        self.assertEqual(self._run(p, out), 0)
        # a completion marker with another plan sha, or its run record turned into a failed run:
        # analysis rejects it, and resume repairs it instead of trusting it
        marker = sorted((out / "done").glob("*.A"))[0]
        marker.write_text(json.dumps(dict(json.loads(marker.read_text()), plan_sha256="0" * 64)))
        runs = out / "runs.jsonl"
        recs = [json.loads(l) for l in runs.read_text().splitlines()]
        target = json.loads(sorted((out / "done").glob("*.A"))[1].read_text())["run_id"]
        runs.write_text("".join(json.dumps(dict(r, rc=1) if r["run_id"] == target else r) + "\n" for r in recs))
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "INVALID")
        before = len(runs.read_text().splitlines())
        self.assertEqual(self._run(p, out), 0)
        self.assertGreater(len(runs.read_text().splitlines()), before)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "PASS")
        # a stale INVALID marker (made against another reference) is not honoured
        stale = sorted((out / "done").glob("*.A"))[2]
        good = json.loads(stale.read_text())
        stale.write_text(json.dumps(dict(good, status="INVALID", ref_sha256="1" * 64, error="old")))
        self.assertEqual(self._run(p, out), 0)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "PASS")

    def test_damaged_failure_marker_keeps_the_retry_count(self):
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["B"]["env"]["FAKE_TRUNCATE"] = "1"
        (self.dir / "arms.json").write_text(json.dumps(arms))
        p = self._plan("holdout")
        out = self.dir / "fc"
        case = "en-02.L256"
        self.assertEqual(self._run(p, out, ["--cases", case]), 1)                 # B fails once
        marker = out / "done" / f"{case}.B"
        text = marker.read_text()
        marker.write_text(text[:len(text) // 2])                               # damaged in transfer
        self.assertEqual(self._run(p, out, ["--cases", case]), 1)                 # second failure
        b_runs = lambda: sum(1 for l in (out / "runs.jsonl").read_text().splitlines()
                             if json.loads(l)["arm"] == "B")
        self.assertEqual(json.loads(marker.read_text())["attempts"], 2)
        n = b_runs()
        self.assertEqual(self._run(p, out, ["--cases", case]), 1)                 # terminal: no third run
        self.assertEqual(b_runs(), n)

    def test_stale_success_and_journal_repair(self):
        knob = self.dir / "noise.txt"
        knob.write_text("0.005")
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["A"] = {"env": {"FAKE_NOISE_FILE": str(knob)}}
        (self.dir / "arms.json").write_text(json.dumps(arms))
        p = self._plan("holdout")
        out = self.dir / "st"
        self.assertEqual(self._run(p, out), 0)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "PASS")
        # keep one case's successful A artifacts, then redo that case with much worse output
        cases = sorted(m.name[:-2] for m in (out / "done").glob("*.A"))
        files = {}
        for case in cases:
            files[case] = {q: q.read_bytes() for q in [out / "done" / f"{case}.A"] +
                           [out / "red" / f"{case}.A{x}" for x in (".npz", ".json")]}
            (out / "done" / f"{case}.A").unlink()
        knob.write_text("1.5")
        self.assertEqual(self._run(p, out), 0)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "FAIL")
        for saved in files.values():                       # restore the older, favourable artifacts
            for q, data in saved.items():
                q.write_bytes(data)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertNotEqual(results["A"]["verdict"], "PASS")
        self.assertEqual(self._run(p, out), 0)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "FAIL")
        # a malformed historical run record: quarantined on resume, the campaign heals
        runs = out / "runs.jsonl"
        recs = [json.loads(l) for l in runs.read_text().splitlines()]
        target = json.loads(next((out / "done").glob("*.B")).read_text())["run_id"]
        runs.write_text("".join(json.dumps({k: v for k, v in r.items() if not (r["run_id"] == target and k == "rc")})
                                + "\n" for r in recs))
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["B"]["verdict"], "INVALID")
        self.assertEqual(self._run(p, out), 0)
        self.assertTrue((out / "resets.jsonl").exists())              # the case was reset, journal untouched
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["B"]["verdict"], "FAIL")
        self.assertFalse(any("INVALID" in n for n in results["B"]["notes"]))
        # restoring the old favourable A artifacts while damaging the newer record never resurrects them
        for saved in files.values():
            for q, data in saved.items():
                q.write_bytes(data)
        recs = [json.loads(l) for l in runs.read_text().splitlines()]
        newest_a = [r["run_id"] for r in recs if r.get("arm") == "A"][-1]
        runs.write_text("".join(json.dumps(dict(r, rc="x") if r.get("run_id") == newest_a else r) + "\n"
                                for r in recs))
        for _ in range(2):
            self._run(p, out)
            _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
            self.assertNotEqual(results["A"]["verdict"], "PASS")
        # a run record without its timing field cannot crash a resume
        recs = [json.loads(l) for l in runs.read_text().splitlines()]
        first_ok = next(r["run_id"] for r in recs if r.get("complete") and "event" not in r)
        runs.write_text("".join(json.dumps({k: v for k, v in r.items() if not (r.get("run_id") == first_ok
                                                                                and k == "wall_seconds")})
                                + "\n" for r in recs))
        self.assertIn(self._run(p, out), (0, 1))

    def test_unterminated_tail_foreign_resets_and_huge_numbers(self):
        knob = self.dir / "noise.txt"
        knob.write_text("0.005")
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["A"] = {"env": {"FAKE_NOISE_FILE": str(knob)}}
        (self.dir / "arms.json").write_text(json.dumps(arms))
        p = self._plan("holdout")
        out = self.dir / "tl"
        case = "en-02.L256"
        self.assertEqual(self._run(p, out, ["--cases", case]), 0)
        saved = {q: q.read_bytes() for q in [out / "done" / f"{case}.A", out / "red" / f"{case}.A.npz",
                                            out / "red" / f"{case}.A.json"]}
        (out / "done" / f"{case}.A").unlink()
        knob.write_text("1.5")
        self.assertEqual(self._run(p, out, ["--cases", case]), 0)          # newer, much worse A run
        for q, data in saved.items():                                      # older favourable files back
            q.write_bytes(data)
        runs = out / "runs.jsonl"
        data = runs.read_bytes()
        newest_a = [json.loads(l) for l in data.decode().splitlines() if json.loads(l).get("arm") == "A"][-1]
        if data.decode().splitlines()[-1] != json.dumps(newest_a, sort_keys=True):
            runs.write_bytes(data + (json.dumps(newest_a, sort_keys=True)).encode())   # newest A unterminated
        else:
            runs.write_bytes(data[:-1])
        for _ in range(2):
            self._run(p, out, ["--cases", case])
            _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
            self.assertNotEqual(results["A"]["verdict"], "PASS")
            ids = [json.loads(l)["run_id"] for l in runs.read_text().splitlines() if l.strip().startswith("{")]
            self.assertIn(newest_a["run_id"], ids)                         # the journal never loses a line
        # a reset journal from another campaign is not honoured and gets repaired
        resets = out / "resets.jsonl"
        bogus = {"case_id": "*", "upto": 10_000, "line_sha256": "0" * 64, "invocation": "x"}
        with open(resets, "a") as f:
            f.write(json.dumps(bogus) + "\n")
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "INVALID")
        self._run(p, out, ["--cases", case])
        self.assertTrue(list((out / "quarantine").glob("resets.jsonl.*")))
        # an absurd number never crashes the validators
        recs = [l for l in runs.read_text().splitlines() if l.strip()]
        first = json.loads(recs[0])
        first["wall_seconds"] = 10 ** 400
        runs.write_text("\n".join([json.dumps(first)] + recs[1:]) + "\n")
        analyze.analyze(self.dir / "m", p, [out])
        self.assertIn(self._run(p, out, ["--cases", case]), (0, 1))

    def test_dangling_events_reset_and_prefix_bound_resets(self):
        p = self._plan("holdout")
        out = self.dir / "ev"
        case = "en-02.L256"
        self.assertEqual(self._run(p, out, ["--cases", case]), 0)
        runs = out / "runs.jsonl"
        some = json.loads(runs.read_text().splitlines()[0])
        event = {"event": "reduction_failed", "run_id": "forged-e1", "of_run": "no-such-run", "case_id": case,
                 "arm": "A", "error": "forged", **{k: some[k] for k in ("plan_sha256", "binary_sha256",
                                                                          "arms_sha256", "model_receipt_sha256")}}
        with open(runs, "a") as f:
            f.write(json.dumps(event) + "\n")
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "INVALID")              # dangling event: never ignored
        self.assertEqual(self._run(p, out, ["--cases", case]), 0)        # its case is reset and redone
        self.assertTrue((out / "resets.jsonl").exists())
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertFalse(any("INVALID" in n for n in results["A"]["notes"]))
        # changing any line covered by a reset makes that reset invalid (prefix-bound, not endpoint-bound)
        lines = runs.read_text().splitlines()
        first = json.loads(lines[0])
        first["wall_seconds"] = first["wall_seconds"] + 1.0
        runs.write_text("\n".join([json.dumps(first, sort_keys=True)] + lines[1:]) + "\n")
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "INVALID")

    def test_retry_ceiling_survives_a_case_reset(self):
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["B"]["env"]["FAKE_TRUNCATE"] = "1"
        (self.dir / "arms.json").write_text(json.dumps(arms))
        p = self._plan("holdout")
        out = self.dir / "rc"
        case = "en-02.L256"
        def distinct_b():
            ids = set()
            for line in (out / "runs.jsonl").read_text().splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("arm") == "B" and "event" not in rec:
                    ids.add(rec["run_id"])
            return len(ids)
        self._run(p, out, ["--cases", case])
        self._run(p, out, ["--cases", case])
        self.assertEqual(distinct_b(), 2)                                   # limit reached
        with open(out / "runs.jsonl", "a") as f:                            # malformed line of this case
            f.write('{"case_id": "%s", "broken": ' % case + "\n")
        for _ in range(2):
            self._run(p, out, ["--cases", case])                            # reset, then resumes
        self.assertTrue((out / "resets.jsonl").exists())
        self.assertEqual(distinct_b(), 2)                                   # never a third attempt
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertNotEqual(results["B"]["verdict"], "PASS")

    def test_failure_recorded_before_its_marker_and_event_lineage(self):
        p = self._plan("holdout")
        out = self.dir / "fl"
        case = "en-02.L256"
        self.assertEqual(self._run(p, out, ["--cases", case]), 0)
        runs = out / "runs.jsonl"
        mark = json.loads((out / "done" / f"{case}.A").read_text())
        some = json.loads(runs.read_text().splitlines()[0])
        ident = {k: some[k] for k in ("plan_sha256", "binary_sha256", "arms_sha256", "model_receipt_sha256")}
        # a failure journaled for A's run, interrupted before the marker changed
        with open(runs, "a") as f:
            f.write(json.dumps({"event": "reduction_failed", "run_id": "x-e1", "of_run": mark["run_id"],
                                "case_id": case, "arm": "A", "error": "interrupted", **ident}) + "\n")
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertNotEqual(results["A"]["verdict"], "PASS")              # a failed run is never a result
        self.assertEqual(self._run(p, out, ["--cases", case]), 0)         # marker settled, one retry, success
        a_runs = [json.loads(l) for l in runs.read_text().splitlines() if l.startswith("{")]
        self.assertEqual(sum(1 for r in a_runs if r.get("arm") == "A" and "event" not in r), 2)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        # only this case ran (others are simply missing): nothing about it is invalid or failed
        self.assertFalse([n for n in results["A"]["notes"] if case in n])
        # an event naming an older run of its pair (not the latest preceding one) resets the case
        older = [r for r in a_runs if r.get("arm") == "A" and "event" not in r][0]["run_id"]
        with open(runs, "a") as f:
            f.write(json.dumps({"event": "reduction_failed", "run_id": "x-e2", "of_run": older,
                                "case_id": case, "arm": "A", "error": "stale target", **ident}) + "\n")
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(results["A"]["verdict"], "INVALID")
        self._run(p, out, ["--cases", case])
        self.assertTrue((out / "resets.jsonl").exists())

    def test_malformed_engine_count_is_a_counted_failure(self):
        knob = self.dir / "text.flag"
        knob.write_text("1")
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["A"]["env"]["FAKE_FORCED_TEXT_FILE"] = str(knob)
        (self.dir / "arms.json").write_text(json.dumps(arms))
        p = self._plan("holdout")
        out = self.dir / "mf"
        case = "en-02.L256"
        for _ in range(3):
            self._run(p, out, ["--cases", case])
        runs = [json.loads(l) for l in (out / "runs.jsonl").read_text().splitlines()]
        a_runs = [r for r in runs if r.get("arm") == "A" and "event" not in r]
        self.assertEqual(len(a_runs), 2)                                   # the ceiling holds
        for r in a_runs:
            self.assertTrue(evidence.run_record(r))                        # schema-valid evidence
            self.assertIsNone(r["forced_steps"])
            self.assertEqual(r["forced_steps_raw"], repr(str(24)))
        self.assertFalse((out / "resets.jsonl").exists())
        knob.unlink()                                                      # the condition goes away
        self._run(p, out, ["--cases", case])
        runs = [json.loads(l) for l in (out / "runs.jsonl").read_text().splitlines()]
        self.assertEqual(sum(1 for r in runs if r.get("arm") == "A" and "event" not in r), 2)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertNotEqual(results["A"]["verdict"], "PASS")

    def test_interrupted_validation_is_a_budget_stop(self):
        from unittest import mock
        p = self._plan("holdout")
        out = self.dir / "iv"
        case = "en-02.L256"
        with mock.patch.object(run.Runner, "validate", lambda self, *a: None):
            self.assertEqual(self._run(p, out, ["--cases", case]), 2)
        runs = [json.loads(l) for l in (out / "runs.jsonl").read_text().splitlines()]
        self.assertTrue(runs and all(r["interrupted"] and not r["complete"] for r in runs))
        self.assertFalse(list((out / "logits").glob("*.bin")))            # no unvalidated file kept
        self.assertEqual(self._run(p, out, ["--cases", case]), 0)
        led = evidence.Ledger([json.loads(l) for l in (out / "runs.jsonl").read_text().splitlines()], [],
                              {k: runs[0][k] for k in evidence.RUN_IDENTITY})
        self.assertEqual(led.failed(case, runs[0]["arm"]), [])

    def test_hashing_retained_logits_stops_at_the_deadline(self):
        from unittest import mock
        p = self._plan("holdout")
        out = self.dir / "hs"
        case = "en-02.L256"
        self.assertEqual(self._run(p, out, ["--cases", case, "--keep-logits"]), 0)
        self.assertTrue(list((out / "logits").glob("*.bin")))
        marks = {q.name: q.read_bytes() for q in (out / "done").iterdir()}

        def stop(self, path):
            raise run.BudgetStop(str(path))
        with mock.patch.object(run.Runner, "logits_sha", stop):
            self.assertEqual(self._run(p, out, ["--cases", case, "--keep-logits"]), 2)
        self.assertEqual({q.name: q.read_bytes() for q in (out / "done").iterdir()}, marks)   # nothing judged
        self.assertFalse((out / "resets.jsonl").exists())
        self.assertEqual(self._run(p, out, ["--cases", case, "--keep-logits"]), 0)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertFalse([n for n in results["A"]["notes"] if case in n])

    def test_damaged_diagnostic_journals_never_block_a_resume(self):
        p = self._plan("holdout")
        out = self.dir / "dd"
        case = "en-02.L256"
        self.assertEqual(self._run(p, out, ["--cases", case]), 0)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        before = (out / "runs.jsonl").read_text()
        verdicts = {k: v["verdict"] for k, v in results.items()}
        for name in ("effective-config.jsonl", "recovery.jsonl", "holdout-ledger.jsonl"):
            path = out / name
            old = path.read_text() if path.exists() else ""
            path.write_text('{"arm": "A", "attem' + old + '{"arm": "B", "rc": 0}\n{"partial')
        for extra in ([], ["--no-effective-config"]):
            self.assertEqual(self._run(p, out, ["--cases", case, *extra]), 0)
        self.assertEqual((out / "runs.jsonl").read_text(), before)          # nothing re-run
        context, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual({k: v["verdict"] for k, v in results.items()}, verdicts)
        self.assertGreater(context["effective_config_damaged_lines"], 0)
        # the damage is a boundary: captures written after it are reused, not repeated forever
        self.assertEqual(self._run(p, out, ["--cases", case]), 0)
        recs = common.diagnostic_records(out / "effective-config.jsonl", after_damage=True)[0]
        last = {r["arm"]: r for r in recs if isinstance(r.get("arm"), str)}
        self.assertTrue(last and all("reused_from" in r for r in last.values()))
        # a torn last line alone is reported too
        (out / "effective-config.jsonl").write_bytes(b'{"partial')
        context, _, _, _ = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(context["effective_config_damaged_lines"], 1)

    def test_fast_parity_decides_and_exact_class_is_reported(self):
        pl = plan.init(self.dir / "m", "holdout", "REF", "CTRL", ["A", "B"], [], [256, 512], bootstrap=500,
                       min_parents=2, repeat_cases=["en-02.L256"], tier="fast-parity", secondary_tier="exact-class")
        pl["secondary"]["endpoints"]["E2"] = {"fixed": 0.01}
        pl["secondary"]["endpoints"]["E3"] = {"fixed": 0.01}
        pl = plan.freeze(pl)
        p = self.dir / "fast.json"
        plan.save(p, pl)
        out = self.dir / "fp"
        self.assertEqual(self._run(p, out), 0)
        context, results, ctrl_report, ctrl_means = analyze.analyze(self.dir / "m", p, [out])
        self.assertEqual(sorted(results["A"]["endpoints"]), ["E1", "E5", "E6"])
        self.assertEqual(results["A"]["secondary"]["tier"], "exact-class")
        self.assertEqual(sorted(results["A"]["secondary"]["endpoints"]), ["E2", "E3", "E4", "E7"])
        self.assertEqual(results["B"]["verdict"], "FAIL")                    # large noise fails parity too
        receipts = self.dir / "fr"
        analyze.write(receipts, context, results, ctrl_report, ctrl_means)
        md = (receipts / "A.receipt.md").read_text()
        self.assertIn("Deciding margins: fast-parity", md)
        rec_a = json.loads((receipts / "A.receipt.json").read_text())
        self.assertIsNone(rec_a["statement"])                                 # short tiers carry none
        self.assertIn("statement", rec_a["secondary"])
        self.assertIn("Also reported (does not decide the verdict): exact-class", md)
        rec = json.loads((receipts / "A.receipt.json").read_text())
        self.assertEqual(rec["verdict"], results["A"]["verdict"])
        self.assertEqual(rec["secondary"]["verdict"], results["A"]["secondary"]["verdict"])

    def test_secondary_shares_the_hard_gates(self):
        pl = plan.init(self.dir / "m", "holdout", "REF", "CTRL", ["A", "B"], [], [256, 512], bootstrap=500,
                       min_parents=2, repeat_cases=["en-02.L256"], tier="fast-parity", secondary_tier="exact-class")
        for spec in pl["secondary"]["endpoints"].values():
            spec.pop("rule", None)
            spec.setdefault("fixed", 0.01)
        p = self.dir / "sg.json"
        plan.save(p, plan.freeze(pl))
        out = self.dir / "sg"
        # partial coverage: neither set may PASS
        some = sorted(c["case_id"] for c in manifest.load(self.dir / "m")[0]["cases"] if c["split"] == "holdout")[:4]
        self._run(p, out, ["--cases", ",".join(some)])
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out])
        for opt in ("A", "B"):
            self.assertNotEqual(results[opt]["secondary"]["verdict"], "PASS")
            self.assertNotEqual(results[opt]["verdict"], "PASS")
        # a shared hard-gate failure (undeclared route) reaches the secondary verdict too
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["A"]["env"]["FAKE_ROUTE"] = "other"
        (self.dir / "arms.json").write_text(json.dumps(arms))
        out2 = self.dir / "sg2"
        self._run(p, out2)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out2])
        self.assertEqual(results["A"]["verdict"], "FAIL")
        self.assertEqual(results["A"]["secondary"]["verdict"], "FAIL")
        self.assertTrue(any("route" in n for n in results["A"]["secondary"]["notes"]))

    def test_exact_class_decides_on_nll_excess_over_control(self):
        pl = plan.init(self.dir / "m", "holdout", "REF", "CTRL", ["A", "B"], [], [256, 512], bootstrap=500,
                       min_parents=2, repeat_cases=["en-02.L256"], secondary_tier="reference-nll")
        self.assertEqual(sorted(pl["endpoints"]), ["E2", "E3", "E4", "E7"])
        self.assertEqual(pl["guards"]["G1"]["endpoint"], "E7")
        self.assertEqual(pl["secondary"]["guards"]["G1"]["endpoint"], "E1")
        pl["endpoints"]["E2"] = {"fixed": 0.01}
        pl["endpoints"]["E3"] = {"fixed": 0.01}
        p = self.dir / "e7.json"
        plan.save(p, plan.freeze(pl))
        out = self.dir / "e7"
        self.assertEqual(self._run(p, out), 0)
        _, results, ctrl_report, _ = analyze.analyze(self.dir / "m", p, [out])
        e7 = results["A"]["endpoints"]["E7"]["estimate"]
        e1 = results["A"]["secondary"]["endpoints"]["E1"]["estimate"]
        # E7 is the option's NLL change minus the control's, per parent (same parents in both)
        self.assertAlmostEqual(e7, e1 - ctrl_report["mean_parent"]["dnll"], places=12)
        self.assertNotAlmostEqual(e7, e1, places=12)
        self.assertIn("rounding-sensitive", results["A"]["secondary"]["statement"])
        self.assertIsNone(results["A"]["statement"])

    def test_clipped_control_e7_with_equal_weights(self):
        import glob as _g
        from collections import defaultdict

        def run_plan(name, clip):
            pl = plan.init(self.dir / "m", "holdout", "REF", "CTRL", ["A", "B"], [], [256, 512], bootstrap=300,
                           min_parents=2, repeat_cases=["en-02.L256"], e7_control_clip=clip, stratum_weights="equal")
            pl["endpoints"]["E2"] = {"fixed": 0.01}
            pl["endpoints"]["E3"] = {"fixed": 0.01}
            p = self.dir / f"{name}.json"
            plan.save(p, plan.freeze(pl))
            out = self.dir / name
            self.assertEqual(self._run(p, out), 0)
            return analyze.analyze(self.dir / "m", p, [out])[1], out

        def manual(out, clip):
            """Mean of stratum means of per-parent means of per-case (A dNLL - clip(CTRL dNLL))."""
            per = defaultdict(list)
            for f in sorted(_g.glob(str(out / "red" / "*.A.npz"))):
                case = Path(f).name[: -len(".A.npz")]
                a, c = np.load(f), np.load(f.replace(".A.npz", ".CTRL.npz"))
                cd = float(np.mean(c["nll_c"] - c["nll_r"]))
                per[case.rsplit(".L", 1)[0]].append(float(np.mean(a["nll_c"] - a["nll_r"]))
                                                    - (cd if clip is None else float(np.clip(cd, -clip, clip))))
            by_s = defaultdict(list)
            for parent, vals in per.items():
                by_s[parent.split("-")[0]].append(float(np.mean(vals)))
            return float(np.mean([np.mean(x) for x in by_s.values()]))

        big, out_big = run_plan("clipbig", 1e9)
        e7 = big["A"]["endpoints"]["E7"]
        self.assertEqual(e7["weights"], "equal")
        self.assertIn("control clipped", e7["label"])
        self.assertAlmostEqual(e7["estimate"], manual(out_big, None), places=12)   # a clip that never binds
        small, out_small = run_plan("clipsmall", 0.001)
        self.assertAlmostEqual(small["A"]["endpoints"]["E7"]["estimate"], manual(out_small, 0.001), places=12)

    def test_stale_partial_capture_is_cleared_before_a_run(self):
        p = self._plan("holdout")
        out = self.dir / "sp"
        case = "en-02.L256"
        (out / "logits").mkdir(parents=True)
        for arm in ("REF", "CTRL", "A", "B"):
            (out / "logits" / f"{case}.{arm}.bin.partial").write_bytes(b"left by a killed run")
        self.assertEqual(self._run(p, out, ["--cases", case]), 0)
        self.assertFalse(list((out / "logits").glob("*.partial")))

    def test_feasibility_checked_only_for_engine_work(self):
        p = self._plan("holdout")
        out = self.dir / "fz"
        case = "en-02.L256"
        self.assertEqual(self._run(p, out, ["--cases", case]), 0)
        # a finished case resumes under an engine prediction no invocation could fit
        self.assertEqual(self._run(p, out, ["--cases", case, "--predict", "5000"]), 0)
        # engine work that can never fit is an error, not a budget stop
        other = next(c["case_id"] for c in manifest.load(self.dir / "m")[0]["cases"]
                     if c["split"] == "holdout" and c["nominal_length"] == 256 and c["case_id"] != case)
        before = (out / "runs.jsonl").read_text()
        self.assertEqual(self._run(p, out, ["--cases", other]), 0)          # the selection itself is valid
        self.assertEqual(self._run(p, out, ["--cases", other, "--predict", "5000"]), 0)   # now finished
        out2 = self.dir / "fz2"
        self.assertEqual(self._run(p, out2, ["--cases", other, "--predict", "5000"]), 1)
        self.assertFalse((out2 / "runs.jsonl").exists() and (out2 / "runs.jsonl").read_text().strip())
        self.assertTrue(before)

    def test_duplicated_or_lost_outcomes_do_not_change_the_retry_count(self):
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["B"]["env"]["FAKE_TRUNCATE"] = "1"
        (self.dir / "arms.json").write_text(json.dumps(arms))
        p = self._plan("holdout")
        out = self.dir / "oc"
        case = "en-02.L256"
        b_runs = lambda: sum(1 for l in (out / "runs.jsonl").read_text().splitlines()
                             if json.loads(l)["arm"] == "B")
        self.assertEqual(self._run(p, out, ["--cases", case]), 1)        # one failure
        runs = out / "runs.jsonl"
        failed_line = next(l for l in runs.read_text().splitlines() if json.loads(l)["arm"] == "B")
        with open(runs, "a") as f:
            f.write((failed_line + "\n") * 2)                            # identical duplicated lines
        self.assertEqual(self._run(p, out, ["--cases", case]), 1)        # still gets its second attempt
        distinct_b = lambda: len({json.loads(l)["run_id"] for l in runs.read_text().splitlines()
                                  if json.loads(l)["arm"] == "B"})
        self.assertEqual(distinct_b(), 2)
        self.assertEqual(self._run(p, out, ["--cases", case]), 1)        # limit reached: no third run
        self.assertEqual(distinct_b(), 2)

    def test_nonfinite_reference_receipt_is_written(self):
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["REF"] = {"env": {"FAKE_NAN_REF": "1"}}
        (self.dir / "arms.json").write_text(json.dumps(arms))
        p = self._plan("holdout")
        out = self.dir / "nr"
        self._run(p, out)
        ctx, results, ctrl, means = analyze.analyze(self.dir / "m", p, [out])
        self.assertIn(results["A"]["verdict"], ("FAIL", "INVALID"))
        analyze.write(self.dir / "nrr", ctx, results, ctrl, means)
        self.assertTrue((self.dir / "nrr" / "A.receipt.json").exists())

    def test_failures_do_not_block_and_calibration_binding(self):
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["B"]["env"]["FAKE_TRUNCATE"] = "1"
        (self.dir / "arms.json").write_text(json.dumps(arms))
        sp = self._plan("search", frozen=False, name="s.json")
        out = self.dir / "fb"
        self.assertEqual(self._run(sp, out), 1)                 # B fails, everything else runs
        marks = {p.name: json.loads(p.read_text()) for p in (out / "done").iterdir()}
        self.assertTrue(all(v["status"] == "FAILED" for k, v in marks.items() if k.endswith(".B")))
        self.assertTrue(all(v["status"] == "COMPLETE" for k, v in marks.items() if k.endswith(".A")))
        ctx, results, ctrl, means = analyze.analyze(self.dir / "m", sp, [out])
        self.assertEqual(results["A"]["verdict"], "PASS")
        self.assertNotEqual(results["B"]["verdict"], "PASS")
        s = analyze.write(self.dir / "rs", ctx, results, ctrl, means)
        self.assertIsNotNone(s["control_means"])
        hp = plan.init(self.dir / "m", "holdout", "REF", "CTRL", ["A", "B"], [{"name": "AB", "members": ["A", "B"]}],
                       [256, 512], bootstrap=500, min_parents=2)
        hp = plan.freeze(plan.calibrate(hp, s))
        plan.save(self.dir / "h.json", hp)
        # the holdout must use the control the margins were calibrated with
        arms["arms"]["CTRL"]["env"]["FAKE_NOISE"] = "0.5"
        (self.dir / "arms.json").write_text(json.dumps(arms))
        out2 = self.dir / "fh"
        self._run(self.dir / "h.json", out2)
        _, results, _, _ = analyze.analyze(self.dir / "m", self.dir / "h.json", [out2])
        self.assertEqual(results["A"]["verdict"], "INVALID")
        self.assertTrue(any("calibrated" in n for n in results["A"]["notes"]))

    def test_holdout_refuses_unfrozen_plan(self):
        p = self._plan("holdout", frozen=False, name="open.json")
        self.assertEqual(self._run(p, self.dir / "out2"), 1)
        self.assertFalse((self.dir / "out2" / "runs.jsonl").exists())

    def test_search_calibration_and_nonfinite(self):
        p = self._plan("search", frozen=False, name="search.json")
        out = self.dir / "s"
        self.assertEqual(self._run(p, out), 0)
        ctx, results, ctrl, means = analyze.analyze(self.dir / "m", p, [out])
        self.assertFalse(ctx["qualification"])
        s = analyze.write(self.dir / "src", ctx, results, ctrl, means)
        self.assertTrue(json.loads((self.dir / "src" / "A.receipt.json").read_text())["verdict"].startswith("SEARCH-"))
        hp = plan.init(self.dir / "m", "holdout", "REF", "CTRL", ["A"], [], [256, 512])
        cal = plan.calibrate(hp, s)
        self.assertGreaterEqual(cal["endpoints"]["E2"]["fixed"], 0.002)
        self.assertAlmostEqual(cal["endpoints"]["E2"]["calibration"]["control_mean"], means["loss"])
        # non-finite logits in an arm fail that arm
        arms = json.loads((self.dir / "arms.json").read_text())
        arms["arms"]["A"]["env"]["FAKE_NAN"] = "1"
        (self.dir / "arms.json").write_text(json.dumps(arms))
        out2 = self.dir / "s2"
        m, _ = manifest.load(self.dir / "m")
        search = [c["case_id"] for c in m["cases"] if c["split"] == "search"]
        self.assertEqual(self._run(p, out2, ["--cases", ",".join(search)]), 0)
        _, results, _, _ = analyze.analyze(self.dir / "m", p, [out2])
        self.assertEqual(results["A"]["verdict"], "FAIL")
        self.assertTrue(any("non-finite" in n for n in results["A"]["notes"]))


if __name__ == "__main__":
    unittest.main()
