"""Synthetic CPU tests of fast_logit_gate.py (numpy only; no engine)."""
import contextlib
import io
import json
import os
import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fast_logit_gate as g  # noqa: E402

ROWS, WIDTHS, TOKENS = 8, (5, 3), 40


def run(argv):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return g.main(argv)


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).resolve().parents[2])
        self.root = pathlib.Path(self.tmp.name)
        self.rng = np.random.default_rng(7)
        self.tok = self.root / "p.u32"
        self.rng.integers(0, sum(WIDTHS), TOKENS, dtype=np.uint32).astype("<u4").tofile(self.tok)
        self.binary = self.root / "engine"
        self.binary.write_bytes(b"binary")
        self.logits = self.rng.normal(size=(ROWS, sum(WIDTHS))).astype("<f4")

    def tearDown(self):
        self.tmp.cleanup()

    def selection(self, path, mode, swap=False):
        words = []
        for layer in range(2):
            ids = np.tile(np.arange(4, dtype=np.uint32), (ROWS, 2, 1))
            if swap and layer == 1:
                ids[3, 1, 2] = 9
            ids[:, :, 3] = 77 + mode  # beyond the count: ignored
            words += [g.SEL_MAGIC, 1, mode, 16, 0, ROWS, 2, 4] + [3] * (ROWS * 2) + ids.ravel().tolist()
        np.array(words, "<u4").tofile(path)

    def arm(self, name, run_id, logits=None, mode=4, swap=False, seal=True, select=True, witness=None):
        d = self.root / name
        d.mkdir()
        z = self.logits if logits is None else logits
        rc = []
        for rank, (lo, hi) in enumerate(((0, WIDTHS[0]), (WIDTHS[0], sum(WIDTHS)))):
            z[:, lo:hi].astype("<f4").tofile(d / f"q-p-tf-r{rank}.f32")
            if select:
                self.selection(d / f"q-p-sel-r{rank}.u32", mode, swap and rank == 1)
            if seal:
                evidence = []
                if witness:
                    files = {}
                    for kind, ext in [("tf", "f32")] + ([("sel", "u32")] if select else []):
                        f = d / f"q-p-{kind}-r{rank}.{ext}"
                        files[kind] = {"name": f.name, "bytes": f.stat().st_size, "sha256": g.sha256_file(f)}
                        files[kind].update({"width": hi - lo} if kind == "tf" else {"records": 2, "mode": mode})
                    w = {"schema": "strix-fast-execution/v1", "run_id": run_id, "rank": rank,
                         "binary_sha256": g.sha256_file(self.binary), "engine": "ds41",
                         "profile": "fast-prefill" if witness == "candidate" else "exact",
                         "model_sha256": "a" * 64, "numerical_config_sha256": ("c" if witness == "candidate" else "b") * 64,
                         "control": {"kind": "layer0-null", "layer": 0, "magnitude": 1, "unit": "ulp"} if witness == "null" else None,
                         "prompts": {"p": {"tokens": TOKENS, "tokens_sha256": g.sha256_file(self.tok), "rows": ROWS, "files": files}}}
                    wp = d / f"execution-r{rank}.json"
                    wp.write_text(json.dumps(w), encoding="utf-8")
                    evidence = ["--execution-witness", str(wp)]
                rc.append(run(["seal", "--dir", str(d), "--rank", str(rank), "--arm", name, "--run-id", run_id, "--binary",
                               str(self.binary), "--rows", str(ROWS), "--prompt", f"p={self.tok}", "--env", "MODE=x",
                               "--attest", f"sel_mode={mode}"] + (["--select"] if select else []) + evidence))
        return d, rc

    def cmp(self, a, b, *extra):
        return run(["compare", str(a), str(b), "--ranks", "0", "1", "--tokens", f"p={self.tok}", *extra])

    def test_identical_repeat_passes(self):
        a, rc = self.arm("A", "r1")
        b, _ = self.arm("A2", "r2")
        self.assertEqual(rc, [0, 0])
        self.assertEqual(self.cmp(a, b, "--expect-identical"), 0)
        self.assertEqual(self.cmp(a, b), 0)

    def test_small_shift_passes_and_large_fails(self):
        a, _ = self.arm("A", "r1")
        near, _ = self.arm("N", "r2", self.logits + self.rng.normal(scale=1e-3, size=self.logits.shape).astype("<f4"), mode=3)
        far, _ = self.arm("F", "r3", -self.logits, mode=3)
        self.assertEqual(self.cmp(a, near), 0)
        self.assertEqual(self.cmp(a, near, "--expect-identical"), 1)
        self.assertEqual(self.cmp(a, far), 1)

    def test_selection_difference_is_counted(self):
        a, _ = self.arm("A", "r1")
        b, _ = self.arm("X", "r2", mode=3, swap=True)
        out = self.root / "r.json"
        self.assertEqual(self.cmp(a, b, "--json", str(out)), 0)
        self.assertEqual(self.cmp(a, b, "--expect-identical"), 1)
        import json
        sel = json.loads(out.read_text())["prompts"]["p"]["selection"]
        self.assertEqual((sel["layers"], sel["slot_diff"], sel["id_diff"], sel["first_layer"], sel["first_pos"]), (4, 1, 1, 3, 19))
        self.assertEqual(sel["modes"], [4, 3])

    def test_identity_defects_have_no_verdict(self):
        a, _ = self.arm("A", "r1")
        same, _ = self.arm("S", "r1")
        self.assertEqual(self.cmp(a, same), 2)  # one run id on both sides
        unsealed, _ = self.arm("U", "r2", seal=False)
        self.assertEqual(self.cmp(a, unsealed), 2)
        b, _ = self.arm("B", "r3")
        with open(b / "q-p-tf-r1.f32", "r+b") as f:  # changed after sealing
            f.write(b"\0\0\0\0")
        self.assertEqual(self.cmp(a, b), 2)

    def test_seal_rejects_missing_short_stale_reused_and_wrong_mode(self):
        d, _ = self.arm("A", "r1", seal=False)
        base = ["seal", "--dir", str(d), "--arm", "A", "--run-id", "r1", "--binary", str(self.binary), "--rows", str(ROWS),
                "--prompt", f"p={self.tok}", "--select"]
        self.assertEqual(run(base + ["--rank", "0", "--attest", "sel_mode=3"]), 2)  # dump says mode 4
        self.assertEqual(run(base + ["--rank", "0", "--not-before", str(os.path.getmtime(d / "q-p-tf-r0.f32") + 60)]), 2)
        self.assertEqual(run(base + ["--rank", "0"]), 0)
        self.assertEqual(run(base + ["--rank", "0"]), 2)  # sealed twice
        os.remove(d / "q-p-sel-r1.u32")
        self.assertEqual(run(base + ["--rank", "1"]), 2)
        self.selection(d / "q-p-sel-r1.u32", 4)
        with open(d / "q-p-tf-r1.f32", "ab") as f:
            f.write(b"\0\0")
        self.assertEqual(run(base + ["--rank", "1"]), 2)
        bad = self.logits.copy()
        bad[0, 0] = np.nan
        e, rc = self.arm("E", "r9", bad)
        self.assertEqual(rc, [2, 0])

    # ---- floor-relative dNLL rule (DS41 FAST prefill only)
    def noisy(self, scale, seed):
        rng = np.random.default_rng(seed)
        return self.logits + rng.normal(scale=scale, size=self.logits.shape).astype("<f4")

    def floor_args(self, nulls, engine="ds41", profile="fast-prefill"):
        out = ["--dnll-rule", "floor", "--engine", engine, "--profile", profile]
        for n in nulls:
            out += ["--null-arm", str(n)]
        return out

    def test_dsq3_numbers(self):
        nulls = [0.01408, 0.01140, 0.01111]
        mean, sd, limit = g.floor_limit(nulls)
        self.assertAlmostEqual(mean, 0.01220, places=5)
        self.assertAlmostEqual(sd, 0.00164, places=5)
        self.assertAlmostEqual(limit, 0.01547, places=5)
        for dnll, ok, abs_ok in ((0.01203, True, False), (0.01458, True, False), (0.01600, False, False), (0.00485, True, True)):
            v = g.dnll_verdict(dnll, "floor", 0.01, nulls)
            self.assertEqual((v["pass"], v["abs_pass"], v["n_nulls"], v["sigmas"]), (ok, abs_ok, 3, 2.0), dnll)
        v = g.dnll_verdict(0.01203, "abs", 0.01)
        self.assertEqual((v["pass"], v["limit"]), (False, 0.01))
        with self.assertRaises(g.GateError):
            g.floor_limit([0.014, 0.011])
        with self.assertRaises(g.GateError):
            g.floor_limit([0.014, 0.011, float("nan")])

    def test_policy_table(self):
        self.assertEqual(g.allowed_dnll_rules("ds41", "fast-prefill"), ("abs", "floor"))
        for e, p in (("ds41", "exact"), ("qwen", "fast-prefill"), ("minimax", "fast"), ("glm", "fast-prefill")):
            self.assertEqual(g.allowed_dnll_rules(e, p), ("abs",))
            with self.assertRaises(g.GateError):
                g.check_dnll_policy("floor", e, p, 3)
        with self.assertRaises(g.GateError):
            g.check_dnll_policy("floor", "ds41", "fast-prefill", 2)
        with self.assertRaises(g.GateError):
            g.check_dnll_policy("floor", None, None, 3)
        with self.assertRaises(g.GateError):
            g.check_dnll_policy("abs", None, None, 1)
        g.check_dnll_policy("floor", "ds41", "fast-prefill", 3)

    def test_floor_end_to_end(self):
        import json
        a, _ = self.arm("A", "r1", witness="baseline")
        nulls = [self.arm(f"N{i}", f"n{i}", self.noisy(0.3, 10 + i), mode=3, witness="null")[0] for i in range(3)]
        cand, _ = self.arm("C", "c1", self.noisy(0.3, 20), mode=3, witness="candidate")
        out = self.root / "f.json"
        rc = self.cmp(a, cand, *self.floor_args(nulls), "--json", str(out))
        p = json.loads(out.read_text())["pooled"]
        d = p["dnll_rule"]
        vals = [n["dnll"] for n in d["nulls"]]
        self.assertEqual([n["run_id"] for n in d["nulls"]], ["n0", "n1", "n2"])
        self.assertEqual(d["nulls"][0]["execution"]["control"], g.DS41_NULL_CONTROL)
        self.assertAlmostEqual(d["mean"], float(np.mean(vals)), places=12)
        self.assertAlmostEqual(d["sd"], float(np.std(vals, ddof=1)), places=12)
        self.assertAlmostEqual(d["limit"], d["mean"] + 2 * d["sd"], places=12)
        self.assertEqual((d["rule"], d["engine"], d["profile"], d["abs_limit"]), ("floor", "ds41", "fast-prefill", 0.01))
        self.assertEqual(d["abs_pass"], p["dnll"] <= 0.01)
        want = p["kl"] <= 0.02 and p["top1"] >= 0.93 and p["dnll"] <= d["limit"]
        self.assertEqual((rc, p["pass"]), (0 if want else 1, want))
        self.assertIn("floor ds41/fast-prefill", p["rule"])

    def test_floor_refusals_have_no_verdict(self):
        a, _ = self.arm("A", "r1", witness="baseline")
        nulls = [self.arm(f"N{i}", f"n{i}", self.noisy(1e-3, 10 + i), mode=3, witness="null")[0] for i in range(3)]
        cand, _ = self.arm("C", "c1", self.noisy(1e-3, 20), mode=3, witness="candidate")
        self.assertEqual(self.cmp(a, cand, *self.floor_args(nulls[:2])), 2)  # two nulls
        self.assertEqual(self.cmp(a, cand, *self.floor_args([])), 2)  # no nulls
        self.assertEqual(self.cmp(a, cand, *self.floor_args(nulls, "qwen")), 2)  # engine not in the policy table
        self.assertEqual(self.cmp(a, cand, *self.floor_args(nulls, "ds41", "exact")), 2)  # profile not in the table
        self.assertEqual(self.cmp(a, cand, "--dnll-rule", "floor", *sum((["--null-arm", str(n)] for n in nulls), [])), 2)
        self.assertEqual(self.cmp(a, cand, "--null-arm", str(nulls[0])), 2)  # nulls without the floor rule
        self.assertEqual(self.cmp(a, cand, *self.floor_args(nulls), "--expect-identical"), 2)
        self.assertEqual(self.cmp(a, cand, *self.floor_args([nulls[0], nulls[1], nulls[0]])), 2)  # one null twice
        self.assertEqual(self.cmp(a, cand, *self.floor_args([nulls[0], nulls[1], cand])), 2)  # candidate as a null
        other = self.root / "engine2"
        other.write_bytes(b"other binary")
        self.binary, keep = other, self.binary
        foreign, _ = self.arm("NB", "nb", self.noisy(1e-3, 30), mode=3, witness="null")
        self.binary = keep
        self.assertEqual(self.cmp(a, cand, *self.floor_args([nulls[0], nulls[1], foreign])), 2)  # null on another binary
        self.assertIn(self.cmp(a, cand, *self.floor_args(nulls)), (0, 1))  # the valid call reaches a verdict

    def floor_fixture(self, witness="null"):
        # Depress only the next-token logit; the winning alternative stays unchanged.
        ref = np.zeros_like(self.logits)
        labels = np.concatenate((np.fromfile(self.tok, "<u4")[-ROWS + 1:], [0]))
        ref[np.arange(ROWS), (labels + 1) % ref.shape[1]] = 1

        def shifted(dnll):
            z = ref.copy()
            mass = np.exp(1) + z.shape[1] - 1
            z[np.arange(ROWS), labels] = -np.log((mass * np.exp(dnll) - 1) / (mass - 1))
            return z

        a, rc = self.arm("A", "ref", ref, witness="baseline", select=False)
        self.assertEqual(rc, [0, 0])
        cand, rc = self.arm("C", "cand", shifted(0.016), witness="candidate", select=False)
        self.assertEqual(rc, [0, 0])
        nulls = []
        for i, dnll in enumerate((0.04, 0.05, 0.06)):
            d, rc = self.arm(f"layer0-null-{i}", f"null-{i}", shifted(dnll), witness=witness, select=False)
            self.assertEqual(rc, [0, 0])
            nulls.append(d)
        return a, cand, nulls

    def test_floor_rejects_inflated_non_controls_despite_names(self):
        a, cand, nulls = self.floor_fixture(witness="baseline")
        out = self.root / "refused.json"
        self.assertEqual(self.cmp(a, cand, *self.floor_args(nulls), "--json", str(out)), 2)
        self.assertFalse(out.exists())

    def test_floor_rejects_missing_control_witness(self):
        a, cand, nulls = self.floor_fixture(witness=None)
        self.assertEqual(self.cmp(a, cand, *self.floor_args(nulls)), 2)

    def test_floor_witness_incompatibilities(self):
        a, cand, nulls = self.floor_fixture()
        for target, key, value in ((nulls[0], "control", None),
                                   (nulls[0], "control", {**g.DS41_NULL_CONTROL, "layer": 1}),
                                   (nulls[0], "control", {**g.DS41_NULL_CONTROL, "magnitude": 2}),
                                   (nulls[0], "model_sha256", "d" * 64),
                                   (nulls[0], "numerical_config_sha256", "d" * 64),
                                   (cand, "engine", "qwen"), (cand, "profile", "exact"),
                                   (a, "control", g.DS41_NULL_CONTROL)):
            with self.subTest(key=key, value=value):
                paths = [target / f"manifest-r{r}.json" for r in range(2)]
                originals = [p.read_text(encoding="utf-8") for p in paths]
                for p, original in zip(paths, originals):
                    doc = json.loads(original)
                    doc["execution"][key] = value
                    p.write_text(json.dumps(doc), encoding="utf-8")
                self.assertEqual(self.cmp(a, cand, *self.floor_args(nulls)), 2)
                for p, original in zip(paths, originals):
                    p.write_text(original, encoding="utf-8")

    def test_floor_rejects_rank_witness_disagreement(self):
        a, cand, nulls = self.floor_fixture()
        p = nulls[0] / "manifest-r1.json"
        original = json.loads(p.read_text(encoding="utf-8"))
        for value in (None, {**original["execution"], "numerical_config_sha256": "d" * 64}):
            doc = dict(original)
            if value is None:
                del doc["execution"]
            else:
                doc["execution"] = value
            p.write_text(json.dumps(doc), encoding="utf-8")
            self.assertEqual(self.cmp(a, cand, *self.floor_args(nulls)), 2)

    def test_seal_binds_execution_witness(self):
        d, rc = self.arm("A", "r1", witness="null", select=False)
        self.assertEqual(rc, [0, 0])
        p = d / "execution-r0.json"
        original = json.loads(p.read_text(encoding="utf-8"))
        (d / "manifest-r0.json").unlink()
        base = ["seal", "--dir", str(d), "--rank", "0", "--arm", "A", "--run-id", "r1", "--binary", str(self.binary),
                "--rows", str(ROWS), "--prompt", f"p={self.tok}", "--execution-witness", str(p)]
        for key, value in (("run_id", "other"), ("rank", 1), ("binary_sha256", "d" * 64),
                           ("prompts", {}), ("model_sha256", ""), ("control", "layer0-null")):
            with self.subTest(key=key):
                p.write_text(json.dumps({**original, key: value}), encoding="utf-8")
                self.assertEqual(run(base), 2)
                self.assertFalse((d / "manifest-r0.json").exists())
        p.write_text("{", encoding="utf-8")
        self.assertEqual(run(base), 2)
        p.write_text(json.dumps(original), encoding="utf-8")
        self.assertEqual(run(base), 0)

    def test_default_rule_unchanged(self):
        import json
        a, _ = self.arm("A", "r1")
        for i, scale in enumerate((1e-3, 0.3, 1.0)):
            cand, _ = self.arm(f"C{i}", f"c{i}", self.noisy(scale, 40 + i), mode=3)
            out = self.root / f"d{i}.json"
            rc = self.cmp(a, cand, "--json", str(out))
            p = json.loads(out.read_text())["pooled"]
            want = p["kl"] <= 0.02 and p["top1"] >= 0.93 and p["dnll"] <= 0.01
            self.assertEqual((rc, p["pass"]), (0 if want else 1, want))
            self.assertEqual(p["rule"], "KL <= 0.02, top-1 >= 0.93, dNLL <= 0.01")
            self.assertEqual((p["dnll_rule"]["rule"], p["dnll_rule"]["limit"], p["dnll_rule"]["nulls"]), ("abs", 0.01, []))


if __name__ == "__main__":
    unittest.main()
