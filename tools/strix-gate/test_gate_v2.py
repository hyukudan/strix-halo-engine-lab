"""Host tests of the FAST gate v2 statistics (gate_v2.py) and its two adapters (numpy only; no engine)."""
import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gate_v2 as v2  # noqa: E402
import test_fast_logit_gate as tfg  # noqa: E402


def write_routes(path, calls):
    """calls: [(layer, ids[rows, k], weights[rows, k])] in the engine's strix-route-file/v1 layout."""
    with open(path, "wb") as f:
        f.write(b"STXROUT1" + np.array([1, 0], "<u4").tobytes())
        for layer, ids, w in calls:
            ids, w = np.asarray(ids, "<u4"), np.asarray(w, "<f4")
            f.write(np.array([layer, ids.shape[0], ids.shape[1], 0], "<u4").tobytes() + ids.tobytes() + w.tobytes())


def rows_with_mean(n, mean, scale, rng):
    d = np.full(n, mean) + rng.normal(scale=scale, size=n)
    return d - (d.mean() - mean)  # exact mean


def window(name, n, kl, top1, dnll, null_kl, rng, hit=0.5, scale=0.05, null_top1=None, null_dnll=None, null_scale=None,
           ref_nll=1.0, cls=None, null_kl_rows=None):
    agree = np.arange(n) < round(top1 * n)
    d = rows_with_mean(n, dnll, scale, rng)
    nll_ref = np.full(n, ref_nll)
    hits = np.arange(n) < round(hit * n)
    null_nlls = None
    if null_dnll is not None:
        null_nlls = [nll_ref + rows_with_mean(n, null_dnll, null_scale or scale, rng)]
    return v2.window_stats(name, np.full(n, kl), agree, nll_ref, nll_ref + d, [null_kl], hits, hits,
                           null_top1s=None if null_top1 is None else [null_top1], null_nlls=null_nlls, cls=cls,
                           null_kl_rows=null_kl_rows)


class Statistics(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)

    def test_repeated_window_cannot_meet_the_row_minimum(self):
        w = window("one", 400, 0.001, 0.99, 0.0, 0.004, self.rng)
        with self.assertRaisesRegex(v2.GateError, "duplicate window"):
            v2.verdict([w, w])

    def test_window_refuses_invalid_or_unpaired_evidence(self):
        for cand, null in [([1.0], 0.004), ([1.0, np.nan], 0.004), ([1.0, 1.0], np.inf)]:
            with self.subTest(cand=cand, null=null), self.assertRaises(v2.GateError):
                v2.window_stats("bad", [0.01, 0.01], [True, True], [1.0, 1.0], cand, [null])

    def test_noisy_null_disagreement_is_bounded_by_one(self):
        w = window("noisy", 16, 0.001, 0.0, 0.0, 0.004, self.rng, null_top1=0.0)
        self.assertEqual(w["top1_pass"], 0.0)
        self.assertEqual(w["states"]["top1"], "pass")

    def test_null_equal_to_reference_is_refused(self):
        # GLM b8 (refx_moe): a one-f32-ulp residual flip under FP16 input staging left the null bitwise equal to the
        # reference; its KL 0 would floor the limit at 0.02 while the FP16-ulp null of the same window has KL 0.49.
        with self.assertRaises(v2.GateError):
            v2.null_moved("b8", [0.0])
        with self.assertRaises(v2.GateError):
            v2.null_moved("b8", [0.4945, 0.0])
        v2.null_moved("b8", [0.4945, 1e-9])

    def test_per_window_kl_limit_is_null_relative(self):
        noisy = window("ho7-01", 256, 0.040, 0.95, 0.0, 0.030, self.rng)   # chaotic window: limit 1.5 x 0.030 = 0.045
        self.assertAlmostEqual(noisy["kl_limit"], 0.045)
        self.assertTrue(noisy["pass"])
        quiet = window("ho7-00", 256, 0.040, 0.95, 0.0, 0.004, self.rng)   # quiet window: the absolute 0.02 stays
        self.assertEqual(quiet["kl_limit"], 0.02)
        self.assertFalse(quiet["pass"])
        self.assertFalse(quiet["checks"]["kl"])
        with self.assertRaises(v2.GateError):
            v2.window_stats("x", [0.0], [True], [1.0], [1.0], [])

    def test_top1_and_dnll_fail_per_window_when_significant(self):
        w = window("w", 256, 0.001, 0.85, 0.0, 0.5, self.rng)     # 15 % disagreement on 256 rows: limit 0.07 + 2 x 0.016
        self.assertEqual((w["checks"]["top1"], w["pass"]), (False, False))
        w = window("w", 256, 0.001, 0.99, 0.02, 0.5, self.rng)    # se 0.003: limit 0.01 + 0.006
        self.assertEqual((w["checks"]["dnll"], w["pass"]), (False, False))
        # the Qwen 32K receipt that failed (256 rows, dNLL +0.051 se 0.0045) still fails
        w = window("w", 256, 0.001, 0.99, 0.051, 0.5, self.rng, scale=0.072)
        self.assertAlmostEqual(w["dnll_se"], 0.0045, delta=0.001)
        self.assertFalse(w["checks"]["dnll"])

    def test_window_resolution_and_null_relative_top1(self):
        # DS41 D1 s8k-a: 63 rows, dNLL +0.030 se ~0.02 is inside the window's resolution (the one-ulp null failed this rule)
        w = window("s8k-a", 63, 0.0139, 0.9365, 0.030, 0.0117, self.rng, scale=0.157)
        self.assertAlmostEqual(w["dnll_se"], 0.0198, delta=0.005)
        self.assertGreater(w["dnll_limit"], 0.03)
        self.assertTrue(w["checks"]["dnll"])
        w = window("big", 63, 0.0139, 0.99, 0.10, 0.0117, self.rng, scale=0.157)   # a real +0.10 nat loss still fails
        self.assertFalse(w["checks"]["dnll"])
        # d32a-o16384: 10 of 63 flips where the null itself flips 5 of 63 (high-entropy window): L = 1.5 x 0.079
        w = window("o16384", 63, 0.0304, 0.8413, -0.016, 0.0293, self.rng, null_top1=0.9206)
        self.assertAlmostEqual(w["null_top1"], 0.9206)
        self.assertTrue(w["checks"]["top1"])
        w = window("o16384", 63, 0.0304, 0.8413, -0.016, 0.0293, self.rng)      # without the null's top-1: fails
        self.assertFalse(w["checks"]["top1"])
        w = window("bad", 63, 0.0304, 0.70, -0.016, 0.0293, self.rng, null_top1=0.9206)   # 30 % flips: fails
        self.assertFalse(w["checks"]["top1"])
        line = v2.window_line(window("o16384", 63, 0.0304, 0.8413, -0.016, 0.0293, self.rng, null_top1=0.9206))
        self.assertIn("(limit 0.7993, null 0.9206;", line)
        self.assertIn("UNRESOLVED top1", line)
        self.assertIn("null 0.9206", line)

    def test_pooled_rows_se_and_absolute_kl(self):
        ws = [window(f"w{i}", 256, 0.015, 0.97, 0.002, 0.001, self.rng) for i in range(3)]
        with self.assertRaises(v2.GateError):   # 768 < 800 rows: no verdict
            v2.verdict(ws)
        ws.append(window("w3", 64, 0.015, 0.97, 0.002, 0.001, self.rng))
        p = v2.verdict(ws)
        self.assertEqual(p["dnll_rows"], 832)
        d = np.concatenate([w["_rows"]["dnll"] for w in ws])
        self.assertAlmostEqual(p["dnll_se"], d.std(ddof=1) / np.sqrt(d.size))
        self.assertAlmostEqual(p["dnll"], 0.002)
        self.assertTrue(p["pass"])
        # A window passing under its own null limit still counts in the pooled KL, whose limit stays 0.02 while the
        # pooled null is measurably inside the bar (null KL 0.015 on every row: se 0).
        ws = [window(f"w{i}", 256, 0.021, 0.97, 0.0, 0.015, self.rng, null_kl_rows=np.full(256, 0.015)) for i in range(4)]
        p = v2.verdict(ws)
        self.assertTrue(p["checks"]["windows"])
        self.assertFalse(p["checks"]["kl"])
        self.assertFalse(p["null_floor"]["kl"]["engaged"])
        self.assertFalse(p["pass"])

    def test_null_floor_relaxes_only_bars_the_null_itself_fails(self):
        # GLM i4gate shape: 4 windows x 255 rows, pooled null top-1 0.9304 (binomial se 0.008: within 2 se of 0.93),
        # null KL 0.141 and null dNLL +0.052 above their bars. Limits: KL 0.161, top-1 0.9204, dNLL 0.062.
        def gate(top1, kl=0.13, dnll=0.005):
            ws = [window(f"w{i}", 255, kl, top1, dnll, 0.141, self.rng, null_top1=0.9304, null_dnll=0.052,
                         null_kl_rows=np.full(255, 0.141), scale=0.01) for i in range(4)]
            return v2.verdict(ws, allow_null_floor=True)
        p = gate(0.926)
        f = p["null_floor"]
        self.assertEqual(f["applied"], ["kl", "top1", "dnll"])
        self.assertAlmostEqual(f["kl"]["limit"], f["kl"]["null"] + 0.02)
        self.assertAlmostEqual(f["top1"]["limit"], f["top1"]["null"] - 0.01)
        self.assertAlmostEqual(f["dnll"]["limit"], f["dnll"]["null"] + 0.01)
        self.assertTrue(f["top1"]["null"] >= 0.93)          # the null passes the absolute top-1 bar by a hair ...
        self.assertTrue(f["top1"]["engaged"])               # ... but not by 2 se: the floor engages
        self.assertTrue(all(p["checks"][b] for b in ("kl", "top1", "dnll")))
        self.assertEqual(f["decisive"], ["kl", "top1"])     # dNLL 0.005 passes the absolute bar as well
        self.assertEqual(p["state"], "PASS")
        line = v2.pooled_line(p)
        self.assertIn("NULL-FLOOR", line)
        self.assertIn("pass ONLY under the floor: kl,top1", line)
        # a lever 0.032 below the null's top-1 (27-layer arm) fails the floor bar; one above null + 0.02 KL fails too
        self.assertFalse(gate(0.898)["checks"]["top1"])
        self.assertFalse(gate(0.926, kl=0.165)["checks"]["kl"])
        self.assertFalse(gate(0.926, dnll=0.07)["checks"]["dnll"])

    def test_null_floor_is_advisory_by_default(self):
        ws = [window(f"w{i}", 255, 0.13, 0.926, 0.005, 0.141, self.rng,
                     null_top1=0.9304, null_dnll=0.052, null_kl_rows=np.full(255, 0.141),
                     scale=0.01) for i in range(4)]
        strict = v2.verdict(ws)
        self.assertEqual(strict["state"], "FAIL")
        self.assertFalse(strict["null_floor"]["enabled"])
        self.assertFalse(strict["checks"]["kl"])
        self.assertFalse(strict["checks"]["top1"])
        self.assertTrue(strict["null_floor"]["kl"]["pass"])
        self.assertIn("ADVISORY ONLY", v2.pooled_line(strict))
        self.assertIn("<= 0.02", v2.pooled_line(strict))
        allowed = v2.verdict(ws, allow_null_floor=True)
        self.assertTrue(allowed["null_floor"]["enabled"])
        self.assertEqual(allowed["state"], "PASS")
        self.assertTrue(allowed["checks"]["kl"])
        self.assertTrue(allowed["checks"]["top1"])
        self.assertIn("enabled by explicit opt-in", v2.pooled_line(allowed))

    def test_null_floor_is_relax_only_and_off_for_a_clean_null(self):
        # engaged (null 0.9405 within 2 se of 0.93) but null - 0.01 = 0.9305 is above the bar: the bar stays 0.93
        ws = [window(f"w{i}", 255, 0.005, 0.931, 0.0, 0.004, self.rng, null_top1=0.9405, null_dnll=0.0)
              for i in range(4)]
        f = v2.verdict(ws)["null_floor"]
        self.assertTrue(f["top1"]["engaged"])
        self.assertFalse(f["top1"]["relaxed"])
        self.assertEqual(f["top1"]["limit"], 0.93)
        # a clean engine: every pooled null inside its bar, the floor is not applied and says so
        ws = [window(f"w{i}", 255, 0.025, 0.97, 0.0, 0.004, self.rng, null_top1=0.99, null_dnll=0.0,
                     null_kl_rows=np.full(255, 0.004)) for i in range(4)]
        p = v2.verdict(ws)
        self.assertEqual(p["null_floor"]["applied"], [])
        self.assertFalse(p["checks"]["kl"])                 # 0.025 > 0.02, the absolute bar
        self.assertIn("not applied", v2.pooled_line(p))
        # without the null's per-row KL the KL band is 0; without null NLL / top-1 those bars stay absolute
        ws = [window(f"w{i}", 255, 0.025, 0.97, 0.0, 0.019, self.rng) for i in range(4)]
        f = v2.verdict(ws)["null_floor"]
        self.assertFalse(f["kl"]["engaged"])
        self.assertIsNone(f["top1"]["null"])
        self.assertIsNone(f["dnll"]["null"])
        # a wide per-row null KL (mean 0.019, se ~0.008) is not measurably inside 0.02: engaged, limit 0.039
        rows = np.where(np.arange(255) % 10 == 0, 0.181, 0.001)
        ws = [window(f"w{i}", 255, 0.025, 0.97, 0.0, float(rows.mean()), self.rng, null_kl_rows=rows) for i in range(4)]
        f = v2.verdict(ws)["null_floor"]
        self.assertTrue(f["kl"]["engaged"])
        self.assertAlmostEqual(f["kl"]["limit"], float(rows.mean()) + 0.02)

    def test_null_floor_pools_the_mean_of_several_nulls(self):
        rng = self.rng
        ws = []
        for i in range(4):
            nll_ref = np.full(255, 1.0)
            ws.append(v2.window_stats(f"w{i}", np.full(255, 0.01), np.ones(255, bool), nll_ref, nll_ref + 0.012,
                                      [0.009, 0.011], null_top1s=[0.97, 0.98],
                                      null_nlls=[nll_ref + rows_with_mean(255, 0.014, 0.05, rng),
                                                 nll_ref + rows_with_mean(255, 0.010, 0.05, rng)]))
        f = v2.verdict(ws)["null_floor"]
        self.assertAlmostEqual(f["kl"]["null"], 0.010)
        self.assertAlmostEqual(f["top1"]["null"], 0.975)
        self.assertAlmostEqual(f["dnll"]["null"], 0.012)       # mean of both nulls, not the first one
        self.assertAlmostEqual(f["dnll"]["limit"], 0.022)
        self.assertEqual(f["applied"], ["dnll"])

    def test_null_floor_never_clears_a_failed_window(self):
        ws = [window(f"w{i}", 255, 0.12, 0.93, 0.0, 0.141, self.rng, null_top1=0.9304, null_dnll=0.05,
                     null_kl_rows=np.full(255, 0.141)) for i in range(3)]
        ws.append(window("bad", 255, 0.25, 0.93, 0.0, 0.141, self.rng, null_top1=0.9304, null_dnll=0.05,
                         null_kl_rows=np.full(255, 0.141)))   # 0.25 > 1.5 x 0.141 on its own window
        p = v2.verdict(ws, allow_null_floor=True)
        self.assertTrue(p["checks"]["kl"])                  # pooled 0.1525 <= 0.161 under the floor ...
        self.assertFalse(p["checks"]["windows"])            # ... the window still fails
        self.assertEqual(p["state"], "FAIL")

    def test_three_states_and_class_pooling(self):
        # review_rule A.3: a real loss on one content class (dNLL 0.03 = 3 x limit, 63-row windows, se ~0.02) is
        # UNRESOLVED per window, not PASS; the class pooled on >= 800 rows FAILs.
        w = window("c0", 63, 0.005, 0.97, 0.030, 0.004, self.rng, scale=0.157, cls="code")
        self.assertEqual((w["state"], w["states"]["dnll"], w["pass"], w["checks"]["dnll"]), ("unresolved", "unresolved", False, True))
        prose = [window(f"p{i}", 256, 0.005, 0.97, 0.0, 0.004, self.rng, cls="prose") for i in range(4)]
        code = [window(f"c{i}", 63, 0.005, 0.97, 0.030, 0.004, self.rng, scale=0.157, cls="code") for i in range(2)]
        p = v2.verdict(prose + code)          # code class: 126 rows, unresolved -> provisional, not a pass
        self.assertEqual((p["state"], p["unresolved"]), ("UNRESOLVED", ["code"]))
        self.assertEqual(v2.EXIT[p["state"]], 3)
        self.assertIn("UNRESOLVED classes code", v2.pooled_line(p))
        code = [window(f"c{i}", 63, 0.005, 0.97, 0.030, 0.004, self.rng, scale=0.157, cls="code") for i in range(13)]
        p = v2.verdict(prose + code)          # 819 rows of the class: final, FAIL
        c = {x["name"]: x for x in p["classes"]}["code"]
        self.assertEqual((c["final"], c["state"], p["state"]), (True, "fail", "FAIL"))
        self.assertFalse(p["checks"]["classes"])
        # an unresolved window inside a final passing class is cleared by it
        ok = [window(f"q{i}", 63, 0.005, 0.97, 0.0, 0.004, self.rng, scale=0.157, cls="prose") for i in range(13)]
        odd = window("q13", 63, 0.005, 0.97, 0.030, 0.004, self.rng, scale=0.157, cls="prose")
        p = v2.verdict(ok + [odd])
        self.assertTrue(odd["cleared_by_class"])
        self.assertEqual(p["state"], "PASS")

    def test_reference_check_holds(self):
        base = [window(f"b{i}", 256, 0.005, 0.97, 0.0, 0.004, self.rng) for i in range(4)]
        # the lever beats the exact reference by ~10 se on the true text, no null NLL: HOLD, never PASS
        w = window("sus", 63, 0.005, 0.97, -0.19, 0.004, self.rng, scale=0.157)
        self.assertTrue(w["reference_suspect"])
        p = v2.verdict(base + [w])
        self.assertEqual((p["state"], p["reference_suspect"], v2.EXIT["HOLD"]), ("HOLD", ["sus"], 4))
        # a chaotic window: the null moves the same way (paired candidate - null not below -3 se): not suspect
        w = window("chaos", 63, 0.005, 0.97, -0.19, 0.004, self.rng, scale=0.157, null_dnll=-0.19)
        self.assertFalse(w["reference_suspect"])
        self.assertIn("null dNLL -0.19000", v2.window_line(w))
        self.assertEqual(v2.verdict(base + [w])["state"], "PASS")

    def test_se_capped_by_the_null(self):
        # row se 0.0198 but the null's se on the same window is 0.005: the dNLL limit uses 1.5 x 0.005
        w = window("cap", 63, 0.005, 0.97, 0.030, 0.004, self.rng, scale=0.157, null_dnll=0.0, null_scale=0.0397)
        self.assertAlmostEqual(w["dnll_se"], 1.5 * w["null_dnll_se"])
        self.assertGreater(w["dnll_se_rows"], w["dnll_se"])
        self.assertEqual(w["state"], "fail")
        self.assertIn("capped 1.5 x null", v2.window_line(w))

    def test_off_distribution_reference_holds(self):
        # refx_dense: a window opening with chat-template ids had EXACT NLL 7.07 against 0.9-2.0 elsewhere
        ws = [window(f"w{i}", 256, 0.005, 0.97, 0.0, 0.004, self.rng, ref_nll=r) for i, r in enumerate((0.9, 1.3, 2.0))]
        ws.append(window("w2tmpl", 256, 0.005, 0.97, 0.0, 0.004, self.rng, ref_nll=7.07))
        p = v2.verdict(ws)
        self.assertEqual((p["state"], p["off_distribution"]), ("HOLD", ["w2tmpl"]))
        self.assertIn("off-distribution reference w2tmpl", v2.pooled_line(p))

    def test_labels_must_be_held_out(self):
        ws = [window(f"w{i}", 256, 0.001, 1.0, 0.0, 0.001, self.rng, hit=1.0) for i in range(4)]
        with self.assertRaises(v2.GateError):   # every label is the reference's argmax: greedy ids, not text
            v2.verdict(ws)
        ws = [window(f"w{i}", 256, 0.001, 1.0, 0.0, 0.001, self.rng, hit=0.6) for i in range(4)]
        self.assertAlmostEqual(v2.verdict(ws)["held_out"]["label_is_ref_top1"], 0.6, places=2)

    def test_route_files_flips_and_ratio(self):
        with tempfile.TemporaryDirectory() as t:
            t = pathlib.Path(t)
            ref = [(0, [[1, 2], [3, 4]], [[.5, .5], [.6, .4]]), (1, [[5, 6]], [[.7, .3]])]
            cand = [(0, [[2, 1], [3, 9]], [[.5, .5], [.6, .4]]), (1, [[5, 8]], [[.7, .3]])]   # order-only + 2 flips
            null = [(0, [[1, 2], [3, 9]], [[.5, .5], [.6, .4]]), (1, [[5, 6]], [[.7, .3]])]   # 1 flip
            for name, calls in (("ref", ref), ("cand", cand), ("null", null)):
                write_routes(t / f"{name}.rte", calls)
            c = v2.route_flips(t / "ref.rte", t / "cand.rte")
            n = v2.route_flips(t / "ref.rte", t / "null.rte")
            self.assertEqual((c["slots"], c["ids"], c["per_layer"]), (6, 2, {0: [4, 1], 1: [2, 1]}))
            self.assertEqual(v2.flip_ratio(c, n), 2.0)
            self.assertEqual(v2.flip_ratio({"rate": 0.0}, {"rate": 0.0}), 1.0)
            self.assertEqual(v2.flip_ratio({"rate": 0.1}, {"rate": 0.0}), float("inf"))
            ws = [window(f"w{i}", 256, 0.001, 0.99, 0.0, 0.001, self.rng) for i in range(4)]
            p = v2.verdict(ws, [{"cand": c, "null": n}])
            self.assertEqual((p["flips"]["ratio"], p["checks"]["flip_ratio"], p["pass"]), (2.0, False, False))
            write_routes(t / "short.rte", ref[:1])
            with self.assertRaises(v2.GateError):   # another call sequence: no verdict
                v2.route_flips(t / "ref.rte", t / "short.rte")
            write_routes(t / "layer.rte", [(0, [[1, 2], [3, 4]], [[1, 1], [1, 1]]), (2, [[5, 6]], [[1, 1]])])
            with self.assertRaises(v2.GateError):
                v2.route_flips(t / "ref.rte", t / "layer.rte")
            (t / "cut.rte").write_bytes((t / "ref.rte").read_bytes()[:-3])
            with self.assertRaises(v2.GateError):
                v2.read_routes(t / "cut.rte")

    def test_replay_bookkeeping(self):
        ref = {"version": 1, "out": {"path": "/o/routes/c.REF.rte", "records": 96, "rows": 9000}}
        arm = {"version": 1, "in": {"path": "/o/routes/c.REF.rte", "records": 96, "consumed": 96, "rows": 9000, "complete": True},
               "local_flips": {"slots": 90000, "ids": 120, "rows_changed": 110, "per_layer": []}}
        got = v2.check_replay(ref, arm, "c.REF.rte", "c.FAST")
        self.assertEqual((got["records"], got["local_flips"]["ids"]), (96, 120))
        with self.assertRaises(v2.GateError):   # a null that never moved a weight is not a null
            v2.check_replay(ref, arm, "c.REF.rte", "c.NULL", null=True)
        self.assertEqual(v2.check_replay(ref, dict(arm, null={"layer": 0, "seed": 1, "nudged": 5}), "c.REF.rte", "c.NULL",
                                         null=True)["null"]["nudged"], 5)
        ordered = dict(arm, null={"order": "test-reduction@1", "applications": 8})
        self.assertEqual(v2.check_replay(ref, ordered, "c.REF.rte", "c.NULL", null=True)["null"]["applications"], 8)
        ordered["null"]["applications"] = 0
        with self.assertRaises(v2.GateError):
            v2.check_replay(ref, ordered, "c.REF.rte", "c.NULL", null=True)
        short = json.loads(json.dumps(arm))
        short["in"].update(consumed=95, complete=False)
        for bad_ref, bad_arm, f in ((ref, short, "c.REF.rte"), (ref, arm, "d.REF.rte"), (ref, {"version": 1}, "c.REF.rte"),
                                    (dict(ref, null={"nudged": 1}), arm, "c.REF.rte")):
            with self.assertRaises(v2.GateError):
                v2.check_replay(bad_ref, bad_arm, f, "c.X")
        line = b"noise\nstrix-route-file/v1 " + json.dumps(arm).encode() + b"\nmore\n"
        self.assertEqual(v2.route_receipt(line, "x")["in"]["consumed"], 96)
        with self.assertRaises(v2.GateError):
            v2.route_receipt(b"no receipt\n", "x")


class FastLogitGateRuleV2(unittest.TestCase):
    """fast_logit_gate.py compare --rule v2 on sealed synthetic arms (helpers of test_fast_logit_gate)."""
    setUp, tearDown, selection, arm, cmp, noisy = (tfg.Case.setUp, tfg.Case.tearDown, tfg.Case.selection, tfg.Case.arm,
                                                   tfg.Case.cmp, tfg.Case.noisy)

    def test_rule_v2(self):
        a, _ = self.arm("A", "r1")
        near, _ = self.arm("N", "r2", self.noisy(1e-3, 1), mode=3)
        null, _ = self.arm("Z", "r3", self.noisy(1e-3, 2), mode=3)
        self.assertEqual(self.cmp(a, near, "--rule", "v2"), 2)                          # no null: no verdict
        self.assertEqual(self.cmp(a, near, "--rule", "v2", "--null-arm", str(null)), 2)  # 8 rows < 800: no verdict
        with mock.patch.dict(v2.LIMITS, {"min_rows": 1}):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = tfg.g.main(["compare", str(a), str(near), "--ranks", "0", "1", "--tokens", f"p={self.tok}",
                                 "--rule", "v2", "--null-arm", str(null)])
            self.assertEqual(rc, 0, out.getvalue())
            self.assertIn("window p [all]:", out.getvalue())
            self.assertIn("POOLED v2", out.getvalue())
            far, _ = self.arm("F", "r4", self.noisy(2.0, 5), mode=3)
            self.assertEqual(self.cmp(a, far, "--rule", "v2", "--null-arm", str(null)), 1)


class HeadFormat(unittest.TestCase):
    """Prefill quality head dumps (rows x V FP32, labels = the prompt IDs) and trace-diag flips (no route tap)."""
    V, N, R = 32, 80, 64

    def setUp(self):
        self.rng = np.random.default_rng(11)

    def test_read_head_labels_positions_and_last_row(self):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
        from tools.qualification.option_qual.records import read_head
        ids = np.arange(10) * 3
        z = self.rng.normal(size=(4, self.V)).astype("<f4")
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "c-exact.f32"
            p.write_bytes(z.tobytes())
            rows = list(read_head(p, ids=ids, rows=4))
            self.assertEqual([r.position for r in rows], [6, 7, 8])         # the last row (position 9) has no label
            self.assertEqual([r.label for r in rows], [21, 24, 27])         # ids[n - rows + j + 1]
            for j, r in enumerate(rows):
                np.testing.assert_array_equal(r.logits, z[j])
            p.write_bytes(z.tobytes() + b"\0\0\0\0")
            with self.assertRaises(ValueError):
                list(read_head(p, ids=ids, rows=4))                         # not rows x V
            p.write_bytes(z.tobytes())
            with self.assertRaises(ValueError):
                list(read_head(p, ids=np.arange(10) * 9, rows=4))           # ID 81 outside V = 32

    def kit(self, d, cases, null_ids=20, cand_ids=30, layers=4, skip_diag=()):
        d = pathlib.Path(d)
        man, log = [], []
        for c in cases:
            ids = self.rng.integers(0, self.V, self.N)
            (d / f"{c}.ids").write_text(" ".join(map(str, ids)), encoding="utf-8")
            man.append(f"{c}\t{self.N}\t{c}.ids\texact,fast,n0")
            ref = 3.0 * self.rng.normal(size=(self.R, self.V))
            for arm, s in (("exact", 0.0), ("fast", 0.02), ("n0", 0.03)):
                (d / f"{c}-{arm}.f32").write_bytes((ref + s * self.rng.normal(size=ref.shape)).astype("<f4").tobytes())
            for arm, n in (("fast", cand_ids), ("n0", null_ids)):
                if (c, arm) in skip_diag:
                    continue
                lay = ",".join(f'{{"l":{i},"router_rows":{n},"router_tail":0,"router_ids":{n // layers}}}' for i in range(layers))
                log.append(f'{{"diag":1,"case":"{c}","arm":"{arm}","mean_kl":0.001,"top1":64,"kl":[0.001],"layers":[{lay}]}}')
        (d / "m.tsv").write_text("\n".join(man) + "\n", encoding="utf-8")
        (d / "r0.log").write_text("loaded rank=0\n" + "\n".join(log) + "\n", encoding="utf-8")
        return d

    def gate(self, d, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = v2.main(["qual", str(d), "--format", "head", "--manifest", str(d / "m.tsv"), "--ref", "exact",
                          "--cand", "fast", "--null", "n0", "--trace-diag", str(d / "r0.log"), "--json", str(d / "g.json"), *extra])
        return rc, out.getvalue()

    def test_head_gate_pass_rows_and_free_running_flips(self):
        with tempfile.TemporaryDirectory() as d:
            d = self.kit(d, [f"w{i:02d}" for i in range(13)], null_ids=20, cand_ids=28)
            rc, out = self.gate(d)
            self.assertEqual(rc, 0, out)
            self.assertIn("POOLED v2 rows 819 (dNLL rows 819, >= 800)", out)  # 13 windows x 63 labelled rows
            self.assertIn("R1: no route tap", out)
            self.assertIn("ratio 1.40", out)                                 # 28 / 20, same slots
            doc = json.loads((d / "g.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["r1"], "free-running, trace flips on 13 of 13 windows")
            self.assertEqual(doc["windows"][0]["flips"]["cand"]["slots"], self.N * 4)
            self.assertTrue(0.0 < doc["pooled"]["held_out"]["label_is_ref_top1"] < 1.0)

    def test_head_gate_flip_ratio_fails(self):
        with tempfile.TemporaryDirectory() as d:
            rc, out = self.gate(self.kit(d, [f"w{i:02d}" for i in range(13)], null_ids=20, cand_ids=32))
            self.assertEqual(rc, 1, out)
            self.assertIn("FAIL flip_ratio", out)

    def test_head_gate_no_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            rc, out = self.gate(self.kit(d, [f"w{i:02d}" for i in range(12)]))   # 756 rows < 800
            self.assertEqual(rc, 2, out)
            self.assertIn("NO VERDICT: 756 scored dNLL rows", out)
        with tempfile.TemporaryDirectory() as d:
            rc, out = self.gate(self.kit(d, [f"w{i:02d}" for i in range(13)], skip_diag={("w03", "n0")}))
            self.assertEqual(rc, 2, out)
            self.assertIn("w03: trace of only one", out)
        with tempfile.TemporaryDirectory() as d:
            d = self.kit(d, [f"w{i:02d}" for i in range(13)])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = v2.main(["qual", str(d), "--format", "head", "--ref", "exact", "--cand", "fast", "--null", "n0"])
            self.assertEqual(rc, 2)
            self.assertIn("needs --manifest", out.getvalue())
            (d / "w05-n0.f32").unlink()                                       # a lost dump: no verdict, not FAIL
            rc, out = self.gate(d)
            self.assertEqual(rc, 2, out)
            self.assertIn("NO VERDICT: FileNotFoundError", out)

    def test_untraced_windows_pool_without_flips(self):
        with tempfile.TemporaryDirectory() as d:
            d = self.kit(d, [f"w{i:02d}" for i in range(13)], skip_diag={("w12", "fast"), ("w12", "n0")})
            rc, out = self.gate(d)
            self.assertEqual(rc, 0, out)
            self.assertIn("engine trace on 12 of 13 windows", out)


if __name__ == "__main__":
    unittest.main()
