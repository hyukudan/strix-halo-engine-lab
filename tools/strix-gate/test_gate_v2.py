"""Host tests of the FAST gate v2 statistics (gate_v2.py) and its two adapters (numpy only; no engine)."""
import contextlib
import copy
import hashlib
import io
import json
import os
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

    def test_turbo_tier_relaxes_only_its_bars(self):
        # flip ratio 32 / 20 = 1.6: above the FAST 1.5, inside the TURBO 2.0; every other rule is unchanged
        with tempfile.TemporaryDirectory() as d:
            d = self.kit(d, [f"w{i:02d}" for i in range(13)], null_ids=20, cand_ids=32)
            rc, out = self.gate(d, "--tier", "turbo")
            self.assertEqual(rc, 0, out)
            self.assertIn("TIER turbo: KL <= 0.05, top-1 >= 0.9, dNLL <= 0.03, route flip ratio <= 2", out)
            self.assertIn("= 1.60 (<= 2)", out)
            self.assertIn("max(0.05, 1.5 x null", out)
            doc = json.loads((d / "g.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["tier"], "turbo")
            self.assertEqual((doc["limits"]["max_kl"], doc["limits"]["min_rows"]), (0.05, 800))
        with tempfile.TemporaryDirectory() as d:   # turbo keeps the >= 800 row rule
            rc, out = self.gate(self.kit(d, [f"w{i:02d}" for i in range(12)]), "--tier", "turbo")
            self.assertEqual(rc, 2, out)

    def test_fast_tier_default_output_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            d = self.kit(d, [f"w{i:02d}" for i in range(13)], null_ids=20, cand_ids=28)
            rc0, out0 = self.gate(d)
            doc0 = (d / "g.json").read_bytes()
            rc1, out1 = self.gate(d, "--tier", "fast")
            self.assertEqual((rc0, out0, doc0), (rc1, out1, (d / "g.json").read_bytes()))
            self.assertNotIn("TIER", out0)
            self.assertNotIn('"tier"', doc0.decode())
            self.assertIn("max(0.02, 1.5 x null", out0)
            self.assertIn("(pass <= +0.01,", out0)
            self.assertIn("(<= 1.5)", out0)

    def test_turbo_limits_per_window(self):
        rng = np.random.default_rng(5)
        n = 63
        nll = np.full(n, 1.0)
        args = ("w", np.full(n, 0.03), np.arange(n) < 58, nll, nll + rows_with_mean(n, 0.002, 0.001, rng), [0.001])
        fast = v2.window_stats(*args)                          # KL 0.03 > 0.02; disagreement 5/63 = 0.079 > 0.07
        turbo = v2.window_stats(*args, limits=v2.TURBO_LIMITS)
        self.assertEqual((fast["kl_limit"], turbo["kl_limit"]), (0.02, 0.05))
        self.assertFalse(fast["checks"]["kl"])
        self.assertTrue(turbo["checks"]["kl"])
        self.assertAlmostEqual(turbo["top1_pass"], 0.90)
        self.assertEqual(turbo["states"]["top1"], "pass")
        self.assertEqual(set(v2.TURBO_LIMITS), set(v2.LIMITS))
        self.assertEqual({k for k in v2.LIMITS if v2.LIMITS[k] != v2.TURBO_LIMITS[k]},
                         {"max_kl", "min_top1", "max_dnll", "max_flip_ratio"})

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


# Synthetic mixed-set fixture. A longer slice lowers the live pooled bar enough to flag an ordinary essay.
R4_ORIGINAL = {
    **{f"essay-{i}": nll for i, nll in enumerate([2.2, 1.4] + [2.4] * 6 + [2.85] + [2.6] * 3 + [3.25], 101)},
    **{f"listing-{i}": nll for i, nll in enumerate([0.2] * 6 + [0.5] * 5 + [1.0, 1.6], 201)},
    **{f"form-{i}": nll for i, nll in enumerate([0.4] * 7 + [0.7] + [1.1] * 4 + [5.0], 301)},
}
R4_SUBS = [("essay-113", "essay-114", 2.55), ("form-313", "form-314", 1.2), ("essay-102", "essay-102", 0.6)]
R4_ID = {"model": "fixture", "checkpoint_sha256": "1" * 64, "tokenizer_sha256": "2" * 64,
         "tier": "exact", "profile": "reference@1", "device_class": "host-fixture", "regime": "decode"}


def r4_slot(ids, rows, cls, band):
    return {"class": cls, "band": band, "tokens": len(ids), "ids_sha256": v2._r4_token_sha(ids),
            "rows": rows, "labels_sha256": v2._r4_token_sha(ids[-rows:])}


def r4_plan(originals, partition="content"):
    return {"schema": "strix-gate-r4-plan/v1", "set_id": "fixture", "reference": R4_ID.copy(), "format": "head",
            "partition": partition, "originals": originals, "reserves": {}, "extensions": {}}


def r4_reference(nll, key):
    return {"ref_nll": nll, "ref_sha256": hashlib.sha256(key.encode()).hexdigest(), "routes_sha256": None}


def r4_record(plan, nlls, retrospective=False):
    originals = {name: r4_reference(nll, name) for name, nll in nlls.items()}
    return {"schema": "strix-gate-r4-frozen/v2", "plan_sha256": v2._r4_digest(plan), "plan": plan,
            "retrospective": retrospective, "originals": originals, "stats": v2._r4_stats(plan, originals, v2.LIMITS),
            "substitutions": []}


def r4_record_substitution(rec, replaced, by, nll):
    slot, trigger = v2._r4_alternative(rec, replaced, by)
    ref = r4_reference(nll, f"{by}/substitute")
    entry = {"previous_sha256": v2._r4_digest(rec), "replaced": replaced, "by": by, "reason": "reference review",
             "approval": "reference-review-1", "slot": dict(slot), "reference": ref,
             "selection": {**trigger, "ref_repeat_sha256": ref["ref_sha256"], "routes_repeat_sha256": None}}
    rec["substitutions"].append(entry)
    return entry


class R4Frozen(unittest.TestCase):
    """Exercise the frozen-bar statistics and versioned evidence format with synthetic cases."""

    def setUp(self):
        self.rng = np.random.default_rng(5)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = pathlib.Path(self.tmp.name)

    def save(self, rec, name="r4.json", pin=None):
        pin = v2._r4_digest(rec) if pin is None else pin
        path = self.dir / name
        path.write_text(json.dumps(dict(rec, record_sha256=pin)), encoding="utf-8")
        return path, pin

    def fixture(self, partition="content"):
        ids = np.arange(512)
        bands = ("short", "medium", "long")
        plan = r4_plan({name: r4_slot(ids, 63, name.rsplit("-", 1)[0], bands[(i + i // 13) % 3])
                        for i, name in enumerate(R4_ORIGINAL)}, partition)
        for a, b, _ in R4_SUBS:
            slot = plan["originals"][a]
            if a == b:
                plan["extensions"][a] = {"rows": 255, "labels_sha256": v2._r4_token_sha(ids[-255:])}
            else:
                reserve = r4_slot(ids[::-1], 63, slot["class"], slot["band"])
                plan["reserves"][a] = [{"case": b, "slot": reserve}]
        return plan

    def windows(self, nlls, expected=None, rows=64, fail=None):
        ws = [window(n, expected[n][0]["rows"] if expected and n in expected else rows,
                     0.1 if n == fail else 0.0005, 0.99, 0.0, 0.0004, self.rng,
                     ref_nll=nll, cls=n.rsplit("-", 1)[0]) for n, nll in nlls.items()]
        if expected is not None:
            for w in ws:
                if w["name"] in expected:
                    w["r4_reference"] = expected[w["name"]][1].copy()
        return ws

    def judge(self, rec, fail=None, nlls=None):
        fr = v2.load_r4_frozen(*self.save(rec))
        if nlls is None:
            nlls = {name: ref["ref_nll"] for name, (_, ref) in fr[1].items()}
        return v2.verdict(self.windows(nlls, fr[1], fail=fail), r4_frozen=fr)

    def test_original_and_final_matrix_in_content_and_band_partitions(self):
        for partition in ("content", "band"):
            with self.subTest(partition=partition):
                original = r4_record(self.fixture(partition), R4_ORIGINAL, retrospective=True)
                legacy = v2.verdict(self.windows(R4_ORIGINAL))
                self.assertEqual((legacy["state"], legacy["off_distribution"]), ("HOLD", ["essay-113", "form-313"]))
                p = self.judge(original)
                self.assertEqual((p["state"], p["off_distribution"]), ("HOLD", ["form-313"]))
                self.assertEqual(self.judge(original, fail="essay-102")["state"], "FAIL")
                self.assertEqual(p["r4_frozen"]["pooled_diagnostic"]["bar"], 3.0)
                stats = {st["name"]: (st["n"], st["median"], round(st["bar"], 3)) for st in p["r4_frozen"]["classes"]}
                self.assertEqual(stats, {"listing": (13, 0.5, 2.5), "essay": (13, 2.4, 7.2), "form": (13, 0.4, 2.4)})
                final = copy.deepcopy(original)
                for a, b, nll in R4_SUBS[:2]:
                    r4_record_substitution(final, a, b, nll)
                nlls63 = {k: w["ref_nll"] for k, (_, w) in v2._r4_expected(final).items()}
                self.assertEqual((v2.verdict(self.windows(nlls63))["state"], self.judge(final)["state"]), ("PASS", "PASS"))
                self.assertEqual((v2.verdict(self.windows(nlls63, fail="essay-102"))["state"],
                                  self.judge(final, fail="essay-102")["state"]), ("FAIL", "FAIL"))
                r4_record_substitution(final, *R4_SUBS[2])
                nlls255 = {k: w["ref_nll"] for k, (_, w) in v2._r4_expected(final).items()}
                legacy = v2.verdict(self.windows(nlls255))
                self.assertEqual((legacy["state"], legacy["off_distribution"]), ("HOLD", ["essay-109"]))
                p = self.judge(final)
                self.assertEqual((p["state"], p["off_distribution"]), ("PASS", []))
                self.assertAlmostEqual(p["r4_frozen"]["live_bar"], 2.7)
                self.assertEqual(len(p["r4_frozen"]["substitutions"]), 3)
                self.assertTrue(any("highest essay-109 2.850" in line for line in v2._r4_lines(p["r4_frozen"])))
        with self.assertRaisesRegex(v2.GateError, "255 scored dNLL rows pooled"):
            v2.verdict(self.windows({"essay-102": 0.6}, rows=255))

    def test_substitutes_are_judged_by_content_bar_even_below_pooled_bar(self):
        rec = r4_record(self.fixture(), dict(R4_ORIGINAL, **{"listing-213": 4.0}))
        plan = rec["plan"]
        slot = plan["originals"]["listing-213"]
        plan["reserves"]["listing-213"] = [{"case": "listing-new", "slot": slot}]
        rec["plan_sha256"] = v2._r4_digest(plan)
        r4_record_substitution(rec, "listing-213", "listing-new", 2.75)
        r4_record_substitution(rec, *R4_SUBS[1])
        p = self.judge(rec)
        self.assertEqual((p["state"], p["off_distribution"]), ("HOLD", ["listing-new"]))
        self.assertLess(2.75, p["r4_frozen"]["pooled_diagnostic"]["bar"])

    def test_sparse_classes_are_provisional_even_if_the_band_passes(self):
        for sparse in (1, 2):
            plan = self.fixture("band")
            for i in range(sparse):
                plan["originals"][f"essay-{i + 101}"]["class"] = "sparse"
            rec = r4_record(plan, {k: 1.0 for k in R4_ORIGINAL})
            p = self.judge(rec)
            self.assertEqual((p["state"], p["unresolved"]), ("UNRESOLVED", ["sparse"]))
            self.assertIn(f"sparse: {sparse} original windows, provisional", "\n".join(v2._r4_lines(p["r4_frozen"])))

    def test_rejects_nonfinite_negative_and_overflow_even_if_statistics_claim_to_match(self):
        plan = self.fixture()
        for value in (float("inf"), float("-inf"), float("nan"), -0.1, 1.5e308):
            with self.subTest(value=value):
                rec = r4_record(plan, R4_ORIGINAL)
                for evidence in rec["originals"].values():
                    evidence["ref_nll"] = value
                rec["stats"]["pooled"].update(median=value, bar=value * 3)
                message = "derived bar must be finite" if value == 1.5e308 else "finite and nonnegative"
                with self.assertRaisesRegex(v2.GateError, message):
                    v2.load_r4_frozen(*self.save(rec, pin="f" * 64))

    def test_rejects_internal_rehash_with_original_external_pin(self):
        base = r4_record(self.fixture(), R4_ORIGINAL, retrospective=True)
        r4_record_substitution(base, *R4_SUBS[0])
        _, old_pin = self.save(base)
        mutations = (
            lambda r: r.update(retrospective=False),
            lambda r: r["plan"]["reference"].update(profile="other"),
            lambda r: r["plan"].update(set_id="other"),
            lambda r: r["plan"]["originals"]["listing-201"].update(ids_sha256="a" * 64),
            lambda r: r["plan"]["originals"]["listing-201"].update(tokens=513),
            lambda r: r["plan"]["originals"]["listing-201"].update(**{"class": "essay", "band": "other"}),
            lambda r: r["plan"]["originals"]["listing-201"].update(labels_sha256="a" * 64, rows=62),
            lambda r: r["originals"]["listing-201"].update(ref_nll=1.2, ref_sha256="a" * 64, routes_sha256="b" * 64),
            lambda r: r["substitutions"][0].update(reason="changed", approval="changed"),
            lambda r: r["substitutions"].clear(),
        )
        for change in mutations:
            bad = copy.deepcopy(base)
            change(bad)
            bad["plan_sha256"] = v2._r4_digest(bad["plan"])
            bad["stats"] = v2._r4_stats(bad["plan"], bad["originals"], v2.LIMITS)
            with self.assertRaisesRegex(v2.GateError, "independently pinned record digest"):
                v2.load_r4_frozen(self.save(bad)[0], old_pin)
        bad = copy.deepcopy(base)
        bad["stats"]["classes"]["form"]["bar"] += 1
        with self.assertRaisesRegex(v2.GateError, "n/median/bar"):
            v2.load_r4_frozen(*self.save(bad))

    def test_ledger_requires_reason_approval_reference_evidence_and_matching_replacement(self):
        base = r4_record(self.fixture(), R4_ORIGINAL)
        r4_record_substitution(base, *R4_SUBS[1])
        for key, value, message in (("reason", "", "reason"), ("approval", "", "approval"),
                                    ("selection", {}, "reference-only selection evidence"),
                                    ("previous_sha256", "a" * 64, "previous record digest"),
                                    ("by", "form-315", "first preregistered reserve")):
            with self.subTest(key=key):
                bad = copy.deepcopy(base)
                bad["substitutions"][0][key] = value
                with self.assertRaisesRegex(v2.GateError, message):
                    v2.load_r4_frozen(*self.save(bad))
        for key, value in (("ids_sha256", "c" * 64), ("class", "essay"), ("band", "wrong"), ("rows", 62)):
            with self.subTest(key=key):
                bad = copy.deepcopy(base)
                bad["substitutions"][0]["slot"][key] = value
                with self.assertRaisesRegex(v2.GateError, "replacement hash, class, band or scored slice"):
                    v2.load_r4_frozen(*self.save(bad))
        with self.assertRaisesRegex(v2.GateError, "no reference-only outlier trigger"):
            r4_record_substitution(base, *R4_SUBS[0])
        r4_record_substitution(base, *R4_SUBS[1])
        with self.assertRaisesRegex(v2.GateError, "repeated original slot"):
            v2.load_r4_frozen(*self.save(base))

    def test_caps_reserve_order_and_no_chains(self):
        ids = np.arange(512)
        plan = self.fixture()
        # Planned extension of every slot, allowing us to exercise the ledger limits independently.
        plan["extensions"] = {n: {"rows": 255, "labels_sha256": v2._r4_token_sha(ids[-255:])} for n in R4_ORIGINAL}
        rec = r4_record(plan, R4_ORIGINAL)
        for name in ("essay-101", "essay-102", "essay-103", "listing-201"):
            r4_record_substitution(rec, name, name, 1.0)
        with self.assertRaisesRegex(v2.GateError, "at most 10%"):
            v2.load_r4_frozen(*self.save(rec))
        rec = r4_record(plan, R4_ORIGINAL)
        for name in ("essay-101", "essay-104"):
            r4_record_substitution(rec, name, name, 1.0)
        with self.assertRaisesRegex(v2.GateError, "one change per class-band"):
            v2.load_r4_frozen(*self.save(rec))
        plan = self.fixture()
        plan["reserves"]["form-313"].append({"case": "form-315", "slot": plan["reserves"]["form-313"][0]["slot"]})
        rec = r4_record(plan, R4_ORIGINAL)
        with self.assertRaisesRegex(v2.GateError, "first preregistered reserve"):
            r4_record_substitution(rec, "form-313", "form-315", 0.5)
        r4_record_substitution(rec, *R4_SUBS[1])
        with self.assertRaisesRegex(v2.GateError, "must be an original slot"):
            r4_record_substitution(rec, "form-314", "form-313", 0.5)

    def test_missing_and_unknown_reference_windows_are_refused_before_the_statistical_verdict(self):
        original = r4_record(self.fixture(), R4_ORIGINAL)
        with self.assertRaisesRegex(v2.GateError, "missing.*form-313"):
            self.judge(original, nlls={k: v for k, v in R4_ORIGINAL.items() if k != "form-313"})
        with self.assertRaisesRegex(v2.GateError, "missing.*form-301"):
            self.judge(original, nlls={k: v for k, v in R4_ORIGINAL.items() if not k.startswith("form-")})
        with self.assertRaisesRegex(v2.GateError, "unexpected.*essay-114"):
            self.judge(original, nlls={**R4_ORIGINAL, "essay-114": 1.0})
        with self.assertRaisesRegex(v2.GateError, "NLL differs"):
            self.judge(original, nlls={**R4_ORIGINAL, "essay-109": 2.5})


class R4Commands(unittest.TestCase):
    """Trigger refusals through real reference files and the public qualification CLI."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = d = pathlib.Path(tmp.name)
        self.names = [f"w{i:02}" for i in range(14)]
        self.manifest = d / "windows.tsv"
        self.identity = d / "identity.json"
        self.ref_identity = dict(R4_ID, build_sha256="a" * 64)
        self.save_json(self.identity, self.ref_identity)
        self.plan_path, self.record, self.next = (d / name for name in ("plan.json", "r4.json", "r4-next.json"))
        self.doc = d / "gate.json"
        self.rng = np.random.default_rng(24)
        self.ids = {c: self.rng.integers(0, 16, 90) for c in [*self.names, "reserve"]}
        for c, ids in self.ids.items():
            (d / f"{c}.ids").write_text(" ".join(map(str, ids)), encoding="utf-8")
            ref = self.rng.normal(0, 0.2, size=(64, 16)).astype("<f4")
            if c == "w00":
                ref[np.arange(63), ids[-63:]] = -14
            (d / f"{c}-REF.f32").write_bytes(ref.tobytes())
        self.man(self.names)
        self.plan = r4_plan({c: r4_slot(self.ids[c], 63, "prose", "short") for c in self.names})
        self.plan.update(schema="strix-gate-r4-plan/v2", reference=self.ref_identity, vocab=16, registered_at_ns=10,
                         routes={"mode": "none", "layers": [], "top_k": 0})
        self.plan["reserves"]["w00"] = [{"case": "reserve", "slot": r4_slot(self.ids["reserve"], 63, "prose", "short")}]
        self.plan["extensions"]["w01"] = {"rows": 79, "labels_sha256": v2._r4_token_sha(self.ids["w01"][-79:]),
                                          "reason": "declared reference rule"}
        self.cap = ["--out", str(d), "--manifest", str(self.manifest), "--reference-identity", str(self.identity), "--ref", "REF"]
        self.ref_runs = d / "ref-runs.json"
        self.intake = d / "intake.json"
        self.runs = d / "runs.json"
        self.fmt = "head"
        self.approve_plan()

    def approve_plan(self):
        self.save_json(self.plan_path, self.plan)
        self.freeze = ["r4-freeze", str(self.record), *self.cap, "--r4-plan", str(self.plan_path),
                       "--r4-plan-sha256", v2._r4_digest(self.plan)]

    def save_json(self, path, doc):
        path.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
        return v2._r4_digest(doc)

    def logit(self, case, arm):
        if self.fmt == "head":
            return self.dir / f"{case}-{arm}.f32"
        return self.dir / "logits" / f"{case}.{arm}.bin"

    def route(self, case, arm):
        return self.dir / "routes" / f"{case}.{arm}.rte"

    def log(self, case, arm):
        return self.dir / "log" / f"{case}.{arm}.stderr"

    def sha(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "0" * 64

    def run_doc(self, arm, role, identity, slots, started, tag):
        mode = self.plan["routes"]["mode"]
        routes = mode == "replay" and role in ("reference", "reference-repeat", "candidate-free", "null-free")
        stderr = mode == "replay"
        replay = mode == "replay" and role in ("candidate", "null")
        windows = {c: {"slot": slot, "vocab": self.plan["vocab"], "logits_sha256": self.sha(self.logit(c, arm)),
                       "routes_sha256": self.sha(self.route(c, arm)) if routes else None,
                       "stderr_sha256": self.sha(self.log(c, arm)) if stderr else None,
                       "replay_sha256": self.sha(self.route(c, "REF")) if replay else None} for c, slot in slots.items()}
        return {"role": role, "identity": identity, "run_id": f"{tag}-{arm}", "started_at_ns": started,
                "finished_at_ns": started + 1, "windows": windows}

    def receipt_args(self, path, doc):
        return ["--r4-runs", str(path), "--r4-runs-sha256", self.save_json(path, doc)]

    def ref_doc(self, slots, previous=None):
        tag = f"ref-{previous[:8]}" if previous else "original"
        return {"schema": "strix-gate-r4-ref-runs/v1", "plan_sha256": v2._r4_digest(self.plan),
                "previous_sha256": previous,
                "arms": {arm: self.run_doc(arm, role, self.ref_identity, slots, 20, tag)
                         for arm, role in (("REF", "reference"), ("REF2", "reference-repeat"))}}

    def man(self, names):
        self.manifest.write_text("".join(f"{c}\t90\t{c}.ids\n" for c in names), encoding="utf-8")

    def run_gate(self, cmd, rc, message=None):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(v2.main(cmd), rc, out.getvalue())
        if message:
            self.assertIn(message, out.getvalue())
        return out.getvalue()

    def freeze_with_arms(self, retrospective=False):
        for c in self.ids:
            ref = np.fromfile(self.dir / f"{c}-REF.f32", dtype="<f4").reshape(-1, 16)
            for arm, scale in (("CAND", 0.0), ("NULL", 0.001), ("REF2", 0.0)):
                rows = (ref + scale * self.rng.normal(size=ref.shape)).astype("<f4")
                (self.dir / f"{c}-{arm}.f32").write_bytes(rows.tobytes())
        self.capture_args = ["--ref-repeat", "REF2", *self.receipt_args(self.ref_runs, self.ref_doc(self.plan["originals"]))]
        self.run_gate(self.freeze + (["--retrospective"] if retrospective else self.capture_args), 0)
        return json.loads(self.record.read_text(encoding="utf-8"))["record_sha256"]

    def qualification_docs(self, record=None):
        doc = json.loads((record or self.record).read_bytes())
        pin = doc["record_sha256"]
        candidate = {"lever": "8-bit-decode@1", "profile": "q8-compute", "build_sha256": "b" * 64}
        intake = {"schema": "strix-gate-r4-intake/v1", "qualification_id": self.plan["set_id"],
                  "plan_sha256": doc["plan_sha256"], "record_sha256": pin, "approval": "review-fixture", "approved_at_ns": 40,
                  "reuse_approval": None, "candidate": candidate, "tier": "fast", "allow_null_floor": False,
                  "null_order": ["NULL"], "null_controls": {"NULL": "order@1"}}
        slots = {c: slot for c, (slot, _) in v2.load_r4_frozen(record or self.record, pin)[1].items()}
        null = {"reference": self.ref_identity, "control": "order@1"}
        roles = {"CAND": ("candidate", candidate), "NULL": ("null", null)}
        if self.plan["routes"]["mode"] == "replay":
            roles.update(CANDF=("candidate-free", candidate), NULLF=("null-free", null))
        runs = {"schema": "strix-gate-r4-runs/v1", "intake_sha256": v2._r4_digest(intake), "record_sha256": pin,
                "null_order": ["NULL"], "arms": {a: self.run_doc(a, r, i, slots, 50, f"score-{pin[:8]}")
                                                for a, (r, i) in roles.items()}}
        return intake, runs

    def qualification_args(self, intake, runs):
        return ["--r4-intake", str(self.intake), "--r4-intake-sha256", self.save_json(self.intake, intake),
                *self.receipt_args(self.runs, runs)]

    def qual(self, pin, rc, message=None, *extra, record=None, authorization=None):
        if authorization is None:
            authorization = [] if "--r4-diagnostic" in extra else self.qualification_args(*self.qualification_docs(record))
        free = ["--cand-free", "CANDF", "--null-free", "NULLF"] if self.plan["routes"]["mode"] == "replay" else []
        return self.run_gate(["qual", str(self.dir), "--format", self.fmt, "--manifest", str(self.manifest),
                              "--reference-identity", str(self.identity), "--ref", "REF", "--cand", "CAND", "--null", "NULL",
                              "--r4-frozen", str(record or self.record), "--r4-sha256", pin, "--json", str(self.doc),
                              *free, *authorization, *extra], rc, message)

    def substitute(self, pin, rc, message=None, *, replaced="w00", by="reserve", repeat="REF2", ref_doc=None,
                   reason="declared reference rule"):
        if ref_doc is None and repeat is not None:
            base = json.loads(self.record.read_bytes())
            slot, _ = v2._r4_alternative(base, replaced, by)
            ref_doc = self.ref_doc({by: slot}, previous=pin)
        attestation = self.receipt_args(self.ref_runs, ref_doc) if ref_doc else []
        self.run_gate(["r4-substitute", str(self.record), str(self.next), *self.cap, "--r4-sha256", pin,
                       "--replaced", replaced, "--by", by, "--reason", reason, "--approval", "fixture-review",
                       *(["--ref-repeat", repeat] if repeat else []), *attestation], rc, message)

    def remove_scored_arms(self, case):
        for arm in ("REF", "CAND", "NULL"):
            (self.dir / f"{case}-{arm}.f32").unlink()

    def test_freeze_plan_pin_exclusive_write_and_candidate_independence(self):
        self.run_gate([*self.freeze[:-1], "0" * 64], 2, "independently pinned plan")
        full_plan = self.plan_path.read_bytes()
        edited = copy.deepcopy(self.plan)
        edited["originals"]["w01"]["class"] = "code"
        self.save_json(self.plan_path, edited)
        self.run_gate(self.freeze, 2, "independently pinned plan")
        self.plan_path.write_bytes(full_plan)
        first_pin = self.freeze_with_arms()
        self.run_gate(self.freeze, 2, "written once")
        first = self.record.read_bytes()
        self.assertEqual(list(self.dir.glob("*.tmp")), [])
        (self.dir / "w01-CAND.f32").write_bytes(b"entirely different candidate")
        self.run_gate([self.freeze[0], str(self.next), *self.freeze[2:], *self.capture_args], 0, first_pin)
        self.assertEqual(first, self.next.read_bytes())

    def test_unknown_missing_and_explicit_subset(self):
        pin = self.freeze_with_arms()
        self.qual(pin, 2, "unexpected ['reserve']")
        (self.dir / "reserve-REF.f32").unlink()
        self.qual(pin, 2, "unexpected ['reserve']")  # a candidate-only addition is refused too
        for arm in ("CAND", "NULL"):
            (self.dir / f"reserve-{arm}.f32").unlink()
        self.qual(pin, 4, "off-distribution reference w00")
        self.qual(pin, 2, "missing ['w00']", "--cases", *self.names[1:])
        self.man(self.names[1:])
        self.qual(pin, 2, "missing ['w00']")
        self.man(self.names)
        (self.dir / "w00-REF.f32").unlink()
        self.qual(pin, 2, "missing ['w00']")

    def test_identity_fields_and_class_override(self):
        pin = self.freeze_with_arms()
        self.remove_scored_arms("reserve")
        self.qual("", 2, "independently pinned record digest")
        self.qual(pin, 2, "class partition comes from the pinned plan", "--classes", str(self.plan_path))
        self.qual(pin, 2, "regime differs", "--r4-regime", "prefill")
        for key, value in (("checkpoint_sha256", "f" * 64), ("tokenizer_sha256", "e" * 64),
                           ("model", "different"), ("tier", "base"), ("profile", "other"), ("device_class", "other"),
                           ("regime", "prefill")):
            with self.subTest(key=key):
                self.identity.write_text(json.dumps({**self.ref_identity, key: value}), encoding="utf-8")
                self.qual(pin, 2, "reference identity differs")

    def test_changed_content_or_reference_bytes_with_identical_mean(self):
        pin = self.freeze_with_arms()
        self.remove_scored_arms("reserve")
        path = self.dir / "w01.ids"
        changed = self.ids["w01"].copy()
        changed[0] = (changed[0] + 1) % 16  # outside the scored slice
        path.write_text(" ".join(map(str, changed)), encoding="utf-8")
        self.qual(pin, 2, "content hash or token length differs")
        path.write_text(" ".join(map(str, self.ids["w01"])), encoding="utf-8")
        ref = self.dir / "w01-REF.f32"
        logits = np.fromfile(ref, dtype="<f4")
        logits[-1] += 1  # the unscored last row changes no computed NLL
        ref.write_bytes(logits.tobytes())
        self.qual(pin, 2, "reference artifact hash differs")

    def test_reference_repeat_and_complete_effective_set_after_substitution(self):
        pin = self.freeze_with_arms()
        self.man([*self.names, "reserve"])
        self.substitute(pin, 2, "distinct arm names", repeat="REF")
        repeat = self.dir / "reserve-REF2.f32"
        original_bytes = repeat.read_bytes()
        repeat.write_bytes(b"\x00" * len(original_bytes))
        self.substitute(pin, 2, "repeat is not bitwise identical")
        repeat.write_bytes(original_bytes)
        self.substitute(pin, 0)
        self.substitute(pin, 2, "written once")
        next_pin = json.loads(self.next.read_bytes())["record_sha256"]
        self.qual(next_pin, 2, "unexpected ['w00']", record=self.next)
        self.man([*self.names[1:], "reserve"])
        self.remove_scored_arms("w00")
        self.qual(next_pin, 0, "approval fixture-review", record=self.next)
        (self.dir / "reserve-REF.f32").write_bytes((self.dir / "w01-REF.f32").read_bytes())
        self.qual(next_pin, 2, "reference artifact hash differs", record=self.next)

    def extend_w01(self):
        original = np.fromfile(self.dir / "w01-REF.f32", dtype="<f4").reshape(64, 16)
        for arm in ("REF", "REF2", "CAND", "NULL"):
            suffix = np.fromfile(self.dir / f"w01-{arm}.f32", dtype="<f4").reshape(64, 16)
            prefix = original[:16].copy()
            if arm == "NULL":
                prefix[0, 0] += 0.001
            (self.dir / f"w01-{arm}.f32").write_bytes(np.concatenate((prefix, suffix)).tobytes())

    def test_row_extension_binds_nested_slice_to_the_same_prompt(self):
        pin = self.freeze_with_arms()
        self.remove_scored_arms("reserve")
        self.extend_w01()
        self.substitute(pin, 2, "no preregistered row extension", replaced="w02", by="w02", repeat=None)
        self.substitute(pin, 2, "requires an independent reference repeat", replaced="w01", by="w01", repeat=None)
        self.substitute(pin, 0, replaced="w01", by="w01")
        next_pin = json.loads(self.next.read_bytes())["record_sha256"]
        self.qual(next_pin, 4, "preregistered-extension", record=self.next)
        rows = json.loads(self.doc.read_bytes())["windows"]
        self.assertEqual(next(w["rows"] for w in rows if w["name"] == "w01"), 79)

    def test_retrospective_score_never_issues_qualification_verdict(self):
        pin = self.freeze_with_arms(retrospective=True)
        self.remove_scored_arms("reserve")
        self.extend_w01()
        self.substitute(pin, 0, replaced="w01", by="w01", repeat=None)
        pin = json.loads(self.next.read_bytes())["record_sha256"]
        self.qual(pin, 2, "requires --r4-diagnostic", record=self.next)
        self.qual(pin, 2, "DIAGNOSTIC POOLED v2", "--r4-diagnostic", record=self.next)
        p = json.loads(self.doc.read_bytes())["pooled"]
        self.assertEqual((p["state"], p["pass"], p["diagnostic_state"]), ("NO VERDICT", False, "HOLD"))

    def test_scored_arms_require_distinct_roles_and_pinned_bytes(self):
        self.names.remove("w00")
        del self.plan["originals"]["w00"], self.plan["reserves"]["w00"]
        self.man(self.names)
        self.approve_plan()
        pin = self.freeze_with_arms()
        for case in ("w00", "reserve"):
            self.remove_scored_arms(case)
        cand = self.logit("w01", "CAND")
        original = cand.read_bytes()
        bad = np.frombuffer(original, "<f4").copy().reshape(64, 16)
        bad[:-1, 0] += 20
        cand.write_bytes(bad.tobytes())
        approved_bad = self.qualification_args(*self.qualification_docs())
        self.qual(pin, 1, "-> FAIL", authorization=approved_bad)
        self.qual(pin, 2, "distinct arm names", "--cand", "REF", authorization=approved_bad)
        cand.write_bytes(self.logit("w01", "REF").read_bytes())
        self.qual(pin, 2, "run logit artifact hash differs", authorization=approved_bad)
        self.qual(pin, 0, "-> PASS")  # a genuine bitwise-equal candidate has its own receipt
        receipt = json.loads(self.doc.read_bytes())["pooled"]["r4_frozen"]
        self.assertEqual(receipt["candidate"]["lever"], "8-bit-decode@1")
        good = self.qualification_args(*self.qualification_docs())
        cand.unlink()
        os.link(self.logit("w01", "REF"), cand)
        self.qual(pin, 2, "separate artifacts", authorization=good)
        cand.unlink()
        cand.write_bytes(original)
        self.logit("w02", "NULL").write_bytes(self.logit("w03", "NULL").read_bytes())
        self.qual(pin, 2, "run logit artifact hash differs", authorization=good)

    def test_scored_receipt_checks_identity_inventory_and_version(self):
        pin = self.freeze_with_arms()
        self.remove_scored_arms("reserve")
        approved, runs = self.qualification_docs()
        self.qual(pin, 2, "independently pinned intake", authorization=[])
        args = self.qualification_args(approved, runs)
        self.runs.write_text(json.dumps({**runs, "null_order": []}), encoding="utf-8")
        self.qual(pin, 2, "differs from the independently pinned digest", authorization=args)
        for field, value in (("class", "relabeled"), ("rows", 62), ("tokens", 89),
                             ("labels_sha256", "7" * 64), ("ids_sha256", "8" * 64)):
            with self.subTest(field=field):
                changed = copy.deepcopy(runs)
                changed["arms"]["NULL"]["windows"]["w01"]["slot"][field] = value
                self.qual(pin, 2, "run window inventory or vocabulary differs",
                          authorization=self.qualification_args(approved, changed))
        changed = copy.deepcopy(runs)
        changed["arms"]["CAND"]["identity"]["build_sha256"] = "7" * 64
        self.qual(pin, 2, "numerical identity differs", authorization=self.qualification_args(approved, changed))
        changed = copy.deepcopy(runs)
        del changed["arms"]["NULL"]["windows"]["w02"]
        self.qual(pin, 2, "window inventory", authorization=self.qualification_args(approved, changed))
        self.qual(pin, 2, "scoring policy differs", "--allow-null-floor",
                  authorization=self.qualification_args(approved, runs))

    def test_reference_copies_and_postscore_approval_are_refused(self):
        for c in self.names:
            self.logit(c, "REF2").write_bytes(self.logit(c, "REF").read_bytes())
        self.run_gate([*self.freeze, "--ref-repeat", "REF2"], 2, "independently pinned reference run receipts")
        receipt = self.ref_doc(self.plan["originals"])
        receipt["arms"]["REF2"]["run_id"] = receipt["arms"]["REF"]["run_id"]
        self.run_gate([*self.freeze, "--ref-repeat", "REF2", *self.receipt_args(self.ref_runs, receipt)],
                      2, "distinct independent run identities")
        receipt["arms"]["REF2"]["run_id"] = "real-repeat"
        receipt["arms"]["REF2"]["started_at_ns"] = 9
        self.run_gate([*self.freeze, "--ref-repeat", "REF2", *self.receipt_args(self.ref_runs, receipt)],
                      2, "run chronology differs")
        pin = self.freeze_with_arms()
        self.remove_scored_arms("reserve")
        record = json.loads(self.record.read_bytes())
        del record["record_sha256"]
        record["reference_runs"]["finished_at_ns"] = self.plan["registered_at_ns"]
        invalid_pin = v2._r4_digest(record)
        self.save_json(self.next, dict(record, record_sha256=invalid_pin))
        self.qual(invalid_pin, 2, "reference run chronology differs", record=self.next, authorization=[])
        intake, runs = self.qualification_docs()
        intake["approved_at_ns"] = 51
        runs["intake_sha256"] = v2._r4_digest(intake)
        self.qual(pin, 2, "run chronology differs", authorization=self.qualification_args(intake, runs))

    def test_second_plan_and_reused_reference_need_independent_approval(self):
        pin = self.freeze_with_arms()
        self.remove_scored_arms("reserve")
        approved, runs = self.qualification_docs()
        original_args = self.qualification_args(approved, runs)
        self.qual(pin, 4, authorization=original_args)
        self.plan["originals"]["w01"]["class"] = "code"
        self.approve_plan()
        capture = self.receipt_args(self.ref_runs, self.ref_doc(self.plan["originals"]))
        self.run_gate([self.freeze[0], str(self.next), *self.freeze[2:], "--ref-repeat", "REF2", *capture], 0)
        new_pin = json.loads(self.next.read_bytes())["record_sha256"]
        self.qual(new_pin, 2, "intake authorizes a different plan or final record",
                  record=self.next, authorization=original_args)

        second = {**approved, "qualification_id": "different-qualification"}
        self.qual(pin, 2, "explicit authorization for reference reuse",
                  authorization=self.qualification_args(second, runs))
        second["reuse_approval"] = "held-out status reviewed"
        self.qual(pin, 2, "another qualification, record or null order",
                  authorization=self.qualification_args(second, runs))
        runs["intake_sha256"] = v2._r4_digest(second)
        self.qual(pin, 4, authorization=self.qualification_args(second, runs))

    def test_extension_refuses_modified_scored_or_final_overlap(self):
        pin = self.freeze_with_arms()
        self.remove_scored_arms("reserve")
        self.extend_w01()
        self.substitute(pin, 2, "extension reason differs", replaced="w01", by="w01", reason="after seeing scores")
        ref = self.logit("w01", "REF")
        repeat = self.logit("w01", "REF2")
        valid = ref.read_bytes()
        for row in (16, 79):
            with self.subTest(row=row):
                changed = np.frombuffer(valid, "<f4").copy().reshape(80, 16)
                changed[row, 0] += 1
                ref.write_bytes(changed.tobytes())
                repeat.write_bytes(changed.tobytes())
                self.substitute(pin, 2, "changes the frozen reference overlap", replaced="w01", by="w01")
        ref.write_bytes(valid)
        repeat.write_bytes(valid)
        self.substitute(pin, 0, replaced="w01", by="w01")
        doc = json.loads(self.next.read_bytes())
        self.assertEqual(doc["substitutions"][0]["selection"]["overlap_sha256"], doc["originals"]["w01"]["ref_sha256"])
        # A syntactically consistent, rehashed ledger still cannot hide a mismatched live overlap.
        ref.write_bytes(changed.tobytes())
        del doc["record_sha256"]
        doc["substitutions"][0]["reference"]["ref_sha256"] = self.sha(ref)
        doc["substitutions"][0]["selection"]["ref_repeat_sha256"] = self.sha(ref)
        new_pin = v2._r4_digest(doc)
        self.save_json(self.next, dict(doc, record_sha256=new_pin))
        self.qual(new_pin, 2, "changes the frozen reference overlap", record=self.next)

    def test_pinned_vocabulary_refuses_head_prefixes_even_when_both_are_cut(self):
        # Two 32 x 16 prefixes also look like complete 64 x 8 dumps by byte count.
        data = self.logit("w01", "REF").read_bytes()
        cut = data[:32 * 16 * 4]
        for case in self.names:
            self.logit(case, "REF2").write_bytes(self.logit(case, "REF").read_bytes())
        for arm in ("REF", "REF2"):
            self.logit("w01", arm).write_bytes(cut)
        capture = self.receipt_args(self.ref_runs, self.ref_doc(self.plan["originals"]))
        self.run_gate([*self.freeze, "--ref-repeat", "REF2", *capture], 2, "head shape differs")
        for arm in ("REF", "REF2"):
            self.logit("w01", arm).write_bytes(data)
        pin = self.freeze_with_arms()
        self.remove_scored_arms("reserve")
        null = self.logit("w01", "NULL")
        null.write_bytes(null.read_bytes()[:len(cut)])
        self.qual(pin, 2, "head shape differs")

    def test_spool_prompt_and_step_position_are_checked_against_the_plan(self):
        from tools.qualification.option_qual.records import write_spool, write_steps

        (self.dir / "logits").mkdir()
        for fmt in ("spool", "steps"):
            with self.subTest(fmt=fmt):
                self.fmt = self.plan["format"] = fmt
                for c, slot in self.plan["originals"].items():
                    ids = self.ids[c]
                    logits = np.fromfile(self.dir / f"{c}-REF.f32", dtype="<f4").reshape(64, 16)[:63]
                    if fmt == "spool":
                        slot["labels_sha256"] = v2._r4_token_sha(ids[1:64])
                        if c == "w01":
                            ids = ids.copy()
                            ids[0] = (ids[0] + 1) % 16
                        write_spool(self.logit(c, "REF"), ids, logits)
                    else:
                        slot["labels_sha256"] = v2._r4_token_sha(ids[-63:])
                        slot["first_position"] = 26
                        self.logit(c, "REF").unlink()
                        start = 27 if c == "w01" else 26
                        write_steps(self.logit(c, "REF"), [(i, start + i, int(z.argmax()), int(label), z)
                                                           for i, (label, z) in enumerate(zip(ids[-63:], logits))])
                    self.logit(c, "REF2").write_bytes(self.logit(c, "REF").read_bytes())
                self.plan["reserves"].clear()
                self.plan["extensions"].clear()
                self.approve_plan()
                capture = self.receipt_args(self.ref_runs, self.ref_doc(self.plan["originals"]))
                self.run_gate([*self.freeze, "--ref-repeat", "REF2", *capture], 2,
                              "spool prompt IDs" if fmt == "spool" else "pinned position")
                fixed = self.logit("w01", "REF")
                ids = self.ids["w01"]
                rows = np.fromfile(self.dir / "w01-REF.f32", dtype="<f4").reshape(64, 16)[:63]
                if fmt == "spool":
                    write_spool(fixed, ids, rows)
                else:
                    fixed.unlink()
                    write_steps(fixed, [(i, 26 + i, int(z.argmax()), int(label), z)
                                        for i, (label, z) in enumerate(zip(ids[-63:], rows))])
                self.logit("w01", "REF2").write_bytes(fixed.read_bytes())
                capture = self.receipt_args(self.ref_runs, self.ref_doc(self.plan["originals"]))
                self.run_gate([*self.freeze, "--ref-repeat", "REF2", *capture], 0)
                self.record.unlink()

    def test_orphaned_frozen_options_never_return_an_ordinary_verdict(self):
        self.freeze_with_arms()
        self.remove_scored_arms("reserve")
        args = ["qual", str(self.dir), "--format", "head", "--manifest", str(self.manifest),
                "--ref", "REF", "--cand", "CAND", "--null", "NULL"]
        self.run_gate(args, 4, "-> HOLD")
        for extra in (["--r4-diagnostic"], ["--r4-sha256", "1" * 64], ["--r4-regime", "decode"],
                      ["--reference-identity", str(self.identity)], ["--r4-intake", str(self.intake)],
                      ["--r4-runs-sha256", "2" * 64]):
            with self.subTest(extra=extra):
                self.run_gate([*args, *extra], 2, "frozen R4 options require --r4-frozen")

    def test_paired_trace_coverage_cannot_hide_flip_ratio_failure(self):
        self.plan["routes"] = {"mode": "trace", "layers": [0, 2], "top_k": 2}
        self.approve_plan()
        pin = self.freeze_with_arms()
        self.remove_scored_arms("reserve")
        (self.dir / "log").mkdir()
        diagnostics = ["--r4-diagnostic"]
        for case in self.names:
            for arm, flips in (("CAND", 12 if case == "w01" else 1), ("NULL", 1)):
                trace = {"diag": 1, "case": case, "arm": arm,
                         "layers": [{"l": i, "router_ids": flips * (3 if case == "w01" and arm == "CAND" else 1)}
                                    for i in (0, 2)]}
                self.save_json(self.log(case, arm), trace)
                diagnostics.extend(("--trace-diag", str(self.log(case, arm))))
        self.qual(pin, 2, "R1: no route tap", *diagnostics)
        pooled = json.loads(self.doc.read_bytes())["pooled"]
        self.assertFalse(pooled["checks"]["flip_ratio"])
        self.assertEqual(pooled["state"], "NO VERDICT")
        for arm in ("CAND", "NULL"):
            self.log("w01", arm).write_bytes(b"trace missing")
        self.qual(pin, 2, "trace mode is diagnostic-only")
        for arm in ("CAND", "NULL"):
            trace = {"diag": 1, "case": "w01", "arm": arm, "layers": [{"l": 0, "router_ids": 1}]}
            self.save_json(self.log("w01", arm), trace)
        self.qual(pin, 2, "trace mode is diagnostic-only")

    def test_unattested_zero_trace_cannot_turn_frozen_failure_into_pass(self):
        self.plan["routes"] = {"mode": "trace", "layers": [0, 2], "top_k": 2}
        self.use_benign_reference()
        pin = self.freeze_with_arms()
        self.remove_scored_arms("reserve")
        (self.dir / "log").mkdir()
        for case in self.names:
            for arm in ("CAND", "NULL"):
                flips = 32 if (case, arm) == ("w01", "CAND") else 1
                self.save_json(self.log(case, arm), {"diag": 1, "case": case, "arm": arm,
                               "layers": [{"l": i, "router_ids": flips} for i in (0, 2)]})
        self.qual(pin, 2, "trace mode is diagnostic-only")
        # A native diagnostic entry with all the layer names can still represent zero comparisons.
        self.save_json(self.log("w01", "CAND"), {"diag": 1, "case": "w01", "arm": "CAND",
                       "layers": [{"l": i, "router_ids": 0, "router_rows": 0} for i in (0, 2)]})
        self.qual(pin, 2, "trace mode is diagnostic-only")

    def use_benign_reference(self):
        self.logit("w00", "REF").write_bytes(self.logit("w01", "REF").read_bytes())
        self.plan["reserves"].clear()
        self.approve_plan()

    def check_secondary_coverage(self, partition, field, group, names):
        self.plan["partition"] = partition
        for case in names:
            self.plan["originals"][case][field] = group
        self.use_benign_reference()
        pin = self.freeze_with_arms()
        self.remove_scored_arms("reserve")
        self.qual(pin, 3, "-> UNRESOLVED")
        p = json.loads(self.doc.read_bytes())["pooled"]
        self.assertIn((group, 63 * len(names), False),
                      [(c["name"], c["rows"], c["final"]) for c in p["secondary_classes"]])

    def test_band_partition_requires_final_content_classes(self):
        self.check_secondary_coverage("band", "class", "code", self.names[:3])

    def test_content_partition_requires_final_prompt_bands(self):
        self.check_secondary_coverage("content", "band", "long", self.names[:1])

    def replay_fixture(self, *, candidate_flips=True):
        self.plan["routes"] = {"mode": "replay", "layers": [0, 2], "top_k": 2}
        route_calls = [[0, 2], [2, 2], [0, 2]]
        for slot in self.plan["originals"].values():
            slot["route_calls"] = route_calls
        self.use_benign_reference()
        (self.dir / "log").mkdir()
        (self.dir / "routes").mkdir()
        for case in self.names:
            for arm in ("REF", "REF2", "CANDF", "NULLF"):
                ids = [1, 3] if candidate_flips and arm == "CANDF" else [1, 2]
                self.write_replay_output(case, arm, [(layer, [ids] * rows) for layer, rows in route_calls])
            for arm in ("CAND", "NULL"):
                data = {"in": {"path": self.route(case, "REF").name, "records": 3, "consumed": 3,
                               "rows": 6, "complete": True}}
                self.write_replay_log(case, arm, data)

    def write_replay_log(self, case, arm, data):
        if arm in ("NULL", "NULLF"):
            data["null"] = {"order": "order@1", "applications": 1}
        self.log(case, arm).write_text("strix-route-file/v1 " + json.dumps(data) + "\n", encoding="utf-8")

    def write_replay_output(self, case, arm, calls):
        route = self.route(case, arm)
        write_routes(route, [(layer, ids, [[0.25] * len(ids[0])] * len(ids)) for layer, ids in calls])
        self.write_replay_log(case, arm, {"out": {"path": route.name, "records": len(calls),
                                                  "rows": sum(len(ids) for _, ids in calls)}})

    def freeze_replay(self):
        pin = self.freeze_with_arms()
        self.remove_scored_arms("reserve")
        for case in self.names:
            for free, scored in (("CANDF", "CAND"), ("NULLF", "NULL")):
                self.logit(case, free).write_bytes(self.logit(case, scored).read_bytes())
        return pin

    def test_replay_duplicate_experts_cannot_dilute_route_flip_failure(self):
        self.replay_fixture()
        pin = self.freeze_replay()
        self.qual(pin, 1, "-> FAIL flip_ratio")
        for case in self.names:
            calls = self.plan["originals"][case]["route_calls"]
            self.write_replay_output(case, "NULLF", [(layer, [[1, 1]] * rows) for layer, rows in calls])
        self.qual(pin, 2, "repeated expert IDs")

    def test_replay_declared_visits_cannot_be_replaced_with_the_same_total_rows(self):
        self.replay_fixture()
        for arm in ("REF", "REF2"):
            self.write_replay_output("w01", arm, [(0, [[1, 2]] * 2), (0, [[1, 2]] * 2), (2, [[1, 2]] * 2)])
        for case in self.names:
            self.logit(case, "REF2").write_bytes(self.logit(case, "REF").read_bytes())
        receipt = self.receipt_args(self.ref_runs, self.ref_doc(self.plan["originals"]))
        self.run_gate([*self.freeze, "--ref-repeat", "REF2", *receipt], 2, "declared router call sequence")

    def test_replay_routes_are_bound_and_complete_including_free_arms(self):
        self.replay_fixture(candidate_flips=False)
        pin = self.freeze_replay()
        self.qual(pin, 0, "-> PASS")
        p = json.loads(self.doc.read_bytes())["pooled"]
        self.assertTrue(p["checks"]["flip_ratio"])
        self.assertEqual(p["flips"]["cand"]["slots"], len(self.names) * 6 * 2)
        approved, runs = self.qualification_docs()
        bad = copy.deepcopy(runs)
        bad["arms"]["NULL"]["windows"]["w01"]["replay_sha256"] = "e" * 64
        self.qual(pin, 2, "binds a different reference route artifact", authorization=self.qualification_args(approved, bad))
        err = self.log("w01", "CAND")
        original_log = err.read_bytes()
        for extra in (original_log, b"strix-route-file/v1 {"):
            err.write_bytes(original_log + extra)
            self.qual(pin, 2, "expected one strix-route-file/v1 receipt line")
        err.write_bytes(original_log.replace(b'"consumed": 3', b'"consumed": 0, "consumed": 3'))
        self.qual(pin, 2, "duplicate JSON key consumed")
        err.write_bytes(original_log.rstrip()[:-1] + b', "local_flips": [1]}\n')
        self.qual(pin, 2, "invalid route receipt")
        err.unlink()
        self.qual(pin, 2, "FileNotFoundError")
        err.write_bytes(original_log)
        path = self.route("w01", "CANDF")
        data = path.read_bytes()
        path.write_bytes(data[:12] + b"\x01" + data[13:])
        self.qual(pin, 2, "invalid strix-route-file/v1 header flags")
        path.write_bytes(data[:-1])
        self.qual(pin, 2, "corrupt or truncated")
        path.write_bytes(data + data[16:])
        self.qual(pin, 2, "declared router call sequence")
        self.write_replay_output("w01", "CANDF", [(0, [[1, 2]] * 2)])
        self.qual(pin, 2, "incomplete route row coverage")
        path.unlink()
        self.qual(pin, 2, "FileNotFoundError")


if __name__ == "__main__":
    unittest.main()
