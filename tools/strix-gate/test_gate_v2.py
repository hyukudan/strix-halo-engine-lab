"""Host tests of the FAST gate v2 statistics (gate_v2.py) and its two adapters (numpy only; no engine)."""
import contextlib
import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gate_v2 as v2  # noqa: E402
import fast_logit_gate as sealed  # noqa: E402
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
        null_nlls = [nll_ref + rows_with_mean(n, null_dnll, scale if null_scale is None else null_scale, rng)]
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
        base = [window(f"b{i}", 256, 0.005, 0.97, 0.0, 0.004, self.rng, null_dnll=0.0) for i in range(4)]
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

    def test_final_class_reference_check_holds(self):
        windows = [window(f"p{i}", 63, 0.005, 0.97, -0.02, 0.004, self.rng, scale=0.10,
                          null_dnll=0.0, null_scale=0.0, cls="prose") for i in range(13)]
        self.assertFalse(any(w["reference_suspect"] for w in windows))
        result = v2.verdict(windows)
        self.assertEqual(result["state"], "HOLD")
        self.assertEqual(result["reference_suspect_classes"], ["prose"])
        self.assertIn("reference suspect class prose", v2.pooled_line(result))

    def test_null_arm_order_does_not_change_acceptance(self):
        ref = np.ones(800)
        varied = ref + np.tile([-0.2, 0.2], 400)
        cand = ref + np.tile([-0.4, 0.46], 400)
        outcomes = []
        for arms in ([ref, varied], [varied, ref]):
            w = v2.window_stats("multi", np.full(800, 0.005), np.ones(800, bool), ref, cand, [0.001, 0.001],
                                null_nlls=arms)
            outcomes.append((w["state"], w["dnll_se"]))
        self.assertEqual(outcomes[0], outcomes[1])

    def test_exact_null_floor_boundary_is_engaged(self):
        ref = np.ones(800)
        w = v2.window_stats("boundary", np.full(800, 0.018), np.ones(800, bool), ref, ref, [0.02])
        self.assertTrue(v2.verdict([w])["null_floor"]["kl"]["engaged"])

    def test_se_capped_by_the_null(self):
        # row se 0.0198 but the null's se on the same window is 0.005: the dNLL limit uses 1.5 x 0.005
        w = window("cap", 63, 0.005, 0.97, 0.030, 0.004, self.rng, scale=0.157, null_dnll=0.0, null_scale=0.0397)
        self.assertAlmostEqual(w["dnll_se"], 1.5 * w["null_dnll_se"])
        self.assertGreater(w["dnll_se_rows"], w["dnll_se"])
        self.assertEqual(w["state"], "fail")
        self.assertIn("capped 1.5 x null", v2.window_line(w))

    def test_zero_variance_null_caps_the_window_before_class_pooling(self):
        clean = [window(f"ok{i}", 63, 0.005, 0.97, 0.0, 0.004, self.rng, scale=0.0,
                        null_dnll=0.0, null_scale=0.0) for i in range(13)]
        bad = window("bad", 63, 0.005, 0.97, 0.05, 0.004, self.rng, scale=0.5,
                     null_dnll=0.0, null_scale=0.0)
        self.assertEqual(bad["dnll_se"], 0.0)
        self.assertEqual(bad["states"]["dnll"], "fail")
        self.assertEqual(v2.verdict(clean + [bad])["state"], "FAIL")

    def test_reference_check_uses_candidate_uncapped_se(self):
        w = window("noise", 63, 0.005, 0.97, -0.02, 0.004, self.rng, scale=0.10,
                   null_dnll=0.1, null_scale=0.002)
        self.assertLess(w["dnll_se"], 0.002)
        self.assertFalse(w["reference_suspect"])

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

    def test_greedy_window_cannot_be_diluted_by_held_out_windows(self):
        clean = window("document", 800, 0.001, 1.0, 0.0, 0.001, self.rng)
        generated = window("greedy", 64, 0.001, 1.0, 0.0, 0.001, self.rng, hit=1.0)
        with self.assertRaisesRegex(v2.GateError, "greedy: every label"):
            v2.verdict([clean, generated])

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

    def test_route_weight_diagnostic_aligns_by_expert(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d)
            write_routes(path / "ref", [(0, [[1, 2]], [[0.75, 0.25]])])
            write_routes(path / "cand", [(0, [[2, 1]], [[0.5, 0.5]])])
            flips = v2.route_flips(path / "ref", path / "cand")
            self.assertEqual(flips["ids"], 0)
            self.assertEqual(flips["max_abs_weight_delta"], 0.25)

    def test_free_route_receipt_rejects_replay_missing_control_and_wrong_counts(self):
        counts = {"records": 3, "rows": 9}
        recorded = {"out": {"path": "case.CAND_FREE.rte", **counts}}
        path = "case.CAND_FREE.rte"
        v2.check_free_record(recorded, path, counts, "candidate")
        with self.assertRaisesRegex(v2.GateError, "free-running"):
            v2.check_free_record(dict(recorded, **{"in": {}}), path, counts, "candidate")
        with self.assertRaisesRegex(v2.GateError, "does not match"):
            v2.check_free_record(recorded, path, dict(counts, rows=6), "candidate")
        with self.assertRaisesRegex(v2.GateError, "no applied null control"):
            v2.check_free_record(recorded, path, counts, "null", null=True)
        v2.check_free_record(dict(recorded, null={"order": "reduce", "applications": 3}), path, counts, "null", null=True)

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

    def gate(self, d, *extra, stdout=None):
        out = io.StringIO() if stdout is None else stdout
        with contextlib.redirect_stdout(out):
            rc = v2.main(self.gate_args(d, *extra))
        return rc, out.getvalue()

    @staticmethod
    def gate_args(d, *extra):
        return ["qual", str(d), "--format", "head", "--manifest", str(d / "m.tsv"), "--ref", "exact",
                "--cand", "fast", "--null", "n0", "--trace-diag", str(d / "r0.log"), "--json", str(d / "g.json"), *extra]

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

    def test_replay_does_not_hide_free_running_regressions(self):
        with tempfile.TemporaryDirectory() as d:
            cases = [f"w{i:02d}" for i in range(13)]
            d = self.kit(d, cases)
            (d / "log").mkdir()
            (d / "routes").mkdir()
            for case in cases:
                ref = f"{case}.exact.rte"
                control = {"null": {"order": "reduce", "applications": 1}}
                for arm in ("exact", "fast_free", "null_free"):
                    path = d / "routes" / f"{case}.{arm}.rte"
                    weights = [0.1, 0.9] if arm == "fast_free" else [0.75, 0.25]
                    write_routes(path, [(0, [[1, 2]], [weights])])
                    receipt = {"out": {"path": path.name, "records": 1, "rows": 1}}
                    if arm == "null_free":
                        receipt.update(control)
                    (d / "log" / f"{case}.{arm}.stderr").write_text(
                        "strix-route-file/v1 " + json.dumps(receipt) + "\n", encoding="utf-8")
                for arm in ("fast", "n0"):
                    receipt = {"in": {"path": ref, "complete": True, "consumed": 1, "records": 1, "rows": 1}}
                    if arm == "n0":
                        receipt.update(control)
                    (d / "log" / f"{case}.{arm}.stderr").write_text(
                        "strix-route-file/v1 " + json.dumps(receipt) + "\n", encoding="utf-8")
                (d / f"{case}-null_free.f32").write_bytes((d / f"{case}-n0.f32").read_bytes())
                (d / f"{case}-fast_free.f32").write_bytes((d / f"{case}-fast.f32").read_bytes())

            args = ("--cand-free", "fast_free", "--null-free", "null_free")
            self.assertEqual(self.gate(d, *args)[0], 0)
            rc, out = self.gate(d, "--cand-free", "exact", "--null-free", "null_free")
            self.assertEqual(rc, 2, out)
            self.assertIn("must be distinct", out)
            target = d / "w00-fast_free.f32"
            finite = target.read_bytes()
            target.write_bytes((-np.frombuffer(finite, "<f4")).tobytes())
            rc, out = self.gate(d, *args)
            self.assertEqual(rc, 1, out)
            doc = json.loads((d / "g.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["pooled"]["free_run"]["state"], "FAIL")
            self.assertEqual(doc["pooled"]["flips"]["cand"]["ids"], 0)
            self.assertGreater(doc["windows"][0]["flips"]["cand"]["rms_weight_delta"], 0.6)

            route = d / "routes" / "w00.fast_free.rte"
            write_routes(route, [(0, [[1, 1]], [[0.1, 0.9]])])
            rc, out = self.gate(d, *args)
            self.assertEqual(rc, 2, out)
            self.assertIn("repeated expert ID", out)
            write_routes(route, [(0, [[1, 2]], [[0.1, 0.9]])])

            target.write_bytes(np.array([np.nan], "<f4").tobytes() + finite[4:])
            self.assertEqual(self.gate(d, *args)[0], 1)
            doc = json.loads((d / "g.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["pooled"]["free_run"]["nonfinite_candidate"], ["w00"])
            target.write_bytes(finite)
            write_routes(d / "routes" / "w00.fast_free.rte", [(0, [[1, 2]], [[np.inf, 0.9]])])
            self.assertEqual(self.gate(d, *args)[0], 1)
            target.unlink()
            self.assertEqual(self.gate(d, *args)[0], 2)

    def test_sealed_route_only_evidence_cannot_qualify_moe(self):
        args = mock.Mock(mode="fast", route_flips=["w=ref.rte,cand.rte,null.rte"])
        with self.assertRaisesRegex(sealed.GateError, "requires free-running logits"):
            sealed.compare(args)

    def test_nonfinite_candidate_is_a_failure_not_an_evidence_error(self):
        with tempfile.TemporaryDirectory() as d:
            d = self.kit(d, [f"w{i:02d}" for i in range(13)])
            target = d / "w00-fast.f32"
            original = target.read_bytes()
            target.write_bytes(np.array([np.nan], dtype="<f4").tobytes() + original[4:])
            rc, out = self.gate(d)
            self.assertEqual(rc, 1, out)
            self.assertIn("non-finite candidate logits", out)
            doc = json.loads((d / "g.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["pooled"]["nonfinite_candidate"], ["w00"])
            target.write_bytes(original)
            null = d / "w00-n0.f32"
            null.write_bytes(np.array([np.inf], dtype="<f4").tobytes() + null.read_bytes()[4:])
            self.assertEqual(self.gate(d)[0], 2)

    def test_head_gate_flip_ratio_fails(self):
        with tempfile.TemporaryDirectory() as d:
            rc, out = self.gate(self.kit(d, [f"w{i:02d}" for i in range(13)], null_ids=20, cand_ids=32))
            self.assertEqual(rc, 1, out)
            self.assertIn("FAIL flip_ratio", out)

    def test_trace_null_does_not_depend_on_argument_order(self):
        with tempfile.TemporaryDirectory() as d:
            cases = [f"w{i:02d}" for i in range(13)]
            d = self.kit(d, cases, null_ids=20, cand_ids=28)
            for c in cases:
                (d / f"{c}-n1.f32").write_bytes((d / f"{c}-n0.f32").read_bytes())
            log = d / "r0.log"
            before = log.read_text(encoding="utf-8")
            log.write_text(before + "".join(line.replace('"arm":"n0"', '"arm":"n1"')
                                             for line in before.splitlines(keepends=True) if '"arm":"n0"' in line),
                           encoding="utf-8")
            rc, out = self.gate(d, "--null", "n1")
            self.assertEqual(rc, 2, out)
            self.assertIn("--null-free", out)
            rc, out = self.gate(d, "--null", "n1", "--null-free", "n1")
            self.assertEqual(rc, 0, out)

    def test_repeating_a_null_does_not_reweight_the_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            d = self.kit(d, [f"w{i:02d}" for i in range(13)], null_ids=20, cand_ids=28)
            rc, out = self.gate(d, "--null", "n0", "--null-free", "n0")
            self.assertEqual(rc, 2, out)
            self.assertIn("distinct reference, candidate and null arms", out)

    def test_null_witness_binds_the_baseline_configuration_and_capture(self):
        with tempfile.TemporaryDirectory() as d:
            cases = [f"w{i:02d}" for i in range(13)]
            d = self.kit(d, cases, cand_ids=20)
            witness = d / "control.json"
            records = {}
            for case in cases:
                records[case] = {}
                ids = np.array((d / f"{case}.ids").read_text(encoding="utf-8").split(), dtype="<u4")
                for arm in ("exact", "n0"):
                    records[case][arm] = {"binary_sha256": "a" * 64, "model_sha256": "b" * 64,
                                          "numerical_config_sha256": "c" * 64,
                                          "logits_sha256": hashlib.sha256((d / f"{case}-{arm}.f32").read_bytes()).hexdigest(),
                                          "head_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                                          "control": None if arm == "exact" else
                                              {"order": "test-order@1", "applications": self.R}}

            def check(old_exit):
                witness.write_text(json.dumps({"schema": "strix-gate-null-witness/v1", "cases": records}), encoding="utf-8")
                rc, out = self.gate(d, "--null-control-witness", str(witness))
                self.assertEqual(rc, old_exit, out)

            check(0)
            record = records[cases[0]]["n0"]
            for field, invalid in (("logits_sha256", "d" * 64), ("numerical_config_sha256", "d" * 64),
                                   ("control", {"order": "test-order@1", "applications": 0})):
                with self.subTest(field=field):
                    original = record[field]
                    record[field] = invalid
                    check(2)
                    record[field] = original

    def test_nonfinite_last_head_row_is_checked_without_scoring_an_extra_label(self):
        with tempfile.TemporaryDirectory() as d:
            d = self.kit(d, [f"w{i:02d}" for i in range(13)])
            for arm, expected in (("fast", 1), ("exact", 2), ("n0", 2)):
                with self.subTest(arm=arm):
                    target = d / f"w00-{arm}.f32"
                    original = target.read_bytes()
                    target.write_bytes(original[:-4] + np.array([np.inf], dtype="<f4").tobytes())
                    rc, out = self.gate(d)
                    self.assertEqual(rc, expected, out)
                    if expected == 1:
                        doc = json.loads((d / "g.json").read_text(encoding="utf-8"))
                        self.assertEqual(doc["pooled"]["nonfinite_candidate"], ["w00"])
                    target.write_bytes(original)

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

    def test_traces_must_cover_the_same_layer_ids(self):
        with tempfile.TemporaryDirectory() as d:
            d = self.kit(d, [f"w{i:02d}" for i in range(13)])
            log = d / "r0.log"
            lines = log.read_text(encoding="utf-8").splitlines()
            lines[1] = lines[1].replace('"l":2', '"l":7')
            log.write_text("\n".join(lines) + "\n", encoding="utf-8")
            rc, out = self.gate(d)
            self.assertEqual(rc, 2, out)
            self.assertIn("candidate trace layers (0, 1, 3, 7) differ from null layers (0, 1, 2, 3)", out)


class FrozenR4Diagnostic(unittest.TestCase):
    kit, gate = HeadFormat.kit, HeadFormat.gate
    gate_args = staticmethod(HeadFormat.gate_args)
    V, N, R = HeadFormat.V, HeadFormat.N, HeadFormat.R

    def setUp(self):
        self.rng = np.random.default_rng(12)

    def test_r4_assessment_keeps_ordinary_hold_and_qualifies_every_diagnostic_pass_token(self):
        cases = [f"toy-{i}" for i in range(4)]
        cls = "essay-PASS"

        def reduce_pair(ref, arm, _format, **_kwargs):
            with open(ref, "rb") as source, open(arm, "rb"):
                nll = np.full(255, 4.0 if source.read(7) == b"outlier" else 1.0)
            return dict(kl_rc=np.full(nll.size, 1 / 1024), agree=np.ones_like(nll), nll_r=nll, nll_c=nll,
                        top1_r_is_label=np.zeros_like(nll), top1_c_is_label=np.zeros_like(nll),
                        nonfinite_r=np.zeros_like(nll), nonfinite_c=np.zeros_like(nll)), {}

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(v2, "_option_qual_reduce", return_value=reduce_pair):
            d = self.kit(tmp, cases, cand_ids=20)
            for case in cases:
                (d / f"{case}-exact.f32").write_bytes(b"outlier" if case == "toy-3" else b"ordinary")
            classes = d / "classes.json"
            classes.write_text(json.dumps(dict.fromkeys(cases, cls)), encoding="utf-8")
            original = d / "original.json"
            original.write_text(json.dumps({cls: [1.5, 2.0, 2.5]}), encoding="utf-8")
            plain_err = io.StringIO()
            with contextlib.redirect_stderr(plain_err):
                plain = self.gate(d, "--classes", str(classes))
            self.assertEqual(plain_err.getvalue(), "")
            self.assertEqual(plain[0], 4)
            self.assertIn("-> HOLD off-distribution reference toy-3", plain[1])
            plain_json = (d / "g.json").read_bytes()
            # Snapshots of the same synthetic run on origin/main (8820d1a29), with portable JSON line endings.
            self.assertEqual(hashlib.sha256(plain[1].encode()).hexdigest(),
                             "22b9da213c22c9c630ac6c8ad24f111bad980dc636e57884647d46d3a49f50a0")
            self.assertEqual(hashlib.sha256(plain_json.replace(b"\r\n", b"\n")).hexdigest(),
                             "1da7688904a3f72e8d061c655ca77201fb192cc2cf8db22a51439f104e35c766")

            report = io.StringIO()
            with contextlib.redirect_stderr(report):
                scored = self.gate(d, "--classes", str(classes), "--r4-assess", str(original))
            self.assertEqual(scored, plain)
            self.assertEqual((d / "g.json").read_bytes(), plain_json)
            self.assertTrue(all(line.startswith("R4 DIAGNOSTIC ONLY: ") for line in report.getvalue().splitlines()))
            self.assertIn("3 original windows, median 2.000, bar 6.000", report.getvalue())
            self.assertIn("frozen-bar assessment would be PASS", report.getvalue())
            self.assertNotIn("PASS", report.getvalue().replace("would be PASS", ""))
            self.assertNotIn("off-distribution", report.getvalue())

            original.write_text(json.dumps({cls: [1.0, 1.0, 2.0]}), encoding="utf-8")
            report = io.StringIO()
            with contextlib.redirect_stderr(report):
                self.assertEqual(self.gate(d, "--classes", str(classes), "--r4-assess", str(original)), plain)
            self.assertIn('off-distribution "toy-3"', report.getvalue())
            self.assertIn("frozen-bar assessment would be HOLD", report.getvalue())

            original.write_text(json.dumps({cls: [1.0, 1.0, 2.0], "code": [1.0, 2.0]}), encoding="utf-8")
            report = io.StringIO()
            with contextlib.redirect_stderr(report):
                self.assertEqual(self.gate(d, "--classes", str(classes), "--r4-assess", str(original)), plain)
            sparse = next(line for line in report.getvalue().splitlines() if 'content class "code"' in line)
            self.assertIn("UNRESOLVED", sparse)
            self.assertIn("frozen-bar assessment would be HOLD", report.getvalue())

            original.write_text(json.dumps({cls: [1.5, 2.0]}), encoding="utf-8")
            report = io.StringIO()
            with contextlib.redirect_stderr(report):
                self.assertEqual(self.gate(d, "--classes", str(classes), "--r4-assess", str(original)), plain)
            self.assertIn("2 original windows, bar unavailable (need 3 originals); 1020 scored rows (provisional; UNRESOLVED)",
                          report.getvalue())
            self.assertIn("frozen-bar assessment would be UNRESOLVED", report.getvalue())

    def test_r4_assessment_checks_rows_in_both_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = pathlib.Path(tmp) / "original.json"
            for thin_axis in ("content", "band"):
                with self.subTest(thin_axis=thin_axis):
                    code = "code" if thin_axis == "content" else "prose"
                    ref = {"prose": [1.0, 1.0, 1.0], code: [1.0, 1.0, 1.0]}
                    original.write_text(json.dumps(ref), encoding="utf-8")
                    windows = [window(f"toy-{i}", 280, 0.005, 1.0, 0.0, 0.001, self.rng, cls="prose") for i in range(3)]
                    windows.append(window("tail", 280, 0.005, 1.0, 0.0, 0.001, self.rng, cls=code))
                    manifest = {w["name"]: np.zeros(64) for w in windows}
                    if thin_axis == "band":
                        manifest["tail"] = np.zeros(8192)
                    plain = v2.verdict(windows)
                    lines = v2.r4_assessment(original, windows, plain, manifest, v2.LIMITS)
                    thin = ('content class "code": 3 original windows, median 1.000, bar 3.000;'
                            if thin_axis == "content" else 'prompt band "prompt<=16K":')
                    self.assertIn(f"{thin} 280 scored rows (provisional)", lines)
                    self.assertEqual(lines[-1], "frozen-bar assessment would be UNRESOLVED")

    def test_r4_assessment_keeps_ordinary_status_with_valid_or_invalid_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = [f"toy-{i}" for i in range(13)]
            d = self.kit(tmp, cases, cand_ids=32)
            classes = d / "classes.json"
            classes.write_text(json.dumps(dict.fromkeys(cases, "prose")), encoding="utf-8")
            original = d / "original.json"
            args = ("--classes", str(classes))
            for tier, rc, would_be in (("fast", 1, "FAIL"), ("turbo", 0, "UNRESOLVED"), ("turbo", 3, "UNRESOLVED")):
                with self.subTest(tier=tier, rc=rc):
                    if rc == 3:
                        classes.write_text(json.dumps({**dict.fromkeys(cases, "prose"), cases[-1]: "code"}),
                                           encoding="utf-8")
                    tier_args = (*args, "--tier", tier)
                    base = self.gate(d, *tier_args)
                    self.assertEqual(base[0], rc, base[1])
                    base_json = (d / "g.json").read_bytes()
                    for calibration, expected in (
                        ('{"prose": [1, 1, Infinity]}', "diagnostic unavailable"),
                        ('{"prose": [1e308, 1e308, 1e308]}', "diagnostic unavailable"),
                        ('{"prose": [3, 3, 3], "code": [3, 3, 3]}', f"would be {would_be}"),
                    ):
                        original.write_text(calibration, encoding="utf-8")
                        report = io.StringIO()
                        with contextlib.redirect_stderr(report):
                            self.assertEqual(self.gate(d, *tier_args, "--r4-assess", str(original)), base)
                        self.assertIn(expected, report.getvalue())
                        self.assertEqual((d / "g.json").read_bytes(), base_json)

    def diagnostic_fixture(self, tmp):
        cases = [f"toy-{i}" for i in range(13)]
        d = self.kit(tmp, cases)
        classes = d / "classes.json"
        classes.write_text(json.dumps(dict.fromkeys(cases, 'prose')), encoding="utf-8")
        original = d / "original.json"
        original.write_text('{"prose": [1, 1, 1]}', encoding="utf-8")
        return d, ("--classes", str(classes)), original

    def test_r4_assessment_bounds_input_before_json_decoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, args, original = self.diagnostic_fixture(tmp)
            base = self.gate(d, *args)
            self.assertEqual(base[0], 0, base[1])
            receipt = (d / "g.json").read_bytes()
            depth = sys.getrecursionlimit() + 32
            for payload, reason in (
                (b'{"prose":' + b'[' * depth + b'1' + b']' * depth + b'}', "nesting"),
                (b' ' * (1024 * 1024) + b'{}', "size"),
                (b'\xff', "UnicodeDecodeError"),
                (b'{"prose":', "JSONDecodeError"),
            ):
                with self.subTest(reason=reason):
                    original.write_bytes(payload)
                    report = io.StringIO()
                    with contextlib.redirect_stderr(report):
                        self.assertEqual(self.gate(d, *args, "--r4-assess", str(original)), base)
                    self.assertEqual((d / "g.json").read_bytes(), receipt)
                    self.assertRegex(report.getvalue(), rf"^diagnostic unavailable: .*{reason}[^\n]*\n$")

    def test_r4_assessment_isolates_decoder_exceptions_and_self_referential_data(self):
        cycle = []
        cycle.append(cycle)
        with tempfile.TemporaryDirectory() as tmp:
            d, args, original = self.diagnostic_fixture(tmp)
            base = self.gate(d, *args)
            self.assertEqual(base[0], 0, base[1])
            receipt = (d / "g.json").read_bytes()
            loads = json.loads
            original_text = original.read_text(encoding="utf-8")
            for injected, reason in ((RecursionError("injected"), "RecursionError"),
                                     (SystemExit(99), "SystemExit"),
                                     ({"prose": cycle}, "finite, nonnegative numbers")):
                def load(data, *a, **kw):
                    if data != original_text:
                        return loads(data, *a, **kw)
                    if isinstance(injected, BaseException):
                        raise injected
                    return injected

                with self.subTest(reason=reason), mock.patch.object(v2.json, "loads", side_effect=load):
                    report = io.StringIO()
                    with contextlib.redirect_stderr(report):
                        self.assertEqual(self.gate(d, *args, "--r4-assess", str(original)), base)
                    self.assertEqual((d / "g.json").read_bytes(), receipt)
                    self.assertRegex(report.getvalue(), rf"^diagnostic unavailable: .*{reason}[^\n]*\n$")

    def test_r4_assessment_isolates_writer_errors_and_closed_stream(self):
        class FlushFails(io.StringIO):
            def flush(self):
                raise OSError("injected stdout flush failure")

        with tempfile.TemporaryDirectory() as tmp:
            d, args, original = self.diagnostic_fixture(tmp)
            base = self.gate(d, *args)
            self.assertEqual(base[0], 0, base[1])
            receipt = (d / "g.json").read_bytes()
            args += ("--r4-assess", str(original))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(self.gate(d, *args, stdout=FlushFails()), base)
            self.assertEqual((d / "g.json").read_bytes(), receipt)

            closed = io.StringIO()
            closed.close()
            broken, failing = io.StringIO(), io.StringIO()
            broken.write = mock.Mock(side_effect=BrokenPipeError("injected"))
            failing.write = mock.Mock(side_effect=OSError("injected"))
            for stderr in (closed, broken, failing):
                with self.subTest(stderr=stderr), contextlib.redirect_stderr(stderr):
                    self.assertEqual(self.gate(d, *args), base)
                    self.assertEqual((d / "g.json").read_bytes(), receipt)

            write = mock.Mock(side_effect=[UnicodeEncodeError("ascii", "\u00e9", 0, 1, "injected"), None])
            stderr = io.StringIO()
            stderr.write = write
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(self.gate(d, *args), base)
            self.assertEqual((d / "g.json").read_bytes(), receipt)
            self.assertEqual(write.call_count, 2)
            self.assertRegex(write.call_args.args[0], r"^diagnostic unavailable: UnicodeEncodeError [^\n]*\n$")

    def test_r4_assessment_keeps_process_exit_with_broken_stderr_pipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, args, original = self.diagnostic_fixture(tmp)
            command = [sys.executable, v2.__file__, *self.gate_args(d, *args)]
            baseline = subprocess.run(command, capture_output=True, check=True)
            receipt = (d / "g.json").read_bytes()
            read_fd, write_fd = os.pipe()
            os.close(read_fd)
            try:
                assessed = subprocess.run([*command, "--r4-assess", str(original)], stdout=subprocess.PIPE, stderr=write_fd)
            finally:
                os.close(write_fd)
            self.assertEqual((assessed.returncode, assessed.stdout), (baseline.returncode, baseline.stdout))
            self.assertEqual((d / "g.json").read_bytes(), receipt)

    def test_r4_assessment_starts_after_stdout_and_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, args, original = self.diagnostic_fixture(tmp)
            base = self.gate(d, *args)
            receipt = (d / "g.json").read_bytes()
            stdout = io.StringIO()
            flushed = mock.Mock()
            stdout.flush = flushed
            actual_read = v2._read_r4_original
            observed = []

            def read(path):
                observed.append((stdout.getvalue(), (d / "g.json").read_bytes(), flushed.call_count))
                return actual_read(path)

            with mock.patch.object(v2, "_read_r4_original", side_effect=read), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(self.gate(d, *args, "--r4-assess", str(original), stdout=stdout), base)
            self.assertEqual(observed, [(base[1], receipt, 1)])

    def test_r4_assessment_refuses_shared_stdout_and_stderr_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, args, original = self.diagnostic_fixture(tmp)
            base = self.gate(d, *args)
            receipt = (d / "g.json").read_bytes()
            shared = io.StringIO()
            with contextlib.redirect_stderr(shared):
                assessed = self.gate(d, *args, "--r4-assess", str(original), stdout=shared)
            self.assertEqual(assessed, base)
            self.assertEqual((d / "g.json").read_bytes(), receipt)

    def test_r4_assessment_times_out_blocked_stdout_flush(self):
        started, release = threading.Event(), threading.Event()

        class BlockingFlush(io.StringIO):
            def flush(self):
                started.set()
                release.wait(3)

        with tempfile.TemporaryDirectory() as tmp:
            d, args, original = self.diagnostic_fixture(tmp)
            base = self.gate(d, *args)
            receipt = (d / "g.json").read_bytes()
            with mock.patch.object(v2, "_read_r4_original") as read, contextlib.redirect_stderr(io.StringIO()):
                result = []
                caller = threading.Thread(target=lambda: result.append(
                    self.gate(d, *args, "--r4-assess", str(original), stdout=BlockingFlush())))
                caller.start()
                try:
                    self.assertTrue(started.wait(2))
                    caller.join(1.5)
                    self.assertFalse(caller.is_alive())
                    read.assert_not_called()
                    self.assertEqual(result, [base])
                    self.assertEqual((d / "g.json").read_bytes(), receipt)
                finally:
                    release.set()
                    caller.join()

    def test_r4_assessment_refuses_open_files_after_paths_are_replaced(self):
        def open_for_append(path):
            if os.name != "nt":
                return path.open("ab")
            # Allow replacement of the pathname while this destination remains open.
            import ctypes
            import msvcrt
            from ctypes import wintypes

            create_file = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
            create_file.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                    wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
            create_file.restype = wintypes.HANDLE
            handle = create_file(str(path), 4, 7, None, 3, 0, None)
            if handle == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            return os.fdopen(msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_APPEND | os.O_BINARY), "ab")

        with tempfile.TemporaryDirectory() as tmp:
            d, args, original = self.diagnostic_fixture(tmp)
            base = self.gate(d, *args)
            receipt = d / "g.json"
            for source in (receipt, d / "classes.json", d / "toy-0-n0.f32"):
                with self.subTest(source=source):
                    expected = source.read_bytes()
                    moved = source.with_suffix(".saved")

                    class RotateOnFlush(io.StringIO):
                        def flush(self):
                            source.rename(moved)
                            source.write_bytes(b"replacement")

                    with open_for_append(source) as stderr, contextlib.redirect_stderr(stderr), \
                            mock.patch.object(v2, "_r4_refusal") as refusal:
                        self.assertEqual(self.gate(d, *args, "--r4-assess", str(original), stdout=RotateOnFlush()), base)
                    self.assertEqual(moved.read_bytes(), expected)
                    self.assertEqual(source.read_bytes(), b"replacement")
                    refusal.assert_called_once()
                    moved.replace(source)

    def test_r4_assessment_bounds_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, args, original = self.diagnostic_fixture(tmp)
            base = self.gate(d, *args)
            receipt = (d / "g.json").read_bytes()
            original.write_text(json.dumps({"prose": [1, 1, 1], "large" * 4096: [1, 1, 1]}), encoding="utf-8")
            report = io.StringIO()
            with contextlib.redirect_stderr(report):
                self.assertEqual(self.gate(d, *args, "--r4-assess", str(original)), base)
            self.assertRegex(report.getvalue(), r"^diagnostic unavailable: .*output exceeds size limit[^\n]*\n$")
            self.assertLessEqual(len(report.getvalue().encode("ascii")), 4096)
            self.assertEqual((d / "g.json").read_bytes(), receipt)

    def test_r4_assessment_preserves_receipt_with_stderr_redirections(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, args, original = self.diagnostic_fixture(tmp)
            command = [sys.executable, v2.__file__, *self.gate_args(d, *args)]
            baseline = subprocess.run(command, capture_output=True, check=True, timeout=10)
            receipt = d / "g.json"
            expected = receipt.read_bytes()
            alias = d / "receipt-link"
            os.link(receipt, alias)
            relative_receipt = d / ".." / d.name / "g.json"
            redirects = [(relative_receipt, "wb"), (receipt, "ab"), (alias, "r+b")]
            if (d / "G.JSON").exists():
                redirects.append((d / "G.JSON", "ab"))
            symlink = d / "receipt-symlink"
            try:
                symlink.symlink_to(receipt.name)
            except OSError:  # Some Windows hosts require a privilege for symlinks.
                pass
            else:
                redirects.append((symlink, "ab"))
            for dest, mode in redirects:
                with self.subTest(dest=dest, mode=mode), dest.open(mode) as stderr:
                    assessed = subprocess.run([*command, "--r4-assess", str(original)], stdout=subprocess.PIPE,
                                               stderr=stderr, timeout=10)
                    self.assertEqual((assessed.returncode, assessed.stdout), (baseline.returncode, baseline.stdout))
                    self.assertEqual(receipt.read_bytes(), expected)

            with receipt.open("ab") as stderr, contextlib.redirect_stderr(stderr), \
                    mock.patch.object(v2, "_r4_refusal") as refusal, mock.patch.object(v2, "_read_r4_original") as read:
                self.assertEqual(self.gate(d, *args, "--r4-assess", str(original))[0], baseline.returncode)
            refusal.assert_called_once()
            read.assert_not_called()
            self.assertEqual(receipt.read_bytes(), expected)

    def test_r4_assessment_preserves_inputs_redirected_to_stderr(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, args, original = self.diagnostic_fixture(tmp)
            command = [sys.executable, v2.__file__, *self.gate_args(d, *args)]
            baseline = subprocess.run(command, capture_output=True, check=True, timeout=10)
            receipt = (d / "g.json").read_bytes()
            for source in (original, d / "classes.json", d / "m.tsv", d / "r0.log", d / "toy-0.ids", d / "toy-0-n0.f32"):
                expected = source.read_bytes()
                with self.subTest(source=source), source.open("ab") as stderr:
                    assessed = subprocess.run([*command, "--r4-assess", str(original)], stdout=subprocess.PIPE,
                                               stderr=stderr, timeout=10)
                    self.assertEqual((assessed.returncode, assessed.stdout), (baseline.returncode, baseline.stdout))
                    self.assertEqual(source.read_bytes(), expected)
                    self.assertEqual((d / "g.json").read_bytes(), receipt)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires POSIX FIFOs")
    def test_r4_assessment_refuses_fifo_without_a_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, args, original = self.diagnostic_fixture(tmp)
            command = [sys.executable, v2.__file__, *self.gate_args(d, *args)]
            baseline = subprocess.run(command, capture_output=True, check=True, timeout=10)
            receipt = (d / "g.json").read_bytes()
            original.unlink()
            os.mkfifo(original)
            assessed = subprocess.run([*command, "--r4-assess", str(original)], capture_output=True, timeout=5)
            self.assertEqual((assessed.returncode, assessed.stdout), (baseline.returncode, baseline.stdout))
            self.assertRegex(assessed.stderr, rb"^diagnostic unavailable: .*regular file[^\r\n]*\r?\n$")
            self.assertEqual((d / "g.json").read_bytes(), receipt)

    def test_r4_assessment_keeps_process_exit_with_full_stderr_pipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, args, original = self.diagnostic_fixture(tmp)
            command = [sys.executable, v2.__file__, *self.gate_args(d, *args)]
            baseline = subprocess.run(command, capture_output=True, check=True, timeout=10)
            receipt = (d / "g.json").read_bytes()
            diagnostic = [*command, "--r4-assess", str(original)]
            assessed = subprocess.run(diagnostic, capture_output=True, timeout=5)
            self.assertIn(b"R4 DIAGNOSTIC ONLY: frozen-bar assessment would be ", assessed.stderr)
            read_fd, write_fd = os.pipe()
            try:
                try:
                    os.set_blocking(write_fd, False)
                except OSError:
                    self.skipTest("requires nonblocking pipes to fill the fixture")
                for size in (512, 1):
                    while True:
                        try:
                            if not os.write(write_fd, b"x" * size):
                                break
                        except BlockingIOError:
                            break
                os.set_blocking(write_fd, True)
                assessed = subprocess.run(diagnostic, stdout=subprocess.PIPE, stderr=write_fd, timeout=5)
                self.assertTrue(os.get_blocking(write_fd))
            finally:
                os.close(write_fd)
                os.close(read_fd)
            self.assertEqual((assessed.returncode, assessed.stdout), (baseline.returncode, baseline.stdout))
            self.assertEqual((d / "g.json").read_bytes(), receipt)

    def test_r4_assessment_leaves_no_writer_or_late_bytes_after_embedded_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, args, original = self.diagnostic_fixture(tmp)
            base = self.gate(d, *args)
            receipt = (d / "g.json").read_bytes()
            read_fd, write_fd = os.pipe()
            with os.fdopen(read_fd, "rb", buffering=0) as reader, os.fdopen(write_fd, "wb", buffering=0) as stderr:
                try:
                    os.set_blocking(write_fd, False)
                except OSError:
                    self.skipTest("requires nonblocking pipes to fill the fixture")
                filled = 0
                for size in (512, 1):
                    while True:
                        try:
                            written = os.write(write_fd, b"x" * size)
                        except BlockingIOError:
                            break
                        if not written:
                            break
                        filled += written
                os.set_blocking(write_fd, True)
                with contextlib.redirect_stderr(stderr):
                    assessed = self.gate(d, *args, "--r4-assess", str(original))
                # Keep the calling process alive and drain only after qual() returns.
                # On Windows a pending write can also block closure of the original handle.
                closer = threading.Thread(target=stderr.close, daemon=True)
                closer.start()
                closer.join(1)
                closed_before_drain = not closer.is_alive()
                drained = []
                drain = threading.Thread(target=lambda: drained.append(reader.read()), daemon=True)
                drain.start()
                drain.join(3)
                closer.join(3)
                self.assertFalse(drain.is_alive(), "diagnostic kept the pipe open after return")
                self.assertEqual(drained[0][filled:], b"", "diagnostic wrote after return")
                self.assertEqual(drained[0], b"x" * filled)
                self.assertTrue(closed_before_drain, "diagnostic blocked closing the caller's pipe")
            self.assertEqual(assessed, base)
            self.assertEqual((d / "g.json").read_bytes(), receipt)

    def test_r4_assessment_keeps_process_exit_with_stalled_input_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, args, original = self.diagnostic_fixture(tmp)
            argv = self.gate_args(d, *args)
            baseline = subprocess.run([sys.executable, v2.__file__, *argv], capture_output=True, check=True, timeout=10)
            receipt = (d / "g.json").read_bytes()
            script = ("import os, sys; sys.path.insert(0, sys.argv.pop(1)); import gate_v2 as gate; "
                      "gate._read_r4_original = lambda _: os.read(0, 1); sys.exit(gate.main())")
            read_fd, write_fd = os.pipe()
            try:
                assessed = subprocess.run([sys.executable, "-c", script, str(pathlib.Path(v2.__file__).parent),
                                           *argv, "--r4-assess", str(original)], stdin=read_fd, capture_output=True,
                                          timeout=5)
            finally:
                os.close(write_fd)
                os.close(read_fd)
            self.assertEqual((assessed.returncode, assessed.stdout), (baseline.returncode, baseline.stdout))
            self.assertEqual((d / "g.json").read_bytes(), receipt)


if __name__ == "__main__":
    unittest.main()
