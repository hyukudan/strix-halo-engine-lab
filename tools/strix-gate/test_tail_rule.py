#!/usr/bin/env python3
"""Host tests of the tail event rule (tools/strix-gate/tail_rule.py); synthetic rows, no model, no GPU."""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fast_logit_gate as g  # noqa: E402
import tail_rule as t  # noqa: E402


def arm(margins, flips):
    """One prompt's compare_logits-like result: rows with the given reference margins, flipped at `flips`."""
    agree = np.ones(len(margins), bool)
    agree[list(flips)] = False
    return {"agree": agree, "margin_ref": np.asarray(margins, float)}


class TailRule(unittest.TestCase):
    def test_ref_margin(self):
        z = np.array([[0.0, 3.0, 1.0, -2.0], [5.0, 5.0, 0.0, 1.0]])
        np.testing.assert_allclose(t.ref_margin(z), [2.0, 0.0])
        # the gap of logits equals the gap of log-probabilities
        np.testing.assert_allclose(t.ref_margin(g.log_softmax(z)), [2.0, 0.0], atol=1e-12)
        with self.assertRaises(ValueError):
            t.ref_margin(np.zeros((3, 1)))

    def test_near_ties_are_not_events(self):
        rng = np.random.default_rng(1)
        m = rng.uniform(0.0, 1.0, 4000)
        flips = np.flatnonzero(m < 0.05)                        # many flips, all near ties
        v = t.tail_verdict([arm(m, flips)], [[arm(m, flips[:3])]], binom_sf=g.binom_sf)
        self.assertEqual(v["events"], 0)
        self.assertTrue(v["pass"])

    def test_confident_flips_fail_against_a_quiet_null(self):
        m = np.full(16000, 6.0)
        cand = arm(m, range(0, 16000, 2000))                    # 8 confident flips in 16,000 rows
        null = [[arm(m, [])]]                                   # the null never flips at 6 nats
        v = t.tail_verdict([cand], null, binom_sf=g.binom_sf)
        self.assertEqual((v["events"], v["null_events"], v["null_rows"]), (8, 0, 16000))
        self.assertAlmostEqual(v["p0"], 1 / 16002)
        self.assertLess(v["p"], 0.01)
        self.assertFalse(v["pass"])
        self.assertEqual(v["first"][0], {"prompt": 0, "row": 0, "margin": 6.0})
        self.assertIn("-> FAIL", t.describe(v, ["p"]))

    def test_rate_comes_from_the_null(self):
        m = np.full(16000, 6.0)
        cand = arm(m, range(0, 16000, 2000))
        noisy = [[arm(m, range(0, 16000, 1600))], [arm(m, range(1, 16000, 1600))]]  # 10 + 10 null events
        v = t.tail_verdict([cand], noisy, binom_sf=g.binom_sf)
        self.assertEqual(v["null_events"], 20)
        self.assertTrue(v["pass"])                              # 8 events at the null's own rate is not a tail
        # one or two events on a quiet null are within rounding (no single-row verdict)
        self.assertTrue(t.tail_verdict([arm(m, [5])], [[arm(m, [])]], binom_sf=g.binom_sf)["pass"])

    def test_pools_prompts_and_checks_shapes(self):
        rows, events, found = t.count_events([arm([3.0, 0.5], [0, 1]), arm([2.0, 9.0], [1])])
        self.assertEqual((rows, events), (4, 2))
        self.assertEqual([(p, r) for p, r, _ in found], [(0, 0), (1, 1)])
        with self.assertRaises(ValueError):
            t.count_events([{"agree": np.ones(3, bool), "margin_ref": np.ones(2)}])


if __name__ == "__main__":
    unittest.main()
