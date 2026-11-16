#!/usr/bin/env python3
"""receipt.py --request-json: the request window is the true request (start .. generation_request_stop), with
prefill and decode sub-windows judged separately (audit of the full-mode anchor in the Qwen length matrix)."""
import os
import re
import shlex
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_runner import load, make_run, receipt, sample  # noqa: E402

LENGTH_MATRIX = os.path.join(HERE, "..", "qwen", "length-matrix.sh")
R_MS = 4000.0  # request: start 16.0, prefill block line 20.0, result object stamped 20.5
BLOCK_T, RESULT_T = 20.0, 20.5


def full_mode_args():
    """The receipt arguments length-matrix.sh binds for full mode (the real binding, not a copy)."""
    text = open(LENGTH_MATRIX).read()
    base = re.search(r"KIT_RECEIPT_ARGS='([^']*)'", text).group(1)
    extra = re.search(r"if \[\[ \$m == full \]\]; then\s+KIT_RECEIPT_ARGS\+='([^']*)'", text).group(1)
    return shlex.split(base + extra)


class RequestWindow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def run_arm(self, evict_at=(), extra=None, stamp_result=True, block=True):
        d = os.path.join(self.tmp.name, "arm")
        ev, rows = 0, []
        for i in range(0, 101):  # samples every 0.25 s over [0, 25]
            t = i * 0.25
            ev += sum(100 for e in evict_at if t - 0.25 < e <= t)
            rows.append(sample(t, ev=ev))
        lines = [(1.0, "loading weights"), (16.1, "qpf2k HC out16 active (rows 8192)")]
        if block:
            lines.append((BLOCK_T, "qpf2k sched block rows 8192 detached_shared 0 late 0"))
        lines.append((22.0, "strix power: {}"))
        out = [(0.5, "banner"), (RESULT_T, "{"), (RESULT_T, '  "generation_request_elapsed_ms":%.2f,' % R_MS),
               (RESULT_T, '  "generated_tokens": [279]'), (RESULT_T, "}")]
        make_run(d, lines, rows, stdout="".join(l + "\n" for _, l in out))
        if stamp_result:
            with open(os.path.join(d, "stdout.ts"), "w") as f:
                f.write("".join("%.3f %s\n" % (t, l) for t, l in out))
        with open(os.devnull, "w") as null:
            old, sys.stdout = sys.stdout, null
            try:
                receipt.main(["summarize", d, "--tag", "arm", "--tokens", "8192"] + (full_mode_args() if extra is None else extra))
            finally:
                sys.stdout = old
        return load(d)

    def test_full_mode_binding_uses_request_windows(self):
        args = full_mode_args()
        self.assertIn("--request-json", args)
        self.assertEqual(args[args.index("--scope") + 1], "prefill")

    def test_windows_span_the_true_request(self):
        rec = self.run_arm()
        w = rec["windows"]
        self.assertEqual((w["request"]["t0"], w["request"]["t1"]), (16.0, RESULT_T))
        self.assertEqual((w["prefill"]["t0"], w["prefill"]["t1"]), (16.0, BLOCK_T))
        self.assertEqual((w["decode"]["t0"], w["decode"]["t1"]), (BLOCK_T, RESULT_T))
        self.assertTrue(rec["valid"] and rec["valid_prefill"] and rec["valid_decode"] and rec["valid_request"])
        self.assertEqual(rec["prefill_ms_bounds"], [R_MS - 500.0, R_MS])
        self.assertEqual(rec["decode_ms_bound"], 500.0)
        self.assertIn("upper bound", rec["prefill_source"])
        self.assertEqual(rec["request_ms"], R_MS)

    def test_decode_eviction_counts_in_request_not_prefill(self):
        # The audited regression: an eviction after the prefill block line but before the request stop.
        rec = self.run_arm(evict_at=(20.3,))
        self.assertGreater(rec["windows"]["decode"]["evicted_ms"], 0)
        self.assertGreater(rec["windows"]["request"]["evicted_ms"], 0)
        self.assertEqual(rec["windows"]["prefill"]["evicted_ms"], 0)
        self.assertTrue(rec["valid"] and rec["valid_prefill"])
        self.assertFalse(rec["valid_decode"] or rec["valid_request"])
        self.assertTrue(any("request receipt INVALID" in w for w in rec["warnings"]))
        req = self.run_arm(evict_at=(20.3,), extra=full_mode_args()[:-2] + ["--scope", "request"])
        self.assertFalse(req["valid"])
        self.assertIn("kfd evicted_ms +100 in the request window", req["invalid_reasons"])

    def test_prefill_eviction_keeps_decode_valid(self):
        rec = self.run_arm(evict_at=(18.0,))
        self.assertFalse(rec["valid"] or rec["valid_prefill"] or rec["valid_request"])
        self.assertTrue(rec["valid_decode"])

    def test_load_and_teardown_are_outside_the_request(self):
        rec = self.run_arm(evict_at=(5.0, 23.0))
        self.assertTrue(rec["valid_prefill"] and rec["valid_decode"] and rec["valid_request"])
        self.assertEqual(rec["evicted_ms_run"], 200)
        self.assertTrue(any("outside the request window" in w for w in rec["warnings"]))

    def test_unstamped_result_or_missing_witness_fails_closed(self):
        for kw in (dict(stamp_result=False), dict(block=False)):
            rec = self.run_arm(**kw)
            self.assertFalse(rec["valid"] or rec["valid_decode"] or rec["valid_request"], kw)

    def test_prefill_json_alias(self):
        args = ["--need-prefill", "--prefill-json", "generation_request_elapsed_ms", "--anchor-regex",
                "^qpf2k.sched.block.rows"]
        rec = self.run_arm(evict_at=(20.3,), extra=args)
        self.assertEqual(rec["scope"], "request")
        self.assertFalse(rec["valid"])


if __name__ == "__main__":
    unittest.main()
