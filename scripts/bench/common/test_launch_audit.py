#!/usr/bin/env python3
"""Hand-computed idle-before and device-busy figures through launch_audit (python3 -m unittest test_launch_audit.py)."""
import contextlib
import csv
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import launch_audit as la

COLS = ["Kernel_Name", "Start_Timestamp", "End_Timestamp", "Queue_Id", "Workgroup_Size_X", "Grid_Size_X",
        "Arch_VGPR_Count", "LDS_Block_Size"]


def row(name, s_us, e_us, queue=1):
    return [name, int(s_us * 1000), int(e_us * 1000), queue, 256, 10240, 32, 0]


class LaunchAudit(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.dir.cleanup()

    def run_csv(self, rows, *extra):
        path = os.path.join(self.dir.name, "trace_kernel_trace.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLS)
            w.writerows(rows)
        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["launch_audit.py", path, "--mp", "40", *extra]), contextlib.redirect_stdout(out):
            la.main()
        lines = out.getvalue().splitlines()
        table = {ln.split(" | ")[0]: ln.split(" | ") for ln in lines[3:]}
        return lines, table

    def test_long_gaps_are_kept(self):
        # three 10 us launches with two 600 us gaps: before the fix every gap >= 50 us was dropped and the
        # first launch recorded a fake 0, so the median read 0.00
        _, t = self.run_csv([row("k_a()", 0, 10), row("k_a()", 610, 620), row("k_a()", 1220, 1230)])
        self.assertEqual(t["k_a"][8:12], ["2", "600.00", "600.00", "600.00"])

    def test_unsorted_rows(self):
        # rocprofv3 CSV not in start order; true gaps 20 and 30 us
        _, t = self.run_csv([row("k_a()", 70, 80), row("k_a()", 0, 10), row("k_a()", 30, 40)])
        self.assertEqual(t["k_a"][8:12], ["2", "25.00", "29.50", "30.00"])

    def test_first_launch_has_no_gap(self):
        # two kernels, one gap g = 15 us: the gap belongs to the second launch only (before: median(0, g) = g / 2)
        _, t = self.run_csv([row("k_a()", 0, 10), row("k_b()", 25, 35)])
        self.assertEqual(t["k_a"][8], "0")
        self.assertEqual(t["k_a"][9], "nan")
        self.assertEqual(t["k_b"][8:10], ["1", "15.00"])

    def test_streams_and_overlap(self):
        # queue 1: a long launch then a short one that starts inside it and a third after the long one ends;
        # the stream's busy end is the latest end, so the third gap is 5 us (not 85 us past the short launch)
        # and the overlapping launch records 0. Queue 2 runs alone and its first launch adds nothing.
        rows = [row("k_a()", 0, 100, 1), row("k_b()", 10, 20, 1), row("k_c()", 105, 110, 1),
                row("k_d()", 300, 310, 2), row("k_d()", 320, 330, 2)]
        lines, t = self.run_csv(rows)
        self.assertEqual(t["k_b"][8:10], ["1", "0.00"])
        self.assertEqual(t["k_c"][8:10], ["1", "5.00"])
        self.assertEqual(t["k_d"][8:10], ["1", "10.00"])
        # span 0..330 us, busy 0..100 + 105..110 + 300..310 + 320..330 = 125 us
        self.assertTrue(lines[1].startswith("span ms 0.330 | device busy ms 0.125 | device idle ms 0.205"), lines[1])

    def test_short_band_is_a_separate_column(self):
        rows = [row("k_a()", 0, 10), row("k_a()", 15, 20), row("k_a()", 27, 30), row("k_a()", 630, 640)]
        lines, t = self.run_csv(rows, "--short-us", "50")
        self.assertIn("idle<50us p50 (n)", lines[2])
        self.assertEqual(t["k_a"][8:13], ["3", "7.00", "540.70", "600.00", "6.00 (2)"])


if __name__ == "__main__":
    unittest.main()
