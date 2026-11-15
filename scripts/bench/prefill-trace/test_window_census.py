#!/usr/bin/env python3
"""Hand-computed layout through window_census (python3 -m unittest test_window_census.py)."""
import csv
import os
import tempfile
import unittest

import window_census as wc

COLS = ["Kernel_Name", "Start_Timestamp", "End_Timestamp", "Queue_Id", "Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z",
        "Workgroup_Size_X", "Workgroup_Size_Y", "Workgroup_Size_Z", "LDS_Block_Size"]


def row(name, s_us, e_us, queue, ctas, lds, wg=256):
    return [name, s_us * 1000, e_us * 1000, queue, ctas * wg, 1, 1, wg, 1, 1, lds]


class Census(unittest.TestCase):
    def setUp(self):
        rows = [row("embedding_kernel", 0, 10, 1, 1, 0)]
        # two 16-CTA flash launches at a time on 4 queues for 400 us: 32 of 40 CUs, no idle
        for i in range(4):
            for q in (0, 1):
                rows.append(row("void strix::attn_core::sparse::flash_rows_kernel<P, 32u>", 100 + 100 * i, 200 + 100 * i,
                                1 + q + 2 * (i % 2), 16, 60416))
        # a select pass that cannot share a CU with a flash CTA (LDS) fills the other 8 CUs for the whole loop
        rows.append(row("void strix::ds41::attn::hip::select_slice_kernel<false>", 100, 500, 4, 8, 52736, 1024))
        rows.append(row("dense_kernel", 500, 600, 1, 100, 0))  # ends the cluster
        fd, self.path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLS)
            w.writerows(rows)

    def tearDown(self):
        os.unlink(self.path)

    def test_layout(self):
        k = wc.load(self.path)
        r = wc.census(k, wc.DS41_KINDS, "flash_rows", 40, 50 * 1000)
        self.assertEqual(r["clusters"], 1)
        self.assertEqual(r["loop_launches"], 9)
        self.assertAlmostEqual(r["span_ms"], 0.4)
        self.assertAlmostEqual(r["idle_ms"], 0.0)
        self.assertAlmostEqual(r["deficit_ms"], 0.4 * 8 / 40)  # 32 flash CTAs of 40 CUs for 400 us
        self.assertAlmostEqual(r["deficit_edge_ms"], 0.1 * 8 / 40)  # 50 us at each end
        self.assertAlmostEqual(r["free_ms"], 0.0)  # the select CTAs take the other 8 CUs
        self.assertEqual(r["queues"], {1: 2, 2: 2, 3: 2, 4: 3})

    def test_cli_last_run(self):
        self.assertEqual(wc.main([self.path, "--run", "-1"]), 0)


if __name__ == "__main__":
    unittest.main()
