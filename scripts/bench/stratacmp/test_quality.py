import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

import quality


class QualityTests(unittest.TestCase):
    def test_native_first_rows_preserve_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            (d / "ids").write_text("0 1 2 3")
            (d / "positions").write_text("0 2")
            for i in range(2):
                np.full(248320, i, dtype="<f4").tofile(d / ("first." + str(i)))
            quality.convert_first(d / "first", d / "ids", d / "positions", d / "out")
            got = list(quality.spool.read(d / "out"))
            self.assertEqual([r[0] for r in got], [0, 2])
            self.assertEqual(got[1][3][17], 1)

    def test_spread_has_true_next_token(self):
        self.assertEqual(quality.selections(9, 3), [3, 6])
        self.assertEqual(quality.selections(10, 3), [3, 6])
        with self.assertRaises(ValueError):
            quality.selections(9, 0)

    def test_strata_conversion_drops_only_unlabeled_row(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            (d / "ids").write_text("0 1 2 3")
            rows = np.zeros((3, 248320), dtype="<f4")
            rows[1, 7] = 3
            (d / "raw").write_bytes(struct.pack("<ii", 248320, 3) + rows.tobytes())
            quality.convert(d / "raw", d / "ids", 2, d / "out")
            got = list(quality.spool.read(d / "out"))
            self.assertEqual([(p, label) for p, _, label, _ in got], [(2, 3)])
            self.assertEqual(got[0][3][7], 3)
            with self.assertRaises(ValueError):
                quality.convert(d / "raw", d / "ids", 1, d / "bad")

    def test_comparison_and_different_input_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            a = np.array([[1., 2., 3.]], dtype="<f4")
            quality.spool.write(d / "E", [0, 1], [0], a)
            quality.spool.write(d / "T", [0, 1], [0], a + 3)
            result = quality.compare(d / "E", ["T=" + str(d / "T")])["arms"]["T"]
            self.assertAlmostEqual(result["kl_ref_run"], 0)
            self.assertAlmostEqual(result["dnll_run_minus_ref"], 0)
            self.assertEqual(result["top1_agreement"], 1)
            self.assertFalse(result["bitwise_equal"])
            quality.spool.write(d / "bad", [0, 2], [0], a)
            with self.assertRaises(ValueError):
                quality.compare(d / "E", ["T=" + str(d / "bad")])


if __name__ == "__main__":
    unittest.main()
