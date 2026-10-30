import unittest
from dspark_pairs import intervals


class PairsTest(unittest.TestCase):
    def setUp(self):
        self.key = ["json", 1, "decode_tok_s", "original"]
        self.rows = [dict(key=self.key, block=i, off=10, auto=11, complete=True, exact=True) for i in range(8)]

    def test_interval_and_boundary(self):
        self.assertEqual(intervals(self.rows, [self.key])[0]["lower"], 1)
        for r in self.rows:
            r["auto"] = 9
        self.assertEqual(intervals(self.rows, [self.key])[0]["decision"], "reject_slower")
        for r in self.rows:
            r["auto"] = 9 if r["block"] % 2 else 11
        self.assertEqual(intervals(self.rows, [self.key])[0]["decision"], "not_demonstrated_slower")
        for r in self.rows:
            r["auto"] = 10
        self.assertEqual(intervals(self.rows, [self.key])[0]["upper"], 0)
        self.assertEqual(intervals(self.rows, [self.key])[0]["decision"], "not_demonstrated_slower")

    def test_fail_closed(self):
        for rows in (self.rows[:-1], self.rows + self.rows[:1], [dict(self.rows[0], auto=float("nan"))] + self.rows[1:],
                     [dict(self.rows[0], complete=False)] + self.rows[1:], [dict(self.rows[0], exact=False)] + self.rows[1:]):
            with self.assertRaises(ValueError):
                intervals(rows, [self.key])
        with self.assertRaises(ValueError):
            intervals(self.rows, [self.key, ["code", 1, "decode_tok_s", "original"]])
        with self.assertRaises(ValueError):
            intervals(self.rows, [self.key, self.key])


if __name__ == "__main__":
    unittest.main()
