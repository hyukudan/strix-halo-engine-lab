import os
import tempfile
import unittest

import numpy as np

import glm_score_nll as g


class GlmScoreNll(unittest.TestCase):
    def write(self, d, name, logits):
        half = logits.shape[1] // 2
        logits[:, :half].astype("<f4").tofile(os.path.join(d, f"{name}-tf-r0.f32"))
        logits[:, half:].astype("<f4").tofile(os.path.join(d, f"{name}-tf-r1.f32"))

    def test_identical_and_perturbed(self):
        rng = np.random.default_rng(1)
        rows, vocab, n = 8, 16, 40
        toks = rng.integers(0, vocab, n).astype("<u4")
        ref = rng.normal(size=(rows, vocab))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "p.u32")
            toks.tofile(p)
            self.write(d, "X1", ref)
            self.write(d, "Y1", ref.copy())
            bumped = ref.copy()
            tgt = toks[n - rows + 1:].astype(int)
            bumped[np.arange(len(tgt)), tgt] += 1.0  # raises the target logit: test NLL must drop
            self.write(d, "Z1", bumped)
            same = g.compare(g.log_softmax("X1", d, rows), g.log_softmax("Y1", d, rows), toks, rows)
            self.assertEqual(same["offset1"]["dnll"], 0.0)
            self.assertAlmostEqual(same["kl_mean"], 0.0, places=12)
            self.assertEqual(same["top1"], 1.0)
            moved = g.compare(g.log_softmax("X1", d, rows), g.log_softmax("Z1", d, rows), toks, rows)
            self.assertLess(moved["offset1"]["dnll"], 0.0)
            self.assertGreater(moved["kl_mean"], 0.0)
            self.assertEqual(moved["offset1"]["rows"], rows - 1)

    def test_short_prompt_refused(self):
        with self.assertRaises(SystemExit):
            g.compare(np.zeros((4, 2)), np.zeros((4, 2)), np.zeros(4, dtype="<u4"), 4)


if __name__ == "__main__":
    unittest.main()
