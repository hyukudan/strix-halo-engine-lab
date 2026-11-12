#!/usr/bin/env python3
"""Tests of tools/dense_fast_quantize.py: python -m unittest tools/test_dense_fast_quantize.py (numpy only)."""
import os, sys, unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense_fast_quantize as q  # noqa: E402


def reference_q4k(raw):
    """Scalar decode of block_q4_K written from the GGUF layout, independent of unpack_super."""
    out = []
    for o in range(0, len(raw), 144):
        b = raw[o:o + 144]
        d = float(np.frombuffer(b[0:2], "<f2")[0])
        dmin = float(np.frombuffer(b[2:4], "<f2")[0])
        s = b[4:16]
        qs = b[16:144]
        for j in range(8):
            if j < 4:
                sc, m = s[j] & 63, s[j + 4] & 63
            else:
                sc, m = (s[j + 4] & 15) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4)
            for l in range(32):
                byte = qs[32 * (j // 2) + l]
                code = byte >> 4 if j & 1 else byte & 15
                out.append(np.float32(d * sc) * np.float32(code) - np.float32(dmin * m))
    return np.array(out, np.float32)


class DenseFastQuantize(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.w = (rng.standard_normal((6, 768)) * 0.03).astype(np.float32)
        self.w[1, :96] = 0.0        # flat blocks
        self.w[2, 300] = 0.9        # outlier
        self.w[3] = np.abs(self.w[3])  # one-sided blocks (min clamps to 0)

    def test_sizes_and_error(self):
        for fmt, size, bound in (("q4_k", 144, 0.11), ("q5_k", 176, 0.055)):
            raw, num, den = q.quantize_matrix(self.w, fmt)
            self.assertEqual(len(raw), self.w.size // 256 * size)
            self.assertLess((num / den) ** 0.5, bound)
            y = q.unpack_super(raw, fmt).reshape(self.w.shape)
            self.assertTrue(np.all(y[1, :96] == 0.0))
            self.assertAlmostEqual(float(((y - self.w).astype(np.float64) ** 2).sum()), num, places=9)

    def test_q4k_layout_matches_scalar_reference(self):
        raw, _, _ = q.quantize_matrix(self.w, "q4_k")
        np.testing.assert_array_equal(q.unpack_super(raw, "q4_k").reshape(-1), reference_q4k(raw))

    def test_q5k_high_bits(self):
        s = 3
        d = np.full(s, 0.01, np.float32)
        dmin = np.full(s, 0.02, np.float32)
        ls = np.tile(np.arange(8, dtype=np.uint8) * 9, (s, 1))
        lm = np.tile(63 - np.arange(8, dtype=np.uint8) * 7, (s, 1))
        codes = (np.arange(s * 256, dtype=np.uint32) * 7 % 32).astype(np.uint8).reshape(s, 8, 32)
        raw = q.pack_super(d, dmin, ls, lm, codes, "q5_k").tobytes()
        y = q.unpack_super(raw, "q5_k").reshape(s, 8, 32)
        want = (q.f16(d)[:, None] * ls)[..., None] * codes - (q.f16(dmin)[:, None] * lm)[..., None]
        np.testing.assert_array_equal(y, want.astype(np.float32))

    def test_q6k_layout_and_error(self):
        raw, num, den = q.quantize_matrix(self.w, "q6_k")
        self.assertEqual(len(raw), self.w.size // 256 * 210)
        self.assertLess((num / den) ** 0.5, 0.03)
        y = q.unpack_super_q6(raw)
        self.assertTrue(np.all(y.reshape(self.w.shape)[1, :96] == 0.0))
        ref = []
        for o in range(0, len(raw), 210):  # scalar decode of block_q6_K
            b = raw[o:o + 210]
            d = np.frombuffer(b[208:210], "<f2")[0].astype(np.float32)
            sc = np.frombuffer(b[192:208], np.int8)
            for half in range(2):
                ql, qh = b[64 * half:64 * half + 64], b[128 + 32 * half:160 + 32 * half]
                y4 = np.empty(128, np.float32)
                for l in range(32):
                    i = l // 16
                    codes = ((ql[l] & 15) | ((qh[l] & 3) << 4), (ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4),
                             (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4), (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4))
                    for k in range(4):
                        y4[l + 32 * k] = d * np.float32(sc[8 * half + i + 2 * k]) * np.float32(codes[k] - 32)
                ref.append(y4)
        np.testing.assert_array_equal(y.reshape(-1), np.concatenate(ref))

    def test_refine_never_worse(self):
        for fmt in ("q4_k", "q5_k"):  # the min types
            x = self.w.reshape(-1, 8, 32)
            w = (np.sqrt((x * x).mean(-1, keepdims=True)) + np.abs(x)).astype(np.float32)
            nmax = q.FORMATS[fmt][1]

            def err(refine):
                d, dmin, ls, lm, codes = q.quantize_super(x, w, nmax, refine)
                y = (d[:, None] * ls)[..., None] * codes - (dmin[:, None] * lm)[..., None]
                return float((w * (y - x) ** 2).sum())
            self.assertLessEqual(err(True), err(False) * (1 + 1e-6))


if __name__ == "__main__":
    unittest.main()
