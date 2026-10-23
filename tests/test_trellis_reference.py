import hashlib
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("trellis_reference", Path(__file__).resolve().parents[1] / "tools/trellis_reference.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


class TrellisReferenceTest(unittest.TestCase):
    def test_bulk_reader_against_scalar_bit_reader_all_rates(self):
        rng = np.random.default_rng(23)
        for bits in range(1, 9):
            p = rng.integers(0, 65536, (3, 2, 16 * bits), dtype=np.uint16)
            np.testing.assert_array_equal(t.decode_bulk(p, bits).view(np.uint16), t.decode(p, bits).view(np.uint16))

    def test_exhaustive_codebook_known_dyadic_hash(self):
        self.assertEqual(hashlib.sha256(t.BOOK.tobytes()).hexdigest(),
                         "bc48d02cb1c14939dc47b90f870dd689d63e3b1ab69a157e65538db68677a6c8")

    def test_ring_and_fragment_mapping(self):
        for bits in range(1, 9):
            p = np.zeros((1, 1, 16 * bits), dtype=np.uint16)
            # Last stream bit is bit0 of the last uint32; it enters the
            # first window at bit index bits, then leaves the window.
            p[0, 0, -2] = 1
            q = t.decode(p, bits)
            for pos in range(256):
                lane, f = divmod(pos, 8)
                i = 2*(lane % 4) + f % 2 + 8*((f//2) % 2)
                o = lane//4 + 8*(f//4)
                end = (pos+1)*bits
                expected = 1 << end if end <= 15 else 0
                if pos == 255:
                    expected = 1
                self.assertEqual(q[i, o].view(np.uint16), t.BOOK[expected].view(np.uint16))

    def test_hadamard_against_explicit_sylvester(self):
        h = np.array([[1]], dtype=np.float64)
        for _ in range(7):
            h = np.block([[h, h], [h, -h]])
        x = np.arange(256, dtype=np.float32).reshape(2, 128) / 128
        np.testing.assert_allclose(t.hadamard(x), x.astype(np.float64) @ h / np.sqrt(128), rtol=2e-6, atol=2e-6)

    def test_pair_tree_and_splits_against_exact_integer_dot(self):
        rng = np.random.default_rng(23)
        a = rng.integers(-3, 4, (8, 128)).astype(np.float16)
        q = rng.integers(-3, 4, (128, 256)).astype(np.float16)
        expected = a.astype(np.float64) @ q.astype(np.float64)
        for splits in (1, 3, 4, 8):
            p = t.core_gemv(a, q, splits)
            np.testing.assert_array_equal(np.sum(p, axis=0), expected)


if __name__ == "__main__":
    unittest.main()
