import importlib.util
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
spec = importlib.util.spec_from_file_location("native_first", HERE / "strata-first.py")
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


class SelectionSliceTests(unittest.TestCase):
    def test_disjoint_shards_reconstruct_frozen_order(self):
        frozen = [64, 128, 256, 512, 1024]
        first, a = native.selection_slice(frozen, 0, 2)
        last, b = native.selection_slice(frozen, 2, 3)
        self.assertEqual(first + last, frozen)
        self.assertFalse(a["complete"])
        self.assertFalse(b["complete"])
        self.assertEqual(b, {"start": 2, "count": 3, "total": 5, "complete": False})

    def test_default_is_complete_and_invalid_slices_fail(self):
        self.assertTrue(native.selection_slice([64, 128])[1]["complete"])
        for start, count in [(-1, 1), (0, 0), (1, 2), (2, None)]:
            with self.assertRaises(ValueError):
                native.selection_slice([64, 128], start, count)


if __name__ == "__main__":
    unittest.main()
