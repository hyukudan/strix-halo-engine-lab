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
    def test_request_cannot_change_frozen_teacher_arithmetic(self):
        native.validate_teacher_kv({}, "fp16")
        native.validate_teacher_kv({"teacher_kv": "int8"}, "int8")
        for protocol, requested in (({}, "int8"), ({"teacher_kv": "int8"}, "fp16")):
            with self.assertRaises(ValueError):
                native.validate_teacher_kv(protocol, requested)

    def test_teacher_kv_is_explicit_and_overrides_serving_config(self):
        config = {"exe": "engine", "args": ["--kv", "int8", "--max-context", "100"]}
        for requested, expected in ((None, "fp16"), ("int8", "int8")):
            argv = native.engine_args(config, 522, *([] if requested is None else [requested]))
            self.assertEqual(argv.count("--kv"), 1)
            self.assertEqual(argv[argv.index("--kv") + 1], expected)
            self.assertEqual(argv[argv.index("--max-context") + 1], "522")
        with self.assertRaises(ValueError):
            native.engine_args(config, 522, "unknown")

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
