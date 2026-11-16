"""Protocol tests: a shared window is insufficient without a shared committed prefix."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('mtp_study', Path(__file__).with_name('mtp-study-analyze.py'))
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)

def input_hash(inputs):
    h = 14695981039346656037
    for token in inputs:
        for shift in range(0, 32, 8):
            h = ((h ^ ((token >> shift) & 255)) * 1099511628211) & ((1 << 64) - 1)
    return h

def write_dump(path, windows):
    with path.open('wb') as f:
        for base, inputs in windows:
            for row, consumed in enumerate(inputs):
                f.write(study.HEADER.pack(study.MAGIC, base + row, base, row, len(inputs), 3,
                                          consumed, input_hash(inputs[:row + 1]), 0, 1))
                f.write(np.array([0., 1., 2.], dtype='<f4').tobytes())

def run(ids, request_sha='same'):
    request = dict(token_ids=ids, prompt=100, lane=0, request_sha256=request_sha)
    return [request], lambda ns: (0, request), {0: []}, {}

class ContextMatching(unittest.TestCase):
    def test_speculative_input_need_not_match_eventual_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows.bin'
            write_dump(path, [(100, [5, 42])])
            indexed, counts = study.indexed_logits(path, run([5, 8]))
            self.assertEqual(counts['indexed'], 2)
            self.assertEqual(len(indexed), 2)

    def test_identical_current_input_does_not_hide_changed_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            left, right = Path(tmp) / 'l.bin', Path(tmp) / 'r.bin'
            write_dump(left, [(100, [5]), (102, [9])])
            write_dump(right, [(100, [5]), (102, [9])])
            result = study.compare_logits(left, right, run([5, 6, 9]), run([5, 7, 9]))
            self.assertEqual(result['matched_context_rows'], 1)
            self.assertEqual(result['classes']['code']['bitwise_rows'], 1)

    def test_prompt_identity_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows.bin'
            write_dump(path, [(100, [5])])
            with self.assertRaisesRegex(ValueError, 'prompt identity'):
                study.compare_logits(path, path, run([5], 'left'), run([5], 'right'))

    def test_corrupt_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows.bin'
            path.write_bytes(b'partial')
            with self.assertRaisesRegex(ValueError, 'truncated'):
                list(study.dump_records(path))

if __name__ == '__main__':
    unittest.main()
