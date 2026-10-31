"""Language regression and incomplete-reference checks on synthetic receipts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("quality", Path(__file__).parents[2] / "tools/gemma4/exl3_quality.py")
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def docs():
    return {f"{language}_{parent}": dict(parent=f"{language}_{parent}", language=language, rows=10,
             matches=9, nll_positions=9, sum_dnll=0.09, teacher_sha256="fixture",
             metric="quality", kl_full=[0.01, 0.02])
            for language in quality.LANGUAGES for parent in range(5)}


class Quality(unittest.TestCase):
    def test_pooled_gain_cannot_hide_one_language(self):
        reference, candidate = docs(), docs()
        for value in candidate.values():
            value["kl_full"] = [0.001, 0.002]
            value["matches"] = 10
        candidate["ja_0"]["matches"] = 0
        candidate["ja_0"]["kl_full"] = [1.0, 2.0]
        result = quality.compare(reference, candidate)
        self.assertGreater(result["metrics"]["pooled"]["top1"], quality.summarize(reference)["pooled"]["top1"])
        self.assertFalse(result["observed_kl_top1_pass"])
        self.assertFalse(result["language_gates"]["ja"]["kl"])
        self.assertFalse(result["language_gates"]["ja"]["top1"])
        candidate["ja_0"]["teacher_sha256"] = "different"
        with self.assertRaises(ValueError): quality.compare(reference, candidate)

    def test_complete_reference_and_mixed_identity(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "records.jsonl"
            metadata = dict(identity="test", model_hash="a", rope_hash="b", prefill_identity="test-prefill")
            records = [metadata] + list(docs().values())
            p.write_text("\n".join(json.dumps(x) for x in records))
            loaded, _ = quality.load([p])
            self.assertEqual(quality.summarize(loaded)["pooled"]["rows"], 400)
            p.write_text("\n".join(json.dumps(x) for x in records[:-1]))
            with self.assertRaises(ValueError): quality.load([p])
            records.append(dict(metadata, model_hash="changed"))
            p.write_text("\n".join(json.dumps(x) for x in records))
            with self.assertRaises(ValueError): quality.load([p])


if __name__ == "__main__":
    unittest.main()
