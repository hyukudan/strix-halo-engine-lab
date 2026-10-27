"""Expected tool failures are evidence only when they match the exact probe."""
import copy
import importlib.util
from pathlib import Path
import unittest


DIRECTORY = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_compatibility_tests", DIRECTORY / "compatibility.py")
compatibility = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compatibility)


class CompatibilityEvidenceTests(unittest.TestCase):
    def receipt(self):
        return {"status": "ERROR", "accepted_under_model": False,
                "decision": {"return_code": 1, "tool_verdict": None},
                "timed_out": False, "logs_truncated": False, "input_identities_stable": True,
                "stdout": "", "stderr": compatibility.PROBES[1][2] + "\n",
                "native_qualified": False, "model_qualified": False, "bitwise_verified": False}

    def test_timeout_truncation_or_changed_input_cannot_establish_a_limitation(self):
        receipt = self.receipt()
        diagnostic = compatibility.PROBES[1][2]
        self.assertTrue(compatibility.limitation_reproduced(receipt, diagnostic))
        for key, value in (("timed_out", True), ("logs_truncated", True), ("input_identities_stable", False)):
            with self.subTest(key=key):
                invalid = dict(receipt, **{key: value})
                self.assertFalse(compatibility.limitation_reproduced(invalid, diagnostic))

    def test_other_error_or_aggregate_verdict_cannot_match_the_probe(self):
        receipt = self.receipt()
        diagnostic = compatibility.PROBES[1][2]
        unrelated = dict(receipt, stderr="error: missing executable\n")
        self.assertFalse(compatibility.limitation_reproduced(unrelated, diagnostic))
        for decision in ({"return_code": 0, "tool_verdict": None},
                         {"return_code": 1, "tool_verdict": "REJECT"}):
            invalid = copy.deepcopy(receipt)
            invalid["decision"] = decision
            self.assertFalse(compatibility.limitation_reproduced(invalid, diagnostic))


if __name__ == "__main__":
    unittest.main()
