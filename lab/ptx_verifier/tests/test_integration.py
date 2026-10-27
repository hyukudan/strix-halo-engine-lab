"""Regression checks for the real-smoke harness's evidence requirements.

These test receipt interpretation only; they do not run an external verifier.
"""
import importlib.util
from pathlib import Path
import sys
import unittest


DIRECTORY = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


integration = load_module("_smoke_integration", "integration.py")
runner = load_module("_smoke_runner", "runner.py")


def receipt(stdout, returncode):
    decision = runner.parse_verdict(stdout, returncode, 1025)
    return {"status": decision["outcome"], "decision": decision, "stdout": stdout,
            "accepted_under_model": decision["accepted_under_model"],
            "native_qualified": False, "model_qualified": False, "bitwise_verified": False}


class SmokeEvidenceTests(unittest.TestCase):
    def test_malformed_or_short_accept_is_not_a_missing_output_refutation(self):
        check = next(check for check in integration.CHECKS if check.expected == "MISSING_OUTPUT")
        for stdout in ("VERDICT: ACCEPT\n",
                       "VC check: 0.001s (decision procedure 0.001s)  elements: 1024\nVERDICT: ACCEPT\n"):
            with self.subTest(stdout=stdout):
                result = receipt(stdout, 0)
                self.assertEqual(result["status"], "ERROR")
                self.assertFalse(integration.evaluate(check, result, 3)[0])

    def test_missing_output_requires_its_aggregate_index_witness(self):
        check = next(check for check in integration.CHECKS if check.expected == "MISSING_OUTPUT")
        stdout = ("VERDICT: REJECT\n"
                  "  equivalence: valid does not have the same set of indices "
                  "(1 never written, first valid[0], 0 written outside it)\n")
        self.assertTrue(integration.evaluate(check, receipt(stdout, 1), 1)[0])
        unrelated = "VERDICT: REJECT\n  kernel: memory safety failed\n"
        self.assertFalse(integration.evaluate(check, receipt(unrelated, 1), 1)[0])

    def test_numeric_mutation_requires_refutation_from_the_aggregate_decision(self):
        check = next(check for check in integration.CHECKS if check.expected == "REFUTED")
        stdout = ("NOT EQUIVALENT: 1 mismatched element(s) vs spec\nVERDICT: REJECT\n"
                  "  equivalence: 1 element(s) refuted by a numeric counterexample, first valid[0]\n")
        self.assertTrue(integration.evaluate(check, receipt(stdout, 1), 1)[0])
        unrelated = stdout.replace("  equivalence: 1 element(s) refuted by a numeric counterexample, first valid[0]",
                                   "  kernel: memory safety failed")
        self.assertFalse(integration.evaluate(check, receipt(unrelated, 1), 1)[0])


if __name__ == "__main__":
    unittest.main()
