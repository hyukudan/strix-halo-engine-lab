import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("ctest_evidence", Path(__file__).resolve().parents[1] / "tools/qualification/ctest_evidence.py")
EVIDENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVIDENCE)


class CTestEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.expected = self.root / "expected.json"
        self.xml = self.root / "junit.xml"
        self.expected.write_text(json.dumps(["alpha", "beta"]), encoding="utf-8")

    def collect(self, xml, expected=None):
        if expected is not None:
            self.expected.write_text(json.dumps(expected), encoding="utf-8")
        self.xml.write_text(xml, encoding="utf-8")
        return EVIDENCE.collect(self.expected, self.xml, "native")

    def test_complete_suite_retains_samples_but_not_native_or_quality_proof(self):
        result = self.collect('<testsuite tests="2" failures="0" errors="0" skipped="0"><testcase name="alpha" time="0.1"/><testcase name="beta" status="run"/></testsuite>')
        self.assertTrue(result["all_requested_tests_passed"])
        self.assertEqual(result["counts"], dict(passed=2, failed=0, skipped=0))
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["seconds_reported"], 0.1)
        self.assertIsNone(result["results"][1]["seconds_reported"])
        self.assertIsNone(result["qualification"])
        self.assertIsNone(result["native_execution_observed"])
        self.assertEqual(result["stage_declared"], "native")
        self.assertEqual(result["inputs"]["junit"]["sha256"], EVIDENCE.hashlib.sha256(self.xml.read_bytes()).hexdigest())

    def test_skips_are_retained_and_never_count_as_passed(self):
        result = self.collect('<testsuite tests="2" failures="0" skipped="2" disabled="0"><testcase name="alpha" status="notrun"><skipped message="no device"/></testcase><testcase name="beta"><skipped>not admitted</skipped></testcase></testsuite>')
        self.assertTrue(result["complete_coverage"])
        self.assertFalse(result["all_requested_tests_passed"])
        self.assertEqual(result["counts"], dict(passed=0, failed=0, skipped=2))
        self.assertEqual(result["results"][0]["diagnostics"][0]["message"], "no device")
        self.assertIsNone(result["native_execution_observed"])

    def test_missing_extra_and_empty_suite_keep_incomplete_coverage(self):
        for xml in ('<testsuite tests="1"><testcase name="alpha"/></testsuite>',
                    '<testsuite><testcase name="alpha"/><testcase name="beta"/><testcase name="extra"/></testsuite>',
                    '<testsuite tests="0"/>'):
            with self.subTest(xml=xml):
                result = self.collect(xml)
                self.assertFalse(result["complete_coverage"])
                self.assertFalse(result["all_requested_tests_passed"])
                self.assertTrue(result["missing"] or result["extra"])

    def test_failures_and_errors_retain_diagnostics_and_block_gate(self):
        result = self.collect('<testsuite tests="2" failures="1" errors="1"><testcase name="alpha"><failure message="mismatch">row1</failure></testcase><testcase name="beta"><error message="launch">details</error></testcase></testsuite>')
        self.assertEqual(result["counts"], dict(passed=0, failed=2, skipped=0))
        self.assertFalse(result["all_requested_tests_passed"])
        self.assertEqual(result["results"][0]["diagnostics"][0]["text"], "row1")

    def test_nested_suite_totals_are_not_double_counted(self):
        result = self.collect('<testsuites tests="2" skipped="0"><testsuite tests="1"><testcase name="alpha"/></testsuite><testsuite tests="1"><testcase name="beta"/></testsuite></testsuites>')
        self.assertEqual(result["reported_tests"], 2)
        self.assertTrue(result["all_requested_tests_passed"])

    def test_ambiguous_corrupt_or_contradictory_reports_reject(self):
        invalid = ['<testsuite><testcase name="alpha"/><testcase name="alpha"/></testsuite>',
                   '<testsuite tests="3"><testcase name="alpha"/><testcase name="beta"/></testsuite>',
                   '<testsuite failures="1"><testcase name="alpha"/></testsuite>',
                   '<testsuite skipped="1"><testcase name="alpha"/></testsuite>',
                   '<testsuite><testcase name="alpha" status="future"/></testsuite>',
                   '<testsuite><testcase name="alpha" time="NaN"/></testsuite>',
                   '<testsuite><testcase name="alpha" time="-1"/></testsuite>',
                   '<testsuite><testcase name="alpha"><failure/><skipped/></testcase></testsuite>',
                   '<testsuite><testcase name="alpha"><testcase name="beta"/></testcase></testsuite>',
                   '<testsuites><testcase name="alpha"/></testsuites>',
                   '<!DOCTYPE testsuite [<!ENTITY bad "alpha">]><testsuite><testcase name="&bad;"/></testsuite>',
                   '<other/>']
        for xml in invalid:
            with self.subTest(xml=xml), self.assertRaises(ValueError):
                self.collect(xml)
        for names in ([], ["alpha", "alpha"], [True], {}, [""], ["a" * 4097]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.collect('<testsuite/>', names)

    def test_nonzero_skip_gate_and_no_overwrite(self):
        self.xml.write_text('<testsuite><testcase name="alpha"><skipped/></testcase><testcase name="beta"/></testsuite>', encoding="utf-8")
        output = self.root / "receipt.json"
        args = ["--expected-names", str(self.expected), "--junit", str(self.xml), "--stage", "native", "--output", str(output)]
        self.assertEqual(EVIDENCE.main(args), 3)
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            EVIDENCE.main(args)
        self.assertEqual(output.read_bytes(), before)
        with self.assertRaises(FileExistsError):
            EVIDENCE.main(args[:-1] + [str(self.xml)])

    def test_byte_bounds_and_encoding_fail_closed(self):
        self.xml.write_bytes(b"x" * (EVIDENCE.MAX_NAMES_BYTES + 1))
        with self.assertRaises(ValueError):
            EVIDENCE.read_bounded(self.xml, EVIDENCE.MAX_NAMES_BYTES)
        self.xml.write_bytes('<testsuite/>'.encode('utf-16'))
        with self.assertRaises(ValueError):
            EVIDENCE.collect(self.expected, self.xml, "host")


if __name__ == "__main__":
    unittest.main()
