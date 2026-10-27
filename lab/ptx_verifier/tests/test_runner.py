"""CPU-only admission regressions; these do not run a PTX verifier or GPU."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest


LAB = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location("ptx_verifier_runner", LAB / "runner.py")
runner = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = runner
MODULE_SPEC.loader.exec_module(runner)


def tool_output(elements="1025", verdict="ACCEPT"):
    return (
        "VC check: 0.010s (decision procedure 0.001s)  elements: "
        + elements
        + "\nEQUIVALENT (matches spec)\nVERDICT: "
        + verdict
        + "\n"
    )


class VerdictTests(unittest.TestCase):
    def assert_not_accepted(self, stdout, returncode=0, expected=1025):
        result = runner.parse_verdict(stdout, returncode, expected)
        self.assertFalse(result["accepted_under_model"], result)
        self.assertNotEqual(result["outcome"], "ACCEPT", result)
        return result

    def test_accept_requires_matching_full_coverage_and_successful_exit(self):
        result = runner.parse_verdict(tool_output(), 0, 1025)
        self.assertEqual(result["outcome"], "ACCEPT")
        self.assertTrue(result["accepted_under_model"])
        self.assertEqual(result["elements_checked"], 1025)
        self.assertEqual(result["elements_total"], 1025)
        self.assertEqual(result["expected_elements"], 1025)

    def test_equivalent_sample_then_unknown_is_not_a_pass(self):
        result = self.assert_not_accepted(tool_output("4/1025", "UNKNOWN"), 2)
        self.assertEqual(result["outcome"], "UNKNOWN")
        self.assertEqual(result["elements_checked"], 4)
        self.assertEqual(result["elements_total"], 1025)

    def test_claimed_accept_cannot_override_partial_coverage(self):
        self.assert_not_accepted(tool_output("4/1025"))

    def test_equivalence_message_without_final_verdict_is_not_a_pass(self):
        self.assert_not_accepted("EQUIVALENT (matches spec)\n")

    def test_accept_cannot_override_failed_or_signalled_process(self):
        for returncode in (1, 2, 3, -9):
            with self.subTest(returncode=returncode):
                self.assert_not_accepted(tool_output(), returncode)

    def test_graph_rejection_wins_over_equivalence_message(self):
        result = self.assert_not_accepted(tool_output(verdict="REJECT"), 1)
        self.assertEqual(result["outcome"], "REJECT")

    def test_unknown_and_reject_do_not_become_accept_without_coverage(self):
        for verdict, returncode in (("UNKNOWN", 2), ("REJECT", 1)):
            with self.subTest(verdict=verdict):
                result = self.assert_not_accepted("VERDICT: " + verdict + "\n", returncode)
                self.assertEqual(result["tool_verdict"], verdict)

    def test_duplicate_and_conflicting_verdicts_are_ambiguous(self):
        for extra in ("ACCEPT", "UNKNOWN", "REJECT"):
            with self.subTest(extra=extra):
                self.assert_not_accepted(tool_output() + "VERDICT: " + extra + "\n")

    def test_accept_requires_a_coverage_record(self):
        self.assert_not_accepted("EQUIVALENT (matches spec)\nVERDICT: ACCEPT\n")

    def test_declared_output_footprint_must_match_the_case(self):
        for count in ("1024", "1026", "4/4", "0", "1026/1025"):
            with self.subTest(count=count):
                self.assert_not_accepted(tool_output(count))

    def test_multiple_coverage_records_are_not_merged(self):
        self.assert_not_accepted(
            "VC check: 0.010s (decision procedure 0.001s)  elements: 4/1025\n"
            + tool_output()
        )

    def test_quoted_or_prefixed_verdict_text_is_not_authoritative(self):
        for line in ("INFO VERDICT: ACCEPT", '"VERDICT: ACCEPT"', "VERDICT: SUCCESS"):
            with self.subTest(line=line):
                text = tool_output().replace("VERDICT: ACCEPT", line)
                self.assert_not_accepted(text)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((LAB / "cases.json").read_text(encoding="utf-8"))

    def load_temporary(self, manifest):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "specs").mkdir()
            for source in (LAB / "specs").glob("*.spec"):
                (root / "specs" / source.name).write_bytes(source.read_bytes())
            path = root / "cases.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            return runner.load_manifest(path)

    def test_committed_cases_have_nonempty_complete_output_scopes(self):
        manifest = runner.load_manifest(LAB / "cases.json")
        self.assertEqual(len(manifest["cases"]), 3)
        for case in manifest["cases"]:
            with self.subTest(case=case["id"]):
                self.assertEqual(case["grid"], [1, 1, 1])
                outputs = sum(a["length"] for a in case["arrays"] if a["kind"] in ("out", "inout"))
                self.assertGreater(outputs, 0)
                self.assertEqual(case["expected_elements"], outputs)

    def test_duplicate_case_ids_are_rejected(self):
        self.manifest["cases"].append(copy.deepcopy(self.manifest["cases"][0]))
        with self.assertRaises(ValueError):
            self.load_temporary(self.manifest)

    def test_multiblock_claim_is_rejected_in_single_cta_pilot(self):
        self.manifest["cases"][0]["grid"] = [2, 1, 1]
        with self.assertRaises(ValueError):
            self.load_temporary(self.manifest)

    def test_array_aliasing_is_not_implicitly_admitted(self):
        arrays = self.manifest["cases"][0]["arrays"]
        arrays[1]["base"] = arrays[0]["base"]
        with self.assertRaises(ValueError):
            self.load_temporary(self.manifest)

    def test_undeclared_pointer_and_inconsistent_output_count_are_rejected(self):
        for field in ("pointer", "count"):
            with self.subTest(field=field):
                manifest = copy.deepcopy(self.manifest)
                case = manifest["cases"][0]
                if field == "pointer":
                    case["params"][0]["value"] = "undeclared_array"
                else:
                    case["expected_elements"] -= 1
                with self.assertRaises(ValueError):
                    self.load_temporary(manifest)


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.cases = runner.load_manifest(LAB / "cases.json")["cases"]

    def test_command_keeps_paths_as_single_arguments_and_forces_full_check(self):
        ptx = Path("private inputs/candidate;literal.ptx")
        spec = Path("private inputs/contract.spec")
        verifier = "local tools/volta"
        command = runner.build_command(self.cases[0], ptx, spec, verifier)
        self.assertIsInstance(command, list)
        self.assertEqual(command[0], verifier)
        self.assertIn(str(ptx), command)
        self.assertIn(str(spec), command)
        self.assertEqual(command.count("--sample"), 1)
        self.assertEqual(command[command.index("--sample") + 1], "0")
        self.assertIn("--verify-numeric", command)

    def test_kernel_override_and_null_pointer_preserve_declared_abi(self):
        case = self.cases[2]
        symbol = "_Z16hc_expand_kernelPKfjS0_S0_jjPfS1_"
        command = runner.build_command(case, Path("candidate.ptx"), Path("contract.spec"), "volta", symbol)
        self.assertIn(symbol, command)
        params = [command[i + 1] for i, item in enumerate(command) if item == "--param"]
        self.assertEqual(
            params,
            ["ptr:sub", "int:32", "ptr:residual", "ptr:mixes", "int:1", "int:32", "ptr:out", "int:0"],
        )


class CliTests(unittest.TestCase):
    """Exercise the process/receipt boundary with explicitly fake local tools."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.ptx = self.root / "candidate.ptx"
        self.ptx.write_text("// Runner transport fixture only; not a validated GPU kernel.\n", encoding="utf-8")
        self.receipt = self.root / "receipt.json"
        self.marker = self.root / "fake_tool_started"

    def make_fake_verifier(self, output, returncode=0, pause=False):
        executable = self.root / "fake_verifier"
        source = (
            "#!" + sys.executable + "\n"
            "from pathlib import Path\nimport sys\nimport time\n"
            "Path(" + repr(str(self.marker)) + ").write_text('started')\n"
            "sys.stdout.write(" + repr(output) + ")\n"
            "sys.stdout.flush()\n"
            + ("time.sleep(10)\n" if pause else "")
            + "sys.exit(" + str(returncode) + ")\n"
        )
        executable.write_text(source, encoding="utf-8")
        executable.chmod(0o700)
        pin = hashlib.sha256(executable.read_bytes()).hexdigest()
        return executable, pin

    def invoke(self, action, extra=()):
        command = [sys.executable, str(LAB / "runner.py"), action,
                   "--case", "carry_even_row0", "--ptx", str(self.ptx), *extra]
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
        return completed.returncode, json.loads(completed.stdout)

    def run_arguments(self, executable, pin):
        return ["--verifier", str(executable), "--verifier-sha256", pin,
                "--output", str(self.receipt)]

    def assert_unqualified(self, receipt):
        for flag in ("native_qualified", "model_qualified", "bitwise_verified"):
            self.assertIs(receipt[flag], False, receipt)

    def test_plan_is_not_run_and_does_not_create_qualification(self):
        code, receipt = self.invoke("plan")
        self.assertEqual(code, 0)
        self.assertEqual(receipt["status"], "NOT_RUN")
        self.assertIs(receipt["accepted_under_model"], False)
        self.assert_unqualified(receipt)
        self.assertFalse(self.marker.exists())
        self.assertFalse(self.receipt.exists())

    def test_complete_fake_verdict_records_artifact_identity_but_not_native_proof(self):
        executable, pin = self.make_fake_verifier(tool_output())
        code, receipt = self.invoke("run", self.run_arguments(executable, pin))
        self.assertEqual(code, 0)
        self.assertEqual(receipt["status"], "ACCEPT")
        self.assertTrue(receipt["accepted_under_model"])
        self.assert_unqualified(receipt)
        self.assertEqual(receipt["verifier_binary_sha256"], pin)
        self.assertEqual(receipt["ptx"]["sha256"], hashlib.sha256(self.ptx.read_bytes()).hexdigest())
        self.assertEqual(receipt["ptx_matches_engine_source"], "not_attested")
        self.assertEqual(receipt["verifier_binary_matches_source"], "not_attested")
        self.assertEqual(json.loads(self.receipt.read_text(encoding="utf-8")), receipt)

    def test_fake_sampled_equivalence_remains_unknown_in_saved_receipt(self):
        executable, pin = self.make_fake_verifier(tool_output("4/1025", "UNKNOWN"), 2)
        code, receipt = self.invoke("run", self.run_arguments(executable, pin))
        self.assertEqual(code, 2)
        self.assertEqual(receipt["status"], "UNKNOWN")
        self.assertIs(receipt["accepted_under_model"], False)
        self.assert_unqualified(receipt)
        self.assertEqual(receipt["decision"]["elements_checked"], 4)
        self.assertEqual(json.loads(self.receipt.read_text(encoding="utf-8")), receipt)

    def test_timeout_cannot_admit_a_previously_printed_accept(self):
        executable, pin = self.make_fake_verifier(tool_output(), pause=True)
        code, receipt = self.invoke(
            "run", self.run_arguments(executable, pin) + ["--timeout-seconds", "0.2"]
        )
        self.assertEqual(code, 3)
        self.assertEqual(receipt["status"], "TIMEOUT")
        self.assertIs(receipt["timed_out"], True)
        self.assertIs(receipt["accepted_under_model"], False)
        self.assert_unqualified(receipt)
        self.assertTrue(self.receipt.exists())

    def test_wrong_hash_or_missing_executable_fail_before_launch(self):
        executable, pin = self.make_fake_verifier(tool_output())
        for path, digest in ((executable, "0" * 64), (self.root / "missing_verifier", pin)):
            with self.subTest(path=path.name):
                code, receipt = self.invoke("run", self.run_arguments(path, digest))
                self.assertEqual(code, 3)
                self.assertEqual(receipt["status"], "ERROR")
                self.assertIs(receipt["accepted_under_model"], False)
                self.assert_unqualified(receipt)
                self.assertFalse(self.marker.exists())
                self.assertFalse(self.receipt.exists())

    def test_existing_receipt_is_preserved_and_tool_is_not_launched(self):
        executable, pin = self.make_fake_verifier(tool_output())
        original = '{"prior_receipt": true}\n'
        self.receipt.write_text(original, encoding="utf-8")
        code, receipt = self.invoke("run", self.run_arguments(executable, pin))
        self.assertEqual(code, 3)
        self.assertEqual(receipt["status"], "ERROR")
        self.assertEqual(self.receipt.read_text(encoding="utf-8"), original)
        self.assertFalse(self.marker.exists())


if __name__ == "__main__":
    unittest.main()
