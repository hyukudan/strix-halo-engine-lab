"""Top-level CLI framing, including CPU-only subprocess receipt checks."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.qualification.longctx_fidelity.build import build_case, write_case
from tools.qualification.longctx_fidelity.common import file_sha
from tools.qualification.longctx_fidelity.run_cli import parse_output, run_cases
from tools.qualification.longctx_fidelity.tok import FakeTokenizer


class RunnerFramingTests(unittest.TestCase):
    result = {"prefix_tokens": [1], "generated_tokens": [65], "stop_reason": "max_new_tokens"}

    def test_nested_objects_are_not_top_level_results(self):
        result = json.dumps(self.result)
        for output in ('{"outer":' + result + ',', '{"outer":' + result + '}',
                       '[' + result + ']', '[{"outer":' + result + '}'):
            with self.subTest(output=output), self.assertRaises(ValueError):
                parse_output(output)

    def test_incomplete_candidate_never_resumes_at_inner_braces(self):
        result = json.dumps(self.result)
        for output in ('{"outer":' + result + ',\n' + result,
                       '{broken\n' + result):
            with self.subTest(output=output), self.assertRaises(ValueError):
                parse_output(output)

    def test_quoted_log_braces_and_escaped_quotes(self):
        quoted = json.dumps('quoted " brace { and } and \\ ' + json.dumps(self.result))
        logs = 'log: ' + quoted + '\n'
        with self.assertRaises(ValueError):
            parse_output(logs)
        self.assertEqual(parse_output(logs + json.dumps(self.result)), self.result)

    def test_duplicate_top_level_results_rejected(self):
        output = json.dumps(self.result, indent=2)
        with self.assertRaises(ValueError):
            parse_output(output + '\nlog\n' + output)

    def test_log_pretty_result_and_trailer(self):
        output = '[INFO] ready\n' + json.dumps(self.result, indent=2) + '\ntrailer\n'
        self.assertIn('\n  "generated_tokens": [\n', output)
        self.assertEqual(parse_output(output), self.result)
        # A complete unrelated value is skipped, including all nested objects.
        outer = json.dumps({"outer": self.result})
        self.assertEqual(parse_output(outer + '\n' + output), self.result)

    def test_fake_cli_nested_result_leaves_receipt_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script, model, receipt = (root / name for name in ("cli.py", "model", "model.receipt"))
            model.write_bytes(b"CPU fixture")
            receipt.write_bytes(b"CPU receipt")
            script.write_bytes(b'import json, sys\n'
                b'if "--print-effective-config" in sys.argv:\n'
                b'    print(json.dumps({"profile": "auto"}))\n'
                b'else:\n'
                b'    end = sys.argv.index("--generate")\n'
                b'    result = {"prefix_tokens": list(map(int, sys.argv[2:end])),\n'
                b'              "generated_tokens": [65] * int(sys.argv[end + 1]),\n'
                b'              "stop_reason": "max_new_tokens"}\n'
                b'    print(\'{"outer":\' + json.dumps(result) + ",")\n')
            text, ids, case = build_case("niah_single", 2048, .5, 42, FakeTokenizer())
            case["tokenizer"] = {"backend": "synthetic-test",
                                 "model_receipt_sha256": file_sha(receipt)}
            manifest = write_case(root, text, ids, case)
            turn = root / "turn.txt"
            turn.write_bytes((manifest.name + '\n').encode("ascii"))
            real_run = subprocess.run
            calls = []
            def launch(argv, **kwargs):
                # Run the fake CLI script portably, including on Windows where
                # executable shebang scripts cannot be launched directly.
                calls.append(argv)
                return real_run([sys.executable, *argv], **kwargs)
            out = root / "out"
            with patch("tools.qualification.longctx_fidelity.run_cli.subprocess.run", side_effect=launch), patch(
                    "tools.qualification.longctx_fidelity.run_cli.argv_check", return_value=1):
                self.assertFalse(run_cases(script, model, receipt, turn, out))
            self.assertEqual(len(calls), 2)
            saved = json.loads((out / (case["case_id"] + ".receipt.json")).read_text())
            self.assertEqual(saved["config_returncode"], 0)
            self.assertEqual(saved["run_returncode"], 0)
            self.assertEqual(saved["status"], "INCOMPLETE")
            self.assertEqual(saved["generated_tokens"], [])
            self.assertIn("exactly one complete generation JSON object", saved["error"])
            self.assertEqual(json.loads((out / "summary.json").read_text()),
                             [{"case_id": case["case_id"], "status": "INCOMPLETE"}])
