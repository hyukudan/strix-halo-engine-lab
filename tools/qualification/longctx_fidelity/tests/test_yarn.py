"""CPU-only explicit YaRN construction and subprocess protocol fixtures."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.qualification.longctx_fidelity.build import build_case, write_case
from tools.qualification.longctx_fidelity.common import canonical, file_sha, mode_budget
from tools.qualification.longctx_fidelity.plan import matrix, predicted
from tools.qualification.longctx_fidelity.run_cli import run_cases, validate_case
from tools.qualification.longctx_fidelity.tok import FakeTokenizer
from tools.qualification.longctx_fidelity.turn import make_script


class YarnTests(unittest.TestCase):
    def test_native_default_unchanged(self):
        args = ("niah_single", 2048, .5, 42, FakeTokenizer())
        default = build_case(*args)
        explicit = build_case(*args, mode="native")
        self.assertEqual(default[:2], explicit[:2])
        self.assertEqual(canonical(default[2]), canonical(explicit[2]))
        self.assertEqual(default[2]["requested"], {"context": "auto", "native_limit": 262144,
                                                "generate": 32, "rope_mode": "native"})
        with self.assertRaisesRegex(ValueError, "--context"):
            build_case(*args, context=300000)

    def test_yarn2_deep_case_and_context_rule(self):
        _, ids, case = build_case("niah_single", 299000, .5, 42, FakeTokenizer(), mode="yarn2")
        self.assertEqual(len(ids), 299000)
        self.assertTrue(case["case_id"].startswith("yarn2.299000."))
        # ceil((299000 + 32)/512)*512 = 299520; max(300000, 299520) = 300000.
        self.assertEqual(case["requested"], {"context": 300000, "rope_mode": "yarn2", "limit": 524288})
        self.assertEqual(mode_budget("yarn2", 300000, 32), (524288, 300032))
        self.assertEqual(mode_budget("yarn2", 299000, 32, 300032), (524288, 300032))
        with self.assertRaisesRegex(ValueError, "524288"):
            build_case("niah_single", 524257, .5, 42, FakeTokenizer(), mode="yarn2")
        for context in (262144, 299031, 524289, 300000.5):
            with self.subTest(context=context), self.assertRaises(ValueError):
                build_case("niah_single", 299000, .5, 42, FakeTokenizer(), mode="yarn2", context=context)

    def test_yarn4_cap_and_matrix(self):
        _, ids, case = build_case("niah_single", 524289, .5, 42, FakeTokenizer(), mode="yarn4")
        self.assertEqual(len(ids), 524289)
        self.assertEqual(case["requested"], {"context": 524800, "rope_mode": "yarn4", "limit": 1048576})
        self.assertEqual(mode_budget("yarn4", 1048544, 32), (1048576, 1048576))
        with self.assertRaisesRegex(ValueError, "1048576"):
            build_case("niah_single", 1048545, .5, 42, FakeTokenizer(), mode="yarn4")
        with self.assertRaises(ValueError):
            mode_budget("other", 2048, 32)
        cells = matrix(["native", "yarn2", "yarn4"], [8192], ["niah_single"], [42], [.5])
        self.assertEqual([c["mode"] for c in cells], ["native", "yarn2", "yarn4"])
        self.assertEqual(cells[1]["requested"]["context"], 300000)
        self.assertAlmostEqual(predicted({"depth": 299000, "generate": 32}), 370.96)
        self.assertAlmostEqual(predicted({"depth": 524256, "generate": 32}), 510.96)
        with self.assertRaises(ValueError):
            matrix(["yarn2"], [524257], ["niah_single"], [42], [.5])

    def test_runner_script_argv_text_and_marker(self):
        # The adapter launches a real CPU Python fixture, portable across Windows/POSIX.
        fixture = '''import json, pathlib, sys
args = sys.argv[1:]
if '--print-effective-config' in args:
    print(json.dumps({'argv': args}))
else:
    ids = list(map(int, pathlib.Path(args[args.index('--prompt-ids')+1]).read_text().split()))
    print(json.dumps({'argv': args, 'prefix_tokens': ids, 'generated_tokens': [65]*32,
                      'stop_reason': 'max_new_tokens'}))
    if 'missing' not in args[0]:
        factor = args[args.index('--rope-yarn')+1]
        context = args[args.index('--context')+1]
        print('Qwen experimental YaRN: factor='+factor+' context='+context+' T=262144', file=sys.stderr)
'''
        for mode in ("yarn2", "yarn4"):
            for marker in (True, False):
                with self.subTest(mode=mode, marker=marker), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    script, receipt = root / "engine.py", root / "model.receipt"
                    script.write_bytes(fixture.encode("ascii"))
                    receipt.write_bytes(b"fixture")
                    model = root / ("model.gguf" if marker else "missing.gguf")
                    model.write_bytes(b"fixture")
                    text, ids, case = build_case("niah_single", 2048, .5, 42, FakeTokenizer(), mode=mode)
                    case["tokenizer"] = {"backend": "synthetic-test", "model_receipt_sha256": file_sha(receipt)}
                    manifest = write_case(root, text, ids, case)
                    self.assertEqual(validate_case(manifest)[1], ids)
                    turn = root / "turn.txt"
                    turn.write_bytes((manifest.name + "\n").encode("ascii"))
                    real_run = subprocess.run
                    def launch(argv, **kwargs):
                        return real_run([sys.executable, str(script), *argv[1:]], **kwargs)
                    out = root / "out"
                    with patch("tools.qualification.longctx_fidelity.run_cli.subprocess.run", side_effect=launch), patch(
                            "tools.qualification.longctx_fidelity.run_cli.argv_check", return_value=1):
                        self.assertEqual(run_cases(sys.executable, model, receipt, turn, out), marker)
                    saved = json.loads((out / (case["case_id"] + ".receipt.json")).read_bytes())
                    prompt = out / (case["case_id"] + ".prompt-ids.txt")
                    self.assertEqual(prompt.read_bytes(), (" ".join(map(str, ids)) + "\n").encode("ascii"))
                    expected = [str(Path(sys.executable).resolve()), str(model.resolve()), "--prompt-ids",
                                str(prompt.resolve()), "--rope-yarn", mode[-1], "--context", "300000", "--generate", "32"]
                    self.assertEqual(saved["argv"], expected)
                    self.assertEqual(saved["config_argv"], expected + ["--print-effective-config"])
                    self.assertEqual(saved["resolved"]["argv"], saved["config_argv"][1:])
                    raw = json.loads((out / (case["case_id"] + ".run.stdout")).read_bytes())
                    self.assertEqual(raw["argv"], expected[1:])
                    self.assertEqual(saved["prompt_ids_text_sha256"], file_sha(prompt))
                    self.assertEqual(saved["status"], "COMPLETE" if marker else "INCOMPLETE")
                    if marker:
                        self.assertIn("Qwen experimental YaRN:", saved["yarn_marker"])
                    else:
                        self.assertIn("missing Qwen experimental YaRN", saved["error"])

    def test_turn_hashes_prompt_ids_text(self):
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            root = Path(os.path.relpath(tmp))
            binary, receipt, model = (root / n for n in ("binary", "receipt", "model"))
            for path in (binary, receipt, model):
                path.write_bytes(b"fixture")
            text, ids, case = build_case("niah_single", 2048, .5, 42, FakeTokenizer(), mode="yarn2")
            manifest = write_case(root, text, ids, case)
            listing = root / "turn.txt"
            listing.write_bytes((manifest.name + "\n").encode("ascii"))
            out = root / "receipts"
            turn = make_script(binary, model, receipt, listing, out, root / "turn.sh")
            prompt = out / (case["case_id"] + ".prompt-ids.txt")
            self.assertEqual(turn["files"][prompt.as_posix()], file_sha(prompt))
            self.assertEqual(list(map(int, prompt.read_text().split())), ids)
