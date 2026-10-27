import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from tools.qualification.longctx_fidelity.common import canonical, atomic, sha, file_sha
from tools.qualification.longctx_fidelity.build import build_case, write_case
from tools.qualification.longctx_fidelity.tok import FakeTokenizer, StrixTokenize
from tools.qualification.longctx_fidelity.run_cli import parse_output, argv_check, clean_env, validate_case, run_cases
from tools.qualification.longctx_fidelity.plan import predicted, pack, matrix, write_plan
from tools.qualification.longctx_fidelity.score import exact, wilson, mutation_self_test, score_case, heatmap
from tools.qualification.longctx_fidelity.turn import make_script


class RunnerScorePlanTests(unittest.TestCase):
    def test_receipt_parse_actual_shape(self):
        output = 'log line\n' + json.dumps({"prefix_tokens": [1, 2], "generated_tokens": [3, 4],
                 "stop_reason": "max_new_tokens", "final_position": 3}) + '\ntrailer'
        self.assertEqual(parse_output(output)["generated_tokens"], [3, 4])
        for bad in ('{"generated_tokens":[1],"stop_reason":"eos"',
                    '{"generated_tokens":[-1],"stop_reason":"eos"}',
                    '{"generated_tokens":[true],"stop_reason":"eos"}', output + output):
            with self.assertRaises(ValueError):
                parse_output(bad)

    def test_clean_env(self):
        with patch.dict(os.environ, {"STRIX_BASELINE": "1", "PATH": "bin", "HOME": "home"}, clear=True):
            self.assertEqual(clean_env(), {"PATH": "bin", "HOME": "home"})
            self.assertEqual(clean_env(["CUSTOM=2"])["CUSTOM"], "2")
            self.assertNotIn("STRIX_BASELINE", clean_env())
        with self.assertRaises(ValueError):
            clean_env(["BROKEN"])

    def test_argv_limits(self):
        self.assertGreater(argv_check(["bin", "model", "123"], {}, 10000, "posix"), 0)
        with self.assertRaisesRegex(ValueError, "ARG_MAX"):
            argv_check(["bin", "1"] * 100, {}, 5000, "posix")
        with self.assertRaisesRegex(ValueError, "Windows"):
            argv_check(["x" * 32768], {}, platform="nt")
        with self.assertRaisesRegex(ValueError, "MAX_ARG_STRLEN"):
            argv_check(["x" * 131072], {}, 1000000, "posix")

    def test_exact_normalization_strict(self):
        self.assertTrue(exact("ABC", "  ABC. \n")["exact"])
        for wrong in ("abc", "ABC..", "answer ABC", "<think>x</think>ABC", None):
            self.assertFalse(exact("ABC", wrong)["exact"])

    def test_pairs_and_mutation(self):
        answer = [["a", "v1"], ["a0", "v2"]]
        result = exact(answer, '[["a","v1"],["a0","wrong"]]')
        self.assertEqual(result["per_value"], [True, False])
        self.assertFalse(result["all_values"])
        self.assertFalse(exact(answer, '[["a","v1"],["a","v2"]]')["exact"])
        self.assertFalse(exact(answer, '[["a0","v2"],["a","v1"]]')["exact"])
        self.assertTrue(mutation_self_test())

    def test_wilson(self):
        self.assertIsNone(wilson(0, 0))
        self.assertAlmostEqual(wilson(0, 10)[1], .27753279986)
        self.assertAlmostEqual(wilson(10, 10)[0], .72246720014)
        with self.assertRaises(ValueError):
            wilson(2, 1)

    def test_status_missing_never_credit(self):
        _, _, c = build_case("niah_single", 2048, .5, 42, FakeTokenizer())
        r = {"case_id": c["case_id"], "status": "COMPLETE", "input_ids_sha256": c["ids_sha256"],
             "ground_truth_sha256": c["ground_truth_sha256"], "generated_tokens": [1]}
        self.assertEqual(score_case(c)["status"], "PLANNED")
        self.assertEqual(score_case(c, r)["status"], "INCOMPLETE")
        okay = score_case(c, r, c["ground_truth"])
        self.assertTrue(okay["exact"])
        self.assertIn("1/1", heatmap([okay]))
        r["status"] = "INCOMPLETE"
        self.assertFalse(score_case(c, r, c["ground_truth"])["exact"])

    def test_plan_packing_and_sort(self):
        cases = matrix(["native"], [32768, 8192], ["niah_single"], [104729], [.95, .05])
        for i, c in enumerate(cases):
            c["case_id"] = str(i)
        turns = pack(list(reversed(cases)), 100)
        self.assertEqual([c for t in turns for c in t["cases"]], [c["case_id"] for c in cases])
        self.assertTrue(all(t["predicted_seconds"] <= 100 for t in turns))
        self.assertEqual(predicted({"depth": 262143, "generate": 1}), 250.03)
        with self.assertRaises(ValueError):
            pack(cases, 10)
        with tempfile.TemporaryDirectory() as tmp:
            text = write_plan(cases, tmp, 100)
            self.assertEqual(Path(tmp, "plan.txt").read_text(), text)
            self.assertIn("PLANNED", text)

    def test_prediction_override(self):
        self.assertEqual(predicted({"depth": 8192, "generate": 10}, {8192: 9}), 49.3)
        with self.assertRaises(ValueError):
            predicted({"depth": 8192, "generate": 1}, {8192: float("nan")})

    def test_fake_execution_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_case(tmp, *build_case("niah_single", 2048, .5, 42, FakeTokenizer()))
            with self.assertRaisesRegex(ValueError, "FakeTokenizer"):
                validate_case(p)

    def test_tokenizer_file_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary, receipt = Path(tmp, "tokenizer"), Path(tmp, "receipt.json")
            binary.write_bytes(b"binary")
            receipt.write_bytes(b"receipt")
            tok = StrixTokenize(binary, "MODEL.gguf", receipt)
            def fake_run(argv, **kwargs):
                content = Path(argv[-1]).read_bytes()
                mode = argv[-2]
                if mode in ("plain", "encode"):
                    self.assertEqual(content, b"full prompt")
                    return subprocess.CompletedProcess(argv, 0, b"1,2,3\n", b"")
                self.assertEqual(content, b"1,2,3")
                return subprocess.CompletedProcess(argv, 0, b"decoded", b"")
            with patch("subprocess.run", side_effect=fake_run):
                self.assertEqual(tok.encode("full prompt"), [1, 2, 3])
                self.assertEqual(tok.plain("full prompt"), [1, 2, 3])
                self.assertEqual(tok.decode([1, 2, 3]), "decoded")

    def test_synthetic_runner_receipts_and_soft_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary, receipt, model = Path(tmp, "bin"), Path(tmp, "model.receipt"), Path(tmp, "model.gguf")
            for p in (binary, receipt, model):
                p.write_bytes(b"fixture")
            text, ids, c = build_case("niah_single", 2048, .5, 42, FakeTokenizer())
            c["tokenizer"] = {"backend": "synthetic-test", "model_receipt_sha256": file_sha(receipt)}
            manifest = write_case(tmp, text, ids, c)
            turn = Path(tmp, "turn.txt")
            turn.write_text(manifest.name + "\n", encoding="utf-8")
            calls = []
            def fake_run(argv, **kwargs):
                calls.append(argv)
                if "--print-effective-config" in argv:
                    return subprocess.CompletedProcess(argv, 0, b'{"profile":"auto"}', b"")
                return subprocess.CompletedProcess(argv, 0, canonical({"prefix_tokens": ids,
                    "generated_tokens": [65] * 32, "stop_reason": "max_new_tokens"}), b"")
            with patch("subprocess.run", side_effect=fake_run), patch(
                    "tools.qualification.longctx_fidelity.run_cli.argv_check", return_value=1):
                self.assertTrue(run_cases(binary, model, receipt, turn, Path(tmp, "out")))
                self.assertEqual(len(calls), 2)
                self.assertFalse(run_cases(binary, model, receipt, turn, Path(tmp, "skip"), soft=1))
            saved = json.loads(Path(tmp, "out", c["case_id"] + ".receipt.json").read_text())
            self.assertEqual(saved["status"], "COMPLETE")
            self.assertEqual(saved["resolved"], {"profile": "auto"})
            skipped = json.loads(Path(tmp, "skip", c["case_id"] + ".receipt.json").read_text())
            self.assertEqual(skipped["status"], "INCOMPLETE")

    def test_turn_generation_preflight_syntax_without_execution(self):
        # Generation needs relative staged paths; never execute the generated turn.
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            # Python >= 3.12 returns an absolute path even for dir=".".
            root = Path(os.path.relpath(tmp))
            binary, receipt, model = root / "binary", root / "model.receipt", root / "model.gguf"
            for p in (binary, receipt, model):
                p.write_bytes(b"fixture")
            case = write_case(root, *build_case("niah_single", 2048, .5, 42, FakeTokenizer()))
            listing = root / "turn.txt"
            listing.write_bytes((case.name + "\n").encode())
            script = root / "turn.sh"
            m = make_script(binary, model, receipt, listing, root / "out", script)
            raw = script.read_bytes()
            self.assertNotIn(b"\r", raw)
            self.assertIn(b"ulimit -s 65536\n", raw)
            self.assertIn(str(binary).replace("\\", "/"), m["files"])
            text = raw.decode()
            embedded = text.split("<<'PY_PREFLIGHT'\n", 1)[1].split("PY_PREFLIGHT\n", 1)[0]
            compile(embedded, "preflight", "exec")
            # bash -n parses only; it does not run any command from the script.
            import shutil
            bash = shutil.which("bash")
            if bash:
                subprocess.run([bash, "-n", script.as_posix()], check=True, capture_output=True)

    def test_matrix_identity_and_boundary_refusal(self):
        cells = matrix(["native"], [8192], ["niah_single"], [104729], [.05, .95])
        self.assertEqual(len({c["case_id"] for c in cells}), 2)
        for args in (([262143], 32, "normal"), ([262143], 1, "normal"), ([262144], 1, "symbol")):
            with self.assertRaises(ValueError):
                matrix(["native"], args[0], ["niah_single"], [104729], [.5], args[1], variant=args[2])
        self.assertEqual(matrix(["native"], [262143], ["niah_single"], [104729], [.95], 1, variant="symbol")[0]["generate"], 1)

    def test_plan_lists_resolve_to_manifests_in_sibling_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = build_case("niah_single", 2048, .5, 42, FakeTokenizer())
            case = write_case(Path(tmp, "cases"), *result)
            plans = Path(tmp, "plans")
            write_plan([result[2]], plans, case_files={result[2]["case_id"]: str(case.resolve())})
            entry = (plans / "turn-001.txt").read_text().strip()
            self.assertEqual((plans / entry).resolve(), case.resolve())

    def test_corrupt_case_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            text, ids, c = build_case("niah_single", 2048, .5, 42, FakeTokenizer())
            c["tokenizer"] = {"backend": "synthetic-test"}
            manifest = write_case(tmp, text, ids, c)
            Path(str(manifest.with_suffix("")) + ".txt").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_case(manifest)

    def test_score_rejects_wrong_manifest_and_ids(self):
        _, _, c = build_case("niah_single", 2048, .5, 42, FakeTokenizer())
        r = {"case_id": c["case_id"], "status": "COMPLETE", "input_ids_sha256": c["ids_sha256"],
             "ground_truth_sha256": c["ground_truth_sha256"], "manifest_sha256": "wrong", "generated_tokens": [1]}
        self.assertFalse(score_case(c, r, c["ground_truth"], "actual")["exact"])
        r["input_ids_sha256"] = "wrong"
        self.assertEqual(score_case(c, r, c["ground_truth"])["status"], "INCOMPLETE")

    def test_deep_cells_get_singleton_turns(self):
        cells = matrix(["native"], [196608], ["niah_single"], [104729], [.05, .95])
        turns = pack(cells)
        self.assertEqual([len(t["cases"]) for t in turns], [1, 1])

    def test_decode_separates_only_declared_terminal_eos(self):
        from tools.qualification.longctx_fidelity.score import decode_receipt
        class Tokenizer:
            def decode(self, ids):
                names = {248044: "<|endoftext|>", 248046: "<|im_end|>"}
                return "".join(names.get(i, chr(i)) for i in ids)
        receipt = {"generated_tokens": [65, 248044], "stop_reason": "eos", "eos_token_id": 248044}
        decoded = decode_receipt(Tokenizer(), receipt)
        self.assertEqual(decoded["text"], "A")
        self.assertEqual(decoded["raw_text"], "A<|endoftext|>")
        self.assertEqual(decoded["terminal_eos_id"], 248044)
        receipt["stop_reason"] = "max_new_tokens"
        self.assertEqual(decode_receipt(Tokenizer(), receipt)["text"], "A<|endoftext|>")
        # Observed on hardware: answer, <|im_end|>, "\n", protocol EOS.
        receipt = {"generated_tokens": [65, 66, 248046, 10, 248044], "stop_reason": "eos",
                   "eos_token_id": 248044}
        decoded = decode_receipt(Tokenizer(), receipt)
        self.assertEqual(decoded["text"], "AB")
        self.assertEqual(decoded["end_of_turn_index"], 2)
        self.assertEqual(decoded["raw_text"], "AB<|im_end|>\n<|endoftext|>")
        receipt["stop_reason"] = "eos"
        receipt["stop_reason"] = "eos"
        receipt["generated_tokens"] = [65]
        with self.assertRaises(ValueError):
            decode_receipt(Tokenizer(), receipt)
