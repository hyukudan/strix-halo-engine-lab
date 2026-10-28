import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools/sessioncache"
spec = importlib.util.spec_from_file_location("gate_identity", TOOLS / "gate_identity.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

class GateTests(unittest.TestCase):
    def test_preserved_turn1(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "checkpoint").mkdir()
            (out / "sessions/A").mkdir(parents=True)
            (out / "checkpoint/session.bin").write_bytes(b"turn1")
            (out / "sessions/A/session.bin").write_bytes(b"turn2")
            gate.restore_turn1(out)
            self.assertEqual((out / "sessions/A/session.bin").read_bytes(), b"turn1")
            self.assertEqual((out / "checkpoint/session.bin").read_bytes(), b"turn1")

    def test_identity_requires_all_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with self.assertRaises(FileNotFoundError): gate.compare_identity(out)
            for name in gate.ARMS:
                (out / (name + ".identity.json")).write_text(json.dumps({"tokens": [1], "sampling": "sampling-v1"}))
            gate.compare_identity(out)
            for name in gate.ARMS:
                path = out / (name + ".identity.json")
                original = path.read_text()
                path.write_text(json.dumps({"tokens": [2], "sampling": "sampling-v1"}))
                with self.assertRaises(RuntimeError): gate.compare_identity(out)
                path.write_text(original)

    def test_transport_timing_does_not_hide_numerical_changes(self):
        response = {"id": "request1", "choices": [{"logprobs": [0.25]}], "usage": {"total_tokens": 2},
                    "strix": {"token_ids": [7], "sampling": "sampling-v1", "decode_ms": 1}}
        before = gate.http_identity(response)
        response["id"] = "request2"
        response["strix"]["decode_ms"] = 2
        self.assertEqual(before, gate.http_identity(response))
        response["strix"]["token_ids"] = [8]
        self.assertNotEqual(before, gate.http_identity(response))

    def test_missing_digest_fails_closed(self):
        with self.assertRaises(RuntimeError): gate.numerical_digests({"generated_tokens": [1]}, "")
        digest = "a" * 64
        result = gate.numerical_digests({}, 'tiled prompt: ' + json.dumps({"state_digest": digest, "prompt_ms": 12}))
        self.assertEqual(result, {"prompt.state_digest": digest})

    def test_unaligned_target_gate_arm(self):
        source = (TOOLS / "qwen-session-cli-mtp-gate.sh").read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
        ast.parse(source)
        # Execute only the pure case selector and workload admission; no engine launch.
        selector = source.split("phase, binary, model, pfile, sfile, output, *flags = sys.argv[1:]", 1)[1].split("if flags[:1]", 1)[0]
        for phase in ("1", "2", "3", "4"):
            scope = {"phase": "unaligned-" + phase}
            exec(selector, scope)
            self.assertTrue(scope["unaligned"])
            self.assertEqual(scope["phase"], phase)
        with self.assertRaises(SystemExit): exec(selector, {"phase": "unaligned-5"})
        admission = source.split("if unaligned and (first != 2", 1)[1].split("if first < 1", 1)[0]
        admission = "if unaligned and (first != 2" + admission
        for first, rows, suffix, flags, valid in ((2,1024,[0]*2046,[],True),
                (1,1024,[0]*2046,[],False), (2,2048,[0]*2046,[],False),
                (2,1024,[0]*2045,[],False), (2,1024,[0]*2046,["--session-mtp-fixed"],False)):
            scope = dict(unaligned=True,first=first,rows=rows,suffix=suffix,flags=flags)
            if valid: exec(admission, scope)
            else:
                with self.assertRaises(SystemExit): exec(admission, scope)
        self.assertIn("alignment_steps=1023", source)

    def test_phased_gate_syntax_and_final_admission(self):
        for name in ("qwen-session-cli-mtp-gate.sh", "qwen-session-http-gate.sh"):
            source = (TOOLS / name).read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
            tree = ast.parse(source)
            passes = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and isinstance(node.value, ast.Constant) and node.value.value == "PASS"]
            self.assertEqual(len(passes), 1)
            self.assertIn("if phase=='7':\n            compare_identity(out)", source)
            if "http" in name:
                self.assertLess(source.index("restore_turn1(out)"), source.index("start('restarted-server')"))

if __name__ == "__main__": unittest.main()
