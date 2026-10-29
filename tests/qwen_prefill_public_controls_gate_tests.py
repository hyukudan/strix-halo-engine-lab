#!/usr/bin/env python3
"""CPU argument-validation regressions for the native qualification gate."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import qwen_prefill_public_controls_gate as gate
import qwen_prefill_mtp_gate as mtp_gate
import copy
import hashlib
import struct


class ReferenceReached(Exception):
    pass


class CommandValidationTests(unittest.TestCase):
    def check_command(self, flags, accepted, kind="state"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("logits", "state"):
                command = ["decoder", "model.gguf", "1"]
                if name == "logits":
                    command.append("--teacher-forced-score")
                if name == kind:
                    command.extend(flags)
                (root / (name + ".json")).write_text(json.dumps(command))
            argv = ["gate", "--reference-logits", str(root / "reference.bin"),
                    "--reference-state", str(root / "manifest.json"),
                    "--output", str(root / "output"),
                    "--logits-command", str(root / "logits.json"),
                    "--state-command", str(root / "state.json")]
            with patch("sys.argv", argv), patch.object(
                    gate, "digest", side_effect=ReferenceReached) as digest:
                if accepted:
                    with self.assertRaises(ReferenceReached):
                        gate.main()
                    digest.assert_called_once_with(root / "reference.bin")
                else:
                    with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                        gate.main()
                    self.assertEqual(error.exception.code, 2)
                    digest.assert_not_called()
                self.assertFalse((root / "output").exists())

    def test_export_budget_allowed(self):
        for flags in ([], ["--validation-final-state-max-bytes", "17179869184"],
                      ["--validation-final-state-max-bytes=17179869184"]):
            with self.subTest(flags=flags):
                self.check_command(flags, accepted=True)

    def test_state_only_long_prompt(self):
        # Long prompts omit teacher logits; a logits reference then makes no sense.
        for with_logits_reference, accepted in ((False, True), (True, False)):
            with self.subTest(with_logits_reference=with_logits_reference), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "state.json").write_text(json.dumps(["decoder", "model.gguf", "1"]))
                argv = ["gate", "--reference-state", str(root / "manifest.json"),
                        "--output", str(root / "output"), "--state-command", str(root / "state.json")]
                if with_logits_reference:
                    argv += ["--reference-logits", str(root / "reference.bin")]
                with patch("sys.argv", argv), patch.object(
                        gate, "state_signature", side_effect=ReferenceReached) as signature:
                    if accepted:
                        with self.assertRaises(ReferenceReached):
                            gate.main()
                        signature.assert_called_once_with(root / "manifest.json")
                    else:
                        with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                            gate.main()
                        self.assertEqual(error.exception.code, 2)
                        signature.assert_not_called()
                    self.assertFalse((root / "output").exists())

    def test_signature_stands_for_discarded_logits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logits.bin").write_bytes(b"logits")
            expected = gate.digest(root / "logits.bin")
            (root / "signature.json").write_text(json.dumps({"logits_sha256": expected, "state": None}))
            self.assertEqual(gate.logits_reference(root / "signature.json"), expected)
            self.assertEqual(gate.logits_reference(root / "logits.bin"), expected)
            (root / "signature.json").write_text(json.dumps({"logits_sha256": None, "state": None}))
            with self.assertRaises(ValueError):
                gate.logits_reference(root / "signature.json")

    def test_artifact_destinations_rejected(self):
        for kind in ("logits", "state"):
            for option in ("--validation-final-state", "--teacher-logits-file"):
                for flags in ([option, "caller-output"], [option + "=caller-output"]):
                    with self.subTest(kind=kind, flags=flags):
                        self.check_command(flags, accepted=False, kind=kind)


class SessionStateTests(unittest.TestCase):
    def fixture(self, identity=b"identity", changed=False):
        def word(x):
            return struct.pack("<Q", x)
        def string(x):
            return word(len(x)) + x
        header = string(b"strix-session-v1") + word(1) + b"m" * 32 + b"e" * 32 + string(identity) + word(0)
        data = header + hashlib.sha256(header).digest()
        data += b"".join(word(x) for x in (1, 1, 1, 0, 42)) + string(b"greedy") + word(1) + word(7)
        names = [b"kv.L3", b"index.L3", b"recurrent_matrix", b"gdn_convolution", b"ple_convolution",
                 b"ngram_history", b"position_history", b"session.metadata"]
        data += word(len(names))
        for name in names:
            payload = identity if name == b"session.metadata" else (b"changed" if changed and name == b"kv.L3" else b"state")
            data += string(name) + word(0) + word(0) + word(len(payload)) + hashlib.sha256(payload).digest()
            data += word(len(data) + 8) + payload
        return data + hashlib.sha256(data).digest()

    def test_session_reference_checks_payload_but_allows_identity_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.bin"
            path.write_bytes(self.fixture())
            original = gate.state_signature(path)
            path.write_bytes(self.fixture(b"new-identity"))
            self.assertEqual(gate.state_signature(path), original)
            path.write_bytes(self.fixture(changed=True))
            self.assertNotEqual(gate.state_signature(path), original)
            broken = bytearray(self.fixture())
            broken[-40] ^= 1
            path.write_bytes(broken)
            with self.assertRaises(ValueError):
                gate.state_signature(path)


class MtpSignatureTests(unittest.TestCase):
    def fixture(self):
        return dict(schema="mtp-prefix-gate-v1", passed=True, full_logits_bits_equal=True,
                    all_seven_committed_state_components_equal=True, initial_position=4096,
                    numerical_profile="profile", target_content="content", state_scope="seven",
                    physical_state_scope="physical-seven", initial_committed_state=["hash"] * 7,
                    initial_physical_state=["hash"] * 7, case_count=1,
                    cases=[dict(rows=2, keep=1, offset=0, tokens=[42], all_physical_logits=["a", "b"],
                                committed_state=["c"] * 7, full_physical_state=["d"] * 7,
                                physical_write_generations=[1] * 7)])

    def test_identity_metadata_can_differ_but_payload_cannot(self):
        old = self.fixture()
        new = copy.deepcopy(old)
        old["serial_execution_identity"] = 1
        new["serial_execution_identity"] = 2
        self.assertEqual(mtp_gate.signature(old), mtp_gate.signature(new))
        new["cases"][0]["committed_state"][6] = "changed"
        self.assertNotEqual(mtp_gate.signature(old), mtp_gate.signature(new))

    def test_incomplete_coverage_fails(self):
        for field in ("committed_state", "full_physical_state", "all_physical_logits"):
            value = self.fixture()
            value["cases"][0][field].pop()
            with self.subTest(field=field), self.assertRaises(ValueError):
                mtp_gate.signature(value)


if __name__ == "__main__":
    unittest.main()
