import json
from pathlib import Path
import tempfile
import unittest

from verify_profile_parity import verify


class ParityTests(unittest.TestCase):
    def test_verifies_capture_and_refuses_drift(self):
        with tempfile.TemporaryDirectory() as root:
            cli, http = (Path(root) / name for name in ("cli", "http"))
            for directory in (cli, http):
                (directory / "state").mkdir(parents=True)
                (directory / "effective.json").write_text(json.dumps({"lever": "on", "entry": directory.name}))
                (directory / "receipt.json").write_text('{"tier":"fast"}')
                (directory / "prompt.u32").write_bytes(b"\x01\0\0\0")
                (directory / "output.u32").write_bytes(b"\x02\0\0\0")
                (directory / "last_logits.f32").write_bytes(b"\0\0\0\0")
                (directory / "state" / "kv.bin").write_bytes(b"state")

            fields = {"entry": "entry-point label"}
            self.assertEqual(verify(cli, http, fields)["schema"], "strix-profile-parity/v1")
            with self.assertRaisesRegex(ValueError, "effective"):
                verify(cli, http, {})
            with self.assertRaisesRegex(ValueError, "stale"):
                verify(cli, http, {"unknown": "stale"})
            for path, error in (("prompt.u32", "prompt"), ("output.u32", "output"), ("last_logits.f32", "logits"),
                                ("state/kv.bin", "state"), ("receipt.json", "receipt")):
                with self.subTest(path=path):
                    file = http / path
                    original = file.read_bytes()
                    file.write_bytes(b'{"changed":true}' if path.endswith(".json") else b"\x03\0\0\0")
                    with self.assertRaisesRegex(ValueError, error):
                        verify(cli, http, fields)
                    file.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
