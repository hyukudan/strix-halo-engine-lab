"""Offline test of fast-gate.sh run: the peer wait handed to kit_pair_arm (no GPU, no peer, stub runlib)."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent

STUB_RUNLIB = r"""
KIT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
KIT_PY=python3
kit_die() { echo "DIE $*"; exit 2; }
kit_same_binary() { :; }
kit_peer_inputs() { :; }
kit_left() { echo 999; }
kit_pair_arm() { echo "peer_wait=${KIT_PEER_WAIT:-unset} base=$(basename "$1")"; }
"""


class FastGateRun(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        kit = self.d / "scripts" / "bench" / "minimax"
        kit.mkdir(parents=True)
        shutil.copy(HERE / "fast-gate.sh", kit / "fast-gate.sh")
        (self.d / "scripts" / "bench" / "common").mkdir(parents=True)
        (self.d / "scripts" / "bench" / "common" / "runlib.sh").write_text(STUB_RUNLIB, encoding="utf-8", newline="\n")
        self.bin = self.d / "rank-bin"
        self.bin.write_text("#!/bin/sh\n", encoding="utf-8", newline="\n")
        self.bin.chmod(0o755)
        (self.d / "model").mkdir()
        (self.d / "p.u32").write_bytes(b"\0" * 8)
        (self.d / "out").mkdir()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def run_kit(self, env_extra):
        env = {k: v for k, v in os.environ.items() if k != "KIT_PEER_WAIT"}
        env.update(env_extra)
        # The PATH bash (on Windows CreateProcess would pick System32's WSL launcher, which re-parses the arguments).
        cmd = [shutil.which("bash") or "bash", "scripts/bench/minimax/fast-gate.sh", "run", "--bin", "rank-bin", "--model", "model",
               "--arm", "m3l40||STRIX_MM_PF_MSA_FLASH=3", "--sessions", "cpros8k=p.u32", "--out", "out"]
        r = subprocess.run(cmd, cwd=self.d, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout

    def test_default_peer_wait_is_short(self):
        # Rank 1 announces itself only after rank 0 connects; the generic 120 s wait would idle every ticket.
        out = self.run_kit({})
        self.assertIn("peer_wait=3 base=m3l40", out)
        self.assertEqual((self.d / "out" / "m3l40.env").read_text(encoding="utf-8").strip(), "STRIX_MM_PF_MSA_FLASH=3")

    def test_explicit_peer_wait_wins(self):
        self.assertIn("peer_wait=7 ", self.run_kit({"KIT_PEER_WAIT": "7"}))


if __name__ == "__main__":
    unittest.main()
