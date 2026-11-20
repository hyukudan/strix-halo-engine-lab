#!/usr/bin/env python3
"""CPU-only checks: a marker must mean that every required benchmark succeeded."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parent / "buckets.sh"

def executable(path, contents):
    path.write_text("#!/usr/bin/env bash\n" + contents)
    path.chmod(0o755)

class CompletionMarkerTests(unittest.TestCase):
    def run_case(self, *, binary=True, profiler=True, profile_fail=False, vendor_fail=False,
                 trace_missing=False, kernel_rows_missing=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = root / "build"
            build.mkdir()
            if binary:
                executable(build / "strix-hip-moe-wmma-prefill-tests", 'echo "bench2 mock"\n')
            prof = root / "rocprofv3"
            if profiler:
                executable(prof, (
                    'if [ "${PROFILE_FAIL:-0}" = 1 ]; then exit 7; fi\n'
                    'if [ "${TRACE_MISSING:-0}" = 1 ]; then echo "bench2 mock"; exit 0; fi\n'
                    'while [ "$#" -gt 0 ]; do\n'
                    '  if [ "$1" = "-d" ]; then mkdir -p "$2"; '
                    'if [ "${ROWS_MISSING:-0}" = 1 ]; then '
                    'printf "Name,TotalDurationNs,Calls,AverageNs\\nother_kernel,1000,1,1000\\n" > "$2/kernel_stats.csv"; '
                    'else printf "Name,TotalDurationNs,Calls,AverageNs\\nprefill_f16x,1000,1,1000\\n" > "$2/kernel_stats.csv"; fi; '
                    'echo "bench2 mock"; exit 0; fi\n'
                    '  shift\n'
                    'done\n'
                ))
            vendor = root / "vendor"
            executable(vendor, 'if [ "${VENDOR_FAIL:-0}" = 1 ]; then exit 8; fi\necho "hb mock"\n')
            out = root / "out"
            env = os.environ.copy()
            env.update(ROCPROF=str(prof), PROFILE_FAIL=str(int(profile_fail)),
                       VENDOR_FAIL=str(int(vendor_fail)),
                       TRACE_MISSING=str(int(trace_missing)),
                       ROWS_MISSING=str(int(kernel_rows_missing)), ATTRIB="0")
            result = subprocess.run(["bash", str(SCRIPT), str(build), str(vendor), str(out), "8"],
                                    env=env, text=True, capture_output=True, timeout=10)
            return result.returncode, (out / "buckets.done").exists()

    def test_missing_binary(self):
        rc, done = self.run_case(binary=False)
        self.assertNotEqual(rc, 0)
        self.assertFalse(done)

    def test_missing_profiler(self):
        rc, done = self.run_case(profiler=False)
        self.assertNotEqual(rc, 0)
        self.assertFalse(done)

    def test_profiler_failure(self):
        rc, done = self.run_case(profile_fail=True)
        self.assertNotEqual(rc, 0)
        self.assertFalse(done)

    def test_missing_trace_file(self):
        rc, done = self.run_case(trace_missing=True)
        self.assertNotEqual(rc, 0)
        self.assertFalse(done)

    def test_missing_required_kernel_rows(self):
        rc, done = self.run_case(kernel_rows_missing=True)
        self.assertNotEqual(rc, 0)
        self.assertFalse(done)

    def test_vendor_failure_in_pipeline(self):
        rc, done = self.run_case(vendor_fail=True)
        self.assertNotEqual(rc, 0)
        self.assertFalse(done)

    def test_complete_campaign(self):
        rc, done = self.run_case()
        self.assertEqual(rc, 0)
        self.assertTrue(done)

if __name__ == "__main__":
    unittest.main()
