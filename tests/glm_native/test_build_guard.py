"""Exercise the actual PowerShell coordination parser without starting a build."""
from pathlib import Path
import subprocess
import tempfile
import unittest

GUARD = Path(__file__).resolve().parents[2]/'src/glm/rtxpf/producer/pc_idle.ps1'


class BuildGuardTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'coord.md'
            path.write_text(text, encoding='utf-8')
            return subprocess.run(['pwsh', '-NoProfile', '-Command',
                '& { param($guard,$coord) . $guard; Get-GpuBusyEvent -CoordinationFile $coord }',
                str(GUARD), str(path)], capture_output=True, text=True)

    def test_quoted_end_does_not_release_active_window(self):
        result = self.parse("- 08:33 worker: GPU-BUSY start (active)\n"
                            "- 08:55 other -> lead: waits for 'GPU-BUSY end'\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '- 08:33 worker: GPU-BUSY start (active)')

    def test_start_end_description_is_not_an_event(self):
        result = self.parse('- 08:33 worker: GPU-BUSY start (active)\n'
                            '- 08:55 other: GPU-BUSY start/end markers are required\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '- 08:33 worker: GPU-BUSY start (active)')

    def test_actual_end_after_prose(self):
        result = self.parse('- 08:33 worker: GPU-BUSY start (active)\n'
                            '- 08:55 other: GPU-BUSY start/end markers are required\n'
                            '- 09:03 worker: **GPU-BUSY end** (done)\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '- 09:03 worker: **GPU-BUSY end** (done)')

    def test_missing_event_fails_closed(self):
        result = self.parse("- 08:55 lead -> worker: ACK after 'GPU-BUSY end'\n")
        self.assertNotEqual(result.returncode, 0)

    def test_placeholder_minute_in_actual_event(self):
        result = self.parse('- 08:33 worker: GPU-BUSY start (active)\n'
                            '- 09:0x worker: GPU-BUSY end (done)\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '- 09:0x worker: GPU-BUSY end (done)')


if __name__ == '__main__':
    unittest.main()
