"""Behavioral tests; HIP/export observation are doubles, never native GPU evidence."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
class NativeHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='gpu-queue-native-helpers-')
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.binaries = {}
        for kind, source in [('memory', 'gpu_queue_native_memory_test.cpp'), ('options', 'gpu_queue_options_test.cpp')]:
            target = Path(cls.tmp.name) / kind
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++20', '-g', '-O1', '-Wall', '-Wextra', '-Werror',
                            '-Wconversion', '-Wshadow', '-fno-omit-frame-pointer', '-fsanitize=address,undefined', '-fno-pie', '-no-pie',
                            '-I', str(ROOT/'tests/gpu_queue_native_shim'), str(ROOT/'tests'/source), '-o', str(target)], check=True)
            cls.binaries[kind] = target
    def case(self, kind, case):
        p = subprocess.run([str(self.binaries[kind]), case], capture_output=True, text=True, timeout=20,
                           env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1:abort_on_error=1'})
        self.assertEqual(p.returncode, 0, p.stdout+p.stderr)
        self.assertIn('PASS '+case, p.stdout)
for kind, cases in [('memory', ['normal','bounds','null_lease','partial_allocation','busy','capture','timeout','device','repeat_allocation']),
                    ('options', ['controller','network','network_no_control','network_only','controller_no_schedule','controller_duplicate','valid','duplex','ack','duplicate','geometry','depth','legacy','epoch','budget'])]:
    for case in cases:
        setattr(NativeHelpers, 'test_'+kind+'_'+case, lambda self, kind=kind, case=case: self.case(kind,case))
if __name__ == '__main__':
    unittest.main()
