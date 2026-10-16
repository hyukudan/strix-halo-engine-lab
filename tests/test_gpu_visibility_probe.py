"""Execute native probe function bodies with a sequential HIP API double.

No ROCm compiler, GPU hardware or cache-coherence proof is provided by this test.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
CASES=('full_bytes','epoch_replay','single_byte','bounds','foreign_device','captured',
       'stale_result','failed_record','timeout','ambiguous_cleanup')
class VisibilityProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='gpu-visibility-')
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.exe=Path(cls.tmp.name)/'test'
        subprocess.run([os.environ.get('CXX','g++'),'-std=c++20','-O1','-g','-Wall','-Wextra','-Werror',
            '-fsanitize=address,undefined', '-fno-pie', '-no-pie','-fno-omit-frame-pointer','-fno-strict-aliasing','-pthread',
            '-I',str(ROOT/'tests/hip_visibility_shim'),'-I',str(ROOT/'include'),'-x','c++',
            str(ROOT/'src/hip_gpu_visibility_probe.hip'),str(ROOT/'tests/hip_gpu_visibility_probe_test.cpp'),
            '-o',str(cls.exe)],check=True)
    def run_case(self,case):
        p=subprocess.run([str(self.exe),case],capture_output=True,text=True,timeout=10,
            env={**os.environ,'ASAN_OPTIONS':'detect_leaks=1:abort_on_error=1'})
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        self.assertIn('PASS '+case,p.stdout)
for c in CASES:
    setattr(VisibilityProbeTests,'test_'+c,lambda self,c=c:self.run_case(c))
if __name__=='__main__': unittest.main(verbosity=2)
