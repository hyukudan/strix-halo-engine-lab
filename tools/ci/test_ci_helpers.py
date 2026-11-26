"""Tests of the CI helpers: change classification (ci_changes.py) and unittest sharding (unittest_shard.py)."""
import pathlib
import subprocess
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ci_changes  # noqa: E402
import unittest_shard  # noqa: E402


def never(_path):
    return False


def always(_path):
    return True


class Classify(unittest.TestCase):
    def check(self, paths, code, hip, grep=never):
        self.assertEqual(ci_changes.classify(paths, grep), {'code': code, 'hip': hip}, paths)

    def test_empty_or_unknown_runs_everything(self):
        self.check([], True, True)

    def test_docs_only_skips_builds(self):
        self.check(['README.md', 'docs/models/x.md', 'docs/media/a.png', 'LICENSE'], False, False)

    def test_doc_read_by_a_test_runs_host_build(self):
        self.check(['docs/getting-started/engine-reference.md'], True, False, grep=always)

    def test_top_level_doc_is_not_grepped(self):
        self.check(['README.es.md'], False, False, grep=always)

    def test_one_code_path_turns_code_on(self):
        self.check(['docs/a.md', 'tools/ci/check_layering.py'], True, False)

    def test_hip_inputs(self):
        for path in ('src/core/x.cpp', 'include/strix/a.hpp', 'CMakeLists.txt', 'cmake/StrixDevice.cmake',
                     'src/hip_shared_expert_launcher.hip', 'tests/hip_launcher_tests.cpp',
                     'lab/uma_residency/hip_residency.hip', '.github/workflows/ci.yml', 'apps/x.cpp',
                     'data/devices/profiles.toml'):
            self.check([path], True, True)

    def test_non_hip_paths(self):
        for path in ('src/cuda/qwen/a.cu', 'include/strix/k.cuh', 'src/cuda/host.cpp', 'tests/cuda/t.cpp',
                     'tools/strix-gate/x.py', 'scripts/bench/a.sh', 'tests/serve_cycles_tests.py',
                     'lab/transport/tcp.cpp', '.github/workflows/other.yml'):
            self.check([path], True, False)

    def test_codegen_and_scratch_contract_changes_run_hip_ci(self):
        for path in ('tools/gen_q8_rows_vopd.py', 'tools/gen_ds41_qk_asm.py', 'tools/ci/check_generated_asm.py',
                     'tools/check_kernel_scratch.py', 'tools/kernel_scratch_allowance.txt'):
            self.check([path], True, True)


class Shard(unittest.TestCase):
    class Fake:
        def __init__(self, name):
            self.name = name

        def id(self):
            return self.name

    def test_shards_partition_every_test_once(self):
        tests = [self.Fake(f't{k:02d}') for k in range(37)]
        for count in (1, 2, 5, 8, 40):
            seen = []
            for index in range(count):
                seen += [t.id() for t in unittest_shard.shard(list(reversed(tests)), index, count)]
            self.assertEqual(sorted(seen), sorted(t.id() for t in tests), count)

    def test_shard_is_order_independent(self):
        tests = [self.Fake(n) for n in ('b', 'a', 'd', 'c')]
        self.assertEqual([t.id() for t in unittest_shard.shard(tests, 1, 2)], ['b', 'd'])

    def test_bad_arguments(self):
        self.assertEqual(unittest_shard.main(['x', 'm', '2', '2']), 2)
        self.assertEqual(unittest_shard.main(['x', 'm']), 2)


class Scratch(unittest.TestCase):
    def test_allowance_requires_decoded_kernel_metadata(self):
        root = pathlib.Path(__file__).resolve().parents[2]
        script = root / 'tools/check_kernel_scratch.py'
        allowance = root / 'tools/kernel_scratch_allowance.txt'
        for name, expected in (('README.md', 2), ('lab/cross_compile/out/probe.gfx1151.hsaco', 0)):
            with self.subTest(artifact=name):
                result = subprocess.run([sys.executable, '-I', '-B', str(script), '--quiet',
                                         '--allowance', str(allowance), str(root / name)],
                                        capture_output=True, text=True, timeout=15, check=False)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
