"""Tests of the CI helpers: change classification (ci_changes.py) and unittest sharding (unittest_shard.py)."""
import pathlib
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
                     'lab/uma_residency/hip_residency.hip', '.github/workflows/ci.yml', 'apps/x.cpp'):
            self.check([path], True, True)

    def test_non_hip_paths(self):
        for path in ('src/cuda/qwen/a.cu', 'include/strix/k.cuh', 'src/cuda/host.cpp', 'tests/cuda/t.cpp',
                     'tools/strix-gate/x.py', 'scripts/bench/a.sh', 'tests/serve_cycles_tests.py',
                     'lab/transport/tcp.cpp', '.github/workflows/other.yml'):
            self.check([path], True, False)


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


if __name__ == '__main__':
    unittest.main()
