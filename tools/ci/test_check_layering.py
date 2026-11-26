#!/usr/bin/env python3
"""Host tests of tools/ci/check_layering.py on a synthetic tree (no repository files read)."""
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_layering as cl  # noqa: E402

TREE = {
    "include/strix/serve/api.hpp": '#include "strix/glm/engine.hpp"\n',          # shared -> model
    "include/strix/glm/engine.hpp": "// glm\n",
    "include/strix/session/plan.hpp": '#include "strix/fnv.hpp"\n',               # shared -> shared: fine
    "include/strix/fnv.hpp": "",
    "src/glm/kernels/tier.hip": '#include "../../minimax/moe/mmx_moe.hpp"\n'      # cross-model, two kernels
                                "__global__ void a() {}\n__global__ void b() {}\n",
    "src/minimax/moe/mmx_moe.hpp": "",
    "src/minimax/xfer/minimax_xfer.hip": '#include "../../xport/channel.hip"\n',  # tu-include (shared TU)
    "src/xport/channel.hip": "",
    "src/hip_ds41_rows.hip": "// __global__ in a comment\n"                        # ds41 by file name
                             'const char* s = "__global__ getenv(";\n'
                             "__global__ void k() { while (__hip_atomic_load(f, 2, 1) < s) {} }\n",
    "src/core/tile.hip": "__global__ void shared_kernel() {}\n",                  # shared home: kernels fine
    "src/serve/options.cpp": 'auto v = std::getenv("X"); auto w = getenv("Y"); x.getenv(1);\n',
    "include/strix/env_flags.hpp": 'inline auto f() { return std::getenv("Z"); }\n',  # env home: not counted
    "src/dist/wait.cpp": "char c = '\"'; while (done.load(std::memory_order_acquire) < n) {}\n",
    "apps/main.cpp": '#include "strix/glm/engine.hpp"\n'                        # composition root
                     'getenv("X"); while (ready.load() == 0) {}\n',
    "tests/t.cpp": "getenv(\"T\");\n",
}


class Layering(unittest.TestCase):
    def counts(self, tree=TREE):
        return cl.scan(list(tree), lambda p: tree.get(p))

    def test_model_of(self):
        self.assertEqual(cl.model_of("src/hip_ds41_rows.hip"), "deepseek41")
        self.assertEqual(cl.model_of("src/cuda/models/qwen/x.cu"), "qwen")
        self.assertEqual(cl.model_of("include/strix/qwen35/config.hpp"), "qwen")
        self.assertEqual(cl.model_of("src/minimax/moe/x.hip"), "minimax")
        self.assertIsNone(cl.model_of("src/moe_core/grouped.hip"))
        self.assertIsNone(cl.model_of("src/hip_mtp_draft_head.hip"))

    def test_legacy_and_new_model_paths_are_ratchet_checked(self):
        model_paths = {
            "src/hy4/hip/new.hip": "hy4",
            "src/inkling2/new.cu": "inkling2",
            "src/ernie/new.hip": "ernie",
            "src/hip_indexed_attention_launcher.hip": "qwen",
            "src/hip_gdn_launcher.hip": "qwen",
            "src/hip_hyper_mix_launcher.hip": "qwen",
            "src/hip_moe_experts_launcher.hip": "qwen",
            "src/hip_ple_launcher.hip": "qwen",
        }
        for path, family in model_paths.items():
            with self.subTest(path=path):
                self.assertEqual(cl.model_of(path), family)
                self.assertIn(("model-kernel", path), self.counts({path: "__global__ void k() {}"}))

    def test_qwen_launcher_kernel_fragments_are_ratchet_checked(self):
        paths = (
            "src/hip_gdn_fp16_chunk.inc",
            "src/hip_qpf2k_hc_output3_body.inc",  # nested under a launcher-owned fragment
            "src/hip_qsa_index_band.inc",
            "src/hip_moe_prefill_commit.inc",
            "src/hip_ple_rows_site.inc",
        )
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(cl.model_of(path), "qwen")
                self.assertEqual(self.counts({path: "__global__ void k() {}"}), {("model-kernel", path): 1})
        self.assertIsNone(cl.model_of("src/hip_gdn_shared_primitive.inc"))

    def test_rules(self):
        c = self.counts()
        self.assertEqual(c.get(("model-include", "include/strix/serve/api.hpp")), 1)
        self.assertNotIn(("model-include", "include/strix/session/plan.hpp"), c)
        self.assertEqual(c.get(("cross-model", "src/glm/kernels/tier.hip")), 1)
        self.assertEqual(c.get(("tu-include", "src/minimax/xfer/minimax_xfer.hip")), 1)
        self.assertEqual(c.get(("model-kernel", "src/glm/kernels/tier.hip")), 2)
        # comments and string literals do not count; the real kernel does
        self.assertEqual(c.get(("model-kernel", "src/hip_ds41_rows.hip")), 1)
        self.assertNotIn(("raw-getenv", "src/hip_ds41_rows.hip"), c)
        self.assertNotIn(("model-kernel", "src/core/tile.hip"), c)
        self.assertEqual(c.get(("raw-getenv", "src/serve/options.cpp")), 2)  # x.getenv() is a member, not libc
        self.assertNotIn(("raw-getenv", "include/strix/env_flags.hpp"), c)
        self.assertEqual(c.get(("spin-wait", "src/hip_ds41_rows.hip")), 1)
        self.assertEqual(c.get(("spin-wait", "src/dist/wait.cpp")), 1)  # a '"' char literal does not open a string
        self.assertEqual(c.get(("raw-getenv", "apps/main.cpp")), 1)
        self.assertEqual(c.get(("spin-wait", "apps/main.cpp")), 1)
        self.assertNotIn(("model-include", "apps/main.cpp"), c)
        self.assertFalse(any(p.startswith("tests/") for _, p in c))

    def test_ratchet(self):
        c = self.counts()
        self.assertEqual(cl.compare(c, dict(c)), [])
        # a count that rises fails, and so does a new (rule, path)
        tree = dict(TREE)
        tree["src/glm/kernels/tier.hip"] += "__global__ void c() {}\n"
        tree["src/serve/new.cpp"] = '#include "strix/glm/engine.hpp"\n'
        msgs = cl.compare(self.counts(tree), dict(c))
        self.assertEqual(len(msgs), 2)
        self.assertTrue(any("model-kernel 3 (rose from 2)" in m for m in msgs))
        self.assertTrue(any("src/serve/new.cpp: model-include 1 (new)" in m for m in msgs))
        # a count that falls must be written down (the list stays the exact to-do list)
        tree = dict(TREE)
        tree["src/glm/kernels/tier.hip"] = "__global__ void a() {}\n"
        msgs = cl.compare(self.counts(tree), dict(c))
        self.assertTrue(any("model-kernel fell to 1 from 2" in m for m in msgs))
        self.assertTrue(any("cross-model fell to 0 from 1" in m for m in msgs))

    def test_allowlist_roundtrip(self):
        c = self.counts()
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "allow.txt"
            p.write_text(cl.format_allowlist(c), encoding="utf-8")
            self.assertEqual(cl.read_allowlist(p), c)
            p.write_text("model-kernel src/a.hip 0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                cl.read_allowlist(p)
            p.write_text("no-such-rule src/a.hip 1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                cl.read_allowlist(p)


if __name__ == "__main__":
    unittest.main()
