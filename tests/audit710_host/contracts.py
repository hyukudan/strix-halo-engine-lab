"""Source integration checks for consumers that require CUDA to build."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]


class Contracts(unittest.TestCase):
    def test_cohort_links_shared_drafter(self):
        cmake = (ROOT / "cmake/QwenCuda.cmake").read_text(encoding="utf-8")
        graph = {}
        for target, dependencies in re.findall(
            r"target_link_libraries\(\s*(\S+)\s+([^)]*)\)", cmake
        ):
            graph.setdefault(target, set()).update(dependencies.split())
        seen = set()

        def visit(target):
            if target not in seen:
                seen.add(target)
                for dependency in graph.get(target, ()):
                    visit(dependency)

        visit("qwen_cohort_e2e")
        self.assertIn("strix_mtp_depth", seen)
        self.assertRegex(cmake, r"add_library\(strix_mtp_depth\b[^)]*src/mtp_context_draft\.cpp")

    def test_server_receipt_sources(self):
        server = (ROOT / "apps/cuda_qwen_serve_main.cu").read_text(encoding="utf-8")
        for receipt, step in (("proposed", "mtp_proposed"), ("accepted", "mtp_accepted"),
                              ("ngram_rows", "context_proposed"), ("ngram_accepted", "context_accepted")):
            self.assertIn(f"s.st.{receipt} += step.{step};", server)
        self.assertIn("lone && step.drafted > 0 && step.context_proposed == 0", server)


if __name__ == "__main__":
    unittest.main()
