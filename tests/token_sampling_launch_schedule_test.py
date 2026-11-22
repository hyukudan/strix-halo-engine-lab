#!/usr/bin/env python3
"""Host-side launch-graph guard for the common HIP/CUDA top-p sampler.

This checks the *enqueued schedule*; device/bitwise qualification remains separate.
"""
from pathlib import Path
import re
import sys
import unittest

SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "src/core/token_sampling_device_body.inl"


class TopPLaunchScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        code = SOURCE.read_text(encoding="utf-8")
        cls.top_p = code.split("if (p.top_p_active) {", 1)[1].split(
            "} else {\n                if (drafts)", 1
        )[0]

    def test_zero_residual_fallback_keeps_its_quantile_and_finish(self):
        # Prefix, initial draw/residual, then a zero-residual fallback draw.
        # The final pass is needed only for drafted rows: topp_finish_kernel
        # resets kind 2 with zero mass to kind 1 instead of publishing a token.
        self.assertEqual(self.top_p.count("quantile();"), 3)
        self.assertEqual(self.top_p.count("STRIX_SAMPLE_LAUNCH(topp_finish_kernel"), 2)
        self.assertRegex(self.top_p, re.compile(
            r"STRIX_SAMPLE_LAUNCH\(topp_finish_kernel[^;]*;\s*"
            r"if \(drafts\) \{\s*quantile\(\);\s*"
            r"STRIX_SAMPLE_LAUNCH\(topp_finish_kernel[^;]*;", re.DOTALL))

    def test_residual_is_conditional_but_final_decision_is_not(self):
        self.assertRegex(
            self.top_p,
            re.compile(
                r"quantile\(\);\s*"
                r"STRIX_SAMPLE_LAUNCH\(topp_after_prefix_kernel[^;]*;\s*"
                r"if \(drafts\) \{\s*"
                r"STRIX_SAMPLE_LAUNCH\(topp_residual_kernel[^;]*;\s*"
                r"STRIX_SAMPLE_LAUNCH\(topp_q_out_kernel[^;]*;\s*"
                r"\}\s*quantile\(\);\s*"
                r"STRIX_SAMPLE_LAUNCH\(topp_finish_kernel[^;]*;",
                re.DOTALL,
            ),
        )


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
