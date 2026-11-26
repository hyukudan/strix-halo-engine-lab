import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from check_tier_monotonicity import HEADER, known_rows, violations


TABLE = f"{HEADER}\n|---|---|---|---|---|---|---|---|---|\n"
RECEIPTS = TABLE + (
    "| M | pair | HTTP | plain | EXACT | 1,102 / 8,192; decode after 508 | 27.0 / 26.9 | 1K 417 / 451, 8K 700 | `base` |\n"
    "| M | pair | HTTP | plain | FAST | same | 27.1 / 26.9 | 1K 415 / 457, 8K 750 | `base` |\n"
    "| M | pair | HTTP | plain | TURBO | same | 26.7 / 26.6 | 1K 422 / 459, 8K 755 | `base` |\n"
)
KNOWN = "base\tM\tpair\tHTTP\tplain\tdecode\t508\tfast\tturbo\t27.0\t26.65\n"


class TierMonotonicityTests(unittest.TestCase):
    def test_new_regression_fails_the_host_check(self):
        modified = RECEIPTS.replace("1K 422 / 459", "1K 320 / 319", 1)
        with tempfile.TemporaryDirectory() as folder:
            readme = Path(folder) / "README.md"
            readme.write_text(modified, encoding="utf-8")
            known = Path(folder) / "known.tsv"
            known.write_text(KNOWN, encoding="utf-8")
            result = subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("check_tier_monotonicity.py")),
                                     "--readme", str(readme), "--known", str(known)], capture_output=True, check=False,
                                    encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"})
        self.assertEqual(result.returncode, 1)
        self.assertIn("new tier slowdown", result.stderr)
        self.assertIn("prefill\t1102\tfast\tturbo", result.stderr)

    def test_resolved_entry_must_leave_known_list(self):
        modified = RECEIPTS.replace("26.7 / 26.6", "27.3 / 27.2", 1)
        self.assertEqual(violations(RECEIPTS), known_rows(KNOWN))
        self.assertEqual(violations(modified), set())

    def test_context_route_and_revision_must_match(self):
        rows = ("| M | pair | CLI | plain | EXACT | 2,048 | 20 | 100 | `base` |\n"
                "| M | pair | HTTP | plain | FAST | 2,048 | 5 | 10 | `base` |\n"
                "| M | pair | CLI | plain | FAST | 4,096 | 5 | 10 | `base` |\n"
                "| M | pair | CLI | plain | FAST | 2,048 | 5 | 10 | `next` |\n")
        self.assertEqual(violations(TABLE + rows), set())

    def test_empty_or_unparseable_table_refused(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            violations("")
        with self.assertRaisesRegex(ValueError, "invalid throughput"):
            violations(RECEIPTS.replace("8K 755", "8K missing", 1))


if __name__ == "__main__":
    unittest.main()
