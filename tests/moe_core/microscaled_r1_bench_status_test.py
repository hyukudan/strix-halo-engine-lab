"""Check the HIP benchmark's launch status handling without a HIP compiler or device."""

from pathlib import Path
import unittest


SOURCE = Path(__file__).with_name("microscaled_r1_hip.hip").read_text(encoding="utf-8")
LAUNCH_PART = SOURCE.split("launch_part(", 1)[1].split("// Read floor", 1)[0]
BENCH = SOURCE.split("void bench(", 1)[1].split("}  // namespace", 1)[0]


class BenchStatusTest(unittest.TestCase):
    def test_whole_launch_status_is_propagated_and_checked(self):
        # The helper can consume the HIP last error, so the caller must check its return value.
        self.assertRegex(LAUNCH_PART, r"if \(part == 0U\)\s*(?:\{\s*)?return r1::launch<")
        warm, timed = BENCH.split("CHECK_HIP(hipEventRecord(start, nullptr));", 1)
        timed = timed.split("CHECK_HIP(hipEventRecord(stop, nullptr));", 1)[0]
        self.assertRegex(warm, r"CHECK_HIP\(\s*launch_part<F>\([^;]*,\s*0U\)\s*\);")
        self.assertRegex(timed, r"CHECK_HIP\(\s*launch_part<F>\([^;]*,\s*part\)\s*\);")

    def test_raw_part_launch_status_is_returned_immediately(self):
        self.assertRegex(
            LAUNCH_PART,
            r"if \(rows == 1U\) d::gate_up<[^;]+;\s*else d::gate_up_rows<[^;]+;\s*}\s*else\s*{[^{}]*?"
            r"if \(rows == 1U\) d::down_combine<[^;]+;\s*else d::down_combine_rows<[^;]+;\s*}"
            r"\s*return hipGetLastError\(\);",
        )

    def test_read_floor_launch_status_is_checked_immediately(self):
        self.assertRegex(BENCH, r"read_floor<<<[^;]+;\s*CHECK_HIP\(hipGetLastError\(\)\);")


if __name__ == "__main__":
    unittest.main()
