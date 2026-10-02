#!/usr/bin/env python3
"""Dependency-free tests for the Q8_0 MoE qualification runner."""

from __future__ import annotations

import copy
import importlib.util
import math
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("qualify.py")
SPEC = importlib.util.spec_from_file_location("moe_q8_qualify", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
qualify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualify)


def valid_result() -> dict[str, object]:
    return {
        "schema": qualify.SCHEMA_ID,
        "device": "AMD Radeon Graphics",
        "gfx_arch": "gfx1151",
        "warp_size": 32,
        "fixture": qualify.FIXTURE_ID,
        "hidden": qualify.EXPECTED_HIDDEN,
        "intermediate": qualify.EXPECTED_INTERMEDIATE,
        "rows": 2,
        "experts": 16,
        "top_k": 10,
        "iterations": 20,
        "assignments": 20,
        "direct_batches": 20,
        "grouped_batches": 12,
        "expert_weight_bytes": 1024,
        "direct_modeled_weight_bytes": 20480,
        "grouped_modeled_weight_bytes": 12288,
        "modeled_weight_bytes_saved": 8192,
        "direct_ms": 2.0,
        "grouped_ms": 1.0,
        "speedup": 2.0,
        "max_direct_reference_error": 1e-4,
        "max_grouped_reference_error": 2e-4,
        "max_direct_grouped_error": 1e-4,
    }


def valid_report() -> dict[str, object]:
    return {
        "schema": qualify.REPORT_SCHEMA_ID,
        "device": "AMD Radeon Graphics",
        "gfx_arch": "gfx1151",
        "fixture": qualify.FIXTURE_ID,
        "hidden": qualify.EXPECTED_HIDDEN,
        "intermediate": qualify.EXPECTED_INTERMEDIATE,
        "experts": 16,
        "top_k": 10,
        "iterations_per_run": 20,
        "repetitions": 2,
        "rows": [
            {
                "row_class": "C2",
                "rows": 2,
                "direct_ms_median": 2.0,
                "grouped_ms_median": 1.0,
                "speedup_direct_over_grouped": 2.0,
                "winner": "grouped",
                "direct_ms_samples": [2.1, 1.9],
                "grouped_ms_samples": [1.1, 0.9],
                "direct_batches": 20,
                "grouped_batches": 12,
                "direct_modeled_weight_bytes": 20480,
                "grouped_modeled_weight_bytes": 12288,
                "modeled_weight_bytes_saved": 8192,
                "max_direct_reference_error": 1e-4,
                "max_grouped_reference_error": 2e-4,
                "max_direct_grouped_error": 1e-4,
            }
        ],
        "all_correct": True,
    }


class QualificationValidationTests(unittest.TestCase):
    def test_valid_result_is_accepted(self) -> None:
        qualify.validate_result(valid_result(), 2, 16, 10, 20, 5e-3, 5e-3, True)

    def test_nonfinite_timing_is_rejected(self) -> None:
        result = valid_result()
        result["direct_ms"] = math.nan
        with self.assertRaises(qualify.QualificationError):
            qualify.validate_result(result, 2, 16, 10, 20, 5e-3, 5e-3, True)

    def test_inconsistent_speedup_is_rejected(self) -> None:
        result = valid_result()
        result["speedup"] = 1.5
        with self.assertRaises(qualify.QualificationError):
            qualify.validate_result(result, 2, 16, 10, 20, 5e-3, 5e-3, True)

    def test_inconsistent_weight_bytes_are_rejected(self) -> None:
        result = valid_result()
        result["modeled_weight_bytes_saved"] = 1
        with self.assertRaises(qualify.QualificationError):
            qualify.validate_result(result, 2, 16, 10, 20, 5e-3, 5e-3, True)

    def test_empty_samples_are_rejected(self) -> None:
        with self.assertRaises(qualify.QualificationError):
            qualify.summarize([], 2)

    def test_metadata_drift_is_rejected(self) -> None:
        first = valid_result()
        second = copy.deepcopy(first)
        second["grouped_batches"] = 11
        with self.assertRaises(qualify.QualificationError):
            qualify.summarize([first, second], 2)

    def test_nonfinite_repeated_timing_is_rejected(self) -> None:
        first = valid_result()
        second = copy.deepcopy(first)
        second["grouped_ms"] = math.inf
        with self.assertRaises(qualify.QualificationError):
            qualify.summarize([first, second], 2)

    def test_valid_report_is_accepted(self) -> None:
        qualify.validate_report(valid_report())

    def test_duplicate_report_row_is_rejected(self) -> None:
        report = valid_report()
        report["rows"].append(copy.deepcopy(report["rows"][0]))
        with self.assertRaises(qualify.QualificationError):
            qualify.validate_report(report)

    def test_report_speedup_mismatch_is_rejected(self) -> None:
        report = valid_report()
        report["rows"][0]["speedup_direct_over_grouped"] = 1.5
        with self.assertRaises(qualify.QualificationError):
            qualify.validate_report(report)

    def test_report_winner_mismatch_is_rejected(self) -> None:
        report = valid_report()
        report["rows"][0]["winner"] = "direct"
        with self.assertRaises(qualify.QualificationError):
            qualify.validate_report(report)

    def test_report_byte_accounting_mismatch_is_rejected(self) -> None:
        report = valid_report()
        report["rows"][0]["modeled_weight_bytes_saved"] = 0
        with self.assertRaises(qualify.QualificationError):
            qualify.validate_report(report)

    def test_report_all_correct_must_be_true(self) -> None:
        report = valid_report()
        report["all_correct"] = False
        with self.assertRaises(qualify.QualificationError):
            qualify.validate_report(report)


if __name__ == "__main__":
    unittest.main()
