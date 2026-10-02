#!/usr/bin/env python3
"""Dependency-free tests for the MoE routing qualification runner."""

from __future__ import annotations

import copy
import importlib.util
import math
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("qualify.py")
SPEC = importlib.util.spec_from_file_location("moe_routing_qualify", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
qualify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualify)


def valid_result() -> dict[str, object]:
    return {
        "device": "AMD Radeon Graphics",
        "gfx_arch": "gfx1151",
        "fixture": qualify.FIXTURE_ID,
        "warp_size": 32,
        "rows": 1,
        "experts": 512,
        "top_k": 10,
        "iterations": 500,
        "normalize_selected": True,
        "exact_indices": True,
        "average_us": 12.5,
        "max_weight_error": 1e-7,
        "kernel_local_bytes": 0,
        "kernel_shared_bytes": 4096,
        "kernel_registers": 32,
        "max_threads_per_block": 1024,
    }


def valid_report() -> dict[str, object]:
    return {
        "schema": "strix.moe-routing-qualification.v1",
        "device": "AMD Radeon Graphics",
        "gfx_arch": "gfx1151",
        "fixture": qualify.FIXTURE_ID,
        "experts": 512,
        "top_k": 10,
        "iterations_per_run": 500,
        "repetitions": 3,
        "rows": [
            {
                "row_class": "C1",
                "rows": 1,
                "serial": {
                    "median_us": 10.0,
                    "samples_us": [10.0, 10.1, 9.9],
                    "resources": {},
                },
                "hierarchical": {
                    "median_us": 5.0,
                    "samples_us": [5.0, 5.1, 4.9],
                    "resources": {},
                },
                "speedup_serial_over_hierarchical": 2.0,
                "winner": "hierarchical",
            }
        ],
        "all_exact": True,
    }


class QualificationValidationTests(unittest.TestCase):
    def test_valid_result_is_accepted(self) -> None:
        qualify.validate_result(
            valid_result(), "serial", 1, 512, 10, 500, 5e-6, True
        )

    def test_nan_timing_is_rejected(self) -> None:
        result = valid_result()
        result["average_us"] = math.nan
        with self.assertRaises(qualify.QualificationError):
            qualify.validate_result(result, "serial", 1, 512, 10, 500, 5e-6, True)

    def test_infinite_weight_error_is_rejected(self) -> None:
        result = valid_result()
        result["max_weight_error"] = math.inf
        with self.assertRaises(qualify.QualificationError):
            qualify.validate_result(result, "serial", 1, 512, 10, 500, 5e-6, True)

    def test_invalid_resource_metadata_is_rejected(self) -> None:
        result = valid_result()
        result["kernel_registers"] = -1
        with self.assertRaises(qualify.QualificationError):
            qualify.validate_result(result, "serial", 1, 512, 10, 500, 5e-6, True)

    def test_empty_sample_set_is_rejected(self) -> None:
        with self.assertRaises(qualify.QualificationError):
            qualify.stable_metadata([], "serial")

    def test_metadata_drift_is_rejected(self) -> None:
        first = valid_result()
        second = copy.deepcopy(first)
        second["device"] = "different-device"
        with self.assertRaises(qualify.QualificationError):
            qualify.stable_metadata([first, second], "serial")

    def test_nonfinite_repeated_timing_is_rejected(self) -> None:
        first = valid_result()
        second = copy.deepcopy(first)
        second["average_us"] = math.inf
        with self.assertRaises(qualify.QualificationError):
            qualify.summarize_samples([first, second], "serial")

    def test_pair_device_mismatch_is_rejected(self) -> None:
        serial = valid_result()
        hierarchical = copy.deepcopy(serial)
        hierarchical["device"] = "different-device"
        with self.assertRaises(qualify.QualificationError):
            qualify.compare_pair(serial, hierarchical, 1)

    def test_valid_report_is_accepted(self) -> None:
        qualify.validate_report(valid_report())

    def test_duplicate_report_row_is_rejected(self) -> None:
        report = valid_report()
        report["rows"].append(copy.deepcopy(report["rows"][0]))
        with self.assertRaises(qualify.QualificationError):
            qualify.validate_report(report)

    def test_inconsistent_report_speedup_is_rejected(self) -> None:
        report = valid_report()
        report["rows"][0]["speedup_serial_over_hierarchical"] = 1.5
        with self.assertRaises(qualify.QualificationError):
            qualify.validate_report(report)

    def test_inconsistent_report_winner_is_rejected(self) -> None:
        report = valid_report()
        report["rows"][0]["winner"] = "serial"
        with self.assertRaises(qualify.QualificationError):
            qualify.validate_report(report)


if __name__ == "__main__":
    unittest.main()
