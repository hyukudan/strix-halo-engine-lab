#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "sweep_serial_ple",
    HERE / "sweep_serial_ple.py",
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def mapped_fixture(elapsed: float) -> dict:
    return {
        "model_content_identity": "cd" * 32,
        "execution_identity": 123,
        "context_tokens": 128,
        "ple_lookup": "mapped",
        "ple_lookup_identity": 10,
        "tokens": [
            {"input_token": 1, "argmax_token": 9, "elapsed_ms": elapsed},
            {"input_token": 2, "argmax_token": 8, "elapsed_ms": elapsed},
        ],
        "decode_elapsed_ms_total": elapsed * 2,
        "decode_elapsed_ms_per_token": elapsed,
        "committed_tokens": 2,
        "next_position": 2,
        "replay": {"reference_id": 77, "passed": True},
    }


mapped = [mapped_fixture(10.0), mapped_fixture(12.0)]
policies = [
    {
        "max_rows_per_io": 4,
        "max_inflight_ios": 4,
        "planned_lookup_identity": 44,
        "planned_decode_ms_per_token_median": 8.0,
        "mapped_decode_ms_per_token_median": 11.0,
        "speedup_mapped_over_planned": 11.0 / 8.0,
    },
    {
        "max_rows_per_io": 2,
        "max_inflight_ios": 2,
        "planned_lookup_identity": 22,
        "planned_decode_ms_per_token_median": 9.0,
        "mapped_decode_ms_per_token_median": 11.0,
        "speedup_mapped_over_planned": 11.0 / 9.0,
    },
]

ranked = module.rank_policies(policies)
assert ranked[0]["max_rows_per_io"] == 4
assert ranked[0]["rank"] == 1
assert ranked[1]["rank"] == 2

report = module.build_sweep_report(
    mapped,
    policies,
    (2, 4),
    (2, 4),
    2,
)
assert report["schema"] == module.REPORT_SCHEMA
assert report["best_policy"]["max_rows_per_io"] == 4
assert report["best_policy"]["max_inflight_ios"] == 4
assert math.isclose(
    report["mapped_baseline"]["decode_ms_per_token_median"],
    11.0,
)

tie = [
    {
        **policies[0],
        "max_rows_per_io": 8,
        "max_inflight_ios": 4,
        "planned_decode_ms_per_token_median": 8.0,
    },
    {
        **policies[0],
        "max_rows_per_io": 4,
        "max_inflight_ios": 2,
        "planned_decode_ms_per_token_median": 8.0,
    },
]
ranked_tie = module.rank_policies(tie)
assert ranked_tie[0]["max_inflight_ios"] == 2
assert ranked_tie[0]["max_rows_per_io"] == 4

print("PLE policy sweep tests passed")
