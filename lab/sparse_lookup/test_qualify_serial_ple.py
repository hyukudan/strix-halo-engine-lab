#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("qualify_serial_ple.py")
SPEC = importlib.util.spec_from_file_location("qualify_serial_ple", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def fixture(mode: str, elapsed: float) -> dict:
    lookup = "planned-pread" if mode == "planned" else "mapped"
    identity = 22 if mode == "planned" else 11
    return {
        "model_content_identity": "ab" * 32,
        "execution_identity": 100,
        "runtime_compatibility_identity": 200 if mode == "mapped" else 201,
        "context_tokens": 128,
        "ple_lookup": lookup,
        "ple_lookup_identity": identity,
        "tokens": [
            {
                "input_token": 1,
                "position": 0,
                "argmax_token": 7,
                "argmax_logit": 1.0,
                "elapsed_ms": elapsed,
                "top5": [],
            },
            {
                "input_token": 2,
                "position": 1,
                "argmax_token": 8,
                "argmax_logit": 2.0,
                "elapsed_ms": elapsed,
                "top5": [],
            },
        ],
        "decode_elapsed_ms_total": elapsed * 2,
        "decode_elapsed_ms_per_token": elapsed,
        "committed_tokens": 2,
        "next_position": 2,
        "replay": {
            "reference_id": 999,
            "logit_values_compared": 4,
            "max_logit_abs_error": 0.0,
            "argmax_match": True,
            "passed": True,
        },
    }


mapped = fixture("mapped", 10.0)
planned = fixture("planned", 8.0)
module.validate_result(mapped, "mapped")
module.validate_result(planned, "planned")
module.compare_identity(mapped, planned)

report = module.build_report(
    [mapped, fixture("mapped", 12.0)],
    [planned, fixture("planned", 9.0)],
    2,
    4,
    4,
)
assert report["schema"] == module.REPORT_SCHEMA
assert report["argmax_equivalent"] is True
assert report["all_replay_passed"] is True
assert math.isclose(
    report["mapped"]["decode_ms_per_token_median"],
    11.0,
)
assert math.isclose(
    report["planned"]["decode_ms_per_token_median"],
    8.5,
)
assert math.isclose(
    report["speedup_mapped_over_planned"],
    11.0 / 8.5,
)

bad = fixture("planned", 8.0)
bad["tokens"][1]["argmax_token"] = 9
try:
    module.compare_identity(mapped, bad)
except module.QualificationError:
    pass
else:
    raise AssertionError("argmax mismatch must reject qualification")

bad_replay = fixture("mapped", 10.0)
bad_replay["replay"]["passed"] = False
try:
    module.validate_result(bad_replay, "mapped")
except module.QualificationError:
    pass
else:
    raise AssertionError("failed replay must reject qualification")

print("PLE serial qualification tests passed")
