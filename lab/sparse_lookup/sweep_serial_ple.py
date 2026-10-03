#!/usr/bin/env python3
"""Sweep planned PLE chunk/inflight policies against one mapped baseline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
QUALIFY_PATH = HERE / "qualify_serial_ple.py"
SPEC = importlib.util.spec_from_file_location("qualify_serial_ple", QUALIFY_PATH)
assert SPEC is not None and SPEC.loader is not None
qualify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualify)

REPORT_SCHEMA = "strix.ple-policy-sweep.v1"


class SweepError(RuntimeError):
    pass


def positive_csv(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in text.split(",") if part)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("values must be comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("values must be positive")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must not contain duplicates")
    return values


def run_policy(
    executable: Path,
    model: Path,
    reference: Path,
    repetitions: int,
    rows_per_io: int,
    inflight_ios: int,
    mapped_samples: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mapped: list[dict[str, Any]] = [] if mapped_samples is None else list(mapped_samples)
    planned: list[dict[str, Any]] = []

    for repetition in range(repetitions):
        if mapped_samples is None:
            mapped_result = qualify.run_once(
                executable,
                model,
                reference,
                False,
                rows_per_io,
                inflight_ios,
            )
            qualify.validate_result(mapped_result, "mapped")
            mapped.append(mapped_result)

        planned_result = qualify.run_once(
            executable,
            model,
            reference,
            True,
            rows_per_io,
            inflight_ios,
        )
        qualify.validate_result(planned_result, "planned")
        planned.append(planned_result)

        qualify.compare_identity(mapped[repetition], planned_result)

    return mapped, planned


def summarize_policy(
    mapped: list[dict[str, Any]],
    planned: list[dict[str, Any]],
    repetitions: int,
    rows_per_io: int,
    inflight_ios: int,
) -> dict[str, Any]:
    report = qualify.build_report(
        mapped,
        planned,
        repetitions,
        rows_per_io,
        inflight_ios,
    )
    speedup = float(report["speedup_mapped_over_planned"])
    if not math.isfinite(speedup) or speedup <= 0.0:
        raise SweepError("policy produced invalid speedup")
    return {
        "max_rows_per_io": rows_per_io,
        "max_inflight_ios": inflight_ios,
        "planned_lookup_identity": report["planned"]["ple_lookup_identity"],
        "planned_decode_ms_per_token_median": report["planned"][
            "decode_ms_per_token_median"
        ],
        "mapped_decode_ms_per_token_median": report["mapped"][
            "decode_ms_per_token_median"
        ],
        "speedup_mapped_over_planned": speedup,
    }


def rank_policies(policies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not policies:
        raise SweepError("no policies to rank")
    ordered = sorted(
        policies,
        key=lambda row: (
            float(row["planned_decode_ms_per_token_median"]),
            int(row["max_inflight_ios"]),
            int(row["max_rows_per_io"]),
        ),
    )
    return [
        {
            **policy,
            "rank": index + 1,
        }
        for index, policy in enumerate(ordered)
    ]


def build_sweep_report(
    mapped: list[dict[str, Any]],
    policies: list[dict[str, Any]],
    rows_values: tuple[int, ...],
    inflight_values: tuple[int, ...],
    repetitions: int,
) -> dict[str, Any]:
    ranked = rank_policies(policies)
    baseline = qualify.summarize(mapped, "mapped")
    first = mapped[0]
    return {
        "schema": REPORT_SCHEMA,
        "model_content_identity": first["model_content_identity"],
        "execution_identity": first["execution_identity"],
        "reference_id": first["replay"]["reference_id"],
        "repetitions": repetitions,
        "grid": {
            "max_rows_per_io": list(rows_values),
            "max_inflight_ios": list(inflight_values),
        },
        "mapped_baseline": baseline,
        "policies": ranked,
        "best_policy": {
            "max_rows_per_io": ranked[0]["max_rows_per_io"],
            "max_inflight_ios": ranked[0]["max_inflight_ios"],
            "planned_decode_ms_per_token_median": ranked[0][
                "planned_decode_ms_per_token_median"
            ],
            "speedup_mapped_over_planned": ranked[0][
                "speedup_mapped_over_planned"
            ],
        },
        "all_replay_passed": True,
        "argmax_equivalent": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep planned PLE max_rows_per_io/max_inflight_ios against one "
            "mapped replay baseline."
        )
    )
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--repetitions", type=qualify.positive_int, default=3)
    parser.add_argument("--rows-per-io", type=positive_csv, default=(1, 2, 4, 8, 16))
    parser.add_argument("--inflight-ios", type=positive_csv, default=(1, 2, 4, 8))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    executable = qualify.executable_path(args.runner)

    # Baseline is intentionally sampled once per repetition and reused across
    # the grid so policy comparisons share identical mapped evidence.
    mapped: list[dict[str, Any]] = []
    for _ in range(args.repetitions):
        result = qualify.run_once(
            executable,
            args.model,
            args.reference,
            False,
            args.rows_per_io[0],
            args.inflight_ios[0],
        )
        qualify.validate_result(result, "mapped")
        mapped.append(result)

    policies: list[dict[str, Any]] = []
    for rows_per_io in args.rows_per_io:
        for inflight_ios in args.inflight_ios:
            _, planned = run_policy(
                executable,
                args.model,
                args.reference,
                args.repetitions,
                rows_per_io,
                inflight_ios,
                mapped_samples=mapped,
            )
            policies.append(
                summarize_policy(
                    mapped,
                    planned,
                    args.repetitions,
                    rows_per_io,
                    inflight_ios,
                )
            )

    report = build_sweep_report(
        mapped,
        policies,
        args.rows_per_io,
        args.inflight_ios,
        args.repetitions,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SweepError, qualify.QualificationError) as error:
        print(f"PLE policy sweep failed: {error}", file=sys.stderr)
        raise SystemExit(2)
