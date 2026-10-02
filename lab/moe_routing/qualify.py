#!/usr/bin/env python3
"""Qualify serial and hierarchical MoE routing candidates on one target GPU."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

FIXTURE_ID = "deterministic-cutoff-tie-v1"
DEFAULT_ROWS = (1, 2, 4, 8)
RESOURCE_KEYS = (
    "kernel_local_bytes",
    "kernel_shared_bytes",
    "kernel_registers",
    "max_threads_per_block",
)


class QualificationError(RuntimeError):
    """Raised when a candidate result cannot be compared safely."""


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def parse_rows(text: str) -> tuple[int, ...]:
    try:
        rows = tuple(int(item) for item in text.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rows must be comma-separated integers") from exc
    if not rows or any(row not in DEFAULT_ROWS for row in rows):
        raise argparse.ArgumentTypeError("rows must be chosen from 1,2,4,8")
    if len(set(rows)) != len(rows):
        raise argparse.ArgumentTypeError("rows must not contain duplicates")
    return rows


def candidate_path(build_dir: Path, name: str) -> Path:
    direct = build_dir / name
    if direct.is_file():
        return direct
    windows = direct.with_suffix(".exe")
    if windows.is_file():
        return windows
    raise QualificationError(f"candidate executable not found: {direct}")


def parse_candidate_json(stdout: str, candidate: str) -> dict[str, Any]:
    begin = stdout.find("{")
    end = stdout.rfind("}")
    if begin < 0 or end < begin:
        raise QualificationError(f"{candidate} did not emit a JSON object")
    try:
        value = json.loads(stdout[begin : end + 1])
    except json.JSONDecodeError as exc:
        raise QualificationError(f"{candidate} emitted invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise QualificationError(f"{candidate} JSON root must be an object")
    return value


def run_candidate(
    executable: Path,
    rows: int,
    experts: int,
    top_k: int,
    iterations: int,
) -> dict[str, Any]:
    command = [
        str(executable),
        "--rows",
        str(rows),
        "--experts",
        str(experts),
        "--top-k",
        str(top_k),
        "--iterations",
        str(iterations),
    ]
    process = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise QualificationError(
            f"{executable.name} failed with exit code {process.returncode}: {detail}"
        )
    return parse_candidate_json(process.stdout, executable.name)


def require_equal(result: dict[str, Any], key: str, expected: Any, candidate: str) -> None:
    actual = result.get(key)
    if actual != expected:
        raise QualificationError(
            f"{candidate} reported {key}={actual!r}; expected {expected!r}"
        )


def validate_result(
    result: dict[str, Any],
    candidate: str,
    rows: int,
    experts: int,
    top_k: int,
    iterations: int,
    max_weight_error: float,
    require_gfx1151: bool,
) -> None:
    require_equal(result, "rows", rows, candidate)
    require_equal(result, "experts", experts, candidate)
    require_equal(result, "top_k", top_k, candidate)
    require_equal(result, "iterations", iterations, candidate)
    require_equal(result, "fixture", FIXTURE_ID, candidate)
    require_equal(result, "normalize_selected", True, candidate)
    require_equal(result, "exact_indices", True, candidate)

    try:
        average_us = float(result["average_us"])
        weight_error = float(result["max_weight_error"])
        warp_size = int(result["warp_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise QualificationError(
            f"{candidate} omitted or malformed timing/correctness metadata"
        ) from exc

    if not math.isfinite(average_us) or average_us <= 0.0:
        raise QualificationError(
            f"{candidate} reported invalid average_us={average_us!r}"
        )
    if (
        not math.isfinite(weight_error)
        or weight_error < 0.0
        or weight_error > max_weight_error
    ):
        raise QualificationError(
            f"{candidate} max_weight_error={weight_error!r} exceeds "
            f"finite range [0, {max_weight_error}]"
        )
    if warp_size != 32:
        raise QualificationError(f"{candidate} reported warp_size={warp_size}; Wave32 required")

    gfx_arch = str(result.get("gfx_arch", ""))
    if require_gfx1151 and gfx_arch.split(":", 1)[0] != "gfx1151":
        raise QualificationError(
            f"{candidate} reported gfx_arch={gfx_arch!r}; gfx1151 required"
        )

    for key in RESOURCE_KEYS:
        if key not in result:
            raise QualificationError(f"{candidate} omitted resource field {key}")
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise QualificationError(
                f"{candidate} reported non-integer resource field {key}={value!r}"
            )
        minimum = 1 if key == "max_threads_per_block" else 0
        if value < minimum:
            raise QualificationError(
                f"{candidate} reported invalid resource field {key}={value!r}"
            )


def stable_metadata(samples: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    if not samples:
        raise QualificationError(f"{candidate} produced no samples")
    first = samples[0]
    stable_keys = ("device", "gfx_arch", "fixture", "warp_size", *RESOURCE_KEYS)
    for sample in samples[1:]:
        for key in stable_keys:
            if sample.get(key) != first.get(key):
                raise QualificationError(
                    f"{candidate} changed {key} across repetitions: "
                    f"{first.get(key)!r} -> {sample.get(key)!r}"
                )
    return {key: first.get(key) for key in stable_keys}


def summarize_samples(samples: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    metadata = stable_metadata(samples, candidate)
    try:
        timings = [float(sample["average_us"]) for sample in samples]
    except (KeyError, TypeError, ValueError) as exc:
        raise QualificationError(
            f"{candidate} omitted or malformed average_us in repeated samples"
        ) from exc
    if any(not math.isfinite(value) or value <= 0.0 for value in timings):
        raise QualificationError(
            f"{candidate} produced non-finite or non-positive timing samples"
        )
    return {
        "median_us": statistics.median(timings),
        "samples_us": timings,
        "resources": {key: metadata[key] for key in RESOURCE_KEYS},
    }


def compare_pair(
    serial: dict[str, Any],
    hierarchical: dict[str, Any],
    rows: int,
) -> None:
    for key in ("device", "gfx_arch", "fixture", "warp_size"):
        if serial.get(key) != hierarchical.get(key):
            raise QualificationError(
                f"C{rows}: candidates disagree on {key}: "
                f"{serial.get(key)!r} != {hierarchical.get(key)!r}"
            )


def validate_report(report: dict[str, Any]) -> None:
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise QualificationError("qualification report contains no row classes")

    seen_rows: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise QualificationError("qualification report row must be an object")
        try:
            row_count = int(row["rows"])
            row_class = str(row["row_class"])
            serial_us = float(row["serial"]["median_us"])
            hierarchical_us = float(row["hierarchical"]["median_us"])
            speedup = float(row["speedup_serial_over_hierarchical"])
            winner = str(row["winner"])
        except (KeyError, TypeError, ValueError) as exc:
            raise QualificationError(
                "qualification report contains malformed row summary"
            ) from exc

        if row_count in seen_rows:
            raise QualificationError(
                f"qualification report contains duplicate C{row_count}"
            )
        seen_rows.add(row_count)
        if row_class != f"C{row_count}":
            raise QualificationError(
                f"qualification report row label {row_class!r} does not match C{row_count}"
            )
        if (
            not math.isfinite(serial_us)
            or not math.isfinite(hierarchical_us)
            or serial_us <= 0.0
            or hierarchical_us <= 0.0
        ):
            raise QualificationError(
                f"C{row_count}: qualification report contains invalid median timing"
            )

        expected_speedup = serial_us / hierarchical_us
        if not math.isfinite(speedup) or not math.isclose(
            speedup, expected_speedup, rel_tol=1e-12, abs_tol=0.0
        ):
            raise QualificationError(
                f"C{row_count}: reported speedup is inconsistent with medians"
            )

        if hierarchical_us < serial_us:
            expected_winner = "hierarchical"
        elif serial_us < hierarchical_us:
            expected_winner = "serial"
        else:
            expected_winner = "tie"
        if winner != expected_winner:
            raise QualificationError(
                f"C{row_count}: reported winner {winner!r} should be "
                f"{expected_winner!r}"
            )

    if report.get("all_exact") is not True:
        raise QualificationError("qualification report must preserve all_exact=true")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the serial and Wave32 hierarchical MoE routing candidates "
            "on identical deterministic fixtures."
        )
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build/moe-routing"),
        help="directory containing the two routing executables",
    )
    parser.add_argument("--iterations", type=positive_int, default=500)
    parser.add_argument("--repetitions", type=positive_int, default=3)
    parser.add_argument("--experts", type=positive_int, default=512)
    parser.add_argument("--top-k", type=positive_int, default=10)
    parser.add_argument("--rows", type=parse_rows, default=DEFAULT_ROWS)
    parser.add_argument(
        "--max-weight-error",
        type=float,
        default=5e-6,
        help="maximum accepted absolute selected-weight error",
    )
    parser.add_argument(
        "--allow-non-target",
        action="store_true",
        help="do not require the reported architecture family to be gfx1151",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="also write the qualification JSON to this path",
    )
    args = parser.parse_args()

    if args.top_k > args.experts:
        raise QualificationError("--top-k cannot exceed --experts")
    if args.max_weight_error < 0.0:
        raise QualificationError("--max-weight-error must be non-negative")

    serial_exe = candidate_path(args.build_dir, "strix-moe-routing")
    hierarchical_exe = candidate_path(
        args.build_dir, "strix-moe-routing-hierarchical"
    )

    row_results: list[dict[str, Any]] = []
    device: str | None = None
    gfx_arch: str | None = None

    for rows in args.rows:
        samples: dict[str, list[dict[str, Any]]] = {
            "serial": [],
            "hierarchical": [],
        }
        for repetition in range(args.repetitions):
            order = (
                ("serial", serial_exe),
                ("hierarchical", hierarchical_exe),
            )
            if repetition % 2 == 1:
                order = tuple(reversed(order))
            for candidate, executable in order:
                result = run_candidate(
                    executable,
                    rows,
                    args.experts,
                    args.top_k,
                    args.iterations,
                )
                validate_result(
                    result,
                    candidate,
                    rows,
                    args.experts,
                    args.top_k,
                    args.iterations,
                    args.max_weight_error,
                    not args.allow_non_target,
                )
                samples[candidate].append(result)

        compare_pair(samples["serial"][0], samples["hierarchical"][0], rows)
        serial_summary = summarize_samples(samples["serial"], "serial")
        hierarchical_summary = summarize_samples(
            samples["hierarchical"], "hierarchical"
        )

        current_device = str(samples["serial"][0]["device"])
        current_arch = str(samples["serial"][0]["gfx_arch"])
        if device is None:
            device = current_device
            gfx_arch = current_arch
        elif device != current_device or gfx_arch != current_arch:
            raise QualificationError(
                "device or architecture changed between row-class measurements"
            )

        serial_us = float(serial_summary["median_us"])
        hierarchical_us = float(hierarchical_summary["median_us"])
        if hierarchical_us < serial_us:
            winner = "hierarchical"
        elif serial_us < hierarchical_us:
            winner = "serial"
        else:
            winner = "tie"

        row_results.append(
            {
                "row_class": f"C{rows}",
                "rows": rows,
                "serial": serial_summary,
                "hierarchical": hierarchical_summary,
                "speedup_serial_over_hierarchical": serial_us / hierarchical_us,
                "winner": winner,
            }
        )

    report = {
        "schema": "strix.moe-routing-qualification.v1",
        "device": device,
        "gfx_arch": gfx_arch,
        "fixture": FIXTURE_ID,
        "experts": args.experts,
        "top_k": args.top_k,
        "iterations_per_run": args.iterations,
        "repetitions": args.repetitions,
        "rows": row_results,
        "all_exact": True,
    }
    validate_report(report)
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QualificationError as error:
        print(f"qualification failed: {error}", file=sys.stderr)
        raise SystemExit(2)
