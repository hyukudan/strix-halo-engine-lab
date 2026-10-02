#!/usr/bin/env python3
"""Qualify the target-dimension Q8_0 direct/grouped MoE reuse candidate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

FIXTURE_ID = "q8-target-dims-overlap-v1"
SCHEMA_ID = "strix.moe-q8-reuse.v1"
DEFAULT_ROWS = (1, 2, 4, 8)
EXPECTED_HIDDEN = 2560
EXPECTED_INTERMEDIATE = 640


class QualificationError(RuntimeError):
    """Raised when a Q8 MoE result cannot be accepted as comparable evidence."""


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def non_negative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
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


def executable_path(build_dir: Path) -> Path:
    direct = build_dir / "strix-moe-q8-0"
    if direct.is_file():
        return direct
    windows = direct.with_suffix(".exe")
    if windows.is_file():
        return windows
    raise QualificationError(f"candidate executable not found: {direct}")


def parse_json(stdout: str) -> dict[str, Any]:
    begin = stdout.find("{")
    end = stdout.rfind("}")
    if begin < 0 or end < begin:
        raise QualificationError("candidate did not emit a JSON object")
    try:
        value = json.loads(stdout[begin : end + 1])
    except json.JSONDecodeError as exc:
        raise QualificationError(f"candidate emitted invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise QualificationError("candidate JSON root must be an object")
    return value


def run_candidate(
    executable: Path,
    rows: int,
    experts: int,
    top_k: int,
    iterations: int,
) -> dict[str, Any]:
    process = subprocess.run(
        [
            str(executable),
            "--rows",
            str(rows),
            "--experts",
            str(experts),
            "--top-k",
            str(top_k),
            "--iterations",
            str(iterations),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise QualificationError(
            f"C{rows} candidate failed with exit code {process.returncode}: {detail}"
        )
    return parse_json(process.stdout)


def require_equal(result: dict[str, Any], key: str, expected: Any, rows: int) -> None:
    actual = result.get(key)
    if actual != expected:
        raise QualificationError(
            f"C{rows} reported {key}={actual!r}; expected {expected!r}"
        )


def numeric(result: dict[str, Any], key: str, rows: int) -> float:
    try:
        value = float(result[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise QualificationError(f"C{rows} omitted or malformed {key}") from exc
    if not math.isfinite(value):
        raise QualificationError(f"C{rows} reported non-finite {key}")
    return value


def validate_result(
    result: dict[str, Any],
    rows: int,
    experts: int,
    top_k: int,
    iterations: int,
    max_reference_error: float,
    max_path_error: float,
    require_gfx1151: bool,
) -> None:
    require_equal(result, "schema", SCHEMA_ID, rows)
    require_equal(result, "fixture", FIXTURE_ID, rows)
    require_equal(result, "hidden", EXPECTED_HIDDEN, rows)
    require_equal(result, "intermediate", EXPECTED_INTERMEDIATE, rows)
    require_equal(result, "rows", rows, rows)
    require_equal(result, "experts", experts, rows)
    require_equal(result, "top_k", top_k, rows)
    require_equal(result, "iterations", iterations, rows)
    require_equal(result, "assignments", rows * top_k, rows)
    require_equal(result, "direct_batches", rows * top_k, rows)

    try:
        warp_size = int(result["warp_size"])
        grouped_batches = int(result["grouped_batches"])
        direct_bytes = int(result["direct_modeled_weight_bytes"])
        grouped_bytes = int(result["grouped_modeled_weight_bytes"])
        saved_bytes = int(result["modeled_weight_bytes_saved"])
    except (KeyError, TypeError, ValueError) as exc:
        raise QualificationError(f"C{rows} omitted required integer metadata") from exc

    if warp_size != 32:
        raise QualificationError(f"C{rows} reported warp_size={warp_size}; Wave32 required")
    gfx_arch = str(result.get("gfx_arch", ""))
    if require_gfx1151 and gfx_arch.split(":", 1)[0] != "gfx1151":
        raise QualificationError(
            f"C{rows} reported gfx_arch={gfx_arch!r}; gfx1151 required"
        )
    if grouped_batches <= 0 or grouped_batches > rows * top_k:
        raise QualificationError(f"C{rows} reported invalid grouped_batches={grouped_batches}")
    if direct_bytes <= 0 or grouped_bytes <= 0:
        raise QualificationError(f"C{rows} reported non-positive modeled weight traffic")
    if grouped_bytes > direct_bytes:
        raise QualificationError(
            f"C{rows} grouped modeled weight traffic exceeds direct traffic"
        )
    if saved_bytes != direct_bytes - grouped_bytes:
        raise QualificationError(f"C{rows} modeled weight byte accounting is inconsistent")

    direct_ms = numeric(result, "direct_ms", rows)
    grouped_ms = numeric(result, "grouped_ms", rows)
    reported_speedup = numeric(result, "speedup", rows)
    direct_error = numeric(result, "max_direct_reference_error", rows)
    grouped_error = numeric(result, "max_grouped_reference_error", rows)
    path_error = numeric(result, "max_direct_grouped_error", rows)

    if direct_ms <= 0.0 or grouped_ms <= 0.0:
        raise QualificationError(f"C{rows} reported non-positive timing")
    expected_speedup = direct_ms / grouped_ms
    if abs(reported_speedup - expected_speedup) > max(1e-6, expected_speedup * 1e-5):
        raise QualificationError(f"C{rows} reported inconsistent speedup")
    if direct_error < 0.0 or direct_error > max_reference_error:
        raise QualificationError(
            f"C{rows} direct/reference error {direct_error} exceeds {max_reference_error}"
        )
    if grouped_error < 0.0 or grouped_error > max_reference_error:
        raise QualificationError(
            f"C{rows} grouped/reference error {grouped_error} exceeds {max_reference_error}"
        )
    if path_error < 0.0 or path_error > max_path_error:
        raise QualificationError(
            f"C{rows} direct/grouped error {path_error} exceeds {max_path_error}"
        )


def summarize(samples: list[dict[str, Any]], rows: int) -> dict[str, Any]:
    first = samples[0]
    stable_keys = (
        "device",
        "gfx_arch",
        "warp_size",
        "fixture",
        "hidden",
        "intermediate",
        "experts",
        "top_k",
        "assignments",
        "direct_batches",
        "grouped_batches",
        "expert_weight_bytes",
        "direct_modeled_weight_bytes",
        "grouped_modeled_weight_bytes",
        "modeled_weight_bytes_saved",
    )
    for sample in samples[1:]:
        for key in stable_keys:
            if sample.get(key) != first.get(key):
                raise QualificationError(
                    f"C{rows} changed {key} across repetitions: "
                    f"{first.get(key)!r} -> {sample.get(key)!r}"
                )

    direct = [float(sample["direct_ms"]) for sample in samples]
    grouped = [float(sample["grouped_ms"]) for sample in samples]
    direct_median = statistics.median(direct)
    grouped_median = statistics.median(grouped)
    speedup = direct_median / grouped_median

    if grouped_median < direct_median:
        winner = "grouped"
    elif direct_median < grouped_median:
        winner = "direct"
    else:
        winner = "tie"

    return {
        "row_class": f"C{rows}",
        "rows": rows,
        "direct_ms_median": direct_median,
        "grouped_ms_median": grouped_median,
        "speedup_direct_over_grouped": speedup,
        "winner": winner,
        "direct_ms_samples": direct,
        "grouped_ms_samples": grouped,
        "direct_batches": int(first["direct_batches"]),
        "grouped_batches": int(first["grouped_batches"]),
        "direct_modeled_weight_bytes": int(first["direct_modeled_weight_bytes"]),
        "grouped_modeled_weight_bytes": int(first["grouped_modeled_weight_bytes"]),
        "modeled_weight_bytes_saved": int(first["modeled_weight_bytes_saved"]),
        "max_direct_reference_error": max(
            float(sample["max_direct_reference_error"]) for sample in samples
        ),
        "max_grouped_reference_error": max(
            float(sample["max_grouped_reference_error"]) for sample in samples
        ),
        "max_direct_grouped_error": max(
            float(sample["max_direct_grouped_error"]) for sample in samples
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the target-dimension Q8_0 direct/grouped MoE candidate over "
            "C1/C2/C4/C8 and reject incomparable or numerically invalid results."
        )
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build/moe-q8-0"),
        help="directory containing strix-moe-q8-0",
    )
    parser.add_argument("--rows", type=parse_rows, default=DEFAULT_ROWS)
    parser.add_argument("--experts", type=positive_int, default=16)
    parser.add_argument("--top-k", type=positive_int, default=10)
    parser.add_argument("--iterations", type=positive_int, default=20)
    parser.add_argument(
        "--repetitions",
        type=positive_int,
        default=1,
        help=(
            "full process repetitions per row class; keep low because every run "
            "also executes the independent CPU packed-weight reference"
        ),
    )
    parser.add_argument(
        "--max-reference-error",
        type=non_negative_float,
        default=5e-3,
    )
    parser.add_argument(
        "--max-path-error",
        type=non_negative_float,
        default=5e-3,
    )
    parser.add_argument(
        "--allow-non-target",
        action="store_true",
        help="do not require the reported architecture family to be gfx1151",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.experts > 512:
        raise QualificationError("--experts must not exceed 512")
    if args.top_k > min(args.experts, 16):
        raise QualificationError("--top-k must not exceed experts or 16")

    executable = executable_path(args.build_dir)
    row_results: list[dict[str, Any]] = []
    device: str | None = None
    gfx_arch: str | None = None

    for rows in args.rows:
        samples: list[dict[str, Any]] = []
        for _ in range(args.repetitions):
            result = run_candidate(
                executable,
                rows,
                args.experts,
                args.top_k,
                args.iterations,
            )
            validate_result(
                result,
                rows,
                args.experts,
                args.top_k,
                args.iterations,
                args.max_reference_error,
                args.max_path_error,
                not args.allow_non_target,
            )
            samples.append(result)

        current_device = str(samples[0].get("device", ""))
        current_arch = str(samples[0].get("gfx_arch", ""))
        if device is None:
            device = current_device
            gfx_arch = current_arch
        elif current_device != device or current_arch != gfx_arch:
            raise QualificationError(
                "device or architecture changed between row-class measurements"
            )
        row_results.append(summarize(samples, rows))

    report = {
        "schema": "strix.moe-q8-qualification.v1",
        "device": device,
        "gfx_arch": gfx_arch,
        "fixture": FIXTURE_ID,
        "hidden": EXPECTED_HIDDEN,
        "intermediate": EXPECTED_INTERMEDIATE,
        "experts": args.experts,
        "top_k": args.top_k,
        "iterations_per_run": args.iterations,
        "repetitions": args.repetitions,
        "rows": row_results,
        "all_correct": True,
    }
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
