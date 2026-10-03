#!/usr/bin/env python3
"""Qualify mapped versus planned PLE lookup in the native serial runner."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "strix.ple-serial-qualification.v1"


class QualificationError(RuntimeError):
    pass


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def executable_path(path: Path) -> Path:
    if path.is_file():
        return path
    exe = path / "strix-hip-decode-token"
    if exe.is_file():
        return exe
    windows = exe.with_suffix(".exe")
    if windows.is_file():
        return windows
    raise QualificationError(f"serial runner not found: {path}")


def parse_json(stdout: str) -> dict[str, Any]:
    begin = stdout.find("{")
    end = stdout.rfind("}")
    if begin < 0 or end < begin:
        raise QualificationError("serial runner did not emit JSON")
    try:
        value = json.loads(stdout[begin : end + 1])
    except json.JSONDecodeError as exc:
        raise QualificationError(f"serial runner emitted invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise QualificationError("serial runner JSON root must be an object")
    return value


def run_once(
    executable: Path,
    model: Path,
    reference: Path,
    planned: bool,
    max_rows_per_io: int,
    max_inflight_ios: int,
) -> dict[str, Any]:
    command = [
        str(executable),
        str(model),
        "--reference",
        str(reference),
        "--ple-max-rows-per-io",
        str(max_rows_per_io),
        "--ple-max-inflight-ios",
        str(max_inflight_ios),
    ]
    if planned:
        command.append("--planned-ple-io")

    process = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        mode = "planned" if planned else "mapped"
        raise QualificationError(
            f"{mode} runner failed with exit code {process.returncode}: {detail}"
        )
    return parse_json(process.stdout)


def finite_positive(result: dict[str, Any], key: str, mode: str) -> float:
    try:
        value = float(result[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise QualificationError(f"{mode} omitted or malformed {key}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise QualificationError(f"{mode} reported invalid {key}={value!r}")
    return value


def validate_result(result: dict[str, Any], mode: str) -> None:
    expected_lookup = "planned-pread" if mode == "planned" else "mapped"
    if result.get("ple_lookup") != expected_lookup:
        raise QualificationError(
            f"{mode} reported ple_lookup={result.get('ple_lookup')!r}"
        )
    if int(result.get("ple_lookup_identity", 0)) == 0:
        raise QualificationError(f"{mode} reported zero PLE backend identity")
    if int(result.get("committed_tokens", 0)) <= 0:
        raise QualificationError(f"{mode} committed no tokens")

    tokens = result.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise QualificationError(f"{mode} emitted no token records")
    for token in tokens:
        if not isinstance(token, dict):
            raise QualificationError(f"{mode} token record is not an object")
        finite_positive(token, "elapsed_ms", mode)
        if "input_token" not in token or "argmax_token" not in token:
            raise QualificationError(f"{mode} token record omitted token ids")

    finite_positive(result, "decode_elapsed_ms_total", mode)
    finite_positive(result, "decode_elapsed_ms_per_token", mode)

    replay = result.get("replay")
    if not isinstance(replay, dict) or replay.get("passed") is not True:
        raise QualificationError(
            f"{mode} replay evidence did not pass declared policy"
        )


def token_signature(result: dict[str, Any]) -> list[tuple[int, int]]:
    return [
        (int(token["input_token"]), int(token["argmax_token"]))
        for token in result["tokens"]
    ]


def compare_identity(mapped: dict[str, Any], planned: dict[str, Any]) -> None:
    for key in (
        "model_content_identity",
        "execution_identity",
        "context_tokens",
        "committed_tokens",
        "next_position",
    ):
        if mapped.get(key) != planned.get(key):
            raise QualificationError(
                f"mapped/planned mismatch for {key}: "
                f"{mapped.get(key)!r} != {planned.get(key)!r}"
            )
    if mapped["replay"].get("reference_id") != planned["replay"].get("reference_id"):
        raise QualificationError("mapped/planned replay reference identities differ")
    if token_signature(mapped) != token_signature(planned):
        raise QualificationError("mapped/planned token or argmax sequence differs")


def summarize(samples: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    if not samples:
        raise QualificationError(f"{mode} produced no samples")

    totals = [finite_positive(sample, "decode_elapsed_ms_total", mode) for sample in samples]
    per_token = [
        finite_positive(sample, "decode_elapsed_ms_per_token", mode)
        for sample in samples
    ]

    first = samples[0]
    signature = token_signature(first)
    for sample in samples[1:]:
        if sample.get("model_content_identity") != first.get("model_content_identity"):
            raise QualificationError(f"{mode} model identity changed across repetitions")
        if sample.get("execution_identity") != first.get("execution_identity"):
            raise QualificationError(f"{mode} execution identity changed across repetitions")
        if sample.get("ple_lookup_identity") != first.get("ple_lookup_identity"):
            raise QualificationError(f"{mode} PLE backend identity changed across repetitions")
        if token_signature(sample) != signature:
            raise QualificationError(f"{mode} token signature changed across repetitions")

    return {
        "ple_lookup_identity": int(first["ple_lookup_identity"]),
        "decode_ms_total_median": statistics.median(totals),
        "decode_ms_per_token_median": statistics.median(per_token),
        "decode_ms_total_samples": totals,
        "decode_ms_per_token_samples": per_token,
    }


def build_report(
    mapped_samples: list[dict[str, Any]],
    planned_samples: list[dict[str, Any]],
    repetitions: int,
    max_rows_per_io: int,
    max_inflight_ios: int,
) -> dict[str, Any]:
    compare_identity(mapped_samples[0], planned_samples[0])
    mapped = summarize(mapped_samples, "mapped")
    planned = summarize(planned_samples, "planned")
    speedup = (
        mapped["decode_ms_per_token_median"]
        / planned["decode_ms_per_token_median"]
    )
    return {
        "schema": REPORT_SCHEMA,
        "model_content_identity": mapped_samples[0]["model_content_identity"],
        "execution_identity": mapped_samples[0]["execution_identity"],
        "reference_id": mapped_samples[0]["replay"]["reference_id"],
        "committed_tokens": mapped_samples[0]["committed_tokens"],
        "repetitions": repetitions,
        "planned_max_rows_per_io": max_rows_per_io,
        "planned_max_inflight_ios": max_inflight_ios,
        "mapped": mapped,
        "planned": planned,
        "speedup_mapped_over_planned": speedup,
        "all_replay_passed": True,
        "argmax_equivalent": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare mapped and planned-pread PLE lookup on identical native "
            "serial replay evidence."
        )
    )
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--repetitions", type=positive_int, default=3)
    parser.add_argument("--max-rows-per-io", type=positive_int, default=4)
    parser.add_argument("--max-inflight-ios", type=positive_int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    executable = executable_path(args.runner)
    samples: dict[str, list[dict[str, Any]]] = {
        "mapped": [],
        "planned": [],
    }

    for repetition in range(args.repetitions):
        order = ("mapped", "planned")
        if repetition % 2:
            order = tuple(reversed(order))
        for mode in order:
            result = run_once(
                executable,
                args.model,
                args.reference,
                mode == "planned",
                args.max_rows_per_io,
                args.max_inflight_ios,
            )
            validate_result(result, mode)
            samples[mode].append(result)

    for index in range(args.repetitions):
        compare_identity(samples["mapped"][index], samples["planned"][index])

    report = build_report(
        samples["mapped"],
        samples["planned"],
        args.repetitions,
        args.max_rows_per_io,
        args.max_inflight_ios,
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
    except QualificationError as error:
        print(f"qualification failed: {error}", file=sys.stderr)
        raise SystemExit(2)
