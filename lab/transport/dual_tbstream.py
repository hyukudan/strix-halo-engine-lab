#!/usr/bin/env python3
"""Run two USB4STREAM benchmark flows concurrently.

This is a physical-link independence harness, not a single-tensor reassembly
benchmark. It deliberately measures two independent flows first.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Lane:
    device: str
    config: str | None


OUTPUT_LIMIT = 1024 * 1024
CLEANUP_GRACE_SECONDS = 1.0


def _private_temp_directory() -> pathlib.Path:
    directory = pathlib.Path(tempfile.gettempdir()).resolve(strict=True)
    if any((parent / ".git").exists() for parent in (directory, *directory.parents)):
        raise ValueError("benchmark temporary outputs must remain outside Git")
    return directory


def _run_lanes(commands: list[list[str]], *, timeout_seconds: float,
               popen=None, now=time.monotonic, pause=time.sleep, kill_group=None) -> tuple[list[dict], float]:
    """Bound two trusted local benchmarks without pipe-reader wait threads.

    Native child groups are POSIX-only, matching the underlying baseline. Files
    spool bounded output privately; each group's descendants are stopped even
    when its leader has already exited. Injected calls are fixture-only.
    """
    if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 3600 or len(commands) != 2:
        raise ValueError("two lanes require a finite deadline within one hour")
    if popen is None:
        if os.name != "posix":
            raise RuntimeError("native transport baselines require POSIX process groups")
        popen = subprocess.Popen
    if kill_group is None:
        kill_group = os.killpg
    processes = []
    outputs = []
    streams = []
    start = last = now()
    if not math.isfinite(start):
        raise ValueError("benchmark clock is unavailable")
    deadline = start + timeout_seconds
    if not math.isfinite(deadline):
        raise ValueError("benchmark deadline overflow")

    def clock():
        nonlocal last
        value = now()
        if not math.isfinite(value) or value < last:
            raise RuntimeError("benchmark clock changed or is unavailable")
        last = value
        return value

    def signal_all(sig):
        errors = []
        for process in processes:
            try:
                kill_group(process.pid, sig)
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(error)
        return errors

    def cleanup():
        # Each group was created exclusively for this invocation. Signal groups,
        # not just leaders; normal parent exit is not descendant cleanup proof.
        signal_all(signal.SIGTERM)
        end = time.monotonic() + CLEANUP_GRACE_SECONDS
        for process in processes:
            try:
                process.wait(timeout=max(0.001, end - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        errors = signal_all(getattr(signal, "SIGKILL", 9))
        end = time.monotonic() + CLEANUP_GRACE_SECONDS
        for process in processes:
            try:
                process.wait(timeout=max(0.001, end - time.monotonic()))
            except subprocess.TimeoutExpired as error:
                errors.append(error)
        if errors:
            raise RuntimeError("benchmark child/group cleanup remains unproved") from errors[0]

    try:
        directory = _private_temp_directory()
        for command in commands:
            if clock() >= deadline:
                raise TimeoutError("dual transport baseline total deadline expired during launch")
            stdout = tempfile.TemporaryFile(dir=directory)
            streams.append(stdout)
            stderr = tempfile.TemporaryFile(dir=directory)
            streams.append(stderr)
            outputs.append((stdout, stderr))
            processes.append(popen(command, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                                   start_new_session=True))
        while True:
            if clock() >= deadline:
                raise TimeoutError("dual transport baseline total deadline expired")
            completed = []
            for process, (stdout, stderr) in zip(processes, outputs):
                if os.fstat(stdout.fileno()).st_size > OUTPUT_LIMIT or os.fstat(stderr.fileno()).st_size > OUTPUT_LIMIT:
                    raise RuntimeError("benchmark output exceeds its bounded retained size")
                code = process.poll()
                if code is not None and code != 0:
                    raise RuntimeError(f"benchmark lane failed with exit code {code}")
                completed.append(code is not None)
            if all(completed):
                break
            pause(min(0.01, max(0.0, deadline - last)))
        if clock() >= deadline:
            raise TimeoutError("dual transport baseline completed after its total deadline")
        wall_seconds = last - start
        # Stop lingering descendants before reading their inherited output files.
        cleanup()
        processes.clear()
        results = []
        for stdout, stderr in outputs:
            for stream in (stdout, stderr):
                if os.fstat(stream.fileno()).st_size > OUTPUT_LIMIT:
                    raise RuntimeError("benchmark output exceeds its bounded retained size")
            stdout.seek(0)
            record = json.loads(stdout.read(OUTPUT_LIMIT + 1).decode("utf-8"))
            if not isinstance(record, dict):
                raise ValueError("benchmark output must be one JSON object")
            results.append(record)
        return results, wall_seconds
    finally:
        try:
            if processes:
                cleanup()
        finally:
            for stream in streams:
                stream.close()


def size_text(total_bytes: int) -> str:
    return str(total_bytes)


def lane_args(binary: str, role: str, lane: Lane, *, portless: bool = True) -> list[str]:
    del portless
    args = [binary, role, "--device", lane.device]
    if lane.config:
        args += ["--config", lane.config]
    return args


def launch_servers(args: argparse.Namespace) -> int:
    lanes = [
        Lane(args.left_device, args.left_config),
        Lane(args.right_device, args.right_config),
    ]
    commands = []
    for lane in lanes:
        command = lane_args(args.binary, "server", lane) + ["--max-bytes", args.max_bytes]
        if args.verify:
            command.append("--verify")
        commands.append(command)
    outputs, wall_seconds = _run_lanes(commands, timeout_seconds=args.timeout_seconds)
    print(json.dumps({"role": "dual_server", "mode": "parallel_flows", "lanes": outputs,
                      "process_wall_seconds": wall_seconds, "measured_overlap": "UNKNOWN",
                      "aggregate_performance_qualified": False}, indent=2))
    return 0


def launch_clients(args: argparse.Namespace) -> int:
    total_bytes = args.bytes
    if total_bytes < 2:
        raise ValueError("--bytes must be at least 2")
    left_bytes = max(1, int(total_bytes * args.left_share))
    left_bytes = min(left_bytes, total_bytes - 1)
    right_bytes = total_bytes - left_bytes

    lanes = [
        (Lane(args.left_device, args.left_config), left_bytes),
        (Lane(args.right_device, args.right_config), right_bytes),
    ]

    commands: list[list[str]] = []
    for lane, lane_bytes in lanes:
        command = lane_args(args.binary, "client", lane)
        command += [
            "--mode",
            args.mode,
            "--bytes",
            size_text(lane_bytes),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
        ]
        if args.verify:
            command.append("--verify")
        commands.append(command)

    results, wall_seconds = _run_lanes(commands, timeout_seconds=args.timeout_seconds)
    lane_seconds = [float(result["measured_seconds"]) for result in results]
    if any(not math.isfinite(value) or value <= 0 for value in lane_seconds):
        raise ValueError("lane measured duration must be finite and positive")
    duration_estimate = max(lane_seconds)
    direction_multiplier = 2 if args.mode == "pingpong" else 1
    payload_bytes = total_bytes * args.iterations * direction_multiplier
    aggregate_gib_s = (
        payload_bytes / (1024**3) / duration_estimate
        if duration_estimate > 0
        else 0.0
    )

    print(
        json.dumps(
            {
                "role": "dual_client",
                "mode": "parallel_flows",
                "payload_mode": args.mode,
                "total_bytes_per_iteration": total_bytes,
                "left_bytes": left_bytes,
                "right_bytes": right_bytes,
                "left_share": args.left_share,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "verify": args.verify,
                "aggregate_duration_estimate_seconds": duration_estimate,
                "process_wall_seconds": wall_seconds,
                "measured_overlap": "UNKNOWN",
                "common_measurement_start_barrier": False,
                "aggregate_performance_qualified": False,
                "aggregate_effective_payload_gib_s": aggregate_gib_s,
                "lanes": results,
            },
            indent=2,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("role", choices=("server", "client"))
    result.add_argument("--binary", default="./build/transport-lab/strix-transport-tbstream")
    result.add_argument("--left-device", required=True)
    result.add_argument("--right-device", required=True)
    result.add_argument("--left-config")
    result.add_argument("--right-config")
    result.add_argument("--verify", action="store_true")

    result.add_argument("--max-bytes", default="64MiB")

    result.add_argument("--bytes", type=int, default=16 * 1024 * 1024)
    result.add_argument("--left-share", type=float, default=0.5)
    result.add_argument("--mode", choices=("stream", "pingpong"), default="stream")
    result.add_argument("--warmup", type=int, default=5)
    result.add_argument("--iterations", type=int, default=50)
    result.add_argument("--timeout-seconds", type=float, default=60.0)
    return result


def main() -> int:
    args = parser().parse_args()
    if not 0.0 < args.left_share < 1.0:
        raise ValueError("--left-share must be between 0 and 1")
    if args.iterations <= 0 or args.warmup < 0:
        raise ValueError("invalid iteration counts")
    if not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= 3600:
        raise ValueError("--timeout-seconds must be finite, positive and at most 3600")
    binary = pathlib.Path(args.binary)
    if not binary.exists():
        raise FileNotFoundError(binary)
    return launch_servers(args) if args.role == "server" else launch_clients(args)


if __name__ == "__main__":
    raise SystemExit(main())
