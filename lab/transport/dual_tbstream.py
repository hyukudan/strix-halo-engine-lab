#!/usr/bin/env python3
"""Run two USB4STREAM benchmark flows concurrently.

This is a physical-link independence harness, not a single-tensor reassembly
benchmark. It deliberately measures two independent flows first.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Lane:
    device: str
    config: str | None


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
    processes: list[subprocess.Popen[str]] = []
    try:
        for lane in lanes:
            command = lane_args(args.binary, "server", lane)
            command += ["--max-bytes", args.max_bytes]
            if args.verify:
                command.append("--verify")
            processes.append(
                subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )

        outputs = []
        for process in processes:
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                sys.stderr.write(stderr)
                return process.returncode or 1
            outputs.append(json.loads(stdout))

        print(
            json.dumps(
                {
                    "role": "dual_server",
                    "mode": "parallel_flows",
                    "lanes": outputs,
                },
                indent=2,
            )
        )
        return 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()


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

    processes: list[subprocess.Popen[str]] = []
    start = time.monotonic()
    try:
        for command in commands:
            processes.append(
                subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )

        results = []
        for process in processes:
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                sys.stderr.write(stderr)
                return process.returncode or 1
            results.append(json.loads(stdout))
        wall_seconds = time.monotonic() - start
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()

    qualified_seconds = max(
        float(result["measured_seconds"]) for result in results
    )
    direction_multiplier = 2 if args.mode == "pingpong" else 1
    payload_bytes = total_bytes * args.iterations * direction_multiplier
    aggregate_gib_s = (
        payload_bytes / (1024**3) / qualified_seconds
        if qualified_seconds > 0
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
                "qualified_seconds": qualified_seconds,
                "process_wall_seconds": wall_seconds,
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
    return result


def main() -> int:
    args = parser().parse_args()
    if not 0.0 < args.left_share < 1.0:
        raise ValueError("--left-share must be between 0 and 1")
    if args.iterations <= 0 or args.warmup < 0:
        raise ValueError("invalid iteration counts")
    binary = pathlib.Path(args.binary)
    if not binary.exists():
        raise FileNotFoundError(binary)
    return launch_servers(args) if args.role == "server" else launch_clients(args)


if __name__ == "__main__":
    raise SystemExit(main())
