#!/usr/bin/env python3
"""Compare explicit single HTTP with AUTO/caps; store evidence outside the checkout."""
import argparse
import concurrent.futures
from contextlib import contextmanager
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request
from batch_service_gpu_gate import completion, require, tokens
from batch_service_gpu_gate import signature as full_signature


def signature(result):
    # Across routes the prompt-state snapshot is taken at different points (the batched route's 8-row prompt window
    # still holds deferred GDN rows then), so single vs N compares the complete IDs, the final frontier and every
    # committed target component; the prompt digest stays a same-route check in batch_service_gpu_gate.py.
    sig = full_signature(result)
    sig.pop("prompt_state_digest")
    return sig


@contextmanager
def server(args, name, mode):
    env = os.environ.copy()
    env.update(STRIX_PROMPT_STATE_DIGEST="1", STRIX_PREFIX_CACHE_DIGEST="1", STRIX_PREFIX_CACHE_DIGEST_COMPONENTS="1",
               STRIX_MTP_BATCH_LOG="1", STRIX_HTTP_EAGER_SLOTS="4")
    command = args.command + ["--http-serve", "--http-port", str(args.port)]
    if mode != "auto":
        command += ["--http-batch", mode]
    with (args.out / (name + ".log")).open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                require(process.poll() is None, "server exited: " + name)
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=1):
                        break
                except OSError:
                    require(time.monotonic() < deadline, "startup timeout: " + name)
                    time.sleep(0.25)
            yield
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            require(process.returncode == 0, "unclean shutdown: " + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", action="append", required=True, help="token ID file; repeat for varied prompts")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--generate", type=int, default=256)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--modes", default="auto,1,2,4,8,16")
    parser.add_argument("--clients", default="1,2,4,8,16")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- executable and pinned model/MTP arguments, without HTTP flags")
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command.pop(0)
    require(args.command, "server command required")
    require(not args.out.resolve().is_relative_to(Path(__file__).resolve().parents[1]), "evidence must stay outside Git")
    args.out.mkdir(parents=True, exist_ok=True)
    prompts = [tokens(path) for path in args.prompt]
    with server(args, "single-reference", "0"):
        reference = [signature(completion(args.port, p, args.generate)) for p in prompts]
    # One server per mode; every client count runs against it (a server start costs minutes on this model).
    lazy_witness = False
    for mode in args.modes.split(","):
        name = f"slots-{mode}"
        with server(args, name, mode):
            for clients in map(int, args.clients.split(",")):
                barrier = threading.Barrier(clients)
                def request(index):
                    barrier.wait()
                    return completion(args.port, prompts[index % len(prompts)], args.generate)
                with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as pool:
                    results = list(pool.map(request, range(clients)))
                for index, result in enumerate(results):
                    require(signature(result) == reference[index % len(prompts)], f"MISMATCH: {name} clients {clients}")
                    headers = {k.lower(): v for k, v in result["http_headers"].items()}
                    require("x-strix-http-slots" in headers and "x-strix-queue-position" in headers, "missing queue headers")
                    require(int(headers["x-strix-queue-position"]) >= 0, "negative queue position")
                    slots = int(headers["x-strix-http-slots"])
                    require(slots >= 1 and (mode == "auto" or slots <= int(mode)), "invalid slot capacity")
                if clients > slots + 1:
                    require(any(int({k.lower(): v for k, v in r["http_headers"].items()}["x-strix-queue-position"]) > 0
                                for r in results), f"NO-WITNESS: no waiting queue: {name} clients {clients}")
                (args.out / f"{name}-clients-{clients}.json").write_text(json.dumps(results, indent=2))
                print(f"EXACT {name} clients {clients}: slots {slots}", flush=True)
        log = (args.out / (name + ".log")).read_text()
        receipts = []
        for line in log.splitlines():
            try:
                row = json.loads(line)
                if row.get("mode") == "native-mtp-batch":
                    receipts.append(row)
            except (ValueError, AttributeError):
                pass
        if slots > 1 and max(map(int, args.clients.split(","))) > 1:
            require(receipts and receipts[-1]["max_active"] > 1, "NO-WITNESS: no request overlap: " + name)
        if slots > 4 and 16 in map(int, args.clients.split(",")):
            require(receipts and receipts[-1]["slots_built"] > 4 and receipts[-1]["lazy_slot_builds"] > 0,
                    "NO-WITNESS: no lazy growth beyond eager four: " + name)
            lazy_witness = True
            require(receipts[-1]["slots_built"] == 4 + receipts[-1]["lazy_slot_builds"], "lazy slot count mismatch")
            require(receipts[-1]["slots_capacity"] == slots, "receipt/header capacity mismatch")
            require(receipts[-1]["lazy_slot_ms"]["sum"] >= receipts[-1]["lazy_slot_ms"]["max"] > 0,
                    "missing lazy build timing")
        if slots > 8 and max(map(int, args.clients.split(","))) > 8:
            require(any(int(n) > 8 for n in re.findall(r"batch cycle sessions=(\d+)", log)),
                    "NO-WITNESS: no decode round above eight: " + name)
        require(all(int(n) <= 8 for n in re.findall(r"joint cycle sessions=(\d+)", log)),
                "controller received more than eight requests")
        line = next((l for l in log.splitlines() if l.startswith("http slots ")), None)
        print(f"{name}: {line} max_active {receipts[-1]['max_active'] if receipts else None}", flush=True)
    require(lazy_witness, "NO-WITNESS: lazy growth requires capacity above four and a 16-client workload")
    print("EXACT: lazy growth witnessed; complete IDs, prompt/final committed state and frontier match explicit single references")


if __name__ == "__main__":
    main()
