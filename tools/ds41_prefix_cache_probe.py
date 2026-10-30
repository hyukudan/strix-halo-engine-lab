#!/usr/bin/env python3
"""Replay frozen streaming chat requests; retain raw SSE and client content-delta timings."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import threading
import time
import urllib.request


def content_delta(event):
    for choice in event.get("choices", []):
        delta = choice.get("delta", {})
        if delta.get("content") or delta.get("reasoning_content"):
            return True
    return False


def token_ids(value):
    if isinstance(value, dict):
        if "token_ids" in value:
            return value["token_ids"]
        for child in value.values():
            found = token_ids(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = token_ids(child)
            if found is not None:
                return found
    return None


def probe(base, payload, output, timeout, barrier=None):
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("DS41_QUAL_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", payload, headers)
    if barrier:
        barrier.wait()
    start = time.perf_counter()
    result = {"ttft_seconds": None, "completion_seconds": None, "content_delta_seconds": [], "token_ids": None}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, output.open("wb") as raw:
            for line in response:
                elapsed = time.perf_counter() - start
                raw.write(line)
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    result["done"] = True
                    break
                event = json.loads(data)
                ids = token_ids(event)
                if ids is not None:
                    result["token_ids"] = ids
                if content_delta(event):
                    if result["ttft_seconds"] is None:
                        result["ttft_seconds"] = elapsed
                    result["content_delta_seconds"].append(elapsed)
        if not result.get("done") or result["ttft_seconds"] is None or result["token_ids"] is None:
            result["error"] = "missing DONE, content delta or complete token IDs"
    except Exception as error:
        result["error"] = str(error)
    result["completion_seconds"] = time.perf_counter() - start
    # Delta gaps may coalesce multiple tokens; they are not a per-token ITL proof.
    times = result["content_delta_seconds"]
    result["content_delta_gaps_seconds"] = [b - a for a, b in zip(times, times[1:])]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--requests", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parallel", action="store_true", help="one barrier arrival; otherwise retain cache across sequential requests")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--deadline-epoch", type=float, required=True, help="latest observation start, seconds since epoch")
    args = parser.parse_args()
    payloads = [path.read_bytes() for path in args.requests]
    for payload in payloads:
        body = json.loads(payload)
        if body.get("stream") is not True or body.get("return_token_ids") is not True:
            parser.error("frozen requests must set stream=true and return_token_ids=true")
    if time.time() >= args.deadline_epoch:
        parser.error("ticket observation-start deadline exceeded")
    args.out.mkdir(parents=True, exist_ok=False)
    barrier = threading.Barrier(len(payloads)) if args.parallel else None
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(payloads) if args.parallel else 1) as pool:
        if args.parallel:
            pending = [pool.submit(probe, args.base, payload, args.out / f"{i}.sse", args.timeout, barrier)
                       for i, payload in enumerate(payloads)]
            results = [future.result() for future in pending]
        else:
            for i, payload in enumerate(payloads):
                if time.time() >= args.deadline_epoch:
                    results.append({"error": "ticket deadline before next request", "request_index": i})
                    break
                results.append(probe(args.base, payload, args.out / f"{i}.sse", args.timeout))
    for i, result in enumerate(results):
        result["request"] = str(args.requests[i])
    (args.out / "measurements.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 1 if len(results) != len(payloads) or any("error" in result for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
