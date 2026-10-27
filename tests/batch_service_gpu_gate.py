#!/usr/bin/env python3
"""Run the actual batched server in fresh processes; keep all receipts outside the checkout."""
import argparse
import concurrent.futures
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import urllib.request


def require(value, message):
    if not value:
        raise RuntimeError(message)


def tokens(path):
    text = Path(path).read_text().strip()
    return json.loads(text) if text.startswith("[") else [int(x) for x in text.split()]


def completion(port, prompt, count, first=None):
    body = json.dumps(dict(prompt=prompt, max_tokens=count, temperature=0,
                           stream=True, return_token_ids=True))
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                     data=body.encode(), headers={"Content-Type": "application/json"})
    result = None
    arrivals = []
    with urllib.request.urlopen(request, timeout=1800) as response:
        headers = dict(response.headers)
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            if "strix" in chunk:
                result = chunk
                final_at = time.monotonic()
            else:
                arrivals.append(time.monotonic())
                if first:
                    first.set()
    require(result is not None, "missing final streaming receipt")
    result["completed_at"] = time.monotonic()
    # A one-token answer can arrive with the final receipt chunk only.
    result["first_at"] = arrivals[0] if arrivals else final_at
    result["http_headers"] = headers
    result["emission_gaps_ms"] = [1000 * (b - a) for a, b in zip(arrivals, arrivals[1:])]
    return result


def signature(result):
    # Committed target state: every registry component except the MTP predictor's (mtp.*). The predictor's cache rows
    # and held row follow the draft batching, which depends on concurrency; they never decide a token.
    s = result["strix"]
    keys = ("token_ids", "prompt_state_digest", "final_state_position", "final_state_components")
    require(all(k in s for k in keys), "NO-REFERENCE: missing IDs or committed-state diagnostics")
    require(s["token_ids"] and s["prompt_state_digest"] and s["final_state_components"], "empty exactness reference")
    sig = {k: s[k] for k in keys[:3]}
    sig["target_state"] = {k: v for k, v in s["final_state_components"].items() if not k.startswith("mtp.")}
    require(sig["target_state"], "NO-REFERENCE: missing committed target components")
    return sig


def predictor_state(result):
    return {k: v for k, v in result["strix"]["final_state_components"].items() if k.startswith("mtp.")}


def run_arm(args, name, budget, cancel, sequential, reuse_reference=None):
    env = os.environ.copy()
    env.update(STRIX_BATCH_SERVICE_MS=str(budget), STRIX_BATCH_CANCEL_POINTS=str(cancel),
               STRIX_BATCH_SERVICE_LOOKAHEAD=str(args.lookahead), STRIX_PROMPT_STATE_DIGEST="1",
               STRIX_PREFIX_CACHE_DIGEST="1", STRIX_PREFIX_CACHE_DIGEST_COMPONENTS="1",
               STRIX_BATCH_NEST_ROWS=str(args.nest_rows if name == "nest-on" else 0),
               STRIX_MTP_BATCH_LOG="a")
    log_path = args.out / (name + ".log")
    command = args.command + ["--http-serve", "--http-batch", "2", "--http-bind", "127.0.0.1",
                              "--http-port", str(args.port)]
    with log_path.open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                require(process.poll() is None, f"{name}: server exited; inspect {log_path.name}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=1):
                        break
                except OSError:
                    require(time.monotonic() < deadline, "server startup timed out")
                    time.sleep(0.25)
            short, long = tokens(args.short), tokens(args.long)
            require(short and long, "prompts must be nonempty")
            cancel_prompt = long.copy()
            cancel_prompt[0] = cancel_prompt[0] - 1 if cancel_prompt[0] else 1
            if sequential:
                results = [completion(args.port, p, args.generate) for p in (short, long, cancel_prompt)]
            elif name in ("nest-on", "nest-off"):
                require(len(long) >= 8192, "nest scenario needs a full long T8192 block")
                require(0 < len(short) <= args.nest_rows, "short prompt exceeds nested row limit")
                first = threading.Event()
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    outer = pool.submit(completion, args.port, long, args.generate, first)
                    deadline = time.monotonic() + args.startup_timeout
                    # This marker is emitted on the worker's first closed native bundle,
                    # before its park decision; admission alone is insufficient evidence.
                    while "batch prefill boundary rows=8192" not in log_path.read_text():
                        require(not outer.done() and not first.is_set(), "long prompt left prefill before native witness")
                        require(process.poll() is None and time.monotonic() < deadline, "no long native prefill witness")
                        time.sleep(0.01)
                    require(not first.is_set() and not outer.done(), "long prompt already completed prefill")
                    nested = pool.submit(completion, args.port, short, args.generate)
                    results = [nested.result(), outer.result()]
                if name == "nest-on":
                    require(results[0]["strix"]["batch_service"]["nested_admits"] > 0,
                            "NO-WITNESS: short prompt did not execute inside parked native call")
                    # Nested: the short request's first token precedes the long request's (completion order depends on lengths).
                    require(results[0]["first_at"] is not None and results[1]["first_at"] is not None and
                            results[0]["first_at"] < results[1]["first_at"], "short prompt's first token did not precede the long prompt's")
            else:
                first = threading.Event()
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    decoding = pool.submit(completion, args.port, short, args.generate, first)
                    require(first.wait(1800), "short request emitted no first token")
                    joining = pool.submit(completion, args.port, long, args.generate)
                    results = [decoding.result(), joining.result()]
            if name == "nest-off":
                require(all(r["strix"]["batch_service"]["nested_admits"] == 0 for r in results),
                        "nest-off executed nested work")
            if name == "service-on":
                require(results[1]["strix"]["batch_service"]["service_rounds"] > 0,
                        "NO-WITNESS: no decode ran inside the joining prompt block; use longer generation")
                # Cancel a lone long prompt before first-token publication, then reuse the slot.
                connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=1800)
                connection.request("POST", "/v1/completions", json.dumps(dict(prompt=cancel_prompt,
                                   max_tokens=args.generate, temperature=0, stream=True)),
                                   {"Content-Type": "application/json"})
                response = connection.getresponse()
                require(response.status == 200, "cancel request was rejected")
                time.sleep(args.cancel_delay)
                if connection.sock:
                    connection.sock.shutdown(socket.SHUT_RDWR)
                response.close()
                connection.close()
                time.sleep(args.cancel_delay)
                reused = completion(args.port, cancel_prompt, args.generate)
                require(signature(reused) == reuse_reference, "cancel/reuse changed IDs or committed state")
                results.append(reused)
            for result in results:
                signature(result)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            require(process.returncode == 0, f"{name}: unclean shutdown; inspect {log_path.name}")
    receipts = []
    for line in log_path.read_text().splitlines():
        try:
            row = json.loads(line)
            if row.get("mode") == "native-mtp-batch":
                receipts.append(row)
        except (ValueError, AttributeError):
            pass
    require(receipts, "missing final batch receipt")
    if name == "service-on":
        require(receipts[-1]["cancel_points_cancelled"] > 0,
                "NO-WITNESS: cancellation missed native boundary; adjust --cancel-delay or prompt length")
    (args.out / (name + ".json")).write_text(json.dumps(dict(requests=results, batch=receipts[-1]), indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short", required=True, help="short prompt token IDs (text or JSON)")
    parser.add_argument("--long", required=True, help="long prompt IDs; include multiple full T blocks and a tail")
    parser.add_argument("--out", type=Path, required=True, help="fresh private output directory outside checkout")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--generate", type=int, default=256)
    parser.add_argument("--service-ms", type=float, default=1)
    parser.add_argument("--lookahead", type=int, default=2)
    parser.add_argument("--cancel-delay", type=float, default=0.5)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--scenario", choices=("service", "nest"), default="service")
    parser.add_argument("--nest-rows", type=int, default=2048)
    parser.add_argument("--arms", help="comma separated fresh-process arms (scenario defaults when omitted)")
    parser.add_argument("--reference", help="alone.json of an earlier run (skips the sequential arm)")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- server executable plus pinned model/MTP args; omit HTTP flags")
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command.pop(0)
    require(args.command, "server command required after --")
    checkout = Path(__file__).resolve().parents[1]
    require(not args.out.resolve().is_relative_to(checkout), "receipts must stay outside Git checkout")
    args.out.mkdir(parents=True, exist_ok=True)
    # --arms splits the matrix over several runs: the sequential reference is stored and reused (--reference).
    arms = {"service-off": (0, 0), "cancel-only": (0, 1), "service-on": (args.service_ms, 1),
            "nest-on": (args.service_ms, 1), "nest-off": (args.service_ms, 1)}
    if args.arms is None:
        args.arms = "nest-on,nest-off" if args.scenario == "nest" else "service-off,cancel-only,service-on"
    if args.reference:
        reference = json.loads(Path(args.reference).read_text())["requests"]
    else:
        reference = run_arm(args, "alone", 0, 0, True)
    for name in [a for a in args.arms.split(",") if a and a != "alone"]:
        budget, cancel = arms[name]
        result = run_arm(args, name, budget, cancel, False, signature(reference[2]) if name == "service-on" else None)
        for k, (x, y) in enumerate(zip(result[:2], reference[:2])):
            if predictor_state(x) != predictor_state(y):
                print(f"note: {name} request {k}: predictor (mtp.*) state differs from the reference (not a token decision)")
        require([signature(x) for x in result[:2]] == [signature(x) for x in reference[:2]],
                f"MISMATCH: {name} differs from sequential reference")
        print(f"EXACT {name}: full IDs, prompt state and committed target state equal the sequential reference")
    print("EXACT: every requested arm")


if __name__ == "__main__":
    main()
