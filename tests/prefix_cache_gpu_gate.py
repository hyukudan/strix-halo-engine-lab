#!/usr/bin/env python3
"""Strict L0 prefix joining/restart gate. Requires a native server and private token fixtures."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

sys.dont_write_bytecode = True
from batch_service_gpu_gate import completion, require, tokens


def prompt_key(prompt):
    value = 14695981039346656037
    for token in prompt:
        for shift in range(0, 32, 8):
            value = ((value ^ ((token >> shift) & 255)) * 1099511628211) & ((1 << 64) - 1)
    return len(prompt), f"{value:016x}"


def run(args, name, prompts, *, joining=False, disk=None, lookup=True, mixed=False, diagnostics=True):
    env = os.environ.copy()
    env.update(STRIX_PREFIX_CACHE_DIGEST="1", STRIX_PREFIX_CACHE_DIGEST_COMPONENTS="1",
               STRIX_FINAL_STATE_DIGEST="1", STRIX_PROMPT_STATE_DIGEST="1",
               STRIX_PREFIX_CACHE_LOOKUP="1" if lookup else "0",
               STRIX_PREFIX_CACHE_INFLIGHT="1" if joining else "0",
               STRIX_PROMPT_BATCH="0", STRIX_PROMPT_WINDOW_COHORT="0",
               STRIX_BATCH_NEST_ROWS="0", STRIX_MTP_BATCH_LOG="a")
    if not diagnostics:
        env.update(STRIX_PREFIX_CACHE_DIGEST="0", STRIX_PREFIX_CACHE_DIGEST_COMPONENTS="0",
                   STRIX_FINAL_STATE_DIGEST="0", STRIX_PROMPT_STATE_DIGEST="0")
    command = args.command + ["--http-serve", "--http-batch", "3", "--http-bind", "127.0.0.1",
                              "--http-port", str(args.port), "--prefix-batch-cert", "l0"]
    if disk:
        command += ["--prefix-cache-dir", str(disk)]
    log_path = args.out / (name + ".log")
    results = []
    def measured_completion(prompt):
        start = time.monotonic()
        result = completion(args.port, prompt, args.generate)
        result["started_at"] = start
        result["ttft_ms"] = 1000 * (result["first_at"] - start)
        return result
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, env=env)
        try:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                require(process.poll() is None, f"{name}: server exited")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=1):
                        break
                except OSError:
                    require(time.monotonic() < deadline, f"{name}: startup timeout")
                    time.sleep(0.25)
            if mixed:
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                    futures = [pool.submit(measured_completion, prompts[0])]
                    time.sleep(args.arrival_ms / 1000)
                    futures.append(pool.submit(measured_completion, prompts[1]))
                    futures.append(pool.submit(measured_completion, prompts[2]))
                    results = [f.result() for f in futures]
            else:
                results = [measured_completion(p) for p in prompts]
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            require(process.returncode == 0, f"{name}: unclean shutdown; see private log")
    finals, batch, receipts = {}, None, []
    for line in log_path.read_text().splitlines():
        if line.startswith("request final: "):
            final = json.loads(line.removeprefix("request final: "))
            require("diagnostic_error" not in final, "NO-REFERENCE: incomplete final diagnostics")
            key = final["prompt_tokens"], final["prompt_fnv"]
            require(key not in finals, "ambiguous duplicate prompt receipt")
            finals[key] = final
        elif line.startswith("tiled prompt: "):
            receipts.append(json.loads(line.removeprefix("tiled prompt: ")))
        elif line.startswith("{"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("mode") == "native-mtp-batch":
                batch = row
    require(batch is not None, "NO-REFERENCE: missing batch summary")
    if joining:
        require(batch.get("inflight_waits", 0) > 0 and batch.get("inflight_restored_tokens", 0) > 0,
                "NO-WITNESS: no joining request reused producer work")
    if not diagnostics:
        ids = {prompt_key(p): {"ids": result["strix"].get("token_ids")} for p, result in zip(prompts, results)}
        require(all(row["ids"] for row in ids.values()), "NO-REFERENCE: missing timing-arm token IDs")
        span = max(r["completed_at"] for r in results) - min(r["started_at"] for r in results)
        args.timings[name] = {"ttft_ms": [r["ttft_ms"] for r in results],
                              "tokens_per_second": sum(len(row["ids"]) for row in ids.values()) / span}
        (args.out / (name + ".json")).write_text(json.dumps(dict(results=results, batch=batch), indent=2))
        return ids
    require(receipts and all("capture_counters_global" in row for row in receipts),
            "NO-REFERENCE: missing snapshot tripwire counters")
    require(all(row["capture_counters_global"]["tripwire"] == 0 for row in receipts), "snapshot tripwire fired")
    prompt_receipts = {(row["prompt_tokens"], row["prompt_fnv"]): row for row in receipts}
    signatures = {}
    for prompt, result in zip(prompts, results):
        key = prompt_key(prompt)
        require(key in finals, "NO-REFERENCE: missing complete request final")
        final = finals[key]
        ids = result["strix"].get("token_ids")
        require(ids and len(ids) == final["generated_ids_count"] == final["output_logits_rows"],
                "NO-REFERENCE: incomplete IDs/raw vocabulary row coverage")
        for field in ("output_logits_chain", "state_digest", "generations_lineage_digest"):
            require(final.get(field) and len(final[field]) == 64, f"NO-REFERENCE: missing {field}")
        require(final["next_position"] == final["committed_tokens"], "uncommitted final frontier")
        require(key in prompt_receipts, "NO-REFERENCE: missing prompt receipt")
        handoff = prompt_receipts[key]
        require(handoff.get("terminal_logits_digest") and handoff.get("state_digest") and handoff.get("terminal_metadata"),
                "NO-REFERENCE: missing prompt logits/state/frontier")
        signatures[key] = {"ids": ids, "prompt": {k: handoff[k] for k in
                           ("terminal_logits_digest", "state_digest", "terminal_metadata")},
                           **{k: v for k, v in final.items() if k != "request_order"}}
    (args.out / (name + ".json")).write_text(json.dumps(dict(results=results, batch=batch, prompts=receipts), indent=2))
    if disk and not mixed and len(prompts) == 1 and name == "restart":
        require(any(row.get("restored_position", 0) > 0
                    for row in receipts), "NO-WITNESS: restart did not restore")
    return signatures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("producer", "joiner", "short"):
        parser.add_argument("--" + name, required=True, help="private token IDs, JSON or whitespace")
    parser.add_argument("--out", type=Path, required=True, help="fresh private receipt directory outside Git")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--generate", type=int, default=64)
    parser.add_argument("--arrival-ms", type=float, default=50)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--timing", action="store_true", help="after exactness, run diagnostics-off ABA x3 and enforce no slower medians")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="-- native binary, model and pinned MTP assets, explicit host budget; omit HTTP/disk/cert flags")
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command.pop(0)
    require(args.command, "server command required")
    require(not args.out.resolve().is_relative_to(Path(__file__).resolve().parents[1]),
            "private receipts must stay outside checkout")
    args.out.mkdir(parents=True, exist_ok=False)
    prompts = [tokens(getattr(args, k)) for k in ("producer", "joiner", "short")]
    require(len({prompt_key(p) for p in prompts}) == 3, "use distinct prompts")
    reference = run(args, "cold", prompts, lookup=False)
    for repeat in range(3):
        for suffix, joining in (("a", False), ("b", True), ("a2", False)):
            actual = run(args, f"aba-{repeat}-{suffix}", prompts, joining=joining, mixed=True)
            require(actual == reference, "MISMATCH: joining/mixed-arrival full logits, IDs or committed state")
    disk = args.out / "prefix-store"
    seed = run(args, "disk-seed", [prompts[0]], disk=disk)
    require(seed[prompt_key(prompts[0])] == reference[prompt_key(prompts[0])], "MISMATCH: disk producer")
    require(list(disk.glob("ckpt-*.tmpc")), "NO-WITNESS: no persisted prefix checkpoint")
    restart = run(args, "restart", [prompts[1]], disk=disk)
    require(restart[prompt_key(prompts[1])] == reference[prompt_key(prompts[1])], "MISMATCH: restart turn")
    print("EXACT: complete emitted vocabulary-row chains, IDs, committed state and lineage; joining ABA x3 and restart")
    if args.timing:
        args.timings = {}
        expected_ids = {key: {"ids": row["ids"]} for key, row in reference.items()}
        off, on = [], []
        for repeat in range(3):
            for suffix, joining in (("a", False), ("b", True), ("a2", False)):
                name = f"timing-{repeat}-{suffix}"
                actual = run(args, name, prompts, joining=joining, mixed=True, diagnostics=False)
                require(actual == expected_ids, "MISMATCH: timing arm tokens differ from qualified reference")
                (on if joining else off).append(args.timings[name])
        medians = {arm: {"ttft_ms": [statistics.median(row["ttft_ms"][i] for row in samples) for i in range(3)],
                         "tokens_per_second": statistics.median(row["tokens_per_second"] for row in samples)}
                   for arm, samples in (("off", off), ("on", on))}
        (args.out / "timing.json").write_text(json.dumps(medians, indent=2))
        require(all(b <= a for a, b in zip(medians["off"]["ttft_ms"], medians["on"]["ttft_ms"])),
                "SLOWER: a producer/joiner/short median TTFT regressed")
        require(medians["on"]["tokens_per_second"] >= medians["off"]["tokens_per_second"],
                "SLOWER: complete mixed-workload throughput regressed")
        print("PASS: diagnostics-off ABA x3 has no slower median TTFT or complete-workload throughput")
    else:
        print("Timing qualification pending: rerun with --timing for diagnostics-off ABA medians.")


if __name__ == "__main__":
    main()
