"""Native EXL3 CLI/server qualification using the existing generation receipts."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Barrier
from time import monotonic, sleep
from urllib.request import ProxyHandler, Request, build_opener


def require(condition, message):
    if not condition:
        raise ValueError(message)


def signature(ids, phase, selection, version, slots, limit):
    require(phase.get("exl3_identity") == f"g4-exl3-{version}" and
            phase.get("exl3_identity_selection") == selection and phase.get("exl3_max_streams") == slots,
            f"expected {selection} {version} at {slots} slots")
    require(phase.get("dflash_requested") == "off" and phase.get("prefill_identity") == "prefill-exact",
            "expected drafting off and exact prefill")
    require(0 < len(ids) <= limit and all(type(token) is int and 0 <= token < 2**32 for token in ids) and
            (phase["finish_reason"] == "stop" or phase["finish_reason"] == "length" and len(ids) == limit),
            "incomplete generated IDs")
    require(phase.get("logits_rows") == len(ids) and re.fullmatch(r"[0-9]{1,20}", phase.get("logits_fnv64", "")),
            "missing raw-logit digest or generation rows")
    require(re.fullmatch(r"[0-9a-f]{64}", phase.get("state_digest", "")) and phase.get("frontier", 0) > 0,
            "missing committed-state digest or frontier")
    return dict(ids=ids, **{key: phase[key] for key in (
        "logits_fnv64", "logits_rows", "state_digest", "frontier", "finish_reason", "model_hash", "rope_hash",
        "tokenizer_hash", "decode_identity", "prefill_identity", "sampling_identity")})


def compare(label, reference, observed):
    differences = [key for key in reference if reference[key] != observed[key]]
    require(not differences, f"{label}: mismatch in {', '.join(differences)}")


# Automatic identity per pack arm (format.hpp auto_version), then the other core forced as a within-identity control.
ARMS = {"M5": (("auto", "v1"), ("forced", "v3")), "M4": (("auto", "v3"), ("forced", "v1")),
        "M6": (("auto", "v3"), ("forced", "v1"))}
EXECUTED = ("executed_max_streams", "executed_beside_prefill_calls")


def overlap(batch, slots=8):
    """Execution receipts, not scheduled views: some decode pass carried all slots' decode rows, and some request
    decoded in a step where another request executed a sliced prefill call."""
    require(batch and all(key in row for row in batch for key in EXECUTED),
            "missing execution receipts for decode width and prefill overlap")
    require(max(row["executed_max_streams"] for row in batch) == slots, f"no executed {slots}-stream decode pass")
    require(any(row["executed_beside_prefill_calls"] > 0 for row in batch),
            "no decode executed beside a sliced prefill call")


def http(url, body=None, timeout=600):
    request = Request(url, body, {"Content-Type": "application/json"})
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
        return json.load(response)


@contextmanager
def server(binary, options, env, slots, port):
    base = f"http://127.0.0.1:{port}"
    with TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "server.log"
        with log_path.open("wb") as log:
            process = subprocess.Popen([binary, *options, "--bind", "127.0.0.1", "--port", port],
                                       env={**env, "STRIX_G4_SERVE_SLOTS": str(slots)}, stdout=log, stderr=log)
        try:
            deadline = monotonic() + 600
            while process.poll() is None and monotonic() < deadline:
                # The readiness line follows listen, and binds /health to this process.
                if b"Gemma 4 service ready" in log_path.read_bytes():
                    http(base + "/health", timeout=5)
                    break
                sleep(0.25)
            else:
                raise ValueError(f"server C{slots} failed to start: {log_path.read_text(errors='replace')}")
            yield base
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def completion(base, body, barrier=None):
    if barrier is not None:
        barrier.wait(timeout=10)
    reply = http(base + "/v1/chat/completions", body)
    require(len(reply["strix"]["phases"]) == 1, "expected one server phase")
    ids = reply["strix"]["generated_ids"]
    require(reply["usage"]["completion_tokens"] == len(ids), "server usage differs from raw IDs")
    return ids, reply["strix"]["phases"][0], reply["usage"]["prompt_tokens"]


def run(decode, serve, model, pack, rope, tokenizer, short, tail, port):
    arm = json.loads((Path(pack) / "manifest.json").read_text(encoding="utf-8"))["arm"]
    require(arm in ARMS, f"unknown EXL3 arm {arm}")
    options = [model, "--backend", "device", "--profile", "quality", "--exl3-arm", pack, "--rope-tables", rope,
               "--tokenizer", tokenizer, "--draft", "off", "--dflash", "off"]
    env = {name: value for name, value in os.environ.items() if not name.startswith("STRIX_G4_")}
    env.update(STRIX_G4_LOGITS_DIGEST="1", STRIX_G4_SERVE_BATCH="1", STRIX_G4_SERVE_PREFIX="0")
    limit = 128
    bodies = [json.dumps(dict(messages=[dict(role="user", content=Path(p).read_text(encoding="utf-8"))],
                             max_tokens=limit, temperature=0, enable_thinking=False, dflash="off")).encode()
              for p in (short, tail)]
    with TemporaryDirectory() as tmp:
        requests = [Path(tmp) / f"request-{i}.json" for i in range(len(bodies))]
        for path, body in zip(requests, bodies):
            path.write_bytes(body)
        for selection, version in ARMS[arm]:
            if selection == "forced":
                env["STRIX_G4_EXL3_IDENTITY"] = version
            reference = []
            for path in requests:
                result = json.loads(subprocess.check_output([decode, *options, "--chat", str(path), "--streams", "1",
                                                             "--output", "json"], env=env, timeout=600))
                require(len(result["receipts"]) == 1, "expected one CLI phase")
                reference.append(signature(result["ids"], result["receipts"][0], selection, version, 1, limit))
            prompt_lengths = []
            for slots in (1, 8):
                with server(serve, options, env, slots, port) as base:
                    if slots == 1:
                        results = [completion(base, body) for body in bodies]
                    else:
                        barrier = Barrier(slots)
                        with ThreadPoolExecutor(max_workers=slots) as workers:
                            results = list(workers.map(lambda body: completion(base, body, barrier), bodies * 4))
                batch = []
                for i, (ids, phase, prompt_tokens) in enumerate(results):
                    prompt = i % len(bodies)
                    record = signature(ids, phase, selection, version, slots, limit)
                    compare(f"{selection} {version} HTTP C{slots} request {i}", reference[prompt], record)
                    stats = phase["serve_batch"]
                    require(stats["slots"] == slots, "server receipt has the wrong slot count")
                    if slots == 1:
                        require(stats["max_batch"] == 1, "server C1 did not run alone")
                        prompt_lengths.append(prompt_tokens)
                    else:
                        require(prompt_tokens == prompt_lengths[prompt], "rendered prompt length changed under load")
                        batch.append(stats | {key: phase[key] for key in EXECUTED if key in phase})
                print(json.dumps(dict(selection=selection, version=version, slots=slots, phase="server",
                                      prompt_lengths=prompt_lengths, batch=batch)), flush=True)
                if slots == 1:
                    require(0 < prompt_lengths[0] < 512 and prompt_lengths[1] > 512 and
                            1 <= (prompt_lengths[1] - 1) % 512 <= 8,
                            f"need a short prompt and a 1–8 row tail after a full 512-row chunk, got {prompt_lengths}")
            overlap(batch)
            print(json.dumps(dict(selection=selection, version=version, verdict="EXACT-WITHIN-IDENTITY",
                                  cli_reference=reference)), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 10:
        sys.exit(f"usage: {sys.argv[0]} DECODE SERVE BF16 EXL3_PACK ROPE TOKENIZER SHORT.txt TAIL.txt PORT")
    run(*sys.argv[1:])
