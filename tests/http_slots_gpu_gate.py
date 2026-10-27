#!/usr/bin/env python3
r"""Compare explicit single HTTP with AUTO/caps; store evidence outside the checkout.

Run each opt-in arm as a separate GPU turn, with --generate 4 (not the general
slots gate's 256-token default). T4.ids contains exactly T+4 tokens; T12.ids
contains the same T block plus 12 suffix tokens. Each command reuses its servers
for all requests and starts only two servers (three for history or roundtrip):
  python3 tests/http_slots_gpu_gate.py --window-state-arm digest-only \
    --export-block-rows 8192 --prompt T4.ids --out /tmp/http-digest --generate 4 \
    -- build/strix-hip-decode-token MODEL.gguf PINNED_MODEL_AND_MTP_ARGS
  python3 tests/http_slots_gpu_gate.py --window-state-arm live-continuation \
    --export-block-rows 8192 --prompt T4.ids --out /tmp/http-live --generate 4 \
    -- build/strix-hip-decode-token MODEL.gguf PINNED_MODEL_AND_MTP_ARGS
  python3 tests/http_slots_gpu_gate.py --window-state-arm history-exhaustion \
    --export-block-rows 8192 --prompt T12.ids --out /tmp/http-history --generate 4 \
    -- build/strix-hip-decode-token MODEL.gguf PINNED_MODEL_AND_MTP_ARGS
  python3 tests/http_slots_gpu_gate.py --export-roundtrip --export-block-rows 8192 \
    --suffix-token 42 --prompt T4.ids --out /tmp/http-export-gate --generate 4 \
    -- build/strix-hip-decode-token MODEL.gguf PINNED_MODEL_AND_MTP_ARGS

Use the same qualified dense-A8 profile, context, T and 2/4/8 prefix-window owners
in PINNED_MODEL_AND_MTP_ARGS, including --mtp-target-window-rows-set 2,4,8 and
all five positive qualified window budgets:
  --mtp-window-arena-bytes ARENA_BYTES --mtp-window-capture-bytes CAPTURE_BYTES
  --mtp-window-checkpoint-bytes CHECKPOINT_BYTES --mtp-window-total-bytes TOTAL_BYTES
  --mtp-window-history-intervals HISTORY_INTERVALS
No HTTP/import/cache flags. History replaces the pinned history intervals with 2;
the other arms and export roundtrip preserve the pinned budgets.
R8 fills the second lineage interval and R4 must fall back to four serial rows.
Before the fix those rows consume state at T and the later flush applies R8
after them: both the final target bytes and live continuation can diverge.
The intermediate-export control folds at T+8, exposing that ordering difference.
Digest-only has no export and ends at pending R4, so removing its prompt-digest
fold must fail even if decode subsequently folds correctly. Live/history disable
prompt digest and export; final-only diagnostics read state after generation.
Roundtrip preserves the saved-component and single-import continuation checks.
Use fresh output directories. Missing cold-window coverage is NO-WITNESS.
"""
import argparse
import concurrent.futures
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import signal
import shutil
import subprocess
import threading
import time
import urllib.request
from batch_service_gpu_gate import completion, require, tokens
from batch_service_gpu_gate import signature as full_signature


def signature(result):
    # Keep the general cross-route prompt-digest exclusion until GPU qualification
    # establishes equality for all prompt/predictor layouts. The digest-only and
    # export-roundtrip arms explicitly compare it for terminal R4 after materialization.
    sig = full_signature(result)
    sig.pop("prompt_state_digest")
    return sig


@contextmanager
def server(args, name, mode, overrides=None, extra=()):
    env = os.environ.copy()
    env.update(STRIX_PROMPT_STATE_DIGEST="1", STRIX_PREFIX_CACHE_DIGEST="1", STRIX_PREFIX_CACHE_DIGEST_COMPONENTS="1",
               STRIX_FINAL_STATE_DIGEST="0", STRIX_MTP_BATCH_LOG="1", STRIX_HTTP_EAGER_SLOTS="4")
    for key in ("STRIX_RTX_EXPORT_STATE", "STRIX_RTX_EXPORT_POSITION"):
        env.pop(key, None)
    if overrides:
        env.update(overrides)
    command = args.command + list(extra) + ["--http-serve", "--http-port", str(args.port)]
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


def prompt_receipts(args, name):
    return [json.loads(line.split("tiled prompt: ", 1)[1])
            for line in (args.out / (name + ".log")).read_text().splitlines()
            if line.startswith("tiled prompt: ")]


def state_components(module, path):
    header, offset = module.read_state_header(path)
    components = {}
    with path.open("rb") as f:
        def digest(at, size):
            f.seek(offset + at)
            h = hashlib.sha256()
            while size:
                data = f.read(min(size, 1 << 20))
                require(data, "truncated export: " + str(path))
                h.update(data)
                size -= len(data)
            return h.hexdigest()
        offsets, rows_end = module.row_offsets(header, int(header["position"]))
        for row, (at, count, bpr) in zip(header["rows"], offsets):
            components[row["id"]] = digest(at, count * bpr)
        for fixed in header["fixed"]:
            components[fixed["id"]] = digest(rows_end + int(fixed["offset"]), int(fixed["bytes"]))
    return header, components


def committed_signature(result):
    # No prompt read is required: final-only diagnostics run after live generation.
    s = result["strix"]
    require(s.get("token_ids") and s.get("final_state_components") and s.get("final_state_position"),
            "NO-REFERENCE: missing live IDs or final committed state")
    target = {k: v for k, v in s["final_state_components"].items() if not k.startswith("mtp.")}
    require({"recurrent_matrix", "gdn_convolution"} <= target.keys(), "NO-REFERENCE: missing recurrent/conv bytes")
    return dict(token_ids=s["token_ids"], final_state_position=s["final_state_position"], target_state=target)


def window_witness(args, name, prompt, rows, serial):
    receipts = prompt_receipts(args, name)
    require(len(receipts) == 1, "NO-WITNESS: expected one cold request: " + name)
    r = receipts[0]
    require(r["prompt_tokens"] == len(prompt) and r["restored_position"] == 0 and
            r["rows"] == args.export_block_rows and r["blocks"] == 1 and r["tail_blocks"] == 0 and
            r.get("window_rows") == rows and r["serial_rows"] == serial,
            "NO-WITNESS: wrong accepted-window/serial boundary: " + name)
    require("native-mtp-batch" in (args.out / (name + ".log")).read_text(), "NO-WITNESS: batch route missing")
    return r


def window_state_arm(args, prompts):
    arm = args.window_state_arm
    history = arm == "history-exhaustion"
    digest = arm == "digest-only"
    require(len(prompts) == 1, "window state arm requires exactly one prompt")
    prompt = prompts[0]
    require(args.export_block_rows > 0 and len(prompt) == args.export_block_rows + (12 if history else 4),
            "window prompt must contain exactly T+12 (history) or T+4 tokens")
    require(1 < args.generate <= 8, "window state arms need a small live continuation: --generate 2..8")
    require(not any(a.startswith(("--prefix-cache", "--serving-prefix", "--rtx-state-import", "--http-")) for a in args.command),
            "window state command must omit cache/import/HTTP options")
    owners = [a.split("=", 1)[1] if "=" in a else args.command[index + 1]
              for index, a in enumerate(args.command) if a.split("=", 1)[0] == "--mtp-target-window-rows-set"]
    require(owners == ["2,4,8"], "window state arms require --mtp-target-window-rows-set 2,4,8")
    if history:
        args = argparse.Namespace(**vars(args))
        args.command = list(args.command)
        budget = "--mtp-window-history-intervals"
        require(any(a.split("=", 1)[0] == budget for a in args.command),
                "history arm requires a pinned --mtp-window-history-intervals budget")
        for index, arg in enumerate(args.command):
            if arg == budget:
                require(index + 1 < len(args.command), "missing history interval budget value")
                args.command[index + 1] = "2"
            elif arg.startswith(budget + "="):
                args.command[index] = budget + "=2"
    common = dict(STRIX_PROMPT_WINDOW="1", STRIX_PREFIX_CACHE_DIGEST="1" if digest else "0",
                  STRIX_FINAL_STATE_DIGEST="1", STRIX_HTTP_EAGER_SLOTS="1")
    results = {}
    # Each producer continues in the very same request. No import/replay can hide
    # pending-window bookkeeping, and final observation cannot repair earlier rows.
    modes = [("single", "0"), ("batch", "1")]
    if history:
        modes.append(("intermediate-export", "1"))
    export_path = (args.out / "intermediate.state").resolve()
    require(not export_path.exists(), "use a fresh output directory: intermediate export exists")
    for mode, batch in modes:
        name = arm + "-" + mode
        env = dict(common)
        if mode == "intermediate-export":
            env.update(STRIX_RTX_EXPORT_STATE=str(export_path),
                       STRIX_RTX_EXPORT_POSITION=str(args.export_block_rows + 8))
        with server(args, name, batch, env):
            result = completion(args.port, prompt, args.generate)
        results[mode] = result
        (args.out / (name + ".json")).write_text(json.dumps(result, indent=2))
        require(len(result["strix"]["token_ids"]) == args.generate, "NO-WITNESS: continuation ended early")
        if not digest:
            require(not result["strix"].get("prompt_state_digest"), "prompt observation unexpectedly enabled")
        if batch == "1":
            r = window_witness(args, name, prompt, 8 if history else 4, 4 if history else 0)
            require(("state_digest" in r) == digest, "unexpected prompt digest witness")
        if digest:
            # Full signature includes the prompt digest. Final-only equality alone
            # would miss a removed digest materialization followed by a good decode fold.
            full_signature(result)
    reference = committed_signature(results["single"])
    for mode, result in results.items():
        require(committed_signature(result) == reference, "MISMATCH: live IDs/final target bytes: " + mode)
        if digest:
            require(full_signature(result) == full_signature(results["single"]), "MISMATCH: digest-only prompt state")
    if history:
        spec = importlib.util.spec_from_file_location("rtx_state_import", Path(__file__).resolve().parents[1] / "tools/rtx_state_import.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        header, _ = module.read_state_header(export_path)
        require(header["position"] == args.export_block_rows + 8, "NO-WITNESS: intermediate T+8 export missing")
    print("EXACT: " + arm + ": live IDs and every final committed target component match single")


def export_roundtrip(args, prompts):
    require(len(prompts) == 1, "export gate requires exactly one prompt")
    prompt = prompts[0]
    require(args.export_block_rows > 0 and len(prompt) == args.export_block_rows + 4,
            "export prompt must contain exactly T+4 tokens")
    require(args.suffix_token is not None and args.suffix_token >= 0, "--suffix-token must be a valid model token")
    require(args.generate > 1, "export gate needs --generate greater than one")
    require(not any(a.startswith(("--prefix-cache", "--serving-prefix", "--rtx-state-import", "--http-")) for a in args.command),
            "export gate command must omit cache/import/HTTP options")
    spec = importlib.util.spec_from_file_location("rtx_state_import", Path(__file__).resolve().parents[1] / "tools/rtx_state_import.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = {mode: (args.out / (mode + ".state")).resolve() for mode in ("batch", "single")}
    require(all(not p.exists() for p in paths.values()), "use a fresh output directory: export file exists")
    results = {}
    common = {"STRIX_PROMPT_WINDOW": "1", "STRIX_HTTP_EAGER_SLOTS": "1"}
    extended = prompt + [args.suffix_token]
    for mode, batch in (("batch", "1"), ("single", "0")):
        name = "export-" + mode
        env = dict(common, STRIX_RTX_EXPORT_STATE=str(paths[mode]), STRIX_RTX_EXPORT_POSITION=str(len(prompt)))
        with server(args, name, batch, env):
            results[mode] = completion(args.port, prompt, 1)
            if mode == "single":
                # Export is pinned at P: Q's serial suffix can overwrite it, so
                # preserve the P snapshot before asking for the references.
                saved = args.out / "single-prompt.state"
                shutil.copyfile(paths[mode], saved)
                reference = {n: completion(args.port, extended, n) for n in (1, args.generate)}
        (args.out / (name + ".json")).write_text(json.dumps(results[mode], indent=2))
        receipts = prompt_receipts(args, name)
        require(receipts and receipts[0]["prompt_tokens"] == len(prompt) and receipts[0]["restored_position"] == 0,
                "NO-WITNESS: cold complete prompt missing")
        if mode == "batch":
            r = receipts[0]
            require(r["rows"] == args.export_block_rows and r["blocks"] == 1 and r["tail_blocks"] == 0
                    and r.get("window_rows") == 4, "NO-WITNESS: T block plus terminal R4 required")
            require("native-mtp-batch" in (args.out / (name + ".log")).read_text(), "NO-WITNESS: batch route missing")
    require(full_signature(results["batch"]) == full_signature(results["single"]), "MISMATCH: prompt digest/IDs/target state")
    batch_header, batch_components = state_components(module, paths["batch"])
    single_header, single_components = state_components(module, saved)
    for key in ("position", "tokens", "rows", "fixed", "layout_identity", "geometry_identity", "producer"):
        require(batch_header[key] == single_header[key], "MISMATCH: export " + key)
    for key in ("next_position", "committed_tokens", "valid", "generations"):
        require(batch_header["session"][key] == single_header["session"][key], "MISMATCH: export session " + key)
    require(batch_header["position"] == len(prompt), "NO-WITNESS: prompt-end export missing")
    require({"recurrent_matrix", "gdn_convolution", "mtp.kv", "mtp.index", "mtp.held"} <= batch_components.keys(),
            "NO-WITNESS: incomplete exported component registry")
    require(batch_components == single_components, "MISMATCH: saved component bytes (including recurrent/conv and mtp.*)")
    imports = {}
    with server(args, "import-single", "0", common, ("--rtx-state-import",)):
        client = module.Server(f"http://127.0.0.1:{args.port}", timeout=1800)
        for count in (1, args.generate):
            report, _ = module.upload(client, paths["batch"], 16 << 20, canonical=True)
            require(report["generations"]["match"], "MISMATCH: imported generations")
            require(report["canonical_matches_file"] and report["commit"]["parked"] and
                    report["commit"]["position"] == len(prompt), "MISMATCH: verified import payload")
            result = completion(args.port, extended, count)
            require(full_signature(result) == full_signature(reference[count]), "MISMATCH: imported continuation")
            if count == 1:
                require(result["strix"]["final_state_components"] == reference[count]["strix"]["final_state_components"],
                        "MISMATCH: canonical one-token predictor state")
            imports[count] = dict(upload=report, result=result)
        client.conn.close()
    receipts = prompt_receipts(args, "import-single")
    require(len(receipts) == 2 and all(r["restored_position"] == len(prompt) and
            r.get("rtx_import", {}).get("sha_ok") and r["rtx_import"]["position"] == len(prompt) for r in receipts),
            "NO-WITNESS: verified prompt-end import was not used")
    (args.out / "export-roundtrip.json").write_text(json.dumps(dict(producer=results, reference=reference,
        components=batch_components, imports=imports), indent=2))
    print("EXACT: terminal R4 saved components, prompt digest, imported IDs and every committed target component")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    arms = parser.add_mutually_exclusive_group()
    arms.add_argument("--export-roundtrip", action="store_true", help="run only the opt-in terminal-R4 export/import gate")
    arms.add_argument("--window-state-arm", choices=("digest-only", "live-continuation", "history-exhaustion"))
    parser.add_argument("--export-block-rows", type=int, default=8192)
    parser.add_argument("--suffix-token", type=int)
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
    if args.export_roundtrip:
        export_roundtrip(args, prompts)
        return
    if args.window_state_arm:
        window_state_arm(args, prompts)
        return
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
    print("EXACT: lazy growth witnessed; complete IDs, final committed state and frontier match explicit single references")


if __name__ == "__main__":
    main()
