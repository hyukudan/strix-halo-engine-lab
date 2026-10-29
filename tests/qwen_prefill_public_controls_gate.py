#!/usr/bin/env python3
"""Native OFF-restoration gate; requires a GPU, model assets and prior plain-FP16 evidence.

Run with --reference-logits previous.bin --reference-state previous-state/manifest.json
--output fresh-private-dir --logits-command teacher-command.json --state-command generation-command.json.
Each command file contains a JSON argv array (executable, model, token IDs, qualified
profile and geometry flags). The logits command uses --teacher-forced-score --generate 1
and all-rows tiled output; the state command uses eager generation with terminal-row
output. Capture prior plain-FP16 references with those same commands and geometry.
The gate adds artifact destinations and startup restrictions, keeping teacher forcing
and final-state exports in separate processes as required by their public contracts.
No timing claims; outputs contain private qualification data.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import struct


def digest(path):
    with path.open("rb") as stream:
        value = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
        return value.hexdigest()


def session_signature(path):
    """Public-main-compatible live-state path; exclude versioned identity metadata.

    Checks the envelope root and every section before comparing all committed
    KV/index rows and all five complete fixed components. Unused cache capacity
    is outside this scope; --state-format final validates full allocations.
    """
    with path.open("rb") as stream:
        total = path.stat().st_size
        if total < 32:
            raise ValueError("truncated session envelope")
        root = hashlib.sha256()
        remaining = total - 32
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("truncated session envelope")
            root.update(chunk)
            remaining -= len(chunk)
        if root.digest() != stream.read(32):
            raise ValueError("session envelope root mismatch")
        stream.seek(0)
        def take(size):
            if size > total - 32 - stream.tell():
                raise ValueError("session field exceeds envelope")
            value = stream.read(size)
            if len(value) != size:
                raise ValueError("truncated session field")
            return value
        def u64():
            return struct.unpack("<Q", take(8))[0]
        def string():
            return take(u64())
        if string() != b"strix-session-v1" or u64() != 1:
            raise ValueError("unsupported session schema")
        model = take(32).hex()
        take(32)  # executable identity may differ
        string()  # compiled configuration identity may differ
        if u64() != 0:
            raise ValueError("CLI state session must be target-only")
        header_end = stream.tell()
        expected_compat = take(32)
        stream.seek(0)
        if hashlib.sha256(take(header_end)).digest() != expected_compat:
            raise ValueError("session compatibility digest mismatch")
        stream.seek(header_end + 32)
        frontier = [u64() for _ in range(5)]
        sampling = string().decode()
        tokens = [u64() for _ in range(u64())]
        if frontier[0] != len(tokens) or frontier[2] != frontier[0] or sampling != "greedy":
            raise ValueError("session frontier incomplete")
        sections = {}
        for _ in range(u64()):
            name = string().decode()
            if name in sections or u64() != 0 or u64() != 0:
                raise ValueError("duplicate section or unsupported rank")
            size = u64()
            expected = take(32)
            if u64() != stream.tell():
                raise ValueError("session section offset mismatch")
            value = hashlib.sha256()
            remaining = size
            while remaining:
                chunk = take(min(1024 * 1024, remaining))
                value.update(chunk)
                remaining -= len(chunk)
            if value.digest() != expected:
                raise ValueError("session section digest mismatch")
            sections[name] = (size, expected.hex())
        if stream.tell() != total - 32:
            raise ValueError("session trailing data")
    sections.pop("session.metadata")  # execution/lineage identities, not numerical payload
    fixed = {"recurrent_matrix", "gdn_convolution", "ple_convolution", "ngram_history", "position_history"}
    if not fixed.issubset(sections) or not any(k.startswith("kv.L") for k in sections) or not any(k.startswith("index.L") for k in sections):
        raise ValueError("NO-REFERENCE: missing one of seven state components")
    return {"model": model, "frontier": frontier, "tokens": tokens, "sampling": sampling}, sections


def state_signature(manifest_path):
    if manifest_path.name == "session.bin":
        return session_signature(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != "strix-final-generation-state-v1":
        raise ValueError("requires a complete native final-generation-state export")
    components = {}
    for component in manifest["components"]:
        path = manifest_path.parent / component["path"]
        if path.stat().st_size != component["bytes"] or digest(path) != component["sha256"]:
            raise ValueError("state payload does not match its manifest")
        components[component["component"]] = (component["bytes"], component["sha256"])
    if len(components) != 7:
        raise ValueError("reference must contain all seven state components")
    fields = ("model_content_identity", "prefix_tokens", "generated_tokens",
              "consumed_input_tokens", "context_capacity", "bonus_pending", "stop_reason")
    return {key: manifest[key] for key in fields}, components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-format", choices=("final", "session"), default="final")
    parser.add_argument("--arm", choices=("off", "on", "public-main"), default="off")
    parser.add_argument("--require-hc-witness", action="store_true", help="require actual OUT16 launch in both processes")
    parser.add_argument("--capture-only", action="store_true", help="record a reference; this is not an exactness verdict")
    parser.add_argument("--reference-logits", type=Path)
    parser.add_argument("--reference-state", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    # Omit --logits-command for long prompts: full-vocabulary teacher logits of every row exceed the
    # engine's teacher-spool bound (about 32 GB at 32K). The committed state then carries the evidence.
    parser.add_argument("--logits-command", type=Path)
    parser.add_argument("--state-command", type=Path, required=True)
    args = parser.parse_args()
    commands = {"state": json.loads(args.state_command.read_text())}
    if args.logits_command:
        commands = {"logits": json.loads(args.logits_command.read_text()), **commands}
    for kind, command in commands.items():
        if not isinstance(command, list) or not command or not all(isinstance(value, str) for value in command):
            parser.error("command files must contain nonempty JSON argv arrays")
        if ("--teacher-forced-score" in command) != (kind == "logits"):
            parser.error("only the logits command must enable teacher-forced scoring")
        if any(value.startswith(("--prefill-disable", "--prefill-exact-fp16", "--config",
                                 "--teacher-logits-file", "--session-save"))
               or value.split("=", 1)[0] == "--validation-final-state" for value in command):
            parser.error("base commands must leave controls and artifact destinations to the gate")
    if not args.capture_only and (not args.reference_state or bool(args.reference_logits) != bool(args.logits_command)):
        parser.error("comparison requires the seven-component state reference, plus a logits reference exactly when logits are captured")
    checkout = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(checkout):
        parser.error("qualification outputs must stay outside the checkout")
    reference_logits = None if args.capture_only or not args.reference_logits else digest(args.reference_logits)
    reference_state = None if args.capture_only else state_signature(args.reference_state)
    args.output.mkdir(parents=False, exist_ok=False)
    env = os.environ.copy()
    for key in ("STRIX_PREFILL_EXACT_FP16", "STRIX_PREFILL_HC_OUTPUT_WMMA"):
        env.pop(key, None)
    env["STRIX_HCW_OUTPUT"] = "1"
    cases = [
        ("cli-individual", ["--prefill-disable", "hc_output_wmma"], {}),
        ("cli-master", ["--prefill-exact-fp16"], {}),
        ("env-individual", [], {"STRIX_PREFILL_HC_OUTPUT_WMMA": "0"}),
        ("env-master", [], {"STRIX_PREFILL_EXACT_FP16": "1"}),
    ]
    for name, body in (("config-individual", 'prefill_disable = ["hc_output_wmma"]\n'),
                       ("config-master", "prefill_exact_fp16 = true\n")):
        config = args.output.resolve() / (name + ".conf")
        config.write_text(body)
        cases.append((name, ["--config", str(config)], {}))
    # Repeat each control with the legacy selector requesting FP16 WMMA output.
    if args.arm != "off":
        cases = [(args.arm, [], {})]
    for name, flags, overrides in cases:
        for legacy in (("1", "2") if args.arm == "off" else ("1",)):
            directory = args.output.resolve() / (name + "-legacy" + legacy)
            directory.mkdir()
            logits = directory / "logits.bin"
            state = directory / "state"
            for kind, command in commands.items():
                destination = (["--teacher-logits-file", str(logits)] if kind == "logits" else
                               ["--session-save" if args.state_format == "session" else "--validation-final-state", str(state)])
                with (directory / (kind + ".json")).open("wb") as stdout, (directory / (kind + ".log")).open("wb") as stderr:
                    subprocess.run(command + flags + destination,
                                   env={**env, **overrides, "STRIX_HCW_OUTPUT": legacy},
                                   stdout=stdout, stderr=stderr, check=True)
                receipt = json.loads((directory / (kind + ".json")).read_text())
                if args.require_hc_witness and "qpf2k HC out16 active" not in (directory / (kind + ".log")).read_text():
                    raise ValueError("NO-WITNESS: HC output WMMA did not execute")
                if args.arm == "off" and (receipt["prefill_identity"] != "fp16-v1" or receipt["prefill_options_active"]
                        or receipt["prefill_identity_hash"] != "a0094634843e16ee"):
                    raise ValueError("OFF receipt did not report plain FP16")
                if args.arm != "public-main" and kind == "logits" and not receipt.get("teacher_forced_tiled_input_rows", 0):
                    raise ValueError("run did not exercise tiled prefill")
                if args.arm != "public-main" and kind == "state" and "logits" not in commands and \
                        not receipt.get("known_prefix_tiled_prefill"):
                    raise ValueError("state-only run did not exercise tiled prefill")
                if args.arm == "on" and (receipt.get("prefill_identity") != "fp16-v1+hcout16" or
                        receipt.get("prefill_options_active") != ["hc_output_wmma"]):
                    raise ValueError("ON receipt did not report HC output WMMA")
            actual = {"logits_sha256": digest(logits) if "logits" in commands else None, "state": state_signature(state / ("session.bin" if args.state_format == "session" else "manifest.json"))}
            (directory / "signature.json").write_text(json.dumps(actual, indent=2) + "\n")
            if not args.capture_only and (actual["logits_sha256"] != reference_logits or actual["state"] != reference_state):
                raise ValueError(args.arm + " logits/state differ from reference: " + directory.name)
            print(directory.name + (": CAPTURED (no equality verdict)" if args.capture_only else ": EXACT"))


if __name__ == "__main__":
    main()
