#!/usr/bin/env python3
"""Bounded host-only preparation and numerical evidence evaluation. Never runs HIP."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time

SCHEMA = "strix-native-numerical-campaign-v1"
ROOT = Path(__file__).resolve().parents[3]
MAX_JSON = 16 * 1024 * 1024
MAX_PLAN_JSON = 256 * 1024 * 1024
MAX_LINE = 8 * 1024 * 1024
MAX_LOGITS_FILE = 8 * 1024**3
MAX_MODEL = 512 * 1024**3
MAX_POSITIONS = 4096
MAX_VOCAB = 262144
STATE_CLASSES = ("kv", "recurrent", "convolution", "host_ngram", "sparse_index", "position")
FRESH_RAW_HISTORY = "single_row_eager_from_fresh_initialized_state"
FRESH_GRAPH_HISTORY = "single_row_graph_from_fresh_initialized_state"
KINDS = {"logits", "greedy", "json", "tool", "state", "context"}
THRESHOLDS = {"max_abs_logit_error": 0.05, "max_mean_kl": 0.001,
              "max_nll_increase": 0.02, "max_teacher_forced_argmax_mismatch_rate": 0.01,
              "max_generated_token_edit_rate": 0.01,
              "semantic_failures": 0, "state_byte_mismatches": 0}
ROW_CONTRACT = ("strix-row-quant-v1:f64:nearest-even:divide:no-ftz:finite-only:"
                "symmetric-127-or-7:zero-scale-one:exact-groups:ordered-groups")


def floating_profile(input_storage):
    require(input_storage in {"bf16", "fp32"}, "unknown floating activation storage")
    return (f"strix-experimental-float-lm-head-v1:decode:C1:M1:{input_storage}-source-widen-f32:"
            "original-ggml-weight-decode-f32:256-thread-feature-stride:fp32-sum-binary-tree:"
            "fp32-logits:compiler-bound-arithmetic:no-whole-model-precision-claim")


def integer_profile(candidate, input_storage):
    require(candidate["activation_class"] in {"W4A8", "W4A4"} and input_storage in {"bf16", "fp32"}
            and candidate["encoding"] in {"q4_0", "iq4_xs"} and candidate["operand"] in {"iu4", "iu8"}
            and (candidate["operand"] != "iu4" or candidate["encoding"] == "q4_0" and candidate["activation_class"] == "W4A4"),
            "integer profile needs explicit source storage and activation encoding")
    activation = "symmetric_a8" if candidate["activation_class"] == "W4A8" else "symmetric_a4"
    fnv = 14695981039346656037
    for byte in ROW_CONTRACT.encode() + b"\0" + activation.encode() + b"\0" + struct.pack("<Q", 32):
        fnv = ((fnv ^ byte) * 1099511628211) & ((1 << 64) - 1)
    code = "q4_0-u4-z8" if candidate["encoding"] == "q4_0" else "iq4_xs-codebook-i8"
    core = (f"strix-integer-projection-v1:{code}:{activation}:{candidate['operand']}:"
            f"gfx1151-w32-g32-i32-no-clamp-i64-correction-f64-scale-left-to-right-no-contract:{fnv}")
    return (f"strix-experimental-lm-head-v1:decode:C1:M1:{input_storage}-exact-widen-f64:{core}"
            ":f64-group-order:binary32-integer-RNE-gradual-signed-zero:reject-nonfinite:whole-logits-status-gated-commit")


class Invalid(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise Invalid(message)


def keys(value, required, optional=()):
    require(isinstance(value, dict) and set(required) <= value.keys()
            and value.keys() <= set(required) | set(optional), "unknown or missing object fields")


def integer(value, low=0, high=(1 << 64) - 1):
    require(type(value) is int and low <= value <= high, "integer is outside declared bounds")
    return value


def finite(value, low=None):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and (low is None or value >= low)
    except (OverflowError, ValueError):
        valid = False
    require(valid, "nonfinite or invalid numerical value")
    return float(value)


def digest(value):
    require(isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), "expected lowercase SHA256")
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def text(value, maximum=65536):
    require(isinstance(value, str) and 0 < len(value.encode("utf-8")) <= maximum,
            "empty or excessive text")
    return value


def private_path(value):
    require(isinstance(value, (str, Path)), "artifact path must be text")
    path = Path(value)
    require(path.is_absolute(), "artifact paths must be absolute")
    path = path.resolve()
    require(path != ROOT and ROOT not in path.parents
            and not any((p / ".git").exists() for p in (path, *path.parents)),
            "model, prompts, references and results must remain outside Git/source checkouts")
    return path


def file_hash(path, maximum):
    path = Path(path)
    before = path.stat()
    require(path.is_file() and 0 < before.st_size <= maximum, "missing, empty or excessive artifact")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            result.update(block)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "artifact changed while hashing")
    return result.hexdigest(), after.st_size


def pin(value, maximum=MAX_JSON, private=True):
    path = private_path(value) if private else Path(value).resolve(strict=True)
    sha, size = file_hash(path, maximum)
    return {"path": str(path), "sha256": sha, "bytes": size}


def verify_pin(value, maximum=MAX_JSON, private=True):
    keys(value, ("path", "sha256", "bytes"))
    digest(value["sha256"]); integer(value["bytes"], 1, maximum)
    actual = pin(value["path"], maximum, private)
    require(actual == value, "artifact identity/extent mismatch")
    return Path(actual["path"])


def parse_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    def reject(_):
        raise Invalid("nonfinite JSON constant")
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise Invalid("invalid bounded UTF8 JSON") from error


def load(path, maximum=MAX_JSON):
    path = Path(path)
    require(path.stat().st_size <= maximum, "JSON exceeds read bound")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(0 < len(raw) <= maximum, "empty or excessive JSON")
    return parse_json(raw)


def load_plan(path):
    return load(path, MAX_PLAN_JSON)


def write_new(value, path, maximum=MAX_JSON):
    path = private_path(path)
    encoded = canonical(value) + b"\n"
    require(len(encoded) <= maximum, "JSON artifact exceeds explicit byte admission")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded)


def source_manifest(value):
    data = load(verify_pin(value))
    keys(data, ("schema", "files"))
    require(data["schema"] == "strix-source-manifest-v1" and isinstance(data["files"], list)
            and 1 <= len(data["files"]) <= 512, "invalid complete source identity manifest")
    paths = set()
    for item in data["files"]:
        path = verify_pin(item, MAX_LOGITS_FILE, private=False)
        require(str(path) not in paths, "duplicate source identity path")
        paths.add(str(path))
    return data


def run_host_tool(tool, model, output, expected, timeout=120, output_limit=32 * 1024 * 1024):
    """Only the project's pinned, single-process host auditors; no shell or GPU runner."""
    require(sys.platform.startswith("linux"), "host auditing runs on Linux/WSL with native host tools")
    before = pin(tool, MAX_LOGITS_FILE)
    require(before == expected, "auditor changed after preflight")
    with tempfile.TemporaryFile() as errors, Path(output).open("xb") as out:
        process = subprocess.Popen([str(tool), str(model)], stdout=out, stderr=errors,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        end = time.monotonic() + timeout
        try:
            while process.poll() is None:
                require(time.monotonic() < end, "host auditor timeout")
                require(out.tell() <= output_limit and errors.tell() <= output_limit,
                        "host auditor output exceeds bound")
                time.sleep(0.02)
            require(process.returncode == 0, "host tensor auditor rejected model")
            require(out.tell() <= output_limit and errors.tell() <= output_limit,
                    "host auditor output exceeds bound")
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    require(pin(tool, MAX_LOGITS_FILE) == before, "host auditor bytes changed")
    return before


def audit_model(model, expected_sha, manifest_tool, manifest_sha, audit_tool, audit_sha, output):
    model_pin = pin(model, MAX_MODEL)
    require(model_pin["sha256"] == digest(expected_sha), "GGUF content hash differs from explicit selection")
    with Path(model_pin["path"]).open("rb") as stream:
        require(stream.read(8) == b"GGUF\x03\x00\x00\x00", "requires actual GGUF v3")
    tools = [(manifest_tool, manifest_sha), (audit_tool, audit_sha)]
    expected_tools = []
    for (path, sha), name in zip(tools, ("strix-model-manifest", "strix-audit-execution-tensors")):
        observed_tool = pin(path, MAX_LOGITS_FILE)
        require(Path(observed_tool["path"]).name == name and observed_tool["sha256"] == digest(sha), "host tool name/hash mismatch")
        expected_tools.append(observed_tool)
    output = private_path(output)
    require(not output.exists(), "audit output directory must be fresh")
    output.mkdir(parents=True)
    tool_pins = []
    for (tool, _), name, expected in zip(tools, ("manifest.json", "tensor_audit.json"), expected_tools):
        tool_pins.append(run_host_tool(tool, Path(model_pin["path"]), output / name, expected))
    require(pin(model, MAX_MODEL) == model_pin, "GGUF changed during audit")
    receipt = {"schema": SCHEMA, "kind": "host_tensor_audit", "model": model_pin,
               "tools": tool_pins, "manifest": pin(output / "manifest.json"),
               "tensor_audit": pin(output / "tensor_audit.json"), "native_execution": False}
    verify_audit(receipt)
    write_new(receipt, output / "audit_receipt.json")
    return receipt


def verify_audit(receipt):
    keys(receipt, ("schema", "kind", "model", "tools", "manifest", "tensor_audit", "native_execution"))
    require(receipt["schema"] == SCHEMA and receipt["kind"] == "host_tensor_audit"
            and receipt["native_execution"] is False, "host audit cannot claim native execution")
    verify_pin(receipt["model"], MAX_MODEL)
    require(isinstance(receipt["tools"], list) and len(receipt["tools"]) == 2, "requires both host audit tools")
    for tool in receipt["tools"]:
        verify_pin(tool, MAX_LOGITS_FILE)
    manifest = load(verify_pin(receipt["manifest"]))
    audit = load(verify_pin(receipt["tensor_audit"]))
    keys(manifest, ("architecture", "layout_identity", "total_tensor_bytes", "file_backed_candidate_bytes",
                    "bounded_cache_candidate_bytes", "unknown_tensor_count", "tensors"))
    keys(audit, ("manifest_layout_identity", "requirements", "satisfied", "missing", "ambiguous",
                 "compiled", "tensor_access_plan_identity"))
    require(audit["compiled"] is True and integer(audit["missing"]) == 0 and integer(audit["ambiguous"]) == 0
            and integer(audit["satisfied"], 1) == len(audit["requirements"])
            and integer(audit["tensor_access_plan_identity"], 1) > 0
            and integer(manifest["layout_identity"], 1) == audit["manifest_layout_identity"],
            "tensor audit is incomplete, ambiguous or belongs to another layout")
    require(isinstance(manifest["tensors"], list) and 1 <= len(manifest["tensors"]) <= 100000,
            "invalid tensor inventory")
    require(integer(manifest["unknown_tensor_count"]) == 0, "unclassified model tensors require explicit loader audit before campaign selection")
    names = set()
    tensor_bytes = 0
    for tensor in manifest["tensors"]:
        keys(tensor, ("name", "role", "residency", "placement", "layer", "ggml_type", "byte_size",
                      "absolute_offset", "dimensions"))
        text(tensor["name"]); require(tensor["name"] not in names, "duplicate tensor name")
        names.add(tensor["name"]); integer(tensor["ggml_type"], 0, 255)
        require(isinstance(tensor["dimensions"], list) and 1 <= len(tensor["dimensions"]) <= 4,
                "invalid tensor rank")
        for dim in tensor["dimensions"]:
            integer(dim, 1, (1 << 32) - 1)
        offset = integer(tensor["absolute_offset"]); size = integer(tensor["byte_size"], 1)
        tensor_bytes += size
        require(offset <= receipt["model"]["bytes"] and size <= receipt["model"]["bytes"] - offset,
                "tensor range exceeds actual GGUF")
    require(integer(manifest["total_tensor_bytes"], 1) == tensor_bytes, "aggregate tensor extent mismatch")
    integer(manifest["file_backed_candidate_bytes"], 0, tensor_bytes)
    integer(manifest["bounded_cache_candidate_bytes"], 0, tensor_bytes)
    for row in audit["requirements"]:
        keys(row, ("step_id", "op", "slot", "cardinality", "role", "layer", "name_contains",
                   "match_count", "status", "tensor_indices"))
        require(row["status"] == "satisfied" and integer(row["match_count"], 1) == len(row["tensor_indices"])
                and (row["cardinality"] == "one_or_more" or row["cardinality"] == "exactly_one" and row["match_count"] == 1),
                "native tensor requirement did not pass")
        for index in row["tensor_indices"]:
            integer(index, 0, len(manifest["tensors"]) - 1)
    return manifest, audit


def validate_prompts(data):
    keys(data, ("schema", "cases"))
    require(data["schema"] == "strix-numerical-prompts-v1" and isinstance(data["cases"], list)
            and 1 <= len(data["cases"]) <= 64, "invalid bounded prompt set")
    names = set()
    for case in data["cases"]:
        keys(case, ("id", "kind", "text", "context_min_tokens", "max_new_tokens", "expected"))
        name = text(case["id"], 128); require(name not in names and case["kind"] in KINDS, "duplicate/unknown prompt case")
        names.add(name); text(case["text"],1024*1024)
        integer(case["context_min_tokens"], 1, 32768); integer(case["max_new_tokens"], 1, 256)
        require(case["expected"] is None if case["kind"] not in {"json", "tool", "context"}
                else isinstance(case["expected"], dict), "JSON/tool/context cases need semantic expected object")
        canonical(case["expected"])
    require(KINDS <= {c["kind"] for c in data["cases"]}, "prompts must cover all six numerical/task/state/context gates")
    return data


def candidate_matrix(manifest):
    result = []
    for index, tensor in enumerate(manifest["tensors"]):
        if len(tensor["dimensions"]) not in (2, 3):
            continue
        k, n = tensor["dimensions"][:2]
        if tensor["role"] not in {"output_head", "attention_projection", "dense_projection", "moe_expert", "sparse_index", "mtp"}:
            continue
        experts = tensor["dimensions"][2] if len(tensor["dimensions"]) == 3 else 1
        storage = {0: ("f32", 1, 4), 1: ("f16", 1, 2), 2: ("q4_0", 32, 18),
                   8: ("q8_0", 32, 34), 20: ("iq4_nl", 32, 18),
                   23: ("iq4_xs", 256, 136), 30: ("bf16", 1, 2)}.get(tensor["ggml_type"])
        encoding, block, block_bytes = storage or ("unsupported_storage", 1, 0)
        storage_valid = storage is not None and k % block == 0 and tensor["byte_size"] == experts * n * (k // block) * block_bytes
        for phase, rows_list in (("decode", (1,)), ("verify", (2, 4, 8, 16)), ("prefill", (16, 64, 512))):
            for rows in rows_list:
                for activations in ("W4A16", "BF16_activation_control", "FP32_activation_control", "W4A8", "W4A4"):
                    is_integer = activations in {"W4A8", "W4A4"}
                    is_control = activations in {"BF16_activation_control", "FP32_activation_control"}
                    valid = storage_valid and (tensor["ggml_type"] in {0, 1, 8, 20, 23, 30} if is_control else encoding in {"q4_0", "iq4_xs"})
                    instruction = "floating_control_unbound" if not is_integer else (
                        "iu4" if encoding == "q4_0" and activations == "W4A4" else "iu8")
                    integrated = valid and activations != "W4A16" and tensor["role"] == "output_head" and phase == "decode" and rows == 1 and len(tensor["dimensions"]) == 2
                    tile_cols = min(1024, n); tiles = (n + 1023) // 1024
                    padded_m = (rows + 15) // 16 * 16; padded_n = (n + 15) // 16 * 16
                    costs = {"checkpoint_tensor_bytes": tensor["byte_size"], "storage_block_elements": block if valid else None,
                             "scale_group_elements": 32 if valid and not is_control else None,
                             "wmma_tile": [16, 16, 16] if is_integer else None,
                             "logical_mnk": [rows, n, k], "physical_mn": [padded_m, padded_n] if is_integer else None,
                             "useful_tile_fraction": (rows * n) / (padded_m * padded_n) if is_integer else None,
                             "output_bytes_fp32": rows * n * 4, "tile_columns": tile_cols if is_integer else None,
                             "column_tiles": tiles if is_integer else None,
                             "preparation_elements_per_complete_projection": rows * k * tiles if is_integer else 0,
                             "activation_code_storage": "one_u8_per_code_even_A4" if is_integer else "no_activation_quantization",
                             "staged_weight_bytes_per_largest_tile": tile_cols * (k // 2 + (k // 32) * 8) if valid and is_integer else None,
                             "group_accumulator_bytes_per_largest_tile": rows * tile_cols * (k // 32) * 4 if valid and is_integer else None,
                             "checkpoint_bytes_per_expert_projection": n * (k // block) * block_bytes if valid else None,
                             "expert_count": experts, "expert_rows": "unknown_requires_observed_histogram" if experts != 1 else rows,
                             "timing_categories": ["input_widen", "activation_prepare", "weight_repack", "projection", "scale_correction",
                                                   "output_conversion", "status_sync", "logits_commit", "complete_wall"],
                             "measured": "unknown"}
                    if valid and is_integer:
                        g = k // 32
                        core = rows * k * 9 + tile_cols * (k // 2 + g * 8) + rows * g * 12 + rows * tile_cols * (16 + g * 4) + 8
                        costs["requested_internal_peak_bytes"] = core + (rows * n * 4 + 4 if integrated else 0)
                        costs["internal_allocation_count"] = 12 if integrated else 10
                        costs["within_256MiB_requested_peak"] = costs["requested_internal_peak_bytes"] <= 256 * 1024**2
                    else:
                        costs["requested_internal_peak_bytes"] = None
                        costs["internal_allocation_count"] = None
                        costs["within_256MiB_requested_peak"] = None
                    costs["actual_retained_allocation_bytes"] = "unknown_requires_whole_owners_and_native_extent_probe"
                    costs["cost_model"] = "existing_C1_tiled_lm_head" if integrated and is_integer else (
                        "planned_column_tiles_diagnostic_only_no_model_dispatch" if is_integer else "floating_layout_unbound")
                    result.append({"id": f"t{index}:{phase}:M{rows}:{activations}", "tensor_index": index,
                                   "tensor_name": tensor["name"], "encoding": encoding, "ggml_type": tensor["ggml_type"], "phase": phase,
                                   "activation_class": activations, "operand": instruction, "rows": rows,
                                   "mathematically_representable": valid, "graph_integer_combination": "unsupported",
                                   "integration": ("experimental_C1_integer_lm_head_eager" if is_integer else
                                                   "floating_control_needs_exact_profile") if integrated else "projection_diagnostic_only",
                                   "reason": "exact audited storage blocks" if valid else "unsupported encoding or partial storage block",
                                   "costs": costs})
                    if encoding == "q4_0" and activations == "W4A4":
                        alternate = copy.deepcopy(result[-1])
                        alternate.update(id=alternate["id"]+":iu8",operand="iu8")
                        result.append(alternate)
                    require(len(result) <= 100000, "projection candidate inventory exceeds bound")
    require(0 < len(result) <= 100000, "no bounded projection candidate inventory")
    return result


def prepare_plan(audit_receipt, prompts):
    audit_pin = pin(audit_receipt); manifest, _ = verify_audit(load(audit_receipt))
    prompt_pin = pin(prompts); cases = validate_prompts(load(prompts))["cases"]
    return {"schema": SCHEMA, "kind": "numerical_plan", "audit_receipt": audit_pin, "prompts": prompt_pin,
            "model_sha256": load(audit_receipt)["model"]["sha256"], "thresholds": dict(THRESHOLDS),
            "cases": cases, "candidates": candidate_matrix(manifest), "runtime": {"status": "unknown"},
            "references": {case["id"]: {"status": "unknown", "reason": "trusted model baseline not recorded"} for case in cases},
            "state_classes": list(STATE_CLASSES), "hardware_execution": "not_done", "qualification": False}


def validate_runtime(runtime):
    if runtime == {"status": "unknown"}:
        return False
    keys(runtime, ("status", "executable", "source_manifest", "mode", "candidate_id", "numerical_identity",
                   "kernel_plan_identity", "prefill_identity", "state_history_identity", "input_storage"))
    require(runtime["status"] == "recorded" and runtime["mode"] in {"eager_integer_lm_head", "eager_float", "graph_float"},
            "unsupported runtime path; integer graph composition is not implemented")
    verify_pin(runtime["executable"], MAX_LOGITS_FILE); source_manifest(runtime["source_manifest"])
    for name in ("numerical_identity", "kernel_plan_identity", "prefill_identity", "state_history_identity"):
        text(runtime[name], 4096)
    text(runtime["candidate_id"], 256)
    require(runtime["input_storage"] in {"bf16", "fp32"}, "unsupported runtime activation source storage")
    return True


def validate_plan(plan):
    keys(plan, ("schema", "kind", "audit_receipt", "prompts", "model_sha256", "thresholds", "cases", "candidates",
               "runtime", "references", "state_classes", "hardware_execution", "qualification"))
    require(plan["schema"] == SCHEMA and plan["kind"] == "numerical_plan" and plan["qualification"] is False
            and plan["hardware_execution"] == "not_done", "preparation must not manufacture native qualification")
    receipt = load(verify_pin(plan["audit_receipt"])); manifest, _ = verify_audit(receipt)
    require(plan["model_sha256"] == receipt["model"]["sha256"], "plan model identity mismatch")
    prompts = validate_prompts(load(verify_pin(plan["prompts"])))
    require(plan["cases"] == prompts["cases"] and plan["candidates"] == candidate_matrix(manifest)
            and plan["state_classes"] == list(STATE_CLASSES), "plan shape/prompt/state contract was substituted")
    keys(plan["thresholds"], THRESHOLDS)
    for name, number in plan["thresholds"].items():
        finite(number, 0)
    require(plan["thresholds"]["max_teacher_forced_argmax_mismatch_rate"] <= 1
            and plan["thresholds"]["max_generated_token_edit_rate"] <= 1 and plan["thresholds"]["semantic_failures"] == 0
            and plan["thresholds"]["state_byte_mismatches"] == 0, "semantic/state gates require exact results")
    require(set(plan["references"]) == {c["id"] for c in plan["cases"]}, "reference coverage mismatch")
    for reference in plan["references"].values():
        if reference.get("status") == "unknown":
            keys(reference, ("status", "reason")); text(reference["reason"])
        else:
            keys(reference, ("status", "artifact")); require(reference["status"] == "recorded", "unknown reference status")
            verify_pin(reference["artifact"])
    if validate_runtime(plan["runtime"]):
        runtime = plan["runtime"]
        candidate = next((c for c in plan["candidates"] if c["id"] == runtime["candidate_id"]), None)
        require(candidate is not None and candidate["mathematically_representable"] and candidate["phase"] == "decode"
                and candidate["rows"] == 1, "current model runtime cannot substitute prefill/verify projection diagnostics")
        integer_mode = runtime["mode"] == "eager_integer_lm_head"
        require(candidate["integration"] == ("experimental_C1_integer_lm_head_eager" if integer_mode else
                                              "floating_control_needs_exact_profile"), "runtime arithmetic/path mismatch")
        require(runtime["numerical_identity"] == integer_profile(candidate, runtime["input_storage"]) if integer_mode else
                candidate["activation_class"] == ("BF16_activation_control" if runtime["input_storage"] == "bf16" else "FP32_activation_control")
                and runtime["numerical_identity"] == floating_profile(runtime["input_storage"]),
                "runtime numerical profile disagrees with candidate storage/activation/operand contract")
    return plan


def semantic_equal(a, b):
    if type(a) in (int, float) and type(b) in (int, float):
        return math.isfinite(a) and math.isfinite(b) and a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(semantic_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(semantic_equal(x, y) for x, y in zip(a, b))
    return a == b


def logits_rows(artifact, vocab):
    path = verify_pin(artifact, MAX_LOGITS_FILE)
    with path.open("rb") as stream:
        position = 0
        while line := stream.readline(MAX_LINE + 1):
            require(len(line) <= MAX_LINE, "logit record exceeds transient read bound")
            row = parse_json(line)
            keys(row, ("position", "target_token", "logits"))
            require(integer(row["position"], 0, MAX_POSITIONS - 1) == position, "logit positions missing/reordered")
            integer(row["target_token"], 0, vocab - 1)
            require(isinstance(row["logits"], list) and len(row["logits"]) == vocab, "empty/truncated vocabulary logits")
            for value in row["logits"]:
                require(abs(finite(value)) <= float.fromhex("0x1.fffffep+127"), "logit exceeds finite binary32 output range")
            yield row
            position += 1
        require(position > 0, "empty logits cannot pass")


def logsumexp(values):
    largest = max(values)
    return largest + math.log(math.fsum(math.exp(x - largest) for x in values))


def compare_logits(reference, actual, vocab):
    import itertools
    count = mismatches = bits_mismatched = 0; maximum = 0.0; divergence = nll_delta = 0.0
    for a, b in itertools.zip_longest(logits_rows(reference, vocab), logits_rows(actual, vocab)):
        require(a is not None and b is not None and a["target_token"] == b["target_token"], "truncated/mismatched reference positions")
        x, y = a["logits"], b["logits"]
        zx, zy = logsumexp(x), logsumexp(y)
        maximum = max(maximum, max(abs(u - v) for u, v in zip(x, y)))
        bits_mismatched += sum(struct.pack("<f",u) != struct.pack("<f",v) for u,v in zip(x,y))
        # exp underflow contributes zero probability; logits stay finite.
        kl = math.fsum(math.exp(u - zx) * ((u - zx) - (v - zy)) for u, v in zip(x, y))
        divergence += max(0.0, kl)  # roundoff can make identical KL a tiny negative.
        token = a["target_token"]; nll_delta += (zy - y[token]) - (zx - x[token])
        mismatches += max(range(vocab), key=lambda i: x[i]) != max(range(vocab), key=lambda i: y[i])
        count += 1
    require(count > 0, "zero teacher-forced positions")
    metrics = {"max_abs_logit_error": maximum, "max_mean_kl": divergence / count,
               "max_nll_increase": nll_delta / count, "max_teacher_forced_argmax_mismatch_rate": mismatches / count,
               "teacher_forced_positions": count, "binary32_logit_bits_mismatched": bits_mismatched}
    for value in metrics.values():
        finite(value)
    return metrics


def generated_trace(artifact, tokens, vocab):
    count = 0
    for row in logits_rows(artifact, vocab):
        require(count < len(tokens) and row["target_token"] == tokens[count]
                and max(range(vocab), key=lambda i: row["logits"][i]) == tokens[count],
                "generated greedy token disagrees with actual generation logits/tie rule")
        count += 1
    require(count == len(tokens), "truncated generated decision trace")


def token_edit_rate(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1] / max(len(a), len(b))


def validate_input_token_evidence(artifact, model_sha256, case, input_tokens, vocab):
    data = load(verify_pin(artifact))
    try:
        from tensormeld.llamacpp_tokenization import validate_tokenization_receipt
        spec = validate_tokenization_receipt(data,require_observed=True).config
    except ImportError as error:
        raise Invalid("observed input-token evidence requires the pinned TensorMeld tokenizer adapter") from error
    except (ValueError,OSError) as error:
        raise Invalid("invalid observed input-token evidence") from error
    require(spec["model"]["sha256"] == model_sha256 and spec["prompt_contract_sha256"] == identity(case)
            and spec["prompt"]["sha256"] == hashlib.sha256(case["text"].encode("utf-8")).hexdigest()
            and spec["controls"]["vocabulary_size"] == vocab and data["tokens"] == input_tokens,
            "observed tokenizer belongs to another GGUF/rendered text/prefix/vocabulary")
    verify_pin(artifact)
    return data["result_sha256"]


def record(artifact, plan, case, vocab, candidate=False):
    data = load(verify_pin(artifact))
    keys(data, ("schema", "model_sha256", "prompt_sha256", "executable", "source_manifest", "numerical_identity",
                "kernel_plan_identity", "prefill_identity", "state_history_identity", "history_sha256", "input_tokens", "logits",
                "teacher_forced_target_tokens", "generation_logits", "output_tokens", "output_text", "state", "timings_ms", "capture_contract"),
         optional=("input_token_evidence",))
    require(data["schema"] == "strix-numerical-record-v1" and data["model_sha256"] == plan["model_sha256"]
            and data["prompt_sha256"] == identity(case), "record belongs to another model/prompt contract")
    verify_pin(data["executable"], MAX_LOGITS_FILE); source_manifest(data["source_manifest"])
    for name in ("numerical_identity", "kernel_plan_identity", "prefill_identity", "state_history_identity"):
        text(data[name], 4096)
    digest(data["history_sha256"])
    capture = data["capture_contract"]
    keys(capture, ("origin", "history_contract", "restoration_record"))
    text(capture["history_contract"], 4096)
    require(capture["origin"] in {"fresh", "restored"}, "unknown state capture origin")
    if capture["origin"] == "fresh":
        require(capture["restoration_record"] is None, "fresh capture cannot claim restoration evidence")
    else:
        require(capture["history_contract"] not in {FRESH_RAW_HISTORY,FRESH_GRAPH_HISTORY}, "fresh RAW history cannot become restored lineage")
        restored = load(verify_pin(capture["restoration_record"]))
        keys(restored, ("schema", "origin", "model_sha256", "executable_sha256", "source_manifest_sha256",
                       "history_sha256", "state_history_identity", "history_contract", "restore_log", "numerical_identity",
                       "kernel_plan_identity", "prefill_identity"))
        require(restored["schema"] == "strix-declared-state-restoration-v1" and restored["origin"] == "restored"
                and restored["model_sha256"] == data["model_sha256"]
                and restored["executable_sha256"] == data["executable"]["sha256"]
                and restored["source_manifest_sha256"] == data["source_manifest"]["sha256"]
                and restored["history_sha256"] == data["history_sha256"]
                and restored["state_history_identity"] == data["state_history_identity"]
                and all(restored[name] == data[name] for name in ("numerical_identity", "kernel_plan_identity", "prefill_identity"))
                and restored["history_contract"] == capture["history_contract"], "restoration declaration belongs to another producer/history")
        verify_pin(restored["restore_log"])
    require(isinstance(data["input_tokens"], list) and case["context_min_tokens"] <= len(data["input_tokens"]) <= 32768
            and isinstance(data["output_tokens"], list) and 1 <= len(data["output_tokens"]) <= case["max_new_tokens"],
            "empty/truncated or unrealistic context/token observations")
    for token in data["input_tokens"] + data["output_tokens"]:
        integer(token, 0, vocab - 1)
    if "input_token_evidence" in data:
        validate_input_token_evidence(data["input_token_evidence"],data["model_sha256"],case,data["input_tokens"],vocab)
    require(isinstance(data["teacher_forced_target_tokens"], list)
            and 1 <= len(data["teacher_forced_target_tokens"]) <= MAX_POSITIONS, "missing teacher-forced targets")
    for token in data["teacher_forced_target_tokens"]:
        integer(token, 0, vocab - 1)
    require(isinstance(data["output_text"], str) and len(data["output_text"].encode("utf-8")) <= 1024 * 1024,
            "excessive output text")
    verify_pin(data["logits"], MAX_LOGITS_FILE)
    targets = data["teacher_forced_target_tokens"]
    count = 0
    for row in logits_rows(data["logits"], vocab):
        require(count < len(targets) and row["target_token"] == targets[count], "teacher-forced target/trace substitution")
        count += 1
    require(count == len(targets), "truncated teacher-forced trace")
    generated_trace(data["generation_logits"], data["output_tokens"], vocab)
    keys(data["state"], STATE_CLASSES)
    for value in data["state"].values():
        verify_pin(value, MAX_LOGITS_FILE)
    categories = ("input_widen", "activation_prepare", "weight_repack", "projection", "scale_correction",
                  "output_conversion", "status_sync", "logits_commit", "complete_wall")
    keys(data["timings_ms"], categories)
    for value in data["timings_ms"].values():
        if value is not None:
            finite(value, 0)
    wall = data["timings_ms"]["complete_wall"]
    require(wall is None or wall > 0, "complete measured wall time must be positive or explicitly unknown")
    if candidate:
        runtime = plan["runtime"]
        for name in ("executable", "source_manifest", "numerical_identity", "kernel_plan_identity", "prefill_identity", "state_history_identity"):
            require(data[name] == runtime[name], "candidate runtime/source/numerical/history identity substituted")
    return data


def evaluate(plan, observations):
    validate_plan(plan)
    keys(observations, ("schema", "plan_sha256", "cases"))
    require(observations["schema"] == "strix-numerical-observations-v1" and observations["plan_sha256"] == identity(plan),
            "observations are not bound to exact predetermined plan/thresholds")
    expected_ids = {c["id"] for c in plan["cases"]}
    require(isinstance(observations["cases"], dict) and set(observations["cases"]) == expected_ids,
            "full declared prompt coverage required; empty/subset/extra cases cannot pass")
    if not validate_runtime(plan["runtime"]):
        return {"status": "unknown", "reason": "runtime/profile not bound", "qualification": False, "native_execution_claim": False,
                "performance_evidence": False}
    candidate = next(c for c in plan["candidates"] if c["id"] == plan["runtime"]["candidate_id"])
    vocab = candidate["costs"]["logical_mnk"][1]
    require(1 <= vocab <= MAX_VOCAB, "vocabulary exceeds bounded evaluator")
    results = {}; blocked = False; checked = []
    for case in plan["cases"]:
        observation = observations["cases"][case["id"]]
        keys(observation, ("result", "replay"))
        result = record(observation["result"], plan, case, vocab, candidate=True)
        checked.append((observation["result"], case, True))
        reference = plan["references"][case["id"]]
        if reference["status"] == "unknown":
            blocked = True; results[case["id"]] = {"status": "unknown", "reason": reference["reason"]}; continue
        baseline = record(reference["artifact"], plan, case, vocab)
        checked.append((reference["artifact"], case, False))
        require(result["input_tokens"] == baseline["input_tokens"], "tokenizer/prompt/context token sequence changed")
        metrics = compare_logits(baseline["logits"], result["logits"], vocab)
        require(result["teacher_forced_target_tokens"] == baseline["teacher_forced_target_tokens"]
                and metrics["teacher_forced_positions"] == len(baseline["teacher_forced_target_tokens"]),
                "teacher-forced targets/coverage differ")
        metrics["max_generated_token_edit_rate"] = token_edit_rate(baseline["output_tokens"], result["output_tokens"])
        passed = all(metrics[name] <= plan["thresholds"][name] for name in THRESHOLDS if name.startswith("max_"))
        semantic = None
        if case["kind"] in {"json", "tool", "context"}:
            semantic = semantic_equal(parse_json(result["output_text"]), case["expected"])
            passed &= semantic
        state = None
        missing_restore = False
        if case["kind"] in {"state", "context"}:
            missing_restore = observation["replay"] is None
            blocked |= missing_restore
            if not missing_restore:
                replay = record(observation["replay"], plan, case, vocab, candidate=True)
                checked.append((observation["replay"], case, True))
                require(result["capture_contract"]["origin"] == "fresh" and replay["capture_contract"]["origin"] == "restored",
                        "two fresh captures do not establish resumed-state correctness")
                require(replay["input_tokens"] == result["input_tokens"] and replay["history_sha256"] == result["history_sha256"],
                    "resume used another canonical producer history")
                state = replay["output_tokens"] == result["output_tokens"] and all(
                replay["state"][name]["sha256"] == result["state"][name]["sha256"]
                and replay["state"][name]["bytes"] == result["state"][name]["bytes"] for name in STATE_CLASSES)
                replay_metrics = compare_logits(result["logits"], replay["logits"], vocab)
                state &= replay_metrics["binary32_logit_bits_mismatched"] == 0 and replay_metrics["max_teacher_forced_argmax_mismatch_rate"] == 0
                if replay["output_tokens"] == result["output_tokens"]:
                    generation_metrics = compare_logits(result["generation_logits"], replay["generation_logits"], vocab)
                    state &= generation_metrics["binary32_logit_bits_mismatched"] == 0
                passed &= state
        elif observation["replay"] is not None:
            raise Invalid("undeclared replay record")
        results[case["id"]] = {"status": "fail" if not passed else "unknown" if missing_restore else "pass", "metrics": metrics,
                                "semantic_match": semantic, "within_plan_state_match": state, "timings_ms": result["timings_ms"],
                                "input_token_binding":"observed-upstream-tool" if "input_token_evidence" in result else "external-operator-assertion"}
        if missing_restore:
            results[case["id"]]["reason"] = "restored-state capture producer/evidence unavailable"
    # Input identity is rechecked after evaluation; observations are trusted
    # quiescent recordings, never proof that these bytes were executed on HIP.
    validate_plan(plan)
    for artifact, case, is_candidate in checked:
        record(artifact, plan, case, vocab, candidate=is_candidate)
    status = "fail" if any(r["status"] == "fail" for r in results.values()) else "unknown" if blocked else "pass"
    return {"status": status, "plan_sha256": identity(plan), "observations_sha256": identity(observations),
            "cases": results, "qualification": False, "native_execution_claim": False,
            "performance_evidence": False,
            "pending": ["actual native execution/provenance", "independent model fidelity", "complete workload/performance",
                        "prefill/verify model dispatch", "sampling/speculation", "integer graph composition"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-model")
    for name in ("model", "model-sha256", "manifest-tool", "manifest-tool-sha256", "tensor-audit-tool", "tensor-audit-tool-sha256", "out"):
        audit.add_argument("--" + name, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--audit", required=True); prepare.add_argument("--prompts", required=True); prepare.add_argument("--out", required=True)
    for name in ("validate", "evaluate"):
        action = sub.add_parser(name); action.add_argument("--plan", required=True)
        if name == "evaluate":
            action.add_argument("--observations", required=True); action.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "audit-model":
            audit_model(args.model, args.model_sha256, args.manifest_tool, args.manifest_tool_sha256,
                        args.tensor_audit_tool, args.tensor_audit_tool_sha256, args.out)
        elif args.command == "prepare":
            plan = prepare_plan(args.audit, args.prompts); validate_plan(plan); write_new(plan, args.out, MAX_PLAN_JSON)
            print(json.dumps({"plan_sha256": identity(plan), "native_execution": "not_done"}))
        elif args.command == "validate":
            plan = validate_plan(load_plan(private_path(args.plan))); print(json.dumps({"plan_sha256": identity(plan), "structurally_valid": True, "qualified": False}))
        else:
            observed_pin = pin(args.observations)
            report = evaluate(load_plan(private_path(args.plan)), load(verify_pin(observed_pin))); verify_pin(observed_pin); write_new(report, args.out)
            print(json.dumps({"status": report["status"], "qualified": False}))
            return {"pass": 0, "unknown": 3, "fail": 4}[report["status"]]
        return 0
    except (Invalid, OSError, OverflowError, ValueError) as error:
        print(f"numerical-campaign: {error}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
