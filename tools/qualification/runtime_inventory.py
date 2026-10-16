"""Collect offline qualification evidence; never launch a decoder or probe.

Inputs are existing decoder, tensor-audit, model-manifest and platform-probe JSON
outputs. CMakeCache records configuration, not a successful build. Missing native
catalog/launcher evidence stays unknown, even when tensor admission succeeds.
"""
import argparse
import hashlib
import json
import math
import re
from decimal import Decimal, InvalidOperation
import os
from pathlib import Path
import struct
import subprocess
import sys


def load_json(path):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def finite_float(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError(f"nonfinite JSON number: {text}")
        return value

    value = json.loads(Path(path).read_text(encoding="utf-8-sig"),
                       object_pairs_hook=unique, parse_float=finite_float,
                       parse_constant=lambda value: (_ for _ in ()).throw(
                           ValueError(f"nonfinite JSON value: {value}")))
    if not isinstance(value, dict):
        raise ValueError(f"JSON evidence must be an object: {path}")
    return value


def fingerprint(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        hasher = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
        digest = hasher.hexdigest()
        after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"evidence changed while hashing: {path}")
    return {"path": str(path), "sha256": digest, "bytes": after.st_size}


def model_identity(digest, size, logical_name="model"):
    """Existing STRIX-MODEL-CONTENT-V1 single-file contract, not file SHA."""
    name = logical_name.encode("utf-8")
    return hashlib.sha256(b"STRIX-MODEL-CONTENT-V1" + struct.pack("<Q", len(name))
                          + name + struct.pack("<Q", size)
                          + bytes.fromhex(digest)).hexdigest()


def cmake_cache(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line or line.startswith(("#", "//")):
            continue
        declaration, separator, value = line.partition("=")
        name, colon, kind = declaration.partition(":")
        if not separator or not colon:
            raise ValueError("malformed CMake cache entry")
        # Unrelated cache settings cannot invalidate this selected configuration.
        if not (name.startswith("STRIX_") or name == "BUILD_TESTING"
                or name in ("CMAKE_BUILD_TYPE", "CMAKE_GENERATOR", "CMAKE_HOME_DIRECTORY")
                or name.startswith(("CMAKE_CXX_", "CMAKE_HIP_", "CMAKE_EXE_LINKER_FLAGS"))):
            continue
        if name in values:
            raise ValueError(f"duplicate CMake cache entry: {name}")
        if kind == "BOOL":
            upper = value.upper()
            if upper in ("ON", "TRUE", "YES", "Y"):
                value = True
            elif upper in ("OFF", "FALSE", "NO", "N", "0", "", "IGNORE", "NOTFOUND") or upper.endswith("-NOTFOUND"):
                value = False
            elif re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", value):
                try:
                    value = Decimal(value) != 0
                except InvalidOperation as error:
                    raise ValueError(f"invalid CMake numeric boolean: {name}") from error
            else:
                raise ValueError(f"unrecognized CMake boolean: {name}")
        values[name] = value
    return values


def source_identity(repo):
    def git(*arguments):
        try:
            result = subprocess.run(["git", "-C", str(repo), *arguments],
                                    capture_output=True, text=True, check=False)
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    revision = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=all")
    return {"repository": str(Path(repo).resolve()), "revision": revision,
            "dirty": bool(status) if status is not None else None,
            "status_porcelain": status.splitlines() if status else ([] if status == "" else None)}


def integer(value, label):
    if type(value) is not int or value < 0 or value > 0xffffffffffffffff:
        raise ValueError(f"{label} must be a uint64 integer")
    return value


def boolean(value, label):
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def hex_identity(value, label, lengths=(64,)):
    if (not isinstance(value, str) or len(value) not in lengths
            or re.fullmatch(r"[0-9a-fA-F]+", value) is None):
        raise ValueError(f"{label} must be a hexadecimal identity of length {lengths}")
    return value.lower()


def closed_keys(value, required, optional, label):
    if not isinstance(value, dict) or not set(required) <= value.keys() or value.keys() - (set(required) | set(optional)):
        raise ValueError(f"{label} has missing or unknown keys")


RUNTIME_ENVIRONMENT_KEYS = {"GPU_MAX_HW_QUEUES", "HSA_ENABLE_SDMA", "HSA_OVERRIDE_GFX_VERSION",
    "AMD_SERIALIZE_KERNEL", "AMD_SERIALIZE_COPY", "HIP_LAUNCH_BLOCKING", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"}


def validate_runtime_environment(value):
    closed_keys(value, {"source", "requested", "effective_hw_queues", "concurrent_execution_observed"}, set(), "runtime environment")
    if value["source"] != "decoder-process" or value["effective_hw_queues"] is not None or value["concurrent_execution_observed"] is not None:
        raise ValueError("requested environment is not effective queue or overlap evidence")
    closed_keys(value["requested"], RUNTIME_ENVIRONMENT_KEYS, set(), "runtime environment requested settings")
    if any(item is not None and not isinstance(item, str) for item in value["requested"].values()):
        raise ValueError("runtime environment settings must be strings or null")


def validate_outer_receipt(receipt):
    if receipt.get("hip_runtime_environment") is not None:
        validate_runtime_environment(receipt["hip_runtime_environment"])
    # Outer receipts and external audit/probe schemas remain open: only fields
    # consumed here are checked. A missing or explicit null identity stays unknown.
    for key in ("execution_identity", "kernel_plan_identity", "launch_plan_identity",
                "runtime_compatibility_identity", "context_tokens", "final_position",
                "next_position", "committed_tokens", "steps_executed", "slot_parallel_q51_submissions"):
        if receipt.get(key) is not None:
            integer(receipt[key], key)
    for key in ("model_sha256", "model_content_identity", "binary_sha256"):
        if receipt.get(key) is not None:
            hex_identity(receipt[key], key)
    if receipt.get("source_revision") is not None:
        hex_identity(receipt["source_revision"], "source_revision", (40, 64))
    for key in ("q4k_a8g32_gate_up", "grouped_exact_q51_down"):
        if key in receipt:
            boolean(receipt[key], key)
    if "q51_slot_parallel_down" in receipt and receipt["q51_slot_parallel_down"] is not None:
        boolean(receipt["q51_slot_parallel_down"], "q51_slot_parallel_down")
    if ("q51_slot_parallel_down_qualification" in receipt and
            receipt["q51_slot_parallel_down_qualification"] != "not_assessed"):
        raise ValueError("q51_slot_parallel_down_qualification must be not_assessed")


def profile(receipt):
    keys = ("requested", "selected", "reason", "automatic_applied")
    values = {key: receipt.get("optimization_profile_" + key) for key in keys}
    if any(value is not None for value in values.values()):
        if any(values[key] is None for key in keys):
            raise ValueError("partial optimization profile receipt")
        if (any(not isinstance(values[key], str) or not values[key]
                for key in keys[:3]) or type(values["automatic_applied"]) is not bool):
            raise ValueError("invalid optimization profile receipt types")
    return values



def reported_opt_in_overlays(receipt):
    # Receipt declarations, not inferred qualification of the combined recipe.
    values = {}
    for key in ("eager_submission_mode", "moe_shared_overlap_mode", "generation_output_mode"):
        value = receipt.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{key} must be a nonempty string or null")
        values[key] = value
    for key in ("batched_moe_activation", "prefetch_q8_lm_head", "wave8_q8_lm_head", "wave8_q8_gdn_input",
                "wave8_q8_gdn_output", "wave8_q8_qsa_output", "prefetch_hc_projection_next_tile", "q51_slot_parallel_down"):
        value = receipt.get(key)
        if value is not None:
            boolean(value, key)
        values[key] = value
    return values


def operator_records(audit, manifest):
    requirements = audit.get("requirements", [])
    if not isinstance(requirements, list):
        raise ValueError("tensor audit requirements must be an array")
    tensors = manifest.get("tensors")
    if tensors is not None and not isinstance(tensors, list):
        raise ValueError("model manifest tensors must be an array")
    operators = {}
    for requirement in requirements:
        if not isinstance(requirement, dict) or not isinstance(requirement.get("op"), str):
            raise ValueError("invalid tensor audit operator requirement")
        if not isinstance(requirement.get("status", "unknown"), str):
            raise ValueError("tensor audit requirement status must be a string")
        operators.setdefault(requirement["op"], []).append(requirement)
    records = []
    for op, uses in sorted(operators.items()):
        formats = set()
        if tensors is not None:
            for use in uses:
                indices = use.get("tensor_indices", [])
                if not isinstance(indices, list):
                    raise ValueError("tensor indices must be an array")
                for index in indices:
                    integer(index, "tensor index")
                    if index >= len(tensors) or not isinstance(tensors[index], dict):
                        raise ValueError("tensor audit index outside manifest")
                    formats.add(integer(tensors[index].get("ggml_type"), "ggml_type"))
        for phase in ("decode", "verify", "prefill"):
            records.append({"operator": op, "phase": phase,
                            "compiled_support": None, "executable_support": None,
                            "allowed_logical_rows": None, "allowed_request_concurrency": None,
                            "tensor_formats_ggml": sorted(formats) if tensors is not None else None,
                            "activation_dtype": None, "state_dtype": None,
                            "graph_safe": None, "locality": None,
                            "numerical_identity": None, "qualification": None,
                            "tensor_requirement_evidence_phase": "decode",
                            "tensor_requirement_statuses": sorted({use.get("status", "unknown") for use in uses})})
    return records


def selected_operator_records(inventory, receipt, tensor_records, conflicts):
    """Consume the decoder exporter, retaining declarations as declarations."""
    if (not isinstance(inventory, dict) or type(inventory.get("schema_version")) is not int
            or inventory.get("schema_version") != 1
            or inventory.get("evidence_kind") != "compiled-selected-launchers"
            or inventory.get("phase") != "decode"
            or inventory.get("verify_admitted_by_this_receipt") is not False
            or inventory.get("prefill_admitted_by_this_receipt") is not False
            or inventory.get("native_execution_observed") is not None
            or inventory.get("qualification") is not None):
        raise ValueError("unsupported execution_inventory receipt or scope")
    # Closed keys apply only to our versioned exporter, never to the enclosing
    # decoder receipt or external model/tensor/platform evidence.
    closed_keys(inventory,
        {"schema_version", "evidence_kind", "phase", "selected_logical_rows",
         "selected_request_concurrency", "verify_admitted_by_this_receipt",
         "prefill_admitted_by_this_receipt", "execution_identity", "kernel_plan_identity",
         "launch_plan_identity", "native_execution_observed", "qualification", "steps"},
        {"whole_plan_graph_safe_declared"}, "execution inventory")
    if "whole_plan_graph_safe_declared" in inventory:
        boolean(inventory["whole_plan_graph_safe_declared"], "whole-plan graph declaration")
    rows = integer(inventory.get("selected_logical_rows"), "selected logical rows")
    concurrency = integer(inventory.get("selected_request_concurrency"), "selected request concurrency")
    if not rows or not concurrency:
        raise ValueError("selected decode geometry must be positive")
    for key in ("execution_identity", "kernel_plan_identity", "launch_plan_identity"):
        if not integer(inventory.get(key), key):
            raise ValueError("execution inventory identities must be nonzero")
        if receipt.get(key) is not None and receipt[key] != inventory[key]:
            conflicts.append({"field": key, "observed": inventory[key], "reported": receipt[key]})
    steps = inventory.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("execution_inventory steps must be a nonempty array")
    formats = {record["operator"]: record for record in tensor_records if record["phase"] == "decode"}
    result = []
    for index, step in enumerate(steps):
        closed_keys(step,
            {"step_id", "operator", "layer", "runtime_step", "launcher_bound",
             "launcher_implementation_identity", "kernel_id", "declared_variant"},
            set(), "execution step")
        if (not isinstance(step, dict) or type(step.get("step_id")) is not int
                or step["step_id"] != index or not isinstance(step.get("operator"), str)
                or not step["operator"] or type(step.get("runtime_step")) is not bool
                or step.get("launcher_bound") is not True
                or not integer(step.get("launcher_implementation_identity"), "launcher identity")):
            raise ValueError("invalid selected launcher step")
        integer(step["step_id"], "step id")
        if step.get("layer") is not None:
            integer(step["layer"], "layer")
        variant = step.get("declared_variant")
        if step["runtime_step"]:
            if variant is not None or step.get("kernel_id") is not None:
                raise ValueError("runtime step cannot claim a kernel variant")
        else:
            if (not isinstance(variant, dict) or not isinstance(step.get("kernel_id"), str)
                    or not step["kernel_id"] or variant.get("id") != step["kernel_id"]
                    or variant.get("phase") != "decode"
                    or type(variant.get("logical_rows")) is not int or variant["logical_rows"] != rows
                    or type(variant.get("request_concurrency")) is not int
                    or variant["request_concurrency"] != concurrency
                    or type(variant.get("graph_safe")) is not bool
                    or variant.get("qualification") not in (
                        "experimental", "oracle_matched", "target_qualified", "production", "rejected")):
                raise ValueError("variant declaration differs from selected decode binding")
            closed_keys(variant,
                {"id", "phase", "logical_rows", "request_concurrency", "graph_safe",
                 "qualification", "input_layout", "output_layout", "compiler_identity", "numerics"},
                {"family", "hidden_size", "auxiliary_size", "wave_size"}, "declared variant")
            for key in ("logical_rows", "request_concurrency", "family", "hidden_size", "auxiliary_size", "wave_size"):
                if key in variant:
                    integer(variant[key], "variant " + key)
            numerics = variant.get("numerics")
            if (not isinstance(numerics, dict) or any(
                    not isinstance(numerics.get(key), str) or not numerics[key]
                    for key in ("input_precision", "accumulator_precision", "output_precision"))):
                raise ValueError("invalid declared variant numerics")
            closed_keys(numerics,
                {"input_precision", "accumulator_precision", "output_precision"},
                {"row_invariant", "deterministic"}, "declared numerics")
            for key in ("row_invariant", "deterministic"):
                if key in numerics:
                    boolean(numerics[key], "numerics " + key)
            for key in ("input_layout", "output_layout", "compiler_identity"):
                if not isinstance(variant.get(key), str) or not variant[key]:
                    raise ValueError("invalid variant layout/compiler declaration")
        for phase in ("decode", "verify", "prefill"):
            admitted = phase == "decode"
            metadata = formats.get(step["operator"], {})
            result.append({"operator": step["operator"], "step_id": index,
                           "layer": step.get("layer"), "phase": phase,
                           "kernel_id": step.get("kernel_id") if admitted else None,
                           "admitted_by_this_receipt": admitted,
                           "selected_logical_rows": rows if admitted else None,
                           "selected_request_concurrency": concurrency if admitted else None,
                           "launcher_bound_declared": True if admitted else None,
                           "declared_variant": variant if admitted else None,
                           "compiled_support": None, "executable_support": None,
                           "allowed_logical_rows": None, "allowed_request_concurrency": None,
                           "tensor_formats_ggml": metadata.get("tensor_formats_ggml"),
                           "activation_dtype": None, "state_dtype": None, "graph_safe": None,
                           "locality": None, "numerical_identity": None, "qualification": None,
                           "evidence": "decoder-receipt/execution_inventory"})
    return result


def collect(repo, cache_path=None, receipt_path=None, audit_path=None, manifest_path=None,
            platform_path=None, model_path=None, binary_path=None,
            capacity=None, occupied=None):
    inputs = {}

    def read(label, path):
        if path is None:
            return {}
        inputs[label] = fingerprint(path)
        value = load_json(path)
        if inputs[label] != fingerprint(path):
            raise ValueError(f"JSON evidence changed while reading: {path}")
        return value

    receipt = read("decoder_receipt", receipt_path)
    audit = read("tensor_audit", audit_path)
    manifest = read("model_manifest", manifest_path)
    platform = read("platform_probe", platform_path)
    cache = {}
    if cache_path:
        inputs["cmake_cache"] = fingerprint(cache_path)
        cache = cmake_cache(cache_path)
        if inputs["cmake_cache"] != fingerprint(cache_path):
            raise ValueError("CMake cache changed while reading")
    source = source_identity(repo)
    validate_outer_receipt(receipt)
    effective = profile(receipt)
    overlays = reported_opt_in_overlays(receipt)
    overlay_present = any(value is not None and value is not False for value in overlays.values())
    overlay_limitations = (["Receipt reports explicit opt-in overlays; the selected base profile label does not "
                            "establish qualification of this combined configuration."] if overlay_present else [])
    conflicts = []

    def agree(label, observed, reported):
        if observed is not None and reported is not None and observed != reported:
            conflicts.append({"field": label, "observed": observed, "reported": reported})

    hip = cache.get("STRIX_ENABLE_HIP_RUNTIME")
    a8 = cache.get("STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME")
    for name, value in (("STRIX_ENABLE_HIP_RUNTIME", hip), ("STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME", a8)):
        if value is not None:
            boolean(value, name)
    if hip is False and a8 is True:
        conflicts.append({"field": "build", "reason": "A8 configured without HIP runtime"})
    for flag in ("q4k_a8g32_gate_up", "grouped_exact_q51_down", "q51_slot_parallel_down"):
        if a8 is False and flag in receipt:
            agree("build/" + flag, False, receipt[flag])
    slot_submissions = receipt.get("slot_parallel_q51_submissions")
    if a8 is False and slot_submissions is not None and slot_submissions > 0:
        agree("build/slot_parallel_q51_submissions", 0, slot_submissions)
    if receipt.get("q51_slot_parallel_down") is True:
        agree("overlay/q51-grouped", True, receipt.get("grouped_exact_q51_down"))
        agree("overlay/q51-A8", True, receipt.get("q4k_a8g32_gate_up"))
    applied = effective["automatic_applied"]
    if applied is not None:
        agree("profile/manual", effective["selected"] == "manual", not applied)
        if effective["requested"] == "manual":
            agree("profile/manual-request", False, applied)
        if applied:
            if a8 is False or hip is False:
                conflicts.append({"field": "profile/build", "reason": "automatic bundle receipt contradicts configured HIP/A8"})
        if effective["reason"] == "a8-runtime-not-built":
            agree("profile/a8-fallback", False, applied)
            agree("profile/a8-fallback-build", False, a8)
    model = fingerprint(model_path) if model_path else {"path": None, "sha256": None, "bytes": None}
    model["content_identity"] = (model_identity(model["sha256"], model["bytes"])
                                 if model_path else None)
    model["content_logical_name"] = "model" if model_path else None
    agree("model_sha256", model["sha256"], hex_identity(receipt["model_sha256"], "model_sha256") if receipt.get("model_sha256") is not None else None)
    agree("model_content_identity", model["content_identity"], hex_identity(receipt["model_content_identity"], "model_content_identity") if receipt.get("model_content_identity") is not None else None)
    agree("source_revision", source["revision"], hex_identity(receipt["source_revision"], "source_revision", (40, 64)) if receipt.get("source_revision") is not None else None)
    binary = fingerprint(binary_path) if binary_path else None
    if binary:
        agree("binary_sha256", binary["sha256"], hex_identity(receipt["binary_sha256"], "binary_sha256") if receipt.get("binary_sha256") is not None else None)
    for evidence, key in ((manifest, "layout_identity"), (audit, "manifest_layout_identity")):
        if evidence.get(key) is not None:
            integer(evidence[key], key)
    if audit and manifest:
        agree("manifest_layout_identity", manifest.get("layout_identity"), audit.get("manifest_layout_identity"))
    capacity_declared, occupied_declared = capacity is not None, occupied is not None
    receipt_capacity = receipt.get("context_tokens")
    if receipt_capacity is not None:
        integer(receipt_capacity, "receipt context_tokens")
    if capacity is not None:
        integer(capacity, "context capacity")
    agree("context_capacity_tokens", capacity, receipt_capacity)
    if capacity is None:
        capacity = receipt_capacity
    prefix = receipt.get("prefix_tokens")
    receipt_occupied = None
    if prefix is not None:
        if not isinstance(prefix, list):
            raise ValueError("receipt prefix_tokens must be an array")
        for token in prefix:
            integer(token, "prefix token")
        receipt_occupied = len(prefix)
    if occupied is not None:
        integer(occupied, "occupied prefix")
    agree("occupied_prefix_tokens", occupied, receipt_occupied)
    if occupied is None:
        occupied = receipt_occupied
    if occupied is not None and capacity is not None and occupied > capacity:
        conflicts.append({"field": "context", "reason": "occupied prefix exceeds capacity"})
    records = operator_records(audit, manifest)
    if "execution_inventory" in receipt:
        records = selected_operator_records(receipt["execution_inventory"], receipt, records, conflicts)
    return {"schema_version": 1, "kind": "offline-runtime-inventory",
            "source": source, "inputs": inputs,
            "configured_build": cache, "build_completed": None,
            "reported_hip_runtime_environment": receipt.get("hip_runtime_environment"),
            "model": model, "reported_model_sha256": receipt.get("model_sha256"),
            "reported_model_content_identity": receipt.get("model_content_identity"),
            "binary": binary, "effective_profile": effective,
            "reported_opt_in_overlays": overlays,
            "profile_overlay_qualification": "not_assessed" if overlay_present else None,
            "profile_evidence": "decoder-receipt" if effective["requested"] is not None else None,
            "context_capacity_tokens": capacity, "occupied_prefix_tokens": occupied,
            "context_evidence": {
                "capacity": "caller-declaration" if capacity_declared else (
                    "decoder-receipt/context_tokens" if receipt_capacity is not None else None),
                "occupied_prefix": "caller-declaration" if occupied_declared else (
                    "decoder-receipt/prefix_tokens" if receipt_occupied is not None else None)},
            "final_position": receipt.get("final_position", receipt.get("next_position")),
            "execution_identity": receipt.get("execution_identity"),
            "runtime_compatibility_identity": receipt.get("runtime_compatibility_identity"),
            "operators": records, "platform_probe": platform or None,
            "native_tests_executed": None, "qualification_state": "not_assessed",
            "consistent": not conflicts, "conflicts": conflicts,
            "limitations": ["No executable, probe, benchmark or GPU operation was launched.",
                            "CMakeCache records configuration; it does not attest a built binary or a receipt's build.",
                            "Tensor requirements and formats do not establish launcher support in any phase.",
                            "Absent effective catalog/native qualification evidence remains null.",
                            "Receipts without source/binary identities are not bound to this checkout/build."] + overlay_limitations}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    for option in ("cmake-cache", "receipt", "tensor-audit", "manifest", "platform-probe", "model", "binary", "output"):
        parser.add_argument("--" + option, type=Path)
    parser.add_argument("--context-capacity-tokens", type=int)
    parser.add_argument("--occupied-prefix-tokens", type=int)
    args = parser.parse_args(argv)
    try:
        if args.output:
            supplied = (args.cmake_cache, args.receipt, args.tensor_audit, args.manifest,
                        args.platform_probe, args.model, args.binary)
            if any(path and args.output.resolve() == path.resolve() for path in supplied):
                raise ValueError("output must not overwrite an evidence input")
        result = collect(args.repo, args.cmake_cache, args.receipt, args.tensor_audit,
                         args.manifest, args.platform_probe, args.model, args.binary,
                         args.context_capacity_tokens, args.occupied_prefix_tokens)
        text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
        return 0 if result["consistent"] else 3
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
