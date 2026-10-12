#!/usr/bin/env python3
"""Import exact paired eager RAW captures; no tokenizer or GPU execution."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
import campaign as c

MAX_STEPS = 32768 + c.MAX_POSITIONS
MAX_METADATA = 64 * 1024 * 1024
MAX_RAW_BYTES = 64 * 1024**3
COMPONENTS = (("attention_kv", "kv", 1, "device"), ("recurrent_matrix", "recurrent", 2, "device"),
              ("gdn_convolution", "convolution", 4, "device"), ("ple_convolution", "convolution", 4, "device"),
              ("sparse_index", "sparse_index", 8, "device"), ("ngram_history", "host_ngram", 16, "host"),
              ("position_history", "position", 32, "host"))
OBSERVED = ("model_content_identity", "gpu_identity", "gpu_architecture", "hip_runtime_version", "hip_driver_version",
            "execution_identity", "kernel_plan_identity", "runtime_compatibility_identity", "layout_identity",
            "context_tokens", "vocabulary_size", "activation_storage")
RECEIPT_FIELDS = ("schema", "complete", "qualified", "performance_evidence", "source_binding", "tokenizer_binding",
                  "evidence_kind", "mode", "state_policy", "model_sha256", "model_content_identity", "executable_sha256",
                  "gpu_identity", "gpu_architecture", "hip_runtime_version", "hip_driver_version", "execution_identity",
                  "kernel_plan_identity", "runtime_compatibility_identity", "layout_identity", "context_tokens",
                  "vocabulary_size", "supplied_tokens", "generated_output_tokens", "maximum_generated_tokens", "eos_token",
                  "stop_reason", "steps_executed", "final_position", "argmax_rule", "history_contract", "state_encoding",
                  "native_byte_order", "steps", "states", "history", "activation_storage")


def tokens(value, vocab, maximum, minimum=1):
    c.require(isinstance(value, list) and minimum <= len(value) <= maximum, "bounded explicit token list required")
    for token in value:
        c.integer(token, 0, vocab - 1)
    return value


class Raw:
    def __init__(self, path, mode):
        self.root = c.private_path(path).resolve(strict=True)
        c.require(self.root.is_dir() and not (self.root / "INCOMPLETE").exists()
                  and not (self.root / "INCOMPLETE").is_symlink(), "incomplete RAW capture")
        self.pins = []; self.total = 0; self.paths = set()
        self.receipt_pin = c.pin(self.root / "receipt.json")
        self.remember(self.receipt_pin)
        r = self.receipt = c.load(self.receipt_pin["path"])
        c.keys(r, RECEIPT_FIELDS)
        c.require(r["schema"] == "strix-raw-eager-baseline-v1" and r["mode"] == mode and r["complete"] is True
                  and r["qualified"] is False and r["performance_evidence"] is False
                  and r["source_binding"] == "external_compile_receipt_required"
                  and r["tokenizer_binding"] == "external_raw_tokens_only"
                  and r["evidence_kind"] in {"native-eager-raw", "injected-fixture"}
                  and r["state_policy"] in {"each", "final"}
                  and r["argmax_rule"] == "first_index_on_exact_tie"
                  and r["history_contract"] == "single_row_eager_from_fresh_initialized_state"
                  and r["state_encoding"] == "opaque_native_bytes" and r["native_byte_order"] == "little",
                  "RAW capture contract mismatch")
        c.require(r["activation_storage"] in {"bf16", "fp32"}, "unknown RAW activation storage")
        for name in ("model_sha256", "model_content_identity", "executable_sha256"):
            c.digest(r[name])
        for name in ("gpu_identity", "gpu_architecture"):
            c.text(r[name], 256)
        for name in ("execution_identity", "kernel_plan_identity", "runtime_compatibility_identity", "layout_identity"):
            c.integer(r[name], 1)
        for name in ("hip_runtime_version", "hip_driver_version"):
            c.integer(r[name], 0, (1 << 31) - 1)
        if r["evidence_kind"] == "native-eager-raw":
            arch = r["gpu_architecture"].split(":")
            features = arch[1:]
            c.require(arch[0] == "gfx1151" and len(features) <= 2
                      and all(f in {"sramecc+","sramecc-","xnack+","xnack-"} for f in features)
                      and len({f[:-1] for f in features}) == len(features), "unsupported native architecture/features")
            c.require(len(r["gpu_identity"]) == 32 and all(x in "0123456789abcdef" for x in r["gpu_identity"])
                      and int(r["gpu_identity"], 16) != 0
                      and r["hip_runtime_version"] > 0 and r["hip_driver_version"] > 0,
                      "native target identity absent or unsupported")
        vocab = c.integer(r["vocabulary_size"], 1, c.MAX_VOCAB)
        steps = c.integer(r["steps_executed"], 1, MAX_STEPS)
        c.require(c.integer(r["final_position"], 1, MAX_STEPS) == steps
                  and steps <= c.integer(r["context_tokens"], 1, (1 << 32) - 1), "RAW position/context mismatch")
        tokens(r["supplied_tokens"], vocab, 32768 + c.MAX_POSITIONS)
        tokens(r["generated_output_tokens"], vocab, 256, 0)
        c.integer(r["maximum_generated_tokens"], 0, 256)
        if r["eos_token"] is not None:
            c.integer(r["eos_token"], 0, vocab - 1)
        history_pin = self.artifact(r["history"], MAX_STEPS * 4)
        c.require(history_pin["bytes"] == steps * 4, "truncated actual fed-token history")
        self.history = list(struct.unpack(f"<{steps}I", Path(history_pin["path"]).read_bytes()))
        tokens(self.history, vocab, MAX_STEPS)
        self.steps = list(self.rows(self.artifact(r["steps"], MAX_METADATA), steps))
        c.require(len(self.steps) == steps, "missing/extra RAW token steps")
        for position, row in enumerate(self.steps):
            c.keys(row, ("position", "input_token", "role", "history_length", "next_position", "argmax_token",
                         "teacher_forced_target", "logits_count", "logits_dtype", "logits"))
            c.require(c.integer(row["position"], 0, MAX_STEPS - 1) == position
                      and c.integer(row["input_token"], 0, vocab - 1) == self.history[position]
                      and c.integer(row["history_length"], 1, MAX_STEPS) == position + 1
                      and c.integer(row["next_position"], 1, MAX_STEPS) == position + 1
                      and c.integer(row["logits_count"], 1, c.MAX_VOCAB) == vocab
                      and row["logits_dtype"] == "fp32-le", "RAW step/history/full-logit contract mismatch")
            c.integer(row["argmax_token"], 0, vocab - 1)
            supplied = position < len(r["supplied_tokens"])
            role = "teacher_forced" if mode == "raw_teacher_forced" else "prefix" if supplied else "generation_fed"
            target = r["supplied_tokens"][position + 1] if mode == "raw_teacher_forced" and position + 1 < len(r["supplied_tokens"]) else None
            if row["teacher_forced_target"] is not None:
                c.integer(row["teacher_forced_target"], 0, vocab - 1)
            c.require(row["role"] == role and row["teacher_forced_target"] == target
                      and (self.history[position] == r["supplied_tokens"][position] if supplied else
                           mode == "raw_greedy" and self.history[position] == self.steps[position - 1]["argmax_token"]),
                      "RAW teacher-forced and autoregressive histories differ")
            row["_pin"] = self.artifact(row["logits"], c.MAX_VOCAB * 4)
            self.logits(row)  # Validate every full row, including unselected prefix/final rows.
        self.validate_stop()
        snapshots = 2 if r["state_policy"] == "final" else steps + 1
        c.require(snapshots <= 4097, "each-state snapshot count exceeds bounded importer")
        self.states = list(self.rows(self.artifact(r["states"], MAX_METADATA), snapshots))
        c.require(len(self.states) == snapshots, "missing/extra complete state snapshots")
        last_generations = [0] * 6; layout = None
        for index, state in enumerate(self.states):
            c.keys(state, ("kind", "next_position", "committed_tokens", "state_valid", "state_generations", "allocations"))
            position = 0 if index == 0 else steps if r["state_policy"] == "final" else index
            kind = "initial" if index == 0 else "final" if r["state_policy"] == "final" else f"post-{index-1}"
            c.require(state["kind"] == kind and c.integer(state["next_position"], 0, MAX_STEPS) == position
                      and c.integer(state["committed_tokens"], 0, MAX_STEPS) == position
                      and type(state["state_valid"]) is int and state["state_valid"] == 63, "state snapshot position/classes mismatch")
            gs = state["state_generations"]
            c.require(isinstance(gs, list) and len(gs) == 6, "missing state generations")
            for old, new in zip(last_generations, gs):
                c.require(c.integer(new) >= old, "state generation regressed")
            last_generations = gs
            c.require(isinstance(state["allocations"], list) and len(state["allocations"]) == 7, "all seven state components required")
            intervals = {"host": [], "device": []}; current_layout = []
            for allocation, expected in zip(state["allocations"], COMPONENTS):
                c.keys(allocation, ("component", "state_class", "state_flag", "storage", "layout_offset", "provisional", "artifact"))
                c.require(tuple(allocation[k] for k in ("component", "state_class", "state_flag", "storage")) == expected
                          and type(allocation["state_flag"]) is int and type(allocation["provisional"]) is bool,
                          "state component/storage/flag mismatch")
                p = allocation["_pin"] = self.artifact(allocation["artifact"], MAX_RAW_BYTES)
                start = c.integer(allocation["layout_offset"]); end = c.integer(start + p["bytes"])
                c.require(all(end <= a or b <= start for a, b in intervals[allocation["storage"]]), "overlapping state components")
                intervals[allocation["storage"]].append((start, end))
                current_layout.append({k: v for k, v in allocation.items() if k not in {"artifact", "_pin"}} | {"bytes": p["bytes"]})
            c.require(layout is None or layout == current_layout, "state allocation layout changed during capture")
            layout = current_layout
        self.layout = layout

    def remember(self, p):
        c.require(p["path"] not in self.paths, "RAW artifact reused by different records")
        self.paths.add(p["path"]); self.pins.append(p); self.total += p["bytes"]
        c.require(self.total <= MAX_RAW_BYTES, "aggregate RAW artifacts exceed 64GiB import bound")
        return p

    def artifact(self, relative, maximum):
        c.keys(relative, ("path", "bytes", "sha256")); name = c.text(relative["path"], 256)
        c.require(Path(name).name == name and name not in {".", ".."} and "/" not in name and "\\" not in name
                  and ":" not in name, "RAW artifacts must be simple relative file names")
        path = (self.root / name).resolve(strict=True)
        c.require(path.parent == self.root, "RAW symlink escapes capture directory")
        p = {**relative, "path": str(path)}
        c.verify_pin(p, maximum)
        return self.remember(p)

    def rows(self, p, maximum):
        with Path(p["path"]).open("rb") as stream:
            for index in range(maximum + 1):
                raw = stream.readline(c.MAX_LINE + 1)
                if not raw:
                    return
                c.require(index < maximum and len(raw) <= c.MAX_LINE, "metadata row/count exceeds bound")
                yield c.parse_json(raw)

    def logits(self, row):
        p = row["_pin"]; vocab = self.receipt["vocabulary_size"]
        c.require(p["bytes"] == vocab * 4, "truncated/extra full-vocabulary binary32 logits")
        values = list(struct.unpack(f"<{vocab}f", Path(p["path"]).read_bytes()))
        c.require(all(math.isfinite(x) for x in values) and max(range(vocab), key=values.__getitem__) == row["argmax_token"],
                  "nonfinite RAW logits or false argmax decision")
        return values

    def validate_stop(self):
        r = self.receipt; supplied = r["supplied_tokens"]; generated = r["generated_output_tokens"]
        if r["mode"] == "raw_teacher_forced":
            c.require(self.history == supplied and not generated and r["stop_reason"] == "raw_input_exhausted",
                      "teacher-forced capture cannot claim generated tokens")
            return
        c.require(generated and r["eos_token"] is not None and 1 <= r["maximum_generated_tokens"] <= 256
                  and self.history == supplied + generated[:-1]
                  and generated == [x["argmax_token"] for x in self.steps[len(supplied)-1:]]
                  and len(generated) <= r["maximum_generated_tokens"], "greedy capture fed/emitted history mismatch")
        eos = r["eos_token"]; reason = r["stop_reason"]
        c.require((generated[-1] == eos and eos not in generated[:-1]) if reason == "eos" else
                  (eos not in generated and (len(generated) == r["maximum_generated_tokens"] and r["final_position"] < r["context_tokens"]
                                             if reason == "max_new_tokens" else reason == "context_capacity" and r["final_position"] == r["context_tokens"])),
                  "greedy stop reason contradicts token/context policy")

    def recheck(self):
        c.require(not (self.root / "INCOMPLETE").exists() and not (self.root / "INCOMPLETE").is_symlink(), "capture became incomplete")
        for p in self.pins:
            c.verify_pin(p, MAX_RAW_BYTES)


def import_record(plan_path, case_id, teacher_path, greedy_path, binding_path, output):
    plan_pin = c.pin(plan_path); binding_pin = c.pin(binding_path)
    plan = c.validate_plan(c.load(plan_path)); binding = c.load(binding_path)
    c.keys(binding, ("schema", "plan_sha256", "case_id", "compile_receipt", "compile_binding", "tokenization", "output_text", "observed", "teacher_forced_targets"))
    c.require(binding["schema"] == "strix-raw-import-binding-v1" and binding["plan_sha256"] == c.identity(plan)
              and binding["case_id"] == case_id, "import binding belongs to another exact plan/case")
    case = next((x for x in plan["cases"] if x["id"] == case_id), None)
    c.require(case is not None and plan["runtime"]["status"] == "recorded" and plan["runtime"]["mode"] == "eager_float",
              "RAW collector supports an explicitly bound eager floating control only")
    runtime = plan["runtime"]
    external = [binding["compile_receipt"], binding["tokenization"], binding["output_text"]]
    for p in external:
        c.verify_pin(p)
    c.load(binding["compile_receipt"]["path"])  # Bounded pinned provenance; authenticity is a separate campaign gate.
    c.keys(binding["compile_binding"], ("executable_sha256", "source_manifest_sha256"))
    c.require(binding["compile_binding"] == {"executable_sha256":runtime["executable"]["sha256"], "source_manifest_sha256":runtime["source_manifest"]["sha256"]},
              "compile/source association differs from exact runtime pins")
    tf = Raw(teacher_path, "raw_teacher_forced"); gr = Raw(greedy_path, "raw_greedy")
    c.keys(binding["observed"], OBSERVED)
    for name in OBSERVED:
        c.require(c.canonical(tf.receipt[name]) == c.canonical(gr.receipt[name]) == c.canonical(binding["observed"][name]), "RAW pair/runtime observation mismatch")
    c.require(tf.receipt["model_sha256"] == gr.receipt["model_sha256"] == plan["model_sha256"]
              and tf.receipt["executable_sha256"] == gr.receipt["executable_sha256"] == runtime["executable"]["sha256"]
              and tf.receipt["evidence_kind"] == gr.receipt["evidence_kind"] and tf.layout == gr.layout,
              "RAW model/executable/evidence/state-layout identity mismatch")
    vocab = gr.receipt["vocabulary_size"]
    candidate = next(x for x in plan["candidates"] if x["id"] == runtime["candidate_id"])
    c.require(candidate["costs"]["logical_mnk"][1] == vocab
              and runtime["kernel_plan_identity"] == str(gr.receipt["kernel_plan_identity"])
              and runtime["input_storage"] == gr.receipt["activation_storage"],
              "RAW vocabulary/kernel plan/activation storage differs from actual bound model runtime")
    c.require([(a["_pin"]["sha256"],a["_pin"]["bytes"]) for a in tf.states[0]["allocations"]]
              == [(a["_pin"]["sha256"],a["_pin"]["bytes"]) for a in gr.states[0]["allocations"]]
              and tf.states[0]["state_generations"] == gr.states[0]["state_generations"],
              "paired fresh captures began with different full initial state")
    prefix = gr.receipt["supplied_tokens"]; generated = gr.receipt["generated_output_tokens"]
    c.require(case["context_min_tokens"] <= len(prefix) <= 32768 and len(generated) <= case["max_new_tokens"]
              and gr.receipt["maximum_generated_tokens"] == case["max_new_tokens"], "actual context/generation policy differs from case")
    targets = tokens(binding["teacher_forced_targets"], vocab, c.MAX_POSITIONS)
    c.require(tf.receipt["supplied_tokens"] == prefix + targets, "teacher-forced targets/prompt substitution")
    tokenization = c.load(binding["tokenization"]["path"])
    c.keys(tokenization, ("schema", "model_sha256", "prompt_sha256", "input_tokens", "output_tokens", "output_text_sha256", "tokenizer_identity", "asserted_by"))
    tokens(tokenization["input_tokens"], vocab, 32768); tokens(tokenization["output_tokens"], vocab, 256)
    c.require(tokenization["schema"] == "strix-external-tokenization-v1" and tokenization["asserted_by"] == "external_operator"
              and tokenization["model_sha256"] == plan["model_sha256"] and tokenization["prompt_sha256"] == c.identity(case)
              and tokenization["input_tokens"] == prefix and tokenization["output_tokens"] == generated
              and tokenization["output_text_sha256"] == binding["output_text"]["sha256"], "external text/tokenization binding mismatch")
    c.text(tokenization["tokenizer_identity"], 4096)
    output_text = Path(binding["output_text"]["path"]).read_text(encoding="utf-8")
    c.require(len(output_text.encode("utf-8")) <= 1024*1024, "decoded text exceeds bounded importer")
    out = c.private_path(output); c.require(not out.exists() and out.parent.is_dir(), "import output directory must be fresh")
    tf.recheck(); gr.recheck(); out.mkdir(); (out / "INCOMPLETE").write_text("typed import incomplete\n", encoding="utf-8")
    def trace(name, raw, begin, emitted):
        path = out / name
        total = 0
        with path.open("xb") as stream:
            for position, target in enumerate(emitted):
                encoded = c.canonical({"position":position,"target_token":target,"logits":raw.logits(raw.steps[begin+position])})+b"\n"
                c.require(len(encoded) <= c.MAX_LINE, "converted full logit row exceeds evaluator bound")
                total += len(encoded)
                c.require(total <= c.MAX_LOGITS_FILE, "converted logit file exceeds 8GiB bound")
                stream.write(encoded)
        return c.pin(path, c.MAX_LOGITS_FILE)
    logits = trace("teacher-forced.jsonl", tf, len(prefix)-1, targets)
    generation = trace("generation.jsonl", gr, len(prefix)-1, generated)
    state = {}
    for state_class in c.STATE_CLASSES:
        path = out / (state_class + ".state")
        with path.open("xb") as stream:
            for allocation in gr.states[-1]["allocations"]:
                if allocation["state_class"] != state_class:
                    continue
                header = c.canonical({k:v for k,v in allocation.items() if k not in {"artifact","_pin"}} | {"bytes":allocation["_pin"]["bytes"]})
                stream.write(struct.pack("<I",len(header))); stream.write(header)
                with Path(allocation["_pin"]["path"]).open("rb") as source:
                    while block := source.read(1024*1024):
                        stream.write(block)
        state[state_class] = c.pin(path, MAX_RAW_BYTES)
    data = {"schema":"strix-numerical-record-v1", "model_sha256":plan["model_sha256"], "prompt_sha256":c.identity(case),
            **{name:runtime[name] for name in ("executable","source_manifest","numerical_identity","kernel_plan_identity","prefill_identity","state_history_identity")},
            "history_sha256":gr.receipt["history"]["sha256"], "input_tokens":prefix, "logits":logits,
            "teacher_forced_target_tokens":targets, "generation_logits":generation, "output_tokens":generated,
            "capture_contract":{"origin":"fresh","history_contract":gr.receipt["history_contract"],"restoration_record":None},
            "output_text":output_text, "state":state, "timings_ms":{name:None for name in candidate["costs"]["timing_categories"]}}
    record_path = out / "record.json"; record_path.write_bytes(c.canonical(data)); record_pin = c.pin(record_path)
    c.record(record_pin, plan, case, vocab, candidate=True)
    tf.recheck(); gr.recheck()
    for p in external + [plan_pin,binding_pin]:
        c.verify_pin(p)
    c.validate_plan(plan)
    receipt = {"schema":"strix-raw-import-receipt-v1", "qualification":False, "performance_evidence":False,
               "native_execution_claim":False, "evidence_kind":gr.receipt["evidence_kind"], "tokenization":"external_operator_assertion",
               "compile_association":"pinned_operator_declaration_not_attestation", "plan":plan_pin,"binding":binding_pin,
               "teacher_forced":tf.receipt_pin,"greedy":gr.receipt_pin,"record":record_pin}
    (out / "import_receipt.json").write_bytes(c.canonical(receipt))
    (out / "INCOMPLETE").unlink()
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("plan", "case", "teacher-forced", "greedy", "binding", "out"):
        parser.add_argument("--"+flag, required=True)
    args = parser.parse_args(argv)
    try:
        import_record(args.plan,args.case,args.teacher_forced,args.greedy,args.binding,args.out)
        print(json.dumps({"status":"imported", "qualification":False})); return 0
    except (c.Invalid, OSError, UnicodeError, OverflowError, ValueError) as error:
        print(str(error),file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
