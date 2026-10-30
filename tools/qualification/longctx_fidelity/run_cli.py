"""Node-only plain production CLI execution. This module never auto-launches.

Use only inside an authorized GPU turn. Output decoding is a separate CPU command.
"""
import argparse
import json
import os
import re
from pathlib import Path
import struct
import subprocess
import time

# Plain CLI uses ModelContract::ple_eos_token_id (include/strix/model_contract.hpp).
# No --eos-token override is supplied by this adapter.
NATIVE_EOS_ID = 248044
try:
    from .common import canonical, atomic, read_ids, file_sha, sha, mode_budget, MODE_FACTORS, prompt_ids_file
    from .plan import predicted
except ImportError:
    from common import canonical, atomic, read_ids, file_sha, sha, mode_budget, MODE_FACTORS, prompt_ids_file
    from plan import predicted


def validate_mtp_timing_receipt(config, result):
    """Require actual MTP work before labelling a completed receipt as MTP timing."""
    if config.get("yarn_decode_route", {}).get("effective") != "mtp":
        raise ValueError("MTP timing requires effective MTP route")
    for name in ("mtp_accepted_candidates", "mtp_proposed_candidates", "mtp_paid_drafts"):
        value = result.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("MTP timing missing valid " + name)
    rows = result.get("mtp_chosen_window_rows_histogram")
    if not isinstance(rows, dict) or not rows or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in rows.values()):
        raise ValueError("MTP timing missing valid window histogram")
    if result["mtp_paid_drafts"] == 0 or result["mtp_proposed_candidates"] == 0:
        raise ValueError("MTP timing has no paid/proposed draft coverage")
    tokens = result.get("generated_tokens")
    if not isinstance(tokens, list) or not tokens or result.get("emitted_tokens") != len(tokens):
        raise ValueError("MTP timing missing complete emitted token count")
    elapsed = result.get("generation_ms")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (float, int)) or not (0 < elapsed < float("inf")):
        raise ValueError("MTP timing missing finite positive generation_ms")


def clean_env(allowed=()):
    env = {k: os.environ[k] for k in ("PATH", "LD_LIBRARY_PATH", "HOME") if k in os.environ}
    for item in allowed:
        name, separator, value = item.partition("=")
        if not separator or not name or "\x00" in item or "=" in name:
            raise ValueError("--allow-env must be NAME=VALUE")
        env[name] = value
    return env


def argv_check(argv, env, arg_max=None, platform=None):
    platform = os.name if platform is None else platform
    size = sum(len(os.fsencode(s)) + 1 for s in argv)
    if platform == "nt":
        units = len(subprocess.list2cmdline(argv).encode("utf-16le")) // 2 + 1
        if units > 32767:
            raise ValueError(f"argv exceeds Windows command-line limit: {units} > 32767")
        return size
    if arg_max is None:
        result = subprocess.run(["getconf", "ARG_MAX"], env=env, capture_output=True, text=True, check=True)
        arg_max = int(result.stdout.strip())
    env_size = sum(len(os.fsencode(k + "=" + v)) + 1 for k, v in env.items())
    pointers = (len(argv) + len(env) + 2) * struct.calcsize("P")
    required = size + env_size + pointers + 4096
    # Linux also limits each individual argument/environment string to 32 pages.
    if any(len(os.fsencode(s)) + 1 > 131072 for s in argv + [k + "=" + v for k, v in env.items()]):
        raise ValueError("individual argv/environment string exceeds Linux MAX_ARG_STRLEN")
    if required > arg_max:
        raise ValueError(f"argv exceeds ARG_MAX: required {required} > {arg_max}; no input-file engine change in this kit")
    return size


def parse_output(stdout):
    """Find a complete top-level JSON result, tolerating log lines before/after it.

    Skip quoted log text, arrays and complete objects as whole values. Stop
    scanning at an incomplete object; never promote its nested objects.
    """
    decoder = json.JSONDecoder()
    results = []
    index, arrays = 0, 0
    quoted = escaped = False
    while index < len(stdout):
        char = stdout[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            index += 1
            continue
        if char == '"':
            quoted = True
        elif char == "[":
            arrays += 1
        elif char == "]":
            arrays = max(0, arrays - 1)
        elif char == "{" and not arrays:
            try:
                value, index = decoder.raw_decode(stdout, index)
            except ValueError:
                # An unfinished top-level value owns the remaining text. Its
                # inner braces cannot introduce independent result objects.
                break
            if "generated_tokens" in value and "stop_reason" in value:
                results.append(value)
            continue
        index += 1
    if len(results) != 1:
        raise ValueError("expected exactly one complete generation JSON object")
    result = results[0]
    ids = result["generated_tokens"]
    if not isinstance(ids, list) or any(type(i) is not int or not 0 <= i <= 0xffffffff for i in ids):
        raise ValueError("invalid generated token IDs")
    return result


def validate_case(path):
    path = Path(path)
    case = json.loads(path.read_text(encoding="utf-8"))
    stem = path.with_suffix("")
    # with_suffix on dotted case IDs must not remove the variant component.
    u32, text = Path(str(stem) + ".u32"), Path(str(stem) + ".txt")
    if case["tokenizer"]["backend"].startswith("fake"):
        raise ValueError("FakeTokenizer cases cannot be executed on a model")
    if file_sha(u32) != case["ids_sha256"] or file_sha(text) != case["text_sha256"]:
        raise ValueError("case content hash mismatch")
    if sha(canonical(case["ground_truth"])) != case["ground_truth_sha256"]:
        raise ValueError("ground truth hash mismatch")
    ids = read_ids(u32)
    mode = case["mode"]
    context = case["requested"].get("context") if mode != "native" else None
    cap, context = mode_budget(mode, case["depth"], case["generate"], context)
    if mode != "native" and case["requested"] != {"context": context, "rope_mode": mode, "limit": cap}:
        raise ValueError("invalid YaRN requested configuration")
    if len(ids) != case["depth"]:
        raise ValueError("invalid prompt budget")
    return case, ids


def engine_argv(binary, model, case, ids, out):
    if case["mode"] == "native":
        prompt = list(map(str, ids))
    else:
        prompt = ["--prompt-ids", str(prompt_ids_file(out, case, ids).resolve()),
                  "--rope-yarn", str(MODE_FACTORS[case["mode"]]),
                  "--context", str(case["requested"]["context"])]
    return [binary, model, *prompt, "--generate", str(case["generate"])]


def run_cases(binary, model, model_receipt, turn_list, out, allowed=(), soft=500, table=None, elapsed_before=0):
    start = time.monotonic() - elapsed_before
    env = clean_env(allowed)
    turn_list = Path(turn_list)
    paths = [turn_list.parent / line.strip() for line in turn_list.read_text().splitlines() if line.strip()]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate case in turn")
    # CPU preflight for all cases before launching even one GPU process.
    cases = [(path, *validate_case(path)) for path in paths]
    binary, model = str(Path(binary).resolve()), str(Path(model).resolve())
    provenance = {"binary_sha256": file_sha(binary), "model_receipt_sha256": file_sha(model_receipt),
                  "turn_list_sha256": file_sha(turn_list), "environment": env}
    commands = []
    for _, case, ids in cases:
        if case["tokenizer"].get("model_receipt_sha256") != provenance["model_receipt_sha256"]:
            raise ValueError("tokenizer/model receipt mismatch")
        argv = engine_argv(binary, model, case, ids, out)
        argv_check(argv + ["--print-effective-config"], env)
        commands.append(argv)
    summary = []
    stopped = False
    for (path, case, ids), argv in zip(cases, commands):
        receipt = {"schema": 1, "case_id": case["case_id"], "status": "INCOMPLETE",
                   "manifest_sha256": file_sha(path), "input_ids_sha256": case["ids_sha256"],
                   "ground_truth_sha256": case["ground_truth_sha256"], "provenance": provenance,
                   "tokenizer": case["tokenizer"],
                   "eos_token_id": NATIVE_EOS_ID,
                   "requested": case["requested"], "observed": {"prompt_tokens": len(ids),
                   "last_input_position": len(ids)-1, "first_output_position": len(ids)},
                   "generated_tokens": [], "resolved": None, "error": None}
        receipt["argv"] = argv
        receipt["config_argv"] = argv + ["--print-effective-config"]
        if case["mode"] != "native":
            receipt["prompt_ids_text_sha256"] = file_sha(argv[argv.index("--prompt-ids") + 1])
            receipt["yarn_marker"] = None
        case_start = time.monotonic()
        output_stem = Path(out) / case["case_id"]
        if stopped or case_start - start + predicted(case, table) > soft:
            receipt["error"] = "soft budget: case not started"
            stopped = True
        else:
            try:
                for label, command in (("config", argv + ["--print-effective-config"]), ("run", argv)):
                    remaining = soft - (time.monotonic() - start)
                    if remaining <= 0:
                        raise TimeoutError("soft deadline reached")
                    result = subprocess.run(command, env=env, capture_output=True, timeout=remaining)
                    atomic(str(output_stem) + "." + label + ".stdout", result.stdout)
                    atomic(str(output_stem) + "." + label + ".stderr", result.stderr)
                    receipt[label + "_stdout_sha256"] = sha(result.stdout)
                    receipt[label + "_stderr_sha256"] = sha(result.stderr)
                    receipt[label + "_returncode"] = result.returncode
                    if result.returncode != 0:
                        raise ValueError(label + " subprocess failed")
                    if label == "config":
                        receipt["resolved"] = json.loads(result.stdout)
                    else:
                        if case["mode"] != "native":
                            marker = re.search(r"^Qwen experimental YaRN: factor=([^\s]+) context=(\d+) T=([^\s]+)[^\r\n]*(?=\r?$)",
                                               result.stderr.decode("utf-8"), re.MULTILINE)
                            if marker is None:
                                raise ValueError("missing Qwen experimental YaRN stderr marker")
                            receipt["yarn_marker"] = marker.group(0)
                            if (float(marker.group(1)) != MODE_FACTORS[case["mode"]] or
                                    int(marker.group(2)) != case["requested"]["context"]):
                                raise ValueError("YaRN stderr marker differs from requested mode/context")
                        parsed = parse_output(result.stdout.decode("utf-8"))
                        if parsed.get("prefix_tokens") != ids:
                            raise ValueError("CLI echoed prompt IDs differ")
                        if case.get("expected_decode_route") == "mtp":
                            validate_mtp_timing_receipt(receipt["resolved"], parsed)
                        receipt["generated_tokens"] = parsed["generated_tokens"]
                        receipt["stop_reason"] = parsed["stop_reason"]
                        if parsed["stop_reason"] == "eos" and parsed["generated_tokens"][-1:] != [NATIVE_EOS_ID]:
                            raise ValueError("EOS stop does not end in the native EOS ID")
                        if not parsed["generated_tokens"] or len(parsed["generated_tokens"]) > case["generate"]:
                            raise ValueError("missing or excess output tokens")
                        if parsed["stop_reason"] not in ("max_new_tokens", "eos"):
                            raise ValueError("generation did not complete normally")
                        if parsed["stop_reason"] == "max_new_tokens" and len(parsed["generated_tokens"]) != case["generate"]:
                            raise ValueError("partial output token count")
                        receipt["status"] = "COMPLETE"
            except (OSError, ValueError, subprocess.SubprocessError, TimeoutError) as error:
                if isinstance(error, subprocess.TimeoutExpired):
                    atomic(str(output_stem) + "." + label + ".stdout", error.stdout or b"")
                    atomic(str(output_stem) + "." + label + ".stderr", error.stderr or b"")
                receipt["error"] = str(error)
                stopped = True
        receipt["wall_seconds"] = time.monotonic() - case_start
        atomic(str(output_stem) + ".receipt.json", canonical(receipt))
        summary.append({"case_id": case["case_id"], "status": receipt["status"]})
    atomic(Path(out) / "summary.json", canonical(summary))
    return all(r["status"] == "COMPLETE" for r in summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model-receipt", required=True)
    p.add_argument("--turn-list", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-env", action="append", default=[])
    p.add_argument("--soft", type=float, default=500)
    p.add_argument("--prediction-table")
    p.add_argument("--turn-start", type=float, help="turn start Unix seconds, including preflight time")
    a = p.parse_args()
    table = {int(k): float(v) for k, v in json.loads(Path(a.prediction_table).read_text()).items()} if a.prediction_table else None
    if not 0 < a.soft <= 500:
        p.error("soft budget must be positive and <=500 seconds")
    elapsed_before = max(0, time.time() - a.turn_start) if a.turn_start is not None else 0
    okay = run_cases(a.binary, a.model, a.model_receipt, a.turn_list, a.out, a.allow_env, a.soft, table, elapsed_before)
    raise SystemExit(0 if okay else 2)


if __name__ == "__main__":
    main()
