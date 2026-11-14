#!/usr/bin/env python3
"""Server / CLI parity gate for the serving engines (CPU only, standard library only).

The server of an engine with ONE request must run what the engine's CLI runs: same resolved configuration, same
plan and chunking, same speed. Two subcommands check the two halves:

  config  Reads the `--print-effective-config` output of the CLI entry point and of the server entry point of one
          engine (any of the resolver formats in the tree: JSON, `effective KEY=VALUE (source)`, `env K=V`,
          `arg A`, `KEY=VALUE`, plus an optional `strix-route-receipt {json}` line), flattens both to KEY -> VALUE
          and compares them. Every key that differs must be named in the explained table
          (tools/serve/cli_parity_explained.json) with the reason serving requires it; an explanation can pin the
          two values, so it stops applying when either side changes. Exit 1 on any unexplained difference.

  ratio   Reads CLI run outputs and server responses of the same prompt and reports prefill tok/s, decode tok/s
          and the server/CLI ratio of each (median over the runs given). Exit 1 when a ratio is below
          --min-ratio (default 0.99: the measurement noise of repeated warm runs) or the output token counts
          differ.

Examples:
  strix-glm5-rank  MODEL --world 2 --rank 0 --transport usb4-auto --decode 64 --print-effective-config > cli.txt 2>&1
  strix-glm5-serve MODEL --world 2 --rank 0 --transport usb4-auto --print-effective-config > serve.txt 2>&1
  serve_cli_parity.py config --engine glm --cli cli.txt --serve serve.txt
  serve_cli_parity.py ratio --engine glm --cli run1.log run2.log run3.log --serve resp1.json resp2.json resp3.json
"""
import argparse
import fnmatch
import json
import os
import re
import statistics
import sys

SCHEMA = "strix-serve-cli-parity-v1"
DEFAULT_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve", "cli_parity_explained.json")
UNSET = "<unset>"
ABSENT = "<absent>"


def _flatten(prefix, value, out):
    if isinstance(value, dict):
        for key in value:
            _flatten(f"{prefix}.{key}" if prefix else str(key), value[key], out)
    elif isinstance(value, list) and any(isinstance(x, (dict, list)) for x in value):
        for i, item in enumerate(value):
            _flatten(f"{prefix}[{i}]", item, out)
    elif isinstance(value, list):
        out[prefix] = " ".join(str(x) for x in value)
    else:
        out[prefix] = json.dumps(value) if isinstance(value, (bool, type(None))) else str(value)


def _json_objects(text):
    """Every top-level JSON object in the text (a line may carry a prefix before the brace)."""
    decoder = json.JSONDecoder()
    at = 0
    while True:
        at = text.find("{", at)
        if at < 0:
            return
        try:
            obj, end = decoder.raw_decode(text, at)
        except ValueError:
            at += 1
            continue
        if isinstance(obj, dict):
            yield text[text.rfind("\n", 0, at) + 1:at].strip(), obj
        at = end


def _arguments(tokens, out):
    """`--flag value...` pairs keep their flag as the key; bare leading tokens are positional."""
    key, positional = None, 0
    for token in tokens:
        if token.startswith("--"):
            key = f"arg.{token}"
            out.setdefault(key, "")
        elif key is None:
            out[f"arg[{positional}]"] = token
            positional += 1
        else:
            out[key] = (out[key] + " " + token).strip()


def parse_config(text):
    """Flatten one entry point's `--print-effective-config` output to {key: value} (both strings)."""
    out, args, notices = {}, [], 0
    spans = []
    for prefix, obj in _json_objects(text):
        if prefix.startswith("strix-route-receipt"):
            for lever in obj.get("levers", []):
                out[f"lever.{lever.get('name')}"] = str(lever.get("effective"))
                if lever.get("reason") is not None:
                    out[f"lever.{lever.get('name')}.reason"] = str(lever.get("reason"))
            for name, value in (obj.get("switches", {}).get("set", {}) or {}).items():
                out[f"switch.{name}"] = str(value)
        elif not prefix:
            _flatten("", obj, out)
    for line in text.splitlines():
        line = line.strip()
        if not line or "{" in line and (line.startswith("{") or line.startswith("strix-route-receipt")):
            continue
        if line.startswith("device:") or re.match(r"^[A-Za-z][\w -]*: \{", line):
            continue  # device banner, prefixed receipts (power, route): not configuration
        if line.startswith("effective env "):
            key, _, value = line[14:].partition("=")
            out[f"env.{key}"] = value.split(" (", 1)[0]
        elif line.startswith("effective "):
            key, _, value = line[10:].partition("=")
            out[key] = value.split(" (", 1)[0]
        elif line.startswith("env "):
            body = line[4:]
            match = re.match(r"^(\S+) (.*) \((set|default)\)$", body)
            if match and "=" not in match.group(1):
                out[f"env.{match.group(1)}"] = match.group(2)
            else:
                key, _, value = body.partition("=")
                out[f"env.{key}"] = value
        elif line.startswith("unset "):
            out[f"env.{line[6:]}"] = UNSET
        elif line.startswith("arg "):
            args.append(line[4:])
        elif line.startswith("app "):
            out["app"] = line[4:]
        elif line.startswith("session "):
            out["session"] = line[8:]
        elif line.startswith("notice "):
            out[f"notice[{notices}]"] = line[7:]
            notices += 1
        elif re.match(r"^[A-Za-z][\w.-]* (\S+=\S*)( \S+=\S*)*$", line):
            head, *pairs = line.split(" ")
            for pair in pairs:
                key, _, value = pair.partition("=")
                out[f"{head}.{key}"] = value
        elif re.match(r"^[A-Za-z_][\w./-]*=", line):
            key, _, value = line.partition("=")
            out[key] = value
        else:
            spans.append(line)
    _arguments(args, out)
    if not out:
        raise ValueError("no effective configuration found (" + "; ".join(spans[:2]) + ")")
    return out


def load_table(path=None):
    with open(path or DEFAULT_TABLE, encoding="utf-8") as handle:
        table = json.load(handle)
    if table.get("schema") != SCHEMA:
        raise ValueError(f"explained table: schema is not {SCHEMA}")
    for engine, rows in table.get("engines", {}).items():
        for row in rows:
            if not row.get("key") or len(str(row.get("reason", "")).strip()) < 12:
                raise ValueError(f"explained table: {engine}: every row needs a key and a real reason: {row}")
    return table


def compare(cli, serve, rows):
    """-> (differences, stale). A difference is (key, cli value, serve value, reason or None)."""
    differences, used = [], set()
    for key in sorted(set(cli) | set(serve)):
        a, b = cli.get(key, ABSENT), serve.get(key, ABSENT)
        if a == b:
            continue
        reason = None
        for index, row in enumerate(rows):
            if not fnmatch.fnmatchcase(key, row["key"]):
                continue
            if "cli" in row and str(row["cli"]) != a or "serve" in row and str(row["serve"]) != b:
                continue
            reason = row["reason"]
            used.add(index)
            break
        differences.append((key, a, b, reason))
    stale = [row["key"] for index, row in enumerate(rows) if index not in used]
    return differences, stale


def check_config(engine, cli_text, serve_text, table):
    rows = list(table.get("engines", {}).get("*", [])) + list(table.get("engines", {}).get(engine, []))
    cli, serve = parse_config(cli_text), parse_config(serve_text)
    differences, stale = compare(cli, serve, rows)
    return dict(engine=engine, keys=len(set(cli) | set(serve)), differences=differences, stale=stale,
                unexplained=[d for d in differences if d[3] is None])


# ---- ratio ----------------------------------------------------------------------------------------------------

def _first(pattern, text):
    match = re.search(pattern, text)
    return match.groups() if match else None


def cli_sample(engine, text):
    """One CLI run -> dict(prompt_tokens, prefill_tps, decode_tps, output_tokens); missing values are None."""
    sample = dict(prompt_tokens=None, prefill_tps=None, decode_tps=None, output_tokens=None)
    if engine == "glm":
        hit = _first(r"prompt (\d+) tokens prefill ([\d.]+) s, (\d+) tokens, [\d.]+ ms/token \(([\d.]+) tok/s\)", text)
        if hit:
            sample.update(prompt_tokens=int(hit[0]), prefill_tps=int(hit[0]) / float(hit[1]) if float(hit[1]) else None,
                          output_tokens=int(hit[2]), decode_tps=float(hit[3]))
    elif engine == "minimax":
        hit = _first(r"prompt (\d+) tokens, TTFT [\d.]+ s, prefill ([\d.]+) tok/s", text)
        if hit:
            sample.update(prompt_tokens=int(hit[0]), prefill_tps=float(hit[1]))
        hit = _first(r"decode (\d+) steps at ctx \d+: .*? ms/token = ([\d.]+) tok/s", text)
        if hit:
            sample.update(output_tokens=int(hit[0]), decode_tps=float(hit[1]))
    elif engine == "ds41":
        hit = _first(r"prefill: (\d+) tokens[^\n]*?([\d.]+) tok/s", text)
        if hit:
            sample.update(prompt_tokens=int(hit[0]), prefill_tps=float(hit[1]))
        hit = _first(r"decode: (\d+) steps after the first, ([\d.]+) tok/s", text)
        if hit:
            sample.update(output_tokens=int(hit[0]) + 1, decode_tps=float(hit[1]))
    else:  # JSON receipts: gemma (timings.*), qwen (generation_request_* keys)
        for _, obj in _json_objects(text):
            timings = obj.get("timings") if isinstance(obj.get("timings"), dict) else None
            ids = obj.get("ids") if isinstance(obj.get("ids"), list) else None
            prompt = obj.get("prompt_tokens", obj.get("prompt_token_count"))
            if timings and "prefill_ms" in timings:
                n = len(ids) if ids is not None else obj.get("completion_tokens")
                sample.update(prompt_tokens=prompt, output_tokens=n)
                if prompt and timings["prefill_ms"]:
                    sample["prefill_tps"] = prompt * 1000.0 / timings["prefill_ms"]
                if n and n > 1 and timings.get("decode_ms"):
                    sample["decode_tps"] = (n - 1) * 1000.0 / timings["decode_ms"]
            elif "generation_request_elapsed_ms" in obj:
                tail = obj.get("known_prefix_serial_tail_and_generation_elapsed_ms_total", obj.get("decode_elapsed_ms_total"))
                step = obj.get("known_prefix_serial_tail_and_generation_elapsed_ms_per_step", obj.get("decode_elapsed_ms_per_step"))
                sample.update(prompt_tokens=prompt, output_tokens=obj.get("generation_request_output_tokens"))
                if prompt and tail is not None and obj["generation_request_elapsed_ms"] > tail:
                    sample["prefill_tps"] = prompt * 1000.0 / (obj["generation_request_elapsed_ms"] - tail)
                if step:
                    sample["decode_tps"] = 1000.0 / step
    return sample


def serve_sample(text):
    """One server response (JSON body, or an SSE stream: the last object that carries a receipt wins)."""
    sample = dict(prompt_tokens=None, prefill_tps=None, decode_tps=None, output_tokens=None, ttft_ms=None)
    for _, obj in _json_objects(text):
        usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else {}
        receipt = obj.get("strix") if isinstance(obj.get("strix"), dict) else None
        timings = obj.get("timings") if isinstance(obj.get("timings"), dict) else None
        if isinstance(usage.get("prompt_tokens"), int):
            sample["prompt_tokens"] = usage["prompt_tokens"]
        if isinstance(usage.get("completion_tokens"), int):
            sample["output_tokens"] = usage["completion_tokens"]
        source = receipt or timings
        if not source:
            continue
        if timings and isinstance(timings.get("completion_tokens"), int):
            sample["output_tokens"] = timings["completion_tokens"]
        if source.get("ttft_ms") is not None:
            sample["ttft_ms"] = float(source["ttft_ms"])
        if source.get("prefill_ms") and sample["prompt_tokens"]:
            sample["prefill_tps"] = sample["prompt_tokens"] * 1000.0 / float(source["prefill_ms"])
        n = sample["output_tokens"]
        if source.get("decode_tok_s"):
            sample["decode_tps"] = float(source["decode_tok_s"])
        elif source.get("decode_ms") and n and n > 1:
            # Same definition as the CLIs: the first token belongs to the prefill, n - 1 steps are decode.
            sample["decode_tps"] = (n - 1) * 1000.0 / float(source["decode_ms"])
    return sample


def _median(samples, key):
    values = [s[key] for s in samples if s.get(key) is not None]
    return statistics.median(values) if values else None


def ratio_report(engine, cli_texts, serve_texts, min_ratio=0.99):
    cli = [cli_sample(engine, t) for t in cli_texts]
    serve = [serve_sample(t) for t in serve_texts]
    report = dict(engine=engine, runs=dict(cli=len(cli), serve=len(serve)), min_ratio=min_ratio, failures=[])
    for key in ("prefill_tps", "decode_tps"):
        a, b = _median(cli, key), _median(serve, key)
        ratio = b / a if a and b else None
        report[key] = dict(cli=a, serve=b, ratio=ratio)
        if ratio is None:
            report["failures"].append(f"{key}: not measured on both entry points")
        elif ratio < min_ratio:
            report["failures"].append(f"{key}: server/CLI {ratio:.4f} below {min_ratio}")
    for key in ("prompt_tokens", "output_tokens"):
        a = {s[key] for s in cli if s.get(key) is not None}
        b = {s[key] for s in serve if s.get(key) is not None}
        report[key] = dict(cli=sorted(a), serve=sorted(b))
        if a and b and a != b:
            report["failures"].append(f"{key}: CLI {sorted(a)} server {sorted(b)}")
    report["ttft_ms"] = _median(serve, "ttft_ms")
    return report


def _read(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    config = sub.add_parser("config", help="diff the resolved configuration of the CLI and of the server")
    config.add_argument("--engine", required=True)
    config.add_argument("--cli", required=True)
    config.add_argument("--serve", required=True)
    config.add_argument("--explained", default=None, help="explained-differences table (default: tools/serve/)")
    config.add_argument("--json", action="store_true")
    ratio = sub.add_parser("ratio", help="server/CLI ratio of prefill and decode tok/s")
    ratio.add_argument("--engine", required=True)
    ratio.add_argument("--cli", nargs="+", required=True)
    ratio.add_argument("--serve", nargs="+", required=True)
    ratio.add_argument("--min-ratio", type=float, default=0.99)
    ratio.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "config":
        result = check_config(args.engine, _read(args.cli), _read(args.serve), load_table(args.explained))
        if args.json:
            print(json.dumps(result, indent=1))
        else:
            print(f"{args.engine}: {result['keys']} resolved keys, {len(result['differences'])} differ, "
                  f"{len(result['unexplained'])} unexplained")
            for key, a, b, reason in result["differences"]:
                print(f"  {'explained  ' if reason else 'UNEXPLAINED'} {key}: cli={a[:70]} serve={b[:70]}"
                      + (f"  [{reason}]" if reason else ""))
            if result["stale"]:
                print("  explanations that matched nothing (remove them or fix the key): " + ", ".join(result["stale"]))
        return 1 if result["unexplained"] else 0
    report = ratio_report(args.engine, [_read(p) for p in args.cli], [_read(p) for p in args.serve], args.min_ratio)
    if args.json:
        print(json.dumps(report, indent=1))
    else:
        def cell(value):
            return "-" if value is None else f"{value:.2f}"
        print(f"{args.engine}: CLI runs {report['runs']['cli']}, server runs {report['runs']['serve']} (medians)")
        for key in ("prefill_tps", "decode_tps"):
            row = report[key]
            print(f"  {key:12s} cli {cell(row['cli']):>10s}  server {cell(row['serve']):>10s}  server/CLI "
                  + ("-" if row["ratio"] is None else f"{row['ratio']:.4f}"))
        print(f"  output tokens cli {report['output_tokens']['cli']} server {report['output_tokens']['serve']}; "
              f"server TTFT ms {cell(report['ttft_ms'])}")
        for failure in report["failures"]:
            print("  FAIL " + failure)
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
