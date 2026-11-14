#!/usr/bin/env python3
"""Server / CLI performance ratio of one engine on the same prompt (any family).

The requirement it checks: a server with one request performs like the engine's CLI. Run both on the same build,
the same prompt and a quiet machine, alternating (server, CLI, server, CLI), then report:

  serve_cli_ratio.py server --url http://127.0.0.1:PORT --prompt-ids IDS.txt [--runs 3] [--max-tokens 64]
                            [--warmup 1] [--chat] --out server.json
  serve_cli_ratio.py cli --family qwen|glm|minimax|ds41|gemma --out cli.json LOG [LOG...]
  serve_cli_ratio.py report server.json cli.json [--min-ratio 0.99] [--skip-decode]

`server` posts the prompt as token ids to /v1/completions (streamed, greedy), and records per request the engine's
own prefill time and decode rate (the `prefill_ms` / `decode_tokens_per_second` timing fields every family's server
returns), the client's time to first token and the produced token ids. `cli` reads the timing lines the family's CLI
prints (one LOG per run). `report` prints the ratio table (median of the runs; ratio = server / CLI tok/s) and exits 1
when a ratio is below --min-ratio or the two entry points produced different tokens.
"""
import argparse
import http.client
import json
import re
import statistics
import sys
import time
import urllib.parse

FLOAT = r"([0-9]+(?:\.[0-9]+)?)"
# The timing line(s) each family's CLI prints: (prefill regex -> prompt tokens, seconds | tok/s), (decode regex -> tok/s).
CLI_PATTERNS = {
    "glm": {"prefill_s": re.compile(r"decode: prompt (\d+) tokens prefill " + FLOAT + r" s"),
            "decode_tps": re.compile(r"ms/token \(" + FLOAT + r" tok/s\)")},
    "minimax": {"prefill_tps": re.compile(r"prompt (\d+) tokens, TTFT " + FLOAT + r" s, prefill " + FLOAT + r" tok/s"),
                "decode_tps": re.compile(r"ms/token = " + FLOAT + r" tok/s")},
    "ds41": {"decode_tps": re.compile(r"decode: \d+ steps after the first, " + FLOAT + r" tok/s")},
}


def find_timings(node):
    """The first object carrying prefill_ms anywhere in a response chunk (families nest it differently)."""
    if isinstance(node, dict):
        if "prefill_ms" in node:
            return node
        for value in node.values():
            found = find_timings(value)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_timings(value)
            if found:
                return found
    return None


def find_ids(node):
    if isinstance(node, dict):
        ids = node.get("token_ids")
        if isinstance(ids, list) and ids and all(isinstance(i, int) for i in ids):
            return ids
        for value in node.values():
            found = find_ids(value)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_ids(value)
            if found:
                return found
    return None


def server_record(prompt_tokens, send, first, done, chunks):
    """One request's record from its decoded stream chunks (pure: the tests feed it canned chunks)."""
    timings, ids, pieces = None, [], []
    for chunk in chunks:
        timings = find_timings(chunk) or timings
        final = chunk.get("strix") if isinstance(chunk, dict) else None
        if isinstance(final, dict) and isinstance(final.get("token_ids"), list):
            ids = final["token_ids"]
        else:
            piece = find_ids(chunk.get("choices") if isinstance(chunk, dict) else chunk)
            if piece:
                pieces += piece
    rec = {"prompt_tokens": prompt_tokens, "ttft_s": (first - send) if first else None, "total_s": done - send,
           "ids": ids or pieces}
    if timings:
        rec["prefill_s"] = timings["prefill_ms"] / 1e3
        rec["prefill_tps"] = prompt_tokens / rec["prefill_s"] if rec["prefill_s"] > 0 else 0.0
        rec["decode_tps"] = timings.get("decode_tokens_per_second")
        rec["queue_ms"] = timings.get("queue_ms")
    return rec


def run_server(args):
    ids = [int(x) for x in open(args.prompt_ids).read().replace(",", " ").split()]
    url = urllib.parse.urlparse(args.url)
    path = "/v1/chat/completions" if args.chat else "/v1/completions"
    body = {"model": args.model, "max_tokens": args.max_tokens, "temperature": 0, "stream": True, "return_token_ids": True}
    if args.chat:
        body["messages"] = [{"role": "user", "content": ids}]
    else:
        body["prompt"] = ids
    records = []
    for run in range(args.warmup + args.runs):
        conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=args.timeout)
        send, first, chunks = time.time(), None, []
        conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            sys.exit("server answered %d: %s" % (response.status, response.read()[:200]))
        for line in response:
            line = line.strip()
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            chunk = json.loads(line[5:])
            if first is None and any(c.get("text") or c.get("token_ids") or (c.get("delta") or {}).get("content")
                                     for c in chunk.get("choices") or []):
                first = time.time()
            chunks.append(chunk)
        rec = server_record(len(ids), send, first, time.time(), chunks)
        rec["warmup"] = run < args.warmup
        records.append(rec)
        conn.close()
    json.dump({"entry": "server", "runs": records}, open(args.out, "w"))
    print("server: %d runs (%d warm-up) -> %s" % (len(records), args.warmup, args.out))


def cli_record(family, text):
    """One CLI run's record from its output text (pure)."""
    rec = {}
    if family in ("qwen", "gemma"):
        # Qwen: phase objects of the power receipt; Gemma: prefill_ms / decode_ms of its JSON timings.
        for name in ("prefill", "decode"):
            m = re.search(r'"%s":\{([^{}]*)\}' % name, text)
            f = dict(re.findall(r'"([a-z_]+)":([0-9.eE+-]+)', m.group(1))) if m else {}
            if "duration_ms" in f and "tokens" in f and float(f["duration_ms"]) > 0:
                rec[name + "_tps"] = float(f["tokens"]) / (float(f["duration_ms"]) / 1e3)
                if name == "prefill":
                    rec["prompt_tokens"], rec["prefill_s"] = int(float(f["tokens"])), float(f["duration_ms"]) / 1e3
        if "prefill_tps" not in rec:
            f = dict(re.findall(r'"(prefill_ms|decode_ms|prompt_tokens|completion_tokens)": *([0-9.eE+-]+)', text))
            if f.get("prefill_ms") and f.get("prompt_tokens"):
                rec["prompt_tokens"], rec["prefill_s"] = int(float(f["prompt_tokens"])), float(f["prefill_ms"]) / 1e3
                rec["prefill_tps"] = rec["prompt_tokens"] / rec["prefill_s"]
            if f.get("decode_ms") and f.get("completion_tokens") and float(f["decode_ms"]) > 0:
                rec["decode_tps"] = float(f["completion_tokens"]) / (float(f["decode_ms"]) / 1e3)
        m = re.search(r'"generated_tokens": *\[([0-9, ]*)\]', text)
        if m and m.group(1).strip():
            rec["ids"] = [int(x) for x in m.group(1).split(",")]
        return rec
    pat = CLI_PATTERNS[family]
    if "prefill_s" in pat and (m := pat["prefill_s"].search(text)):
        rec["prompt_tokens"], rec["prefill_s"] = int(m.group(1)), float(m.group(2))
        rec["prefill_tps"] = rec["prompt_tokens"] / rec["prefill_s"] if rec["prefill_s"] > 0 else 0.0
    if "prefill_tps" in pat and (m := pat["prefill_tps"].search(text)):
        rec["prompt_tokens"], rec["prefill_s"], rec["prefill_tps"] = int(m.group(1)), float(m.group(2)), float(m.group(3))
    if m := pat["decode_tps"].search(text):
        rec["decode_tps"] = float(m.group(1))
    return rec


def run_cli(args):
    records = [cli_record(args.family, open(log, errors="replace").read()) for log in args.logs]
    json.dump({"entry": "cli", "family": args.family, "runs": records}, open(args.out, "w"))
    print("cli: %d runs -> %s" % (len(records), args.out))


def median(values):
    values = [v for v in values if v]
    return statistics.median(values) if values else None


def report(server, cli, min_ratio=0.99, skip_decode=False):
    """(lines, ok): the ratio table and whether every ratio meets the bar and the tokens agree (pure)."""
    s_runs = [r for r in server["runs"] if not r.get("warmup")]
    c_runs = cli["runs"]
    lines, ok = ["metric            server        cli      ratio (server/cli)"], True
    for key, label, gated in (("prefill_tps", "prefill tok/s", True), ("decode_tps", "decode tok/s", not skip_decode)):
        s, c = median([r.get(key) for r in s_runs]), median([r.get(key) for r in c_runs])
        if s is None or c is None:
            lines.append("%-14s %9s %10s      n/a" % (label, "%.1f" % s if s else "n/a", "%.1f" % c if c else "n/a"))
            ok = ok and not gated
            continue
        ratio = s / c
        verdict = "" if not gated else ("ok" if ratio >= min_ratio else "BELOW %.3f" % min_ratio)
        ok = ok and (not gated or ratio >= min_ratio)
        lines.append("%-14s %9.1f %10.1f      %.3f %s" % (label, s, c, ratio, verdict))
    ttft, pre = median([r.get("ttft_s") for r in s_runs]), median([r.get("prefill_s") for r in s_runs])
    if ttft and pre:
        lines.append("server TTFT %.3f s = prefill %.3f s + %.1f ms outside the engine" % (ttft, pre, (ttft - pre) * 1e3))
    s_ids = {tuple(r["ids"]) for r in s_runs if r.get("ids")}
    c_ids = {tuple(r["ids"]) for r in c_runs if r.get("ids")}
    if s_ids and c_ids:
        n = min(len(next(iter(s_ids))), len(next(iter(c_ids))))
        same = len({i[:n] for i in s_ids} | {i[:n] for i in c_ids}) == 1
        lines.append("output tokens: %s (first %d)" % ("equal" if same else "DIFFERENT", n))
        ok = ok and same
    else:
        lines.append("output tokens: not compared (ids missing on %s)" % ("the server" if not s_ids else "the CLI"))
    return lines, ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("server")
    s.add_argument("--url", required=True)
    s.add_argument("--prompt-ids", required=True)
    s.add_argument("--runs", type=int, default=3)
    s.add_argument("--warmup", type=int, default=1)
    s.add_argument("--max-tokens", type=int, default=64)
    s.add_argument("--model", default="default")
    s.add_argument("--timeout", type=float, default=600)
    s.add_argument("--chat", action="store_true")
    s.add_argument("--out", required=True)
    c = sub.add_parser("cli")
    c.add_argument("--family", required=True, choices=["qwen", "gemma"] + sorted(CLI_PATTERNS))
    c.add_argument("--out", required=True)
    c.add_argument("logs", nargs="+")
    r = sub.add_parser("report")
    r.add_argument("server")
    r.add_argument("cli")
    r.add_argument("--min-ratio", type=float, default=0.99)
    r.add_argument("--skip-decode", action="store_true", help="report the decode ratio without gating on it")
    args = parser.parse_args(argv)
    if args.command == "server":
        run_server(args)
    elif args.command == "cli":
        run_cli(args)
    else:
        lines, ok = report(json.load(open(args.server)), json.load(open(args.cli)), args.min_ratio, args.skip_decode)
        print("\n".join(lines))
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
