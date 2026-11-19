#!/usr/bin/env python3
"""Scripted multi-turn chat sessions against an OpenAI-compatible server: time to first token and cached tokens per turn.

Scenarios (all non-streaming, greedy, short answers; the assistant turns sent back are fixed text, so every arm sends
the same token ids whatever the model answers):
  turns     one conversation, turns 1..T on a shared system + tools prompt
  siblings  N new conversations with the same system + tools prompt (agent sub-sessions): turn 1 of each
  edit      the T-turn history with user turn 2 edited, sent again (an edited history)

Usage (as a kit client: scripts/bench/common/serve-decode.sh ... --client-cmd 'python3 .../session_turns.py'):
  session_turns.py [--url http://127.0.0.1:PORT] [--system-words 4500] [--user-words 1100] [--reply-words 600]
                   [--turns 6] [--siblings 3] [--max-tokens 8] [--scenarios turns,siblings,edit] [--out DIR]
KIT_URL / KIT_RUN_DIR (exported by the kit) are the defaults of --url / --out. Writes DIR/session-turns.jsonl (one line
per request) and DIR/client-result.json; prints one line per request.
TTFT here = wall time of a request with max_tokens small (prefill + a few decode steps), cached = the server's
usage.prompt_tokens_details.cached_tokens, answer_sha = SHA-256 of the answer (content + tool calls): equal
answers across arms (greedy) are the identity check.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo "
         "sierra tango uniform victor whiskey xray yankee zulu north south east west river stone cloud field").split()


def text(seed, words):
    """Deterministic filler: `words` words, different per seed."""
    out, x = [], seed * 2654435761 % 4294967296 or 1
    for _ in range(words):
        x = (x * 1103515245 + 12345) % 2147483648
        out.append(WORDS[x % len(WORDS)])
    return " ".join(out)


def tools():
    return [{"type": "function", "function": {
        "name": f"tool_{k}", "description": text(900 + k, 40),
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "line": {"type": "integer"}},
                       "required": ["path"]}}} for k in range(4)]


def post(url, body, timeout):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    ms = (time.perf_counter() - t0) * 1000.0
    usage = data.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    msg = ((data.get("choices") or [{}])[0].get("message") or {})
    answer = json.dumps([msg.get("content"), msg.get("tool_calls")], sort_keys=True)
    return ms, usage.get("prompt_tokens"), cached, hashlib.sha256(answer.encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("KIT_URL", "http://127.0.0.1:18671"))
    ap.add_argument("--out", default=os.environ.get("KIT_RUN_DIR", "."))
    ap.add_argument("--model", default="default")
    ap.add_argument("--system-words", type=int, default=4500)
    ap.add_argument("--user-words", type=int, default=1100)
    ap.add_argument("--reply-words", type=int, default=600)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--siblings", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=100.0)
    ap.add_argument("--scenarios", default="turns,siblings,edit")
    ap.add_argument("--body-extra", default='{"enable_thinking": false}')
    a = ap.parse_args()
    extra = json.loads(a.body_extra)
    system = {"role": "system", "content": "You are a coding agent. " + text(1, a.system_words)}
    tl = tools()
    rows, failed = [], 0

    def run(scenario, conv, turn, messages):
        nonlocal failed
        body = {"model": a.model, "messages": messages, "tools": tl, "max_tokens": a.max_tokens, "temperature": 0,
                "stream": False}
        body.update(extra)
        try:
            ms, prompt, cached, answer = post(a.url, body, a.timeout)
            row = {"scenario": scenario, "conv": conv, "turn": turn, "prompt_tokens": prompt, "cached_tokens": cached,
                   "ttft_ms": round(ms, 1), "answer_sha": answer}
        except Exception as e:  # one failed request is reported, the session goes on
            failed += 1
            row = {"scenario": scenario, "conv": conv, "turn": turn, "error": str(e)[:200]}
        rows.append(row)
        print("session-turns " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    def history(conv, turns, edited=None):
        msgs = [system]
        for t in range(1, turns + 1):
            seed = conv * 1000 + t
            user = text(seed, a.user_words) if t != edited else "EDITED " + text(seed + 500, a.user_words)
            msgs.append({"role": "user", "content": user})
            if t < turns:
                msgs.append({"role": "assistant", "content": text(seed + 100, a.reply_words)})
        return msgs

    scenarios = a.scenarios.split(",")
    if "turns" in scenarios:
        for t in range(1, a.turns + 1):
            run("turns", 0, t, history(0, t))
    if "siblings" in scenarios:
        for c in range(1, a.siblings + 1):
            run("siblings", c, 1, history(c, 1))
    if "edit" in scenarios:
        run("edit", 0, a.turns, history(0, a.turns, edited=2))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "session-turns.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(a.out, "client-result.json"), "w", encoding="utf-8") as f:
        json.dump({"completed": len(rows) - failed, "failed": failed}, f)
    return 0 if failed == 0 and rows else 1


if __name__ == "__main__":
    sys.exit(main())
