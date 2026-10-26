#!/usr/bin/env python3
"""Serve load for strix-minimax-serve (stdlib only): C concurrent streaming clients, each sending R chat requests.

Per request: TTFT (first streamed content byte), decode tok/s (completion tokens after the first / time after it),
total time. Per run: aggregate completion tok/s over the wall time, medians. Prints one JSON line per run plus the
first request's text (so the output can be read).

usage: serve_load.py URL --concurrency 1,4,8 [--requests 2] [--max-tokens 128] [--prompt-file F | --prompt TEXT]
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request


def one(url, messages, max_tokens):
    body = json.dumps({"model": "minimax", "messages": messages, "max_tokens": max_tokens, "stream": True}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    chunks = 0
    text = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            delta = json.loads(data)["choices"][0].get("delta", {}).get("content", "")
            if delta:
                if first is None:
                    first = time.perf_counter()
                chunks += 1
                text.append(delta)
    t1 = time.perf_counter()
    return {"ttft": (first or t1) - t0, "total": t1 - t0, "chunks": chunks, "text": "".join(text),
            "decode_tok_s": (chunks - 1) / (t1 - first) if first and chunks > 1 and t1 > first else 0.0}


def run(url, conc, requests, messages, max_tokens):
    out = []
    lock = threading.Lock()

    def client():
        for _ in range(requests):
            r = one(url, messages, max_tokens)
            with lock:
                out.append(r)

    t0 = time.perf_counter()
    th = [threading.Thread(target=client) for _ in range(conc)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.perf_counter() - t0
    toks = sum(r["chunks"] for r in out)
    return {"concurrency": conc, "requests": len(out), "wall_s": round(wall, 3), "aggregate_tok_s": round(toks / wall, 3),
            "ttft_median_s": round(statistics.median(r["ttft"] for r in out), 3),
            "decode_tok_s_median": round(statistics.median(r["decode_tok_s"] for r in out), 3),
            "completion_chunks": toks}, out[0]["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--concurrency", default="1,4,8")
    ap.add_argument("--requests", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompt", default="Explain in three short paragraphs how a hash table handles collisions.")
    ap.add_argument("--prompt-file")
    a = ap.parse_args()
    prompt = open(a.prompt_file, encoding="utf-8").read() if a.prompt_file else a.prompt
    messages = [{"role": "user", "content": prompt}]
    for c in [int(x) for x in a.concurrency.split(",")]:
        res, text = run(a.url, c, a.requests, messages, a.max_tokens)
        print(json.dumps(res), flush=True)
        print("text: " + json.dumps(text[:600], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
