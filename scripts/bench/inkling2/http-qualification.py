#!/usr/bin/env python3
"""Collect one Mini HTTP timing receipt for a fixed, pretokenized prompt."""

import argparse
import json
from pathlib import Path
import time
import urllib.request


def measure(url, prompt, max_tokens):
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        data=json.dumps(dict(prompt=prompt, max_tokens=max_tokens, temperature=0,
                             stream=True, return_token_ids=True)).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    first = final = None
    done = False
    with urllib.request.urlopen(request, timeout=3600) as reply:
        for line in reply:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                done = True
                break
            chunk = json.loads(payload)
            if "error" in chunk:
                raise ValueError(f"generation failed: {chunk['error']}")
            if first is None and any(choice.get("text") for choice in chunk.get("choices", [])):
                first = time.monotonic()
            if "usage" in chunk:
                final = chunk
    if not done or first is None or final is None:
        raise ValueError("missing first content, terminal usage, or [DONE]")
    if final["usage"]["prompt_tokens"] != len(prompt):
        raise ValueError("server reported a different prompt length")
    stats = final["strix"]
    ids = stats["token_ids"]
    if len(ids) < 2 or len(ids) != final["usage"]["completion_tokens"]:
        raise ValueError("missing complete IDs or no measured decode")
    prefill_ms = stats["prefill_ms"]
    decode_rate = stats["decode_tokens_per_second"]
    if not prefill_ms > 0 or not decode_rate > 0:
        raise ValueError("missing server-side prefill or decode timing")
    return dict(ttft_ms=1000 * (first - start), prompt_tokens=len(prompt),
                prefill_tok_s=1000 * len(prompt) / prefill_ms, decode_tok_s=decode_rate,
                finish=final["choices"][0]["finish_reason"], token_ids=ids,
                receipt=stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("prompt", type=Path, help="whitespace-separated token IDs")
    parser.add_argument("rows", type=int, help="required prompt token count")
    parser.add_argument("--generate", type=int, default=64)
    args = parser.parse_args()
    prompt = [int(value) for value in args.prompt.read_text().split()]
    if len(prompt) != args.rows or not prompt or min(prompt) < 0 or args.generate < 2:
        parser.error("need exactly ROWS nonnegative token IDs and at least 2 generated tokens")
    print(json.dumps(measure(args.url, prompt, args.generate), allow_nan=False))


if __name__ == "__main__":
    main()
