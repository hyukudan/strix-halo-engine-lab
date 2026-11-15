"""Prefill TTFT client for one running Qwen server (OpenAI /v1/completions with token-id prompts).

For every length: a cold request then a warm request, each on a distinct slice of the token file so a prefix
cache cannot hit; prints client TTFT and the server's own prefill_ms (strix.rtx / strix fields when present).
Usage: prefill_client.py BASE_URL IDS_FILE OUT_TXT [--lens 8192,32768] [--runs 2] [--max-tokens 4]
IDS_FILE: whitespace- or comma-separated token ids of one long text (tokenised for this model).
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path


def wait_health(base, limit):
    t0 = time.time()
    while time.time() - t0 < limit:
        try:
            with urllib.request.urlopen(base + '/health', timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except OSError:
            pass
        time.sleep(2)
    raise SystemExit(f'server not healthy after {limit} s')


def stream(base, prompt, max_tokens, timeout):
    body = dict(prompt=prompt, max_tokens=max_tokens, temperature=0, stream=True, return_token_ids=True)
    req = urllib.request.Request(base + '/v1/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    first, strix, error = None, None, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line.startswith(b'data:') or b'[DONE]' in line:
                continue
            try:
                obj = json.loads(line[5:])
            except ValueError:
                continue
            if obj.get('error'):
                error = obj['error']
            if isinstance(obj.get('strix'), dict):
                strix = obj['strix']
            if first is None and any((c.get('text') or '') for c in obj.get('choices') or []):
                first = time.perf_counter()
    ttft = (first - t0) if first else None
    return dict(ttft=ttft, prefill_ms=(strix or {}).get('prefill_ms'), error=error)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base')
    ap.add_argument('ids')
    ap.add_argument('out')
    ap.add_argument('--lens', default='8192,32768')
    ap.add_argument('--runs', type=int, default=2, help='requests per length; the first is cold, the last warm')
    ap.add_argument('--max-tokens', type=int, default=4)
    ap.add_argument('--health-s', type=int, default=900)
    a = ap.parse_args()
    ids = [int(x) for x in Path(a.ids).read_text().replace(',', ' ').split()]
    off = 0
    with open(a.out, 'a') as out:
        def say(s):
            print(s, flush=True)
            out.write(s + '\n')
            out.flush()
        say(f'health_s {wait_health(a.base, a.health_s):.1f}')
        for length in [int(x) for x in a.lens.split(',')]:
            for run in range(a.runs):
                if off + length > len(ids):
                    off = 0
                prompt = ids[off:off + length]
                off += length + 7
                r = stream(a.base, prompt, a.max_tokens, 1800)
                tag = 'cold' if run == 0 else ('warm' if run == a.runs - 1 else f'run{run}')
                pf = r['prefill_ms']
                say(f'pf L={length} {tag} ttft={r["ttft"] and round(r["ttft"], 3)} prefill_ms={pf} '
                    f'tok/s={pf and round(length / (pf / 1e3))} err={r["error"]}')


if __name__ == '__main__':
    main()
