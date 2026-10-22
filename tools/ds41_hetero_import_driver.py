#!/usr/bin/env python3
"""Hetero serving path, PC side: prefill a prompt on the local CUDA device, hand its attention state to the AMD pair's
strix-deepseek41-serve (POST /admin/import-state on rank 0) and decode the first tokens there, timing every part.

  python ds41_hetero_import_driver.py --server http://HOST:PORT --prompt-ids FILE --tokens N
      [--exe strix-cuda-ds41-prefill.exe] [--model MODEL.gguf] [--fast | --exact] [--prefill-arg=--slots --prefill-arg=1 ...]
      [--state-dir DIR] [--max-tokens 17] [--api-key KEY] [--expect 10316,294,...] [--json OUT]
  python ds41_hetero_import_driver.py --server URL --state FILE [...]   (no prefill: post an existing state file)

--fast: --fast-dense --fast-moe, plus --fast-attention from 8192 tokens (the qualified fast arm); --exact: none.
The prompt sent to /v1/completions is the state's own token history (read from the file), so the request continues
the imported state. Timings (s):
  prefill   the CUDA run (the prefill app's "seconds": model resident, tokenization excluded)
  export    the run's end (its JSON line) to the state file written (export from the device + save)
  read      reading the file back (RAM disk / page cache)
  transfer  the POST's wall time minus the server's own time (the bytes over the network + HTTP)
  import    the server's time (SHA-256, parse, rank 1 replication, device import on both ranks)
  first     the completion request to its first token (the state's next token: no decode step)
  second    the completion request to its second token (the first decode step on the pair)
  ttft      prefill + export + read + transfer + import + first (the user's first token, model load excluded)
and the decode rate of the tokens after the first.
"""
import argparse
import hashlib
import http.client
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.parse

STATE_FORMAT = 'strix/ds41-attention-state-v1'


def read_ids(path, n):
    text = open(path, encoding='utf-8').read().replace('\n', ',')
    ids = [int(t) for t in text.split(',') if t.strip()]
    if n is None:
        return ids
    return [ids[i % len(ids)] for i in range(n)]


class Server:
    def __init__(self, url, api_key=None):
        u = urllib.parse.urlparse(url)
        self.host, self.port = u.hostname, u.port or 80
        self.headers = {'Authorization': 'Bearer ' + api_key} if api_key else {}

    def conn(self, timeout=600):
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)

    def get(self, path):
        c = self.conn(10)
        c.request('GET', path, headers=self.headers)
        r = c.getresponse()
        return r.status, r.read()

    def post_state(self, blob, sha):
        c = self.conn()
        h = dict(self.headers)
        h.update({'Content-Type': 'application/octet-stream', 'X-State-Format': STATE_FORMAT, 'X-State-Sha256': sha})
        t0 = time.perf_counter()
        c.request('POST', '/admin/import-state', body=blob, headers=h)
        r = c.getresponse()
        body = r.read()
        t1 = time.perf_counter()
        if r.status != 200:
            raise RuntimeError(f'import-state: HTTP {r.status}: {body[:400]!r}')
        return json.loads(body), t1 - t0

    def complete_stream(self, prompt_ids, max_tokens):
        """Streams /v1/completions; returns (token ids, [arrival time of each text chunk], final usage/engine json)."""
        c = self.conn()
        h = dict(self.headers)
        h['Content-Type'] = 'application/json'
        body = json.dumps({'prompt': prompt_ids, 'max_tokens': max_tokens, 'stream': True, 'temperature': 0,
                           'return_token_ids': True, 'stream_options': {'include_usage': True}})
        t0 = time.perf_counter()
        c.request('POST', '/v1/completions', body=body, headers=h)
        r = c.getresponse()
        if r.status != 200:
            raise RuntimeError(f'completions: HTTP {r.status}: {r.read()[:400]!r}')
        arrivals, ids, last = [], None, None
        buf = b''
        while True:
            chunk = r.read1(65536) if hasattr(r, 'read1') else r.read(1)
            if not chunk:
                break
            now = time.perf_counter() - t0
            buf += chunk
            while b'\n\n' in buf:
                ev, buf = buf.split(b'\n\n', 1)
                for line in ev.split(b'\n'):
                    if not line.startswith(b'data:'):
                        continue
                    data = line[5:].strip()
                    if data == b'[DONE]':
                        continue
                    j = json.loads(data)
                    for ch in j.get('choices', []):
                        if ch.get('text'):
                            arrivals.append(now)
                    eng = j.get('strix') or {}
                    if eng:
                        last = j
                    if 'token_ids' in eng:
                        ids = eng['token_ids']
        return ids, arrivals, last

    def complete_ids(self, prompt_ids, max_tokens):
        """Non-streamed /v1/completions with return_token_ids (the exact token check)."""
        c = self.conn()
        h = dict(self.headers)
        h['Content-Type'] = 'application/json'
        body = json.dumps({'prompt': prompt_ids, 'max_tokens': max_tokens, 'temperature': 0, 'return_token_ids': True})
        t0 = time.perf_counter()
        c.request('POST', '/v1/completions', body=body, headers=h)
        r = c.getresponse()
        data = r.read()
        dt = time.perf_counter() - t0
        if r.status != 200:
            raise RuntimeError(f'completions: HTTP {r.status}: {data[:400]!r}')
        j = json.loads(data)
        return j, dt


def run_prefill(args, n, state_path):
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=args.cuda_device)
    flags = []
    if args.fast:
        flags = ['--fast-dense', '--fast-moe'] + (['--fast-attention'] if n >= 8192 else [])
    cmd = [args.exe, args.model, '--prompt-ids', args.prompt_ids, '--tokens', str(n), '--state-out', state_path] + flags + args.prefill_arg
    print('prefill:', ' '.join(cmd), flush=True)
    t_start = time.perf_counter()
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    out = {'cmd': cmd}
    for line in p.stdout:
        now = time.perf_counter()
        if not line.startswith('{'):
            continue
        j = json.loads(line.replace(chr(92), '/'))  # the app prints Windows paths unescaped
        ev = j.get('event')
        if ev == 'loaded':
            out['loaded'] = {k: j['describe'].get(k) for k in ('device_gb', 'pinned_gb', 'load_s', 'tiers', 'slots')}
            out['load_wall_s'] = now - t_start
            print(f"  loaded: device_gb {out['loaded']['device_gb']:.2f} load_s {out['loaded']['load_s']:.1f}", flush=True)
        elif ev == 'run':
            out['run'] = {k: j.get(k) for k in ('seconds', 'tok_s', 'next_token', 'logits_fnv', 'expert_io_s', 'expert_gb_streamed', 'engram_s')}
            out['t_run'] = now
            print(f"  run: {j['seconds']:.3f} s, next token {j['next_token']}, logits {j['logits_fnv']}", flush=True)
        elif ev == 'state':
            out['state'] = j
            out['t_state'] = now
            print(f"  state: {j['bytes']} B, export_s {j['export_s']:.4f}, digest {j['digest']}", flush=True)
            break  # hand the file on now; the process ends on its own
    return p, out


def import_and_decode(args, srv, blob, prompt_ids, res):
    sha = hashlib.sha256(blob).hexdigest()
    reply, post_s = srv.post_state(blob, sha)
    t_imported = time.perf_counter()
    if reply.get('sha256') != sha or reply.get('position') != len(prompt_ids):
        raise RuntimeError(f'import reply does not match: {reply}')
    res['import_reply'] = reply
    res['post_s'] = post_s
    res['import_s'] = reply['timing']['server_s']
    res['transfer_s'] = post_s - reply['timing']['server_s']
    print(f"  import: position {reply['position']} slot {reply['slot']} next {reply['next_token']} digest {reply['digest']}; "
          f"POST {post_s:.3f} s = transfer {res['transfer_s']:.3f} + server {res['import_s']:.3f} "
          f"(sha {reply['timing']['sha256_s']:.3f}, parse {reply['timing']['parse_s']:.3f}, queue {reply['timing']['queue_s']:.3f}, "
          f"rank1 {reply['timing']['replicate_s']:.3f}, device {reply['timing']['device_s']:.3f})", flush=True)
    ids, arrivals, last = srv.complete_stream(prompt_ids, args.max_tokens)
    res['request_gap_s'] = time.perf_counter() - t_imported
    res['tokens'] = ids
    res['arrivals'] = arrivals
    res['server_stats'] = {k: v for k, v in ((last or {}).get('strix') or {}).items() if k != 'token_ids'}
    # the server's prefill_ms (admission to first token) is ~0 when the request continued the parked state
    res['continued'] = (res['server_stats'].get('prefill_ms', 1e9) or 0) < 50
    if not res['continued']:
        print(f"  WARNING: the request did not continue the imported state (server prefill_ms {res['server_stats'].get('prefill_ms')})", flush=True)
    if arrivals:
        res['first_s'] = arrivals[0]
        if len(arrivals) > 1:
            res['second_s'] = arrivals[1]
            res['decode_tok_s'] = (len(arrivals) - 1) / (arrivals[-1] - arrivals[0]) if arrivals[-1] > arrivals[0] else None
    print(f"  completion: first token {res.get('first_s', float('nan')) * 1e3:.1f} ms, second {res.get('second_s', float('nan')) * 1e3:.1f} ms, "
          f"{len(arrivals)} chunks, decode {res.get('decode_tok_s') or 0:.2f} tok/s; tokens {ids}", flush=True)
    if args.expect:
        want = [int(t) for t in args.expect.split(',')]
        got = (ids or [])[:len(want)]
        res['expect_equal'] = got == want
        print(f"  CHECK vs reference ({len(want)} tokens): {'IDENTICAL' if got == want else 'DIFFERENT'}", flush=True)
        if got != want:
            print(f"    want {want}\n    got  {got}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--server', required=True)
    ap.add_argument('--api-key')
    ap.add_argument('--exe', default='strix-cuda-ds41-prefill.exe', help='the CUDA prefill app (default: from PATH)')
    ap.add_argument('--model', help='MODEL.gguf of the prefill app (required unless --state)')
    ap.add_argument('--prompt-ids')
    ap.add_argument('--tokens', type=int)
    ap.add_argument('--state', help='post this state file instead of running the prefill (needs --prompt-ids/--tokens for its tokens)')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--fast', action='store_true', default=True)
    g.add_argument('--exact', dest='fast', action='store_false')
    ap.add_argument('--prefill-arg', action='append', default=[], help='extra argument of the prefill app (repeat)')
    ap.add_argument('--cuda-device', default='0')
    ap.add_argument('--state-dir', default=tempfile.gettempdir())
    ap.add_argument('--max-tokens', type=int, default=17)
    ap.add_argument('--expect', help='comma-separated reference tokens (the first ones of the completion)')
    ap.add_argument('--json')
    args = ap.parse_args()
    if not args.state and not args.model:
        ap.error('--model is required unless --state is given')
    srv = Server(args.server, args.api_key)
    st, body = srv.get('/health')
    print(f'server {args.server}: /health {st} {body[:200]!r}', flush=True)
    if st != 200:
        sys.exit('server not ready')
    prompt_ids = read_ids(args.prompt_ids, args.tokens)
    res = {'server': args.server, 'tokens_in': len(prompt_ids), 'fast': args.fast, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
    if args.state:
        t0 = time.perf_counter()
        blob = open(args.state, 'rb').read()
        res['read_s'] = time.perf_counter() - t0
        res['state_file'] = args.state
        import_and_decode(args, srv, blob, prompt_ids, res)
    else:
        n = len(prompt_ids)
        path = os.path.join(args.state_dir, f'ds41-hetero-{n}-{os.getpid()}.state')
        p, out = run_prefill(args, n, path)
        res['prefill'] = out
        if 'state' not in out:
            p.wait()
            sys.exit('prefill failed: ' + p.stderr.read()[-2000:])
        res['prefill_s'] = out['run']['seconds']
        res['export_s'] = out['t_state'] - out['t_run']
        t0 = time.perf_counter()
        blob = open(path, 'rb').read()
        res['read_s'] = time.perf_counter() - t0
        import_and_decode(args, srv, blob, prompt_ids, res)
        p.wait()
        try:
            os.remove(path)
        except OSError:
            pass
        res['ttft_s'] = res['prefill_s'] + res['export_s'] + res['read_s'] + res['post_s'] + res.get('first_s', 0.0)
        print(f"TTFT {n} tokens: {res['ttft_s']:.3f} s = prefill {res['prefill_s']:.3f} + export {res['export_s']:.3f} + read {res['read_s']:.3f} "
              f"+ transfer {res['transfer_s']:.3f} + import {res['import_s']:.3f} + first token {res.get('first_s', 0.0):.3f}; "
              f"second token +{(res.get('second_s', 0.0) - res.get('first_s', 0.0)) * 1e3:.1f} ms; decode {res.get('decode_tok_s') or 0:.2f} tok/s", flush=True)
    if args.json:
        with open(args.json, 'a', encoding='utf-8') as f:
            f.write(json.dumps(res) + '\n')


if __name__ == '__main__':
    main()
