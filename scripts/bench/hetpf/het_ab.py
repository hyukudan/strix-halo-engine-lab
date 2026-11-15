#!/usr/bin/env python3
"""hetpf client: the same prompt through three routes, one JSON line per request (appended to --out).

  python het_ab.py --arms A,B,A,B --ids FILE --tokens N --key KEYFILE --label T1 --mode fast --out OUT.jsonl
                   [--service http://127.0.0.1:PORT] [--pair http://NODE:PORT] [--max-tokens 65]

  A  RTX -> pair: POST /v1/prefill on the CUDA prefill service with "handoff": PAIR/admin/import-state (the service runs
     the prefill, exports the state, hashes it and POSTs it; rank 0 relays it to rank 1, both import), then a streamed
     /v1/completions on the pair whose prompt is the same ids (it takes the parked state: first token = the state's next
     token, the rest are pair decode steps).
  B  pair alone: the streamed /v1/completions with the same ids (the pair prefills; --mode names its policy).
  R  RTX alone: the service prefill without a hand-off (its next_token is the first token).
Per record: epoch start/end (for the GPU clock monitors), wall and per-phase seconds, ttft_s, the generated ids, first-token
time, decode tok/s of the tokens after the first, the service and import replies (sha256 / digest / position) and the
server's own stats. A completion that did not continue the imported state (server prefill_ms >= 50) is flagged.
The pair client (Server) is tools/ds41_hetero_import_driver.py's, imported, not copied.
"""
import argparse
import http.client
import json
import os
import sys
import time
import urllib.parse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'tools'))
from ds41_hetero_import_driver import Server, read_ids  # noqa: E402


def service_prefill(url, key, ids, session, handoff=None, reserve=0):
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=1800)
    body = {'tokens': ids, 'session': session}
    if handoff:
        body.update({'handoff': handoff, 'api_key': key, 'reserve_tokens': reserve})
    h = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key, 'x-api-key': key}
    t0 = time.perf_counter()
    c.request('POST', '/v1/prefill', json.dumps(body), h)
    r = c.getresponse()
    reply = json.loads(r.read() or b'{}')
    return r.status, reply, time.perf_counter() - t0


def stream(srv, ids, max_tokens, rec):
    out_ids, arrivals, last = srv.complete_stream(ids, max_tokens)
    stats = {k: v for k, v in ((last or {}).get('strix') or {}).items() if k != 'token_ids'}
    rec.update({'gen_ids': out_ids, 'n_chunks': len(arrivals), 'server_stats': stats})
    if arrivals:
        rec['first_s'] = arrivals[0]
        if len(arrivals) > 1 and arrivals[-1] > arrivals[0]:
            rec['decode_tok_s'] = (len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
            rec['decode_s'] = arrivals[-1] - arrivals[0]
    return arrivals


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arms', required=True)
    p.add_argument('--ids', required=True)
    p.add_argument('--tokens', type=int, required=True)
    p.add_argument('--key', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--mode', default='-', help='pair prefill policy of this process (exact|fast), recorded')
    p.add_argument('--out', required=True)
    p.add_argument('--service')
    p.add_argument('--pair')
    p.add_argument('--max-tokens', type=int, default=65)
    a = p.parse_args()
    key = open(a.key).read().strip()
    ids = read_ids(a.ids, a.tokens)
    srv = Server(a.pair, key) if a.pair else None
    seen = {}
    for i, arm in enumerate(a.arms.split(',')):
        seen[arm] = seen.get(arm, 0) + 1
        rec = {'label': a.label, 'mode': a.mode, 'arm': arm, 'index': i, 'nth_in_process': seen[arm], 'tokens': len(ids),
               'epoch_start': time.time()}
        try:
            if arm in ('A', 'R'):
                handoff = a.pair.rstrip('/') + '/admin/import-state' if arm == 'A' else None
                st, reply, wall = service_prefill(a.service, key, ids, f'{a.label}-{i}-{arm}', handoff, a.max_tokens + 16)
                rec.update({'status': st, 'service_wall_s': wall, 'service': {k: reply.get(k) for k in (
                    'next_token', 'logits_fnv', 'state_bytes', 'sha256', 'digest', 'seconds', 'fast', 'import', 'error')}})
                if st != 200:
                    raise RuntimeError(f'prefill HTTP {st}: {str(reply.get("error"))[:300]}')
                if arm == 'R':
                    rec['ttft_s'] = wall
                else:
                    imp = reply.get('import') or {}
                    if imp.get('sha256') and imp.get('sha256') != reply.get('sha256'):
                        raise RuntimeError('import sha256 differs from the exported state')
                    rec['t_stream0'] = time.perf_counter()
                    stream(srv, ids, a.max_tokens, rec)
                    rec['ttft_s'] = wall + rec.get('first_s', float('nan'))
                    rec['continued'] = (rec['server_stats'].get('prefill_ms', 1e9) or 0) < 50
            elif arm == 'B':
                stream(srv, ids, a.max_tokens, rec)
                rec['ttft_s'] = rec.get('first_s', float('nan'))
            else:
                raise ValueError('arm must be A, B or R')
        except Exception as e:  # recorded, the next arm still runs
            rec['error'] = f'{type(e).__name__}: {e}'[:400]
        rec['epoch_end'] = time.time()
        with open(a.out, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        sec = (rec.get('service') or {}).get('seconds') or {}
        print(f"{a.label} {arm}#{seen[arm]} ttft {rec.get('ttft_s', float('nan')):.3f} s pf {sec.get('prefill', 0):.2f} exp "
              f"{sec.get('export', 0):.3f} xfer {sec.get('transfer', 0):.3f} imp {sec.get('import', 0):.3f} first "
              f"{rec.get('first_s', 0):.3f} dec {rec.get('decode_tok_s') or 0:.2f} tok/s next "
              f"{(rec.get('gen_ids') or [None])[0]} {rec.get('error', '')}", flush=True)


if __name__ == '__main__':
    sys.exit(main())
