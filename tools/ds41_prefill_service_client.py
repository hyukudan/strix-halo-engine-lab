#!/usr/bin/env python3
"""Client of strix-cuda-ds41-prefill-service: POST /v1/prefill of a prompt (token ids), optionally handed to a decode
server's import route, printing the service's JSON reply and the client-side wall time.

  python ds41_prefill_service_client.py --service http://127.0.0.1:8095 --prompt-ids FILE [--tokens N]
      [--handoff http://HOST:PORT/admin/import-state] [--session S] [--repeat R] [--api-key KEY] [--json OUT]
  python ds41_prefill_service_client.py --service URL --describe | --health

--tokens N tiles the prompt file to N ids (as the one-shot app's --tokens with --prompt-ids). Each request line adds
"wall_s" (the client's time from the POST to the reply) and "overhead_s" (wall - prefill: queue, export, SHA-256,
handoff, HTTP).
"""
import argparse
import http.client
import json
import sys
import time
import urllib.parse


def read_ids(path, n):
    text = open(path, encoding='utf-8').read().replace('\n', ',')
    ids = [int(t) for t in text.replace('[', ',').replace(']', ',').split(',') if t.strip()]
    return ids if n is None else [ids[i % len(ids)] for i in range(n)]


def call(url, method, path, body=None, api_key=None, timeout=3600):
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    h = {'Content-Type': 'application/json'}
    if api_key:
        h['Authorization'] = 'Bearer ' + api_key
    c.request(method, path, body=None if body is None else json.dumps(body), headers=h)
    r = c.getresponse()
    data = r.read()
    try:
        return r.status, json.loads(data)
    except ValueError:
        return r.status, {'raw': data[:400].decode(errors='replace')}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--service', default='http://127.0.0.1:8095')
    ap.add_argument('--prompt-ids')
    ap.add_argument('--tokens', type=int)
    ap.add_argument('--handoff')
    ap.add_argument('--session')
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--api-key')
    ap.add_argument('--json')
    ap.add_argument('--describe', action='store_true')
    ap.add_argument('--health', action='store_true')
    a = ap.parse_args()
    if a.describe or a.health:
        st, j = call(a.service, 'GET', '/v1/describe' if a.describe else '/health', api_key=a.api_key)
        print(st, json.dumps(j))
        return 0 if st == 200 else 1
    if not a.prompt_ids:
        sys.exit('--prompt-ids is needed')
    ids = read_ids(a.prompt_ids, a.tokens)
    body = {'tokens': ids}
    if a.handoff:
        body['handoff'] = a.handoff
    rc = 0
    for r in range(a.repeat):
        if a.session:
            body['session'] = f'{a.session}-{r}'
        t0 = time.perf_counter()
        st, j = call(a.service, 'POST', '/v1/prefill', body, a.api_key)
        wall = time.perf_counter() - t0
        j['http_status'] = st
        j['wall_s'] = wall
        if 'seconds' in j and 'prefill' in j['seconds']:
            j['overhead_s'] = wall - j['seconds']['prefill']
        print(json.dumps(j), flush=True)
        if a.json:
            with open(a.json, 'a', encoding='utf-8') as f:
                f.write(json.dumps(j) + '\n')
        if st != 200:
            rc = 1
    return rc


if __name__ == '__main__':
    sys.exit(main())
