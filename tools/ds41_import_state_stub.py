#!/usr/bin/env python3
"""A stand-in for strix-deepseek41-serve's POST /admin/import-state (tests of the prefill service's handoff without
the decode pair): checks X-State-Format and X-State-Sha256 against the body, reads the state's header (position,
next token, layer count) and trailing digest, and answers like the serve route ({"imported":true,"position",
"sha256","next_token","digest","bytes","slot","timing":{sha256_s,parse_s,queue_s,replicate_s,device_s,server_s}}).

  python ds41_import_state_stub.py [--host 127.0.0.1] [--port 8096] [--api-key KEY] [--log FILE] [--save-dir DIR]

GET /health answers {"status":"ok","imports":N}. --log appends one JSON line per import; --save-dir keeps the bodies.
"""
import argparse
import hashlib
import http.server
import json
import os
import struct
import threading
import time

FORMAT = 'strix/ds41-attention-state-v1'
MAGIC = b'DS41ATST'


class Stub(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    imports = 0
    lock = threading.Lock()
    args = None

    def log_message(self, fmt, *a):
        pass

    def reply(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def fail(self, status, msg):
        self.reply(status, {'error': {'message': msg}, 'imported': False})

    def do_GET(self):
        if self.path == '/health':
            return self.reply(200, {'status': 'ok', 'imports': Stub.imports})
        self.fail(404, 'unknown path')

    def do_POST(self):
        t0 = time.perf_counter()
        n = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(n)
        if self.path != '/admin/import-state':
            return self.fail(404, 'unknown path')
        key = Stub.args.api_key
        if key and self.headers.get('Authorization') != 'Bearer ' + key:
            return self.fail(401, 'invalid or missing API key')
        if self.headers.get('X-State-Format', FORMAT) != FORMAT:
            return self.fail(415, 'X-State-Format must be ' + FORMAT)
        if not body:
            return self.fail(400, 'empty body')
        t_sha = time.perf_counter()
        sha = hashlib.sha256(body).hexdigest()
        want = (self.headers.get('X-State-Sha256') or '').lower()
        if want and want != sha:
            return self.fail(400, f'X-State-Sha256 does not match the body ({sha})')
        sha_s = time.perf_counter() - t_sha
        t_parse = time.perf_counter()
        if len(body) < 36 or body[:8] != MAGIC:
            return self.fail(400, 'not a valid state: magic')
        version, position, next_token, layers = struct.unpack_from('<IIII', body, 8)
        digest, = struct.unpack_from('<Q', body, len(body) - 16)
        if version != 1:
            return self.fail(400, 'not a valid state: version')
        parse_s = time.perf_counter() - t_parse
        with Stub.lock:
            Stub.imports += 1
            count = Stub.imports
        if Stub.args.save_dir:
            with open(os.path.join(Stub.args.save_dir, f'import-{count}-{sha[:16]}.state'), 'wb') as f:
                f.write(body)
        out = {'imported': True, 'position': position, 'sha256': sha, 'next_token': next_token, 'digest': f'{digest:016x}',
               'bytes': len(body), 'slot': 0, 'layers': layers,
               'timing': {'sha256_s': sha_s, 'parse_s': parse_s, 'queue_s': 0.0, 'replicate_s': 0.0, 'device_s': 0.0,
                          'server_s': time.perf_counter() - t0}}
        if Stub.args.log:
            with Stub.lock, open(Stub.args.log, 'a', encoding='utf-8') as f:
                f.write(json.dumps(dict(out, time=time.strftime('%Y-%m-%d %H:%M:%S'))) + '\n')
        print(f"import {count}: position {position} next {next_token} digest {digest:016x} {len(body)} B sha {sha[:16]}..", flush=True)
        self.reply(200, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8096)
    ap.add_argument('--api-key')
    ap.add_argument('--log')
    ap.add_argument('--save-dir')
    Stub.args = ap.parse_args()
    srv = http.server.ThreadingHTTPServer((Stub.args.host, Stub.args.port), Stub)
    print(f'import-state stub on http://{Stub.args.host}:{Stub.args.port}/admin/import-state', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
