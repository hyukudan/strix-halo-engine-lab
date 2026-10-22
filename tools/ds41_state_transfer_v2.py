#!/usr/bin/env python3
"""Experimental disk-backed import-state v2 stub. No device state is imported.

The sender streams immutable sections, then supplies an ordered assembly manifest
for the original v1 file. Validation checks the full SHA before atomic commit.
Run explicitly with --root DIR; this does not change the v1 stub or serving defaults.
"""
import argparse
import copy
import hashlib
import http.server
import json
import os
from pathlib import Path
import re
import shutil
import struct
import threading
import time

CHUNK = 4 << 20
IDENTIFIER = re.compile(r'[0-9a-f]{32}\Z')
SHA256 = re.compile(r'[0-9a-f]{64}\Z')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value):
    require(type(value) is int and value >= 0, 'expected a non-negative integer')
    return value


def split_v1(data):
    """Keep cache payloads separate; concatenating the returned segments is byte-exact.

    This parses lengths only. The native v1 reader remains the semantic validator.
    """
    require(len(data) >= 48 and data[:8] == b'DS41ATST', 'invalid v1 header')
    version, position, _, layers = struct.unpack_from('<IIII', data, 8)
    require(version == 1 and 0 < layers <= 64, 'invalid v1 version/layers')
    at, previous = 24, 0
    payloads, segments, final = {}, [], bytearray()

    def rest(end):
        nonlocal previous
        if end > previous:
            segments.append({'section': 'final', 'offset': len(final), 'bytes': end - previous})
            final.extend(data[previous:end])
        previous = end

    for layer in range(layers):
        require(at + 5 <= len(data) - 16, 'truncated layer')
        at += 5
        for kind, width in (('ring', 1), ('kv', 1), ('idx', 1), ('carry', 4), ('topk', 4)):
            require(at + 8 <= len(data) - 16, 'truncated vector size')
            count, = struct.unpack_from('<Q', data, at)
            start, end = at + 8, at + 8 + count * width
            require(end <= len(data) - 16, 'truncated vector')
            if kind in ('kv', 'idx') and end > start:
                rest(start)
                name = f'{kind}.{layer}'
                payloads[name] = memoryview(data)[start:end]
                segments.append({'section': name, 'offset': 0, 'bytes': end - start})
                previous = end
            at = end
    require(at + 8 <= len(data) - 16, 'truncated token size')
    tokens, = struct.unpack_from('<Q', data, at)
    require(tokens == position and at + 8 + 4 * tokens + 16 == len(data), 'invalid v1 token history/trailer')
    rest(len(data))
    payloads['final'] = final
    return position, payloads, segments


class Store:
    """Single-process stub transaction store. All callers serialize through lock.

    Partial transactions are never visible at committed/<id>.state. Restart resume
    and a distributed two-rank commit are deliberately outside this local stub.
    """
    def __init__(self, root, max_bytes=1536 << 20, max_active=1, expected_identity=None, lease_seconds=900):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'committed').mkdir(exist_ok=True)
        self.max_bytes, self.max_active = max_bytes, max_active
        self.expected_identity = expected_identity
        self.lease_seconds = lease_seconds
        self.transactions = {}
        self.lock = threading.Lock()

    def begin(self, key, manifest):
        require(IDENTIFIER.fullmatch(key), 'transaction id must be 32 lowercase hex digits')
        require(type(manifest) is dict, 'manifest must be an object')
        position = integer(manifest.get('position'))
        require(0 < position <= 2097152, 'position outside 1..2M')
        identity = manifest.get('identity')
        require(isinstance(identity, str) and 0 < len(identity) <= 256, 'effective model identity required')
        if self.expected_identity is not None:
            require(identity == self.expected_identity, 'effective model identity differs')
        sections = manifest.get('sections')
        require(type(sections) is list and 0 < len(sections) <= 130, 'invalid sections')
        names, total = set(), 0
        for section in sections:
            require(type(section) is dict, 'invalid section')
            name, capacity = section.get('name'), integer(section.get('bytes'))
            require(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', name) and name not in names, 'invalid/duplicate section name')
            names.add(name)
            total += capacity
        require(total <= self.max_bytes, 'transaction exceeds byte budget')
        now = time.monotonic()
        for old, transaction in list(self.transactions.items()):
            if transaction['stage'] != 'committed' and now - transaction['touched'] > self.lease_seconds:
                self.abort(old)
        if key in self.transactions:
            require(self.transactions[key]['manifest'] == manifest, 'conflicting begin retry')
            self.transactions[key]['touched'] = now
            return {'begun': True, 'id': key, 'retry': True}
        require(sum(t['stage'] != 'committed' for t in self.transactions.values()) < self.max_active, 'active transaction limit')
        require(len(self.transactions) < 128, 'stub receipt limit; use a fresh root/process')
        folder = self.root / key
        require(not folder.exists() and not (self.root / 'committed' / f'{key}.state').exists(), 'transaction already exists on disk')
        folder.mkdir()
        self.transactions[key] = dict(manifest=copy.deepcopy(manifest), folder=folder, stage='begun', touched=now,
            sections={s['name']: dict(capacity=s['bytes'], path=folder / str(i), chunks={}) for i, s in enumerate(sections)})
        return {'begun': True, 'id': key}

    def chunk(self, key, name, offset, body, checksum):
        t = self.transactions[key]
        require(t['stage'] == 'begun', 'chunks require begun transaction')
        require(name in t['sections'], 'unknown section')
        section = t['sections'][name]
        offset = integer(offset)
        require(0 < len(body) <= CHUNK and offset <= section['capacity'] and len(body) <= section['capacity'] - offset, 'chunk outside section or 4 MiB bound')
        require(isinstance(checksum, str) and SHA256.fullmatch(checksum), 'invalid SHA256')
        require(hashlib.sha256(body).hexdigest() == checksum, 'chunk SHA256 mismatch')
        old = section['chunks'].get(offset)
        if old is not None:
            require(old == (len(body), checksum), 'conflicting chunk retry')
            t['touched'] = time.monotonic()
            return {'received': True, 'retry': True}
        require(all(offset + len(body) <= a or a + n <= offset for a, (n, _) in section['chunks'].items()), 'overlapping chunk')
        with open(section['path'], 'r+b' if section['path'].exists() else 'w+b') as out:
            out.seek(offset)
            out.write(body)
        section['chunks'][offset] = (len(body), checksum)
        t['touched'] = time.monotonic()
        return {'received': True, 'bytes': len(body)}

    def finalize(self, key, manifest):
        t = self.transactions[key]
        if t['stage'] != 'begun':
            require(t['assembly'] == manifest, 'conflicting finalize retry')
            t['touched'] = time.monotonic()
            return {'finalized': True, 'retry': True}
        require(type(manifest) is dict and SHA256.fullmatch(manifest.get('sha256', '')), 'final SHA256 required')
        segments = manifest.get('segments')
        require(type(segments) is list and 0 < len(segments) <= 512, 'invalid assembly segments')
        used, total = {}, 0
        for segment in segments:
            name = segment['section']
            offset, count = integer(segment['offset']), integer(segment['bytes'])
            require(name in t['sections'] and count > 0, 'invalid assembly section')
            require(offset == used.get(name, 0) and count <= t['sections'][name]['capacity'] - offset, 'assembly must cover each used section once in order')
            used[name] = offset + count
            total += count
        require(total <= self.max_bytes, 'assembly exceeds byte budget')
        for name, section in t['sections'].items():
            covered = 0
            for offset, (count, _) in sorted(section['chunks'].items()):
                require(offset == covered, 'missing section bytes')
                covered += count
            require(covered == used.get(name, 0), 'assembly/chunk coverage differs')
        t.update(stage='finalized', assembly=copy.deepcopy(manifest), bytes=total, touched=time.monotonic())
        return {'finalized': True, 'bytes': total}

    def validate(self, key):
        t = self.transactions[key]
        if t['stage'] in ('validated', 'committed'):
            t['touched'] = time.monotonic()
            return dict(t['receipt'], retry=True)
        require(t['stage'] == 'finalized', 'finalize required before validation')
        started, digest = time.perf_counter(), hashlib.sha256()
        assembled = t['folder'] / 'assembled.state'
        with open(assembled, 'wb') as out:
            for segment in t['assembly']['segments']:
                with open(t['sections'][segment['section']]['path'], 'rb') as source:
                    source.seek(segment['offset'])
                    remaining = segment['bytes']
                    while remaining:
                        data = source.read(min(remaining, CHUNK))
                        require(data, 'truncated staging file')
                        out.write(data)
                        digest.update(data)
                        remaining -= len(data)
            out.flush()
            os.fsync(out.fileno())
        require(digest.hexdigest() == t['assembly']['sha256'], 'assembled v1 SHA256 mismatch')
        with open(assembled, 'rb') as source:
            header = source.read(24)
        require(len(header) == 24 and header[:8] == b'DS41ATST', 'invalid assembled v1 header')
        version, position, next_token, layers = struct.unpack_from('<IIII', header, 8)
        require(version == 1 and position == t['manifest']['position'] and 0 < layers <= 64, 'assembled v1 metadata differs')
        receipt = dict(validated=True, id=key, sha256=digest.hexdigest(), position=position, next_token=next_token,
                       bytes=t['bytes'], validation_s=time.perf_counter() - started)
        t.update(stage='validated', receipt=receipt, touched=time.monotonic())
        return receipt

    def commit(self, key):
        t = self.transactions[key]
        require(t['stage'] in ('validated', 'committed'), 'validation required before commit')
        if t['stage'] != 'committed':
            require(t['folder'].resolve().parent == self.root, 'staging folder escaped root')
            os.replace(t['folder'] / 'assembled.state', self.root / 'committed' / f'{key}.state')
            t['stage'] = 'committed'
            try:
                shutil.rmtree(t['folder'])
            except OSError:
                pass  # cleanup failure must not undo publication or break an idempotent commit retry
        return dict(t['receipt'], committed=True, imported=False)

    def abort(self, key):
        t = self.transactions[key]
        require(t['stage'] != 'committed', 'committed transaction cannot be aborted')
        require(t['folder'].resolve().parent == self.root, 'staging folder escaped root')
        shutil.rmtree(t['folder'])
        del self.transactions[key]
        return {'aborted': True}


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    store = None
    api_key = None

    def log_message(self, *_):
        pass

    def reply(self, status, value):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_GET(self):
        self.reply(200 if self.path == '/health' else 404, {'status': 'ok', 'stub': True, 'device_import': False})

    def do_POST(self):
        try:
            if self.api_key and self.headers.get('Authorization') != 'Bearer ' + self.api_key:
                return self.reply(401, {'error': 'invalid or missing API key'})
            parts = self.path.split('/')
            require(len(parts) == 6 and parts[1:4] == ['admin', 'import-state', 'v2'] and IDENTIFIER.fullmatch(parts[4]), 'unknown v2 path')
            key, action = parts[4:]
            size = int(self.headers.get('Content-Length', '-1'))
            require(0 <= size <= CHUNK, 'body exceeds 4 MiB bound or lacks Content-Length')
            self.connection.settimeout(30)
            body = self.rfile.read(size)
            require(len(body) == size, 'truncated body')
            with self.store.lock:
                if action == 'chunk':
                    result = self.store.chunk(key, self.headers.get('X-State-Section'), int(self.headers.get('X-State-Offset', '-1')), body, self.headers.get('X-State-Sha256'))
                elif action in ('begin', 'finalize'):
                    result = getattr(self.store, action)(key, json.loads(body))
                elif action in ('validate', 'commit', 'abort'):
                    require(not body or body == b'{}', 'unexpected operation body')
                    result = getattr(self.store, action)(key)
                else:
                    raise ValueError('unknown v2 operation')
            self.reply(200, result)
        except (ValueError, KeyError, TypeError, OSError) as error:
            self.reply(409, {'error': str(error), 'committed': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--api-key')
    parser.add_argument('--expected-identity', required=True)
    args = parser.parse_args()
    Handler.store = Store(args.root, expected_identity=args.expected_identity)
    Handler.api_key = args.api_key
    server = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'v2 local stub on {args.host}:{args.port}; no device import', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
