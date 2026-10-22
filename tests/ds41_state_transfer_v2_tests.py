#!/usr/bin/env python3
"""CPU transport tests; optional --state checks a previously qualified native v1 artifact."""
import argparse
import copy
import hashlib
import http.client
import json
from pathlib import Path
import struct
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import ds41_state_transfer_v2 as v2


def container():
    position = 32768
    data = bytearray(b'DS41ATST' + struct.pack('<IIII', 1, position, 17, 1))
    data += struct.pack('<BI', 11, 2)
    for count, width in ((136*528, 1), (position//2*288, 1), (position//2*68, 1), (1024, 4), (512, 4)):
        data += struct.pack('<Q', count)
        data += bytes((i * 31 + 7) % 256 for i in range(count * width))
    data += struct.pack('<Q', position) + bytes(position * 4) + bytes(16)
    return data  # structural transport fixture; semantic digest is intentionally unspecified


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = container()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = v2.Store(self.tmp.name, expected_identity='fixture')
        self.key = uuid.uuid4().hex
        self.position, self.payloads, self.segments = v2.split_v1(self.data)
        self.manifest = dict(position=self.position, identity='fixture',
                             sections=[dict(name=k, bytes=len(b)) for k, b in self.payloads.items()])

    def fill(self):
        self.store.begin(self.key, self.manifest)
        for name, data in reversed(list(self.payloads.items())):
            for at in reversed(range(0, len(data), v2.CHUNK)):
                piece = data[at:at+v2.CHUNK]
                self.store.chunk(self.key, name, at, piece, hashlib.sha256(piece).hexdigest())

    def finish(self, sha=None):
        assembly = dict(segments=self.segments, sha256=sha or hashlib.sha256(self.data).hexdigest())
        self.store.finalize(self.key, assembly)
        return assembly

    def test_byte_exact_out_of_order_and_idempotent_commit(self):
        self.fill()
        self.assertTrue(self.store.begin(self.key, self.manifest)['retry'])
        name, data = next(iter(self.payloads.items()))
        piece = data[:v2.CHUNK]
        self.assertTrue(self.store.chunk(self.key, name, 0, piece, hashlib.sha256(piece).hexdigest())['retry'])
        assembly = self.finish()
        self.assertFalse((Path(self.tmp.name) / 'committed' / f'{self.key}.state').exists())
        self.assertTrue(self.store.finalize(self.key, assembly)['retry'])
        self.store.validate(self.key)
        self.assertTrue(self.store.validate(self.key)['retry'])
        first = self.store.commit(self.key)
        self.assertEqual(first, self.store.commit(self.key))
        self.assertFalse(first['imported'])
        self.assertEqual((Path(self.tmp.name) / 'committed' / f'{self.key}.state').read_bytes(), self.data)

    def test_loss_corruption_overlap_and_reordered_operations(self):
        self.store.begin(self.key, self.manifest)
        with self.assertRaises(ValueError):
            self.store.commit(self.key)
        with self.assertRaises(ValueError):
            self.store.validate(self.key)
        with self.assertRaises(ValueError):
            self.finish()
        name, data = next(iter(self.payloads.items()))
        piece = data[:32]
        with self.assertRaises(ValueError):
            self.store.chunk(self.key, name, 0, piece, '0'*64)
        self.store.chunk(self.key, name, 0, piece, hashlib.sha256(piece).hexdigest())
        for offset, body in ((0, bytes(32)), (16, piece), (len(data), piece), (-1, piece), (0, bytes(v2.CHUNK+1))):
            with self.assertRaises(ValueError):
                self.store.chunk(self.key, name, offset, body, hashlib.sha256(body).hexdigest())
        self.store.abort(self.key)
        self.assertFalse((Path(self.tmp.name) / self.key).exists())

    def test_final_sha_rejection_never_publishes(self):
        self.fill()
        self.finish('0'*64)
        with self.assertRaises(ValueError):
            self.store.validate(self.key)
        with self.assertRaises(ValueError):
            self.store.commit(self.key)
        self.assertEqual(list((Path(self.tmp.name) / 'committed').iterdir()), [])

    def test_commit_io_failure_and_cleanup_failure_are_retry_safe(self):
        self.fill()
        self.finish()
        self.store.validate(self.key)
        with mock.patch.object(v2.os, 'replace', side_effect=OSError('injected rename failure')):
            with self.assertRaises(OSError):
                self.store.commit(self.key)
        self.assertEqual(list((Path(self.tmp.name) / 'committed').iterdir()), [])
        with mock.patch.object(v2.shutil, 'rmtree', side_effect=OSError('injected cleanup failure')):
            receipt = self.store.commit(self.key)
        self.assertEqual(receipt, self.store.commit(self.key))
        with self.assertRaises(ValueError):
            self.store.abort(self.key)

    def test_staging_corruption_is_detected_at_validate(self):
        self.fill()
        self.finish()
        section = next(iter(self.store.transactions[self.key]['sections'].values()))
        with open(section['path'], 'r+b') as staged:
            staged.write(b'changed-after-ack')
        with self.assertRaises(ValueError):
            self.store.validate(self.key)
        self.assertEqual(list((Path(self.tmp.name) / 'committed').iterdir()), [])

    def test_orphan_lease_releases_admission_without_touching_committed_files(self):
        self.fill()
        folder = self.store.transactions[self.key]['folder']
        touched = self.store.transactions[self.key]['touched']
        new_key = uuid.uuid4().hex
        with mock.patch.object(v2.time, 'monotonic', return_value=touched + 901):
            self.store.begin(new_key, self.manifest)
        self.assertNotIn(self.key, self.store.transactions)
        self.assertFalse(folder.exists())
        self.assertIn(new_key, self.store.transactions)

    def test_begin_limits_identity_conflicts_and_caller_mutation(self):
        for key, patch in (('../escape', {}), (self.key, {'identity': 'other'}),
                           (self.key, {'position': 2097153}), (self.key, {'sections': [{'name': '../file', 'bytes': 1}]}),
                           (self.key, {'sections': [{'name': 'large', 'bytes': 2 << 30}]})):
            bad = dict(self.manifest, **patch)
            with self.assertRaises(ValueError):
                self.store.begin(key, bad)
        self.store.begin(self.key, self.manifest)
        original = copy.deepcopy(self.manifest)
        self.manifest['position'] = 1
        self.assertTrue(self.store.begin(self.key, original)['retry'])
        with self.assertRaises(ValueError):
            self.store.begin(self.key, self.manifest)
        with self.assertRaises(ValueError):
            self.store.begin(uuid.uuid4().hex, original)

    def test_assembly_cannot_duplicate_omit_or_exceed_sections(self):
        self.fill()
        for field, value in (('offset', 1), ('bytes', 1 << 31), ('section', 'missing')):
            segments = copy.deepcopy(self.segments)
            segments[0][field] = value
            with self.assertRaises(ValueError):
                self.store.finalize(self.key, dict(sha256='0'*64, segments=segments))
        with self.assertRaises(ValueError):
            self.store.finalize(self.key, dict(sha256='0'*64, segments=self.segments[:-1]))

    def test_truncated_or_trailing_v1(self):
        for data in (self.data[:24], self.data[:-1], self.data + b'extra'):
            with self.assertRaises(ValueError):
                v2.split_v1(data)

    def test_http_authenticated_flow(self):
        handler = type('TestHandler', (v2.Handler,), dict(store=self.store, api_key='test-secret'))
        server = v2.http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        def shutdown():
            server.shutdown()
            server.server_close()
            thread.join()
        self.addCleanup(shutdown)

        def post(action, body=b'{}', headers=None, authorized=True):
            connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=10)
            h = dict(headers or {})
            if authorized:
                h['Authorization'] = 'Bearer test-secret'
            connection.request('POST', f'/admin/import-state/v2/{self.key}/{action}', body, h)
            response = connection.getresponse()
            status, payload = response.status, json.loads(response.read())
            connection.close()
            return status, payload
        self.assertEqual(post('begin', json.dumps(self.manifest), authorized=False)[0], 401)
        self.assertEqual(post('begin', json.dumps(self.manifest))[0], 200)
        for name, data in self.payloads.items():
            for at in range(0, len(data), v2.CHUNK):
                piece = data[at:at+v2.CHUNK]
                self.assertEqual(post('chunk', piece, {'X-State-Section': name, 'X-State-Offset': str(at),
                    'X-State-Sha256': hashlib.sha256(piece).hexdigest()})[0], 200)
        self.assertEqual(post('commit')[0], 409)
        self.assertEqual(post('finalize', json.dumps(dict(segments=self.segments, sha256=hashlib.sha256(self.data).hexdigest())))[0], 200)
        self.assertEqual(post('validate')[0], 200)
        self.assertTrue(post('commit')[1]['committed'])


def native_artifact(path):
    data = Path(path).read_bytes()
    position, payloads, segments = v2.split_v1(data)
    with tempfile.TemporaryDirectory() as root:
        store, key = v2.Store(root), uuid.uuid4().hex
        store.begin(key, dict(position=position, identity='native-artifact-control',
                             sections=[dict(name=n, bytes=len(b)) for n, b in payloads.items()]))
        for name, body in payloads.items():
            for at in range(0, len(body), v2.CHUNK):
                piece = body[at:at+v2.CHUNK]
                store.chunk(key, name, at, piece, hashlib.sha256(piece).hexdigest())
        store.finalize(key, dict(segments=segments, sha256=hashlib.sha256(data).hexdigest()))
        store.validate(key)
        receipt = store.commit(key)
        assert (Path(root) / 'committed' / f'{key}.state').read_bytes() == data
        print(json.dumps(dict(native_artifact=str(path), **receipt)))


def native_client(executable, paths):
    """Exercise the actual C++ sender without constructing its CUDA engine."""
    with tempfile.TemporaryDirectory() as root:
        class Faults(v2.Handler):
            drops = set()
            drop_every_begin = False

            def reply(self, status, value):
                action = self.path.rsplit('/', 1)[-1]
                once = action in ('begin', 'chunk', 'finalize', 'validate', 'commit') and action not in self.drops
                if status == 200 and (once or (action == 'begin' and self.drop_every_begin)):
                    self.drops.add(action)
                    self.close_connection = True
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
                super().reply(status, value)

        Faults.store = v2.Store(root, expected_identity='native-client-fixture')
        Faults.api_key = 'transport-test-key'
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Faults)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()

        def run(path):
            command = [str(Path(executable).resolve()), str(Path(path).resolve()), '--handoff-only',
                f'http://127.0.0.1:{server.server_port}/admin/import-state', '--handoff-v2',
                '--state-identity', 'native-client-fixture', '--handoff-api-key', Faults.api_key, '--v2-timeout', '5']
            return subprocess.run(command, capture_output=True, text=True, timeout=90,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)

        try:
            for path in paths:
                result = run(path)
                assert result.returncode == 0, (result.stdout, result.stderr)
                receipt = json.loads(result.stdout)
                original = Path(path).read_bytes()
                assert (Path(root) / 'committed' / f"{receipt['id']}.state").read_bytes() == original
                assert receipt['sha256'] == hashlib.sha256(original).hexdigest()
                print(json.dumps(dict(native_client=str(path), **receipt)))
            assert Faults.drops == {'begin', 'chunk', 'finalize', 'validate', 'commit'}
            Faults.drop_every_begin = True
            failed = run(paths[0])
            assert failed.returncode != 0, 'three lost begin ACKs must fail the upload'
            assert all(t['stage'] == 'committed' for t in Faults.store.transactions.values()), 'unacknowledged begin was not aborted'
            print('native client: lost ACKs at all five stages recover; three lost begin ACKs abort staging')
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', action='append', default=[])
    parser.add_argument('--client')
    args, remaining = parser.parse_known_args()
    result = unittest.main(argv=[sys.argv[0]] + remaining, exit=False).result
    if not result.wasSuccessful():
        sys.exit(1)
    for path in args.state:
        native_artifact(path)
    if args.client:
        if not args.state:
            parser.error('--client requires at least one --state')
        native_client(args.client, args.state)
