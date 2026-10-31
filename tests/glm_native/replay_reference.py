#!/usr/bin/env python3
"""CPU replay sender for GLM import v1; no model/device dependencies."""
import argparse
import concurrent.futures
import hashlib
import hmac
import os
from pathlib import Path
import socket
import struct

MAX_FRAME = 1 << 20

def u32(n): return struct.pack('<I', n)
def u64(n): return struct.pack('<Q', n)
def text(s):
    b = s.encode('ascii')
    return u32(len(b)) + b

def layout(frontier, layers):
    for layer in range(layers):
        if layer < 45 and layer % 4 != 3:
            yield layer, 0, 32, 128*128*4
            yield layer, 1, 32, 3*128*3*4
        else:
            for component, rows, stride in ((2, frontier, 1024), (3, frontier, 256),
                                           (4, frontier, 256), (5, frontier//4, 512)):
                if rows:
                    yield layer, component, rows, stride

def exact(sock, n):
    out = bytearray()
    while len(out) < n:
        b = sock.recv(n-len(out))
        if not b:
            raise RuntimeError('receiver disconnected/rejected transaction')
        out.extend(b)
    return bytes(out)

class Sender:
    def __init__(self, sock, key):
        self.sock, self.key = sock, key
        self.nonce, self.sequence = exact(sock, 32), 0
    def packet(self, body):
        size = u32(len(body))
        mac = hmac.digest(self.key, self.nonce+u64(self.sequence)+size+body, 'sha256')
        self.sequence += 1
        self.sock.sendall(size+body+mac)
        return exact(self.sock, 1)[0]

def replay(endpoint, path, rank, *, key, model, prompt, node, mtp, transaction):
    prompt_data = Path(prompt).read_bytes()
    if len(prompt_data) % 4:
        raise ValueError('prompt must contain little-endian uint32 IDs')
    with open(path, 'rb') as f:
        header = f.read(16)
        if len(header) != 16:
            raise ValueError('short state header')
        frontier, total = struct.unpack('<QQ', header)
        if frontier != len(prompt_data)//4-1 or total != len(prompt_data)//4 or not 0 < frontier < total <= 131072:
            raise ValueError('replay requires P=N-1, matching prompt/T, context <=131072')
        layers = 46 if mtp else 45
        expected = 16+sum(rows*stride for _, _, rows, stride in layout(frontier, layers))
        if os.fstat(f.fileno()).st_size != expected:
            raise ValueError('state byte count does not match layout (check --mtp)')
        host, port = endpoint.rsplit(':', 1)
        with socket.create_connection((host, int(port)), timeout=120) as sock:
            sender = Sender(sock, key)
            begin = (u32(1)+u32(1)+u32(2)+u32(rank)+u32(layers)+u64(frontier)+u64(total)+
                     model+hashlib.sha256(prompt_data).digest()+transaction+
                     text('glm-pf-node-replay-v1')+text(node))
            if sender.packet(begin) != 0:
                raise RuntimeError('unexpected begin ACK')
            digests, raw = [], 0
            for layer, component, rows, stride in layout(frontier, layers):
                row = 0
                while row < rows:
                    n = min(rows-row, max(1, MAX_FRAME//stride))
                    payload = f.read(n*stride)
                    if len(payload) != n*stride:
                        raise ValueError('state truncated during replay')
                    body = (u32(2)+u32(layer)+u32(component)+u64(row)+u64(n)+u64(len(payload))+
                            hashlib.sha256(payload).digest()+payload)
                    if sender.packet(body) != 0:
                        raise RuntimeError('unexpected frame ACK')
                    digests.append(hashlib.sha256(body).digest())
                    raw += len(payload)
                    row += n
            if f.read(1):
                raise ValueError('state grew during replay')
            if sender.packet(u32(3)+u64(len(digests))+u64(raw)+b''.join(digests)) != 1:
                raise RuntimeError('missing local seal ACK')
            return dict(rank=rank, frontier=frontier, raw_bytes=raw, frames=len(digests),
                        status='LOCALLY_SEALED (inspect both node receipts for global commit)')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--send', action='append', required=True, help='numeric HOST:PORT, rank order')
    p.add_argument('--state', action='append', required=True, help='rank-local --state-out file, rank order')
    p.add_argument('--rank', type=int, choices=(0, 1), help='required for a single endpoint')
    p.add_argument('--key-file', required=True, help='exactly 32 raw bytes, shared with receivers')
    p.add_argument('--model-sha256', required=True, help='trusted model asset manifest SHA-256')
    p.add_argument('--prompt', required=True, help='N input IDs (not the extra TF label)')
    p.add_argument('--node', choices=('r1', 'r1b', 'r1b-all'), required=True)
    p.add_argument('--mtp', type=int, choices=(0, 1), required=True)
    p.add_argument('--transaction', help='shared 64-hex transaction ID; generated for a paired invocation')
    a = p.parse_args()
    if len(a.send) != len(a.state) or len(a.send) not in (1, 2):
        p.error('supply one or two matching --send/--state pairs')
    if len(a.send) == 1 and (a.rank is None or a.transaction is None):
        p.error('single-rank replay requires --rank and a common --transaction')
    key = Path(a.key_file).read_bytes()
    if len(key) != 32:
        p.error('key must contain exactly 32 raw bytes')
    model = bytes.fromhex(a.model_sha256)
    transaction = bytes.fromhex(a.transaction) if a.transaction else os.urandom(32)
    if len(model) != 32 or len(transaction) != 32:
        p.error('model/transaction must contain 64 hex digits')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(replay, endpoint, state, a.rank if len(a.send)==1 else i,
                            key=key, model=model, prompt=a.prompt, node=a.node,
                            mtp=a.mtp, transaction=transaction)
                for i, (endpoint, state) in enumerate(zip(a.send, a.state))]
        for job in jobs:
            print(job.result())

if __name__ == '__main__':
    main()
