#!/usr/bin/env python3
"""Run real C++ receiver loopback tests. Pass the host-only test executable."""
import hashlib
import importlib.util
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('replay', ROOT/'replay_reference.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
exe = str(Path(sys.argv[1]).resolve())
work_root = Path(sys.argv[2]).resolve()
(work_root/'tmp').mkdir(exist_ok=True)
os.environ['TMP'] = os.environ['TEMP'] = str(work_root/'tmp')


def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def connect(p):
    for _ in range(200):
        try:
            return socket.create_connection(('127.0.0.1', p), timeout=10)
        except ConnectionRefusedError:
            time.sleep(.02)
    raise RuntimeError('loopback receiver failed to listen')


def run():
    subprocess.run([exe], check=True)
    work = Path(sys.argv[2]).resolve()
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work) as td:
        td = Path(td)
        key = bytes(range(32))
        (td/'key').write_bytes(key)
        prompt = struct.pack('<6I', *range(6))
        (td/'prompt').write_bytes(prompt)
        model, tx = b'\x22'*32, b'\x33'*32
        begin = (r.u32(1)+r.u32(1)+r.u32(2)+r.u32(0)+r.u32(46)+r.u64(5)+r.u64(6)+
                 model+hashlib.sha256(prompt).digest()+tx+r.text('glm-pf-node-replay-v1')+r.text('r1'))
        state_hash = hashlib.sha256()
        with open(td/'state', 'wb') as f:
            f.write(struct.pack('<QQ', 5, 6))
            for layer, component, rows, stride in r.layout(5, 46):
                payload = bytes([(layer*7+component)%256])*(rows*stride)
                state_hash.update(payload)
                f.write(payload)
        def server():
            p = port()
            proc = subprocess.Popen([exe, f'127.0.0.1:{p}', str(td/'key'),
                                     hashlib.sha256(prompt).hexdigest(), model.hex()],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return proc, p
        def done(proc, success):
            try:
                out, err = proc.communicate(timeout=60)
            except BaseException:
                proc.kill()
                proc.wait()
                raise
            assert (proc.returncode == 0) == success, (out, err)
            return out, err
        proc, p = server()
        for _ in range(200):
            try:
                result = r.replay(f'127.0.0.1:{p}', td/'state', 0, key=key, model=model,
                                  prompt=td/'prompt', node='r1', mtp=True, transaction=tx)
                break
            except ConnectionRefusedError:
                time.sleep(.02)
        else:
            raise RuntimeError('receiver never started')
        out, _ = done(proc, True)
        assert state_hash.hexdigest() in out, out
        assert result['raw_bytes'] == (td/'state').stat().st_size-16
        print('PASS loopback replay, complete staged state SHA-256')
        for case in ('wrong-key', 'nonce-replay', 'sequence', 'abort', 'missing', 'before-begin',
                     'duplicate-begin', 'corrupt', 'duplicate', 'out-of-order', 'oversized', 'truncated'):
            proc, p = server()
            with connect(p) as sock:
                sender = r.Sender(sock, key if case != 'wrong-key' else b'\0'*32)
                try:
                    if case == 'oversized':
                        sock.sendall(r.u32(8388609))
                    elif case == 'truncated':
                        sock.sendall(r.u32(50)+b'123')
                        sock.shutdown(socket.SHUT_WR)
                    elif case == 'nonce-replay':
                        sender.nonce = b'\0'*32
                        sender.packet(begin)
                    elif case == 'sequence':
                        sender.sequence = 1
                        sender.packet(begin)
                    elif case == 'before-begin':
                        sender.packet(r.u32(3)+r.u64(0)+r.u64(0))
                    else:
                        sender.packet(begin)
                        if case == 'abort':
                            sender.packet(r.u32(4))
                        elif case == 'missing':
                            sender.packet(r.u32(3)+r.u64(0)+r.u64(0))
                        elif case == 'duplicate-begin':
                            sender.packet(begin)
                        elif case in ('corrupt', 'duplicate', 'out-of-order'):
                            payload = b'\x11'*(128*128*4)
                            frame = (r.u32(2)+r.u32(0)+r.u32(0)+r.u64(0)+r.u64(1)+r.u64(len(payload))+
                                     hashlib.sha256(payload).digest()+payload)
                            if case == 'corrupt':
                                frame = frame[:-1]+bytes([frame[-1]^1])
                            if case == 'out-of-order':
                                frame = frame[:12]+r.u64(1)+frame[20:]
                            sender.packet(frame)
                            if case == 'duplicate':
                                sender.packet(frame)
                    # Rejection is a closed connection; drain to avoid a Windows close/reset race.
                    try:
                        sock.recv(1)
                    except OSError:
                        pass
                except (OSError, RuntimeError):
                    pass
            _, err = done(proc, False)
            assert 'REJECT' in err, err
            print('PASS loopback rejection:', case)

if __name__ == '__main__':
    run()
