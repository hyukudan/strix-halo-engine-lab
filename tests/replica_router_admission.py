"""Exercise policy admission with loopback workers that report their instance IDs."""

import contextlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time


class Handler(BaseHTTPRequestHandler):
    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError):
            pass  # The test may stop the router during a poll.

    def do_GET(self):
        status = {"state": "ready", "capacity": {"slots_total": 1}}
        if self.server.worker_id is not None:
            status["worker_instance_id"] = self.server.worker_id
        body = json.dumps(status).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@contextlib.contextmanager
def worker(worker_id):
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        server.worker_id = worker_id
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join()


def run(router, description, a, b, refusal=None):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    cmd = [router, "--description", str(description), "--listen", f"127.0.0.1:{port}",
           "--replica", f"http://127.0.0.1:{a.server_port}",
           "--replica", f"http://127.0.0.1:{b.server_port}", "--poll-ms", "20"]
    with (tempfile.TemporaryFile(mode="w+", encoding="utf-8", dir=description.parent) as log,
          subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT) as proc):
        def output():
            log.seek(0)
            return log.read()

        try:
            if refusal:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired as exc:
                    raise AssertionError(f"router did not refuse {refusal}: {output()}") from exc
                log_text = output()
                assert proc.returncode == 1 and refusal in log_text, log_text
                assert "effective-multi-device" not in log_text, log_text
                return

            def await_health(count):
                deadline = time.monotonic() + 5
                while proc.poll() is None and time.monotonic() < deadline:
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
                    try:
                        conn.request("GET", "/health")
                        if json.load(conn.getresponse())["replicas_healthy"] == count:
                            return
                    except OSError:
                        pass
                    finally:
                        conn.close()
                    time.sleep(0.02)
                raise AssertionError(f"router did not report {count} healthy workers: {output()}")

            await_health(2)
            b.worker_id = a.worker_id
            await_health(0)
            b.worker_id = "replacement"
            await_health(2)
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp, worker("first") as a, worker("first") as b:
        description = Path(temp) / "model.toml"
        description.write_text("name = 'router-test'\n[multi_device]\nversion = 1\ndefault = 'replica'\n"
                               "[multi_device.forms.replica]\nranks = 2\n", encoding="utf-8")
        run(sys.argv[1], description, a, b, "worker instance is already configured")
        b.worker_id = None
        run(sys.argv[1], description, a, b, "worker_instance_id")
        b.worker_id = "second"
        run(sys.argv[1], description, a, b)
    print("replica router admission: duplicate and missing IDs refused, distinct IDs served and rechecked")
