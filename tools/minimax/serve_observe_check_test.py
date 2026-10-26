#!/usr/bin/env python3
"""Tests of serve_observe_check.py compare/verify on synthetic sets (stdlib unittest, no server):
    python3 tools/minimax/serve_observe_check_test.py"""

import copy
import socket
import threading
import time
import importlib.util
import json
import os
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("serve_observe_check", os.path.join(HERE, "serve_observe_check.py"))
chk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chk)

ORDER = ["N1", "A1", "N2", "A2", "P"]
ROLES = {"N1": "unarmed", "N2": "unarmed", "A1": "armed", "A2": "extra", "P": "extra"}
MODES = {"N1": "unarmed", "N2": "unarmed", "A1": "armed", "A2": "armed", "P": "ids-only"}


def ids_for(case):
    base = sum(map(ord, case))
    return [(base * 7 + i * 31) % 200000 for i in range(47)] + [200020]


def make_set(label, started, rate=20.0):
    cases = []
    modes = ["non-stream"] if MODES[label] == "ids-only" else ["non-stream", "stream"]
    for name, prompt in chk.PROMPTS:
        for m in modes:
            c = {"case": name, "mode": m, "request_id": f"{label}.{name}.{m}", "body": chk.canonical(chk.body_for(prompt, m == "stream")),
                 "ids": ids_for(name), "finish": "length", "usage": {"completion_tokens": 48}}
            if m == "non-stream":
                c["timings"] = {"decode_tok_s": rate, "prefill_ms": 500.0}
            cases.append(c)
    return {"label": label, "mode": MODES[label], "url": "", "started": started, "cases": cases, "problems": []}


def identity():
    return {l: {"build": "b" + l, "binary_sha256": "0" * 64, "model": "m", "tokenizer_sha256": "1" * 64} for l in ORDER}


class Rig:
    def __init__(self):
        self.dir = tempfile.mkdtemp()
        self.sets = {l: make_set(l, 1000.0 + i) for i, l in enumerate(ORDER)}
        self.identity = identity()

    def run(self, expect_order=ORDER):
        paths = {}
        for l, s in self.sets.items():
            paths[l] = os.path.join(self.dir, l + ".json")
            with open(paths[l], "w", encoding="utf-8") as f:
                json.dump(s, f)
        ident = os.path.join(self.dir, "arms.json")
        with open(ident, "w", encoding="utf-8") as f:
            json.dump(self.identity, f)
        self.receipt = os.path.join(self.dir, "receipt.json")
        args = types.SimpleNamespace(unarmed=[paths["N1"], paths["N2"]], armed=[paths["A1"]], extra=[paths["A2"], paths["P"]],
                                     identity=ident, expect_order=expect_order, out=self.receipt)
        return chk.compare(args)

    def verify(self):
        return chk.verify(types.SimpleNamespace(receipt=self.receipt))

    def load(self):
        with open(self.receipt, encoding="utf-8") as f:
            return json.load(f)


class CompareTests(unittest.TestCase):
    def test_good(self):
        r = Rig()
        self.assertEqual(r.run(), chk.EXIT_OK)
        self.assertEqual(r.verify(), chk.EXIT_OK)
        rec = r.load()
        self.assertEqual(rec["summary"]["identical"], 36)
        self.assertEqual(rec["summary"]["total"], 36)
        self.assertEqual(rec["trial_order"], ORDER)

    def test_empty_identity_fails(self):
        r = Rig()
        r.identity["P"] = {}
        self.assertEqual(r.run(), chk.EXIT_FAILED)

    def test_changed_body_fails(self):
        r = Rig()
        r.sets["A1"]["cases"][0]["body"] = r.sets["A1"]["cases"][2]["body"]
        self.assertEqual(r.run(), chk.EXIT_FAILED)

    def test_missing_case_fails(self):
        r = Rig()
        r.sets["P"]["cases"].pop()
        self.assertEqual(r.run(), chk.EXIT_FAILED)

    def test_duplicate_case_fails(self):
        r = Rig()
        r.sets["N2"]["cases"].append(copy.deepcopy(r.sets["N2"]["cases"][0]))
        self.assertEqual(r.run(), chk.EXIT_FAILED)

    def test_finish_reason_differs_fails(self):
        r = Rig()
        r.sets["A2"]["cases"][1]["finish"] = "stop"
        self.assertEqual(r.run(), chk.EXIT_FAILED)

    def test_divergence_reported(self):
        r = Rig()
        r.sets["P"]["cases"][1]["ids"][5] += 1
        self.assertEqual(r.run(), chk.EXIT_FAILED)
        rec = r.load()
        arm = [c for c in rec["cases"] if c["case"] == chk.PROMPTS[1][0] and c["mode"] == "non-stream"][0]["arms"]["P"]
        self.assertEqual((arm["equal"], arm["first_divergence"]), (False, 5))
        self.assertEqual(r.verify(), chk.EXIT_FAILED)  # not identical: verify recomputes the same and reports failure

    def test_wrong_order_fails(self):
        r = Rig()
        r.sets["A1"]["started"], r.sets["N2"]["started"] = r.sets["N2"]["started"], r.sets["A1"]["started"]
        self.assertEqual(r.run(), chk.EXIT_FAILED)

    def test_live_problem_fails(self):
        r = Rig()
        r.sets["A1"]["problems"] = ["x: echo None"]
        self.assertEqual(r.run(), chk.EXIT_FAILED)

    def test_overhead_measured(self):
        r = Rig()
        for c in r.sets["A1"]["cases"]:
            if "timings" in c:
                c["timings"]["decode_tok_s"] = 18.0  # 10 % slower
        self.assertEqual(r.run(), chk.EXIT_OVERHEAD)

    def test_noise_is_inconclusive(self):
        r = Rig()
        for c in r.sets["N2"]["cases"]:
            if "timings" in c:
                c["timings"]["decode_tok_s"] = 21.0  # 5 % spread between the unarmed sets
        self.assertEqual(r.run(), chk.EXIT_INCONCLUSIVE)

    def test_small_delta_passes(self):
        r = Rig()
        for c in r.sets["A1"]["cases"]:
            if "timings" in c:
                c["timings"]["decode_tok_s"] = 19.9  # -0.5 %
        self.assertEqual(r.run(), chk.EXIT_OK)

    def test_tampered_receipt_fails_verify(self):
        r = Rig()
        self.assertEqual(r.run(), chk.EXIT_OK)
        rec = r.load()
        rec["cases"][0]["arms"]["P"]["token_ids"][0] += 1  # ids changed, recorded "equal" kept
        with open(r.receipt, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        self.assertEqual(r.verify(), chk.EXIT_FAILED)
        rec = r.load()
        rec["cases"][0]["arms"]["P"]["token_ids"][0] -= 1
        rec["cases"][0]["arms"]["P"]["body"] = "{}"
        with open(r.receipt, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        self.assertEqual(r.verify(), chk.EXIT_FAILED)

    def test_client_deadline(self):
        import time
        with self.assertRaises(chk.Deadline):
            chk.Client("", deadline=time.time() - 1).budget()
        self.assertLessEqual(chk.Client("", deadline=time.time() + 3).budget(), 3.0)
        self.assertEqual(chk.Client("").budget(), 120)

    def test_redeclared_mode_fails(self):
        r = Rig()
        r.sets["A2"]["mode"] = "ids-only"
        r.sets["A2"]["cases"] = [c for c in r.sets["A2"]["cases"] if c["mode"] == "non-stream"]
        self.assertEqual(r.run(), chk.EXIT_FAILED)

    def test_completion_tokens_checked(self):
        r = Rig()
        r.sets["N2"]["cases"][3]["usage"]["completion_tokens"] = 999
        self.assertEqual(r.run(), chk.EXIT_FAILED)
        r = Rig()
        self.assertEqual(r.run(), chk.EXIT_OK)
        rec = r.load()
        for c in rec["cases"]:
            for arm in c["arms"].values():
                arm["usage"]["completion_tokens"] = 999
        with open(r.receipt, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        self.assertEqual(r.verify(), chk.EXIT_FAILED)

    def test_finish_reason_required(self):
        r = Rig()
        for l in ORDER:
            for c in r.sets[l]["cases"]:
                c["finish"] = None
        self.assertEqual(r.run(), chk.EXIT_FAILED)
        r = Rig()
        self.assertEqual(r.run(), chk.EXIT_OK)
        rec = r.load()
        for c in rec["cases"]:
            for arm in c["arms"].values():
                arm.pop("finish")
        with open(r.receipt, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        self.assertEqual(r.verify(), chk.EXIT_FAILED)

    def test_role_from_input_ignored(self):
        r = Rig()
        r.identity["A2"]["role"] = "armed"  # an identity file cannot redeclare the experiment
        self.assertEqual(r.run(), chk.EXIT_OK)


    def test_pack_round_trip(self):
        ids = list(range(1000, 1600))
        packed = chk.pack_ids(ids)
        self.assertIsInstance(packed, dict)
        self.assertEqual(chk.unpack_ids(packed), ids)
        self.assertEqual(chk.pack_ids([1, 2]), [1, 2])


class DripServer:
    """Answers every connection with a valid response, but one byte per `pace` seconds (headers, body or a stream)."""

    def __init__(self, kind, pace=0.05):
        # the loopback address, from the resolver (no node name and no passive flag = the loopback interface)
        self.host = socket.getaddrinfo(None, 0, socket.AF_INET, socket.SOCK_STREAM)[0][4][0]
        self.sock = socket.create_server((self.host, 0))
        self.port = self.sock.getsockname()[1]
        body = json.dumps({"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 1}})
        if kind == "stream":
            payload = "data: " + json.dumps({"choices": [{"delta": {"content": "x"}, "finish_reason": None}]}) + "\n\n"
            self.data = ("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n" + payload * 50).encode()
        else:
            self.data = (f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n" + body).encode()
        self.slow_from = 0 if kind == "headers" else self.data.index(b"\r\n\r\n") + 4
        self.pace, self.stop = pace, False
        threading.Thread(target=self.serve, daemon=True).start()

    def serve(self):
        while not self.stop:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            try:
                conn.recv(65536)
                conn.sendall(self.data[:self.slow_from])
                for i in range(self.slow_from, len(self.data)):
                    conn.sendall(self.data[i:i + 1])
                    time.sleep(self.pace)
            except OSError:
                pass
            finally:
                conn.close()

    def close(self):
        self.stop = True
        self.sock.close()


class DeadlineTests(unittest.TestCase):
    def check_bounded(self, kind, call):
        srv = DripServer(kind)
        try:
            client = chk.Client(f"http://{srv.host}:{srv.port}", deadline=time.time() + 0.3)
            t0 = time.monotonic()
            with self.assertRaises((chk.Deadline, OSError, chk.http.client.HTTPException, ValueError)):
                call(client)
            self.assertLess(time.monotonic() - t0, 1.0, f"{kind}: the exchange outlived the deadline")
        finally:
            srv.close()

    def test_slow_headers(self):
        self.check_bounded("headers", lambda c: c.get("/health"))

    def test_slow_body(self):
        self.check_bounded("body", lambda c: c.bounded(chk.non_stream, c, "x", "deadline1", []))

    def test_slow_stream(self):
        self.check_bounded("stream", lambda c: c.bounded(chk.stream, c, "x", "deadline2", []))


if __name__ == "__main__":
    unittest.main()
