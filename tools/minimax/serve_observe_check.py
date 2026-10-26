#!/usr/bin/env python3
"""strix-minimax-serve observability check and exactness/overhead receipt (stdlib only).

run:     one set of fixed greedy chat requests against a running server, written as JSON.
           serve_observe_check.py run URL --label N1 --mode unarmed|armed|ids-only --out N1.json [--warmup]
         unarmed: only GET /health and plain GET /api/status are used (the server must report armed=false);
         armed:   GET /metrics first (arms), then /metrics and /api/status?detail=1 after every request;
         ids-only: non-stream requests only (a build without streaming token ids).
compare: token identity across sets and the armed-vs-unarmed overhead, as a hyperloom/ab-receipt-v1 document.
           serve_observe_check.py compare --unarmed N1.json N2.json --armed A1.json --extra A2.json P.json \
               --identity arms.json --expect-order N1 A1 N2 A2 P --out receipt.json
verify:  recomputes, offline, the structure, the corpus digest and every equality of a receipt.
           serve_observe_check.py verify receipt.json
Exit codes (compare): 0 token-identical and no measurable overhead; 1 a failed check (identity, structure, live checks);
2 token-identical but the overhead is inconclusive (noise above the limit); 3 token-identical with overhead measured.
Every request sends X-Request-Id; replies are checked for the echo, strix.token_ids (stop token included, length ==
usage.completion_tokens), the stream event order (content, final chunk with finish_reason and ids, usage chunk, [DONE]).
"""

import argparse
import base64
import gzip
import hashlib
import http.client
import json
import socket
import sys
import threading
import time
import urllib.parse

PROMPTS = [
    ("prose-en", "Describe a quiet harbour town at dawn in a short paragraph."),
    ("prose-es", "Escribe un párrafo breve sobre una biblioteca antigua en una ciudad pequeña."),
    ("code-py", "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."),
    ("json", "Return a JSON object with the keys name, version and tags for a small command-line tool."),
]
WARMUP = ("warmup", "Say hello in three languages.")
MAX_TOKENS = 48
WARMUP_MAX_TOKENS = 16


def canonical(body):
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class Deadline(Exception):
    pass


class Client:
    """One connection per exchange. With a deadline, bounded() runs an exchange in a worker thread and waits at most
    until the deadline: then it shuts the open sockets down and raises Deadline, so no exchange (connect, send,
    headers, body, stream) keeps the caller past it, whatever the peer's pace. Socket timeouts bound inactivity."""

    def __init__(self, url, timeout=120, deadline=0.0):
        u = urllib.parse.urlsplit(url)
        self.host, self.port, self.timeout, self.deadline = u.hostname, u.port or 80, timeout, deadline
        self.open = set()
        self.lock = threading.Lock()

    def budget(self):
        if not self.deadline:
            return self.timeout
        left = self.deadline - time.time()
        if left <= 0:
            raise Deadline("deadline reached")
        return min(self.timeout, left)

    def bounded(self, fn, *args, **kwargs):
        if not self.deadline:
            return fn(*args, **kwargs)
        self.budget()
        box = {}

        def work():
            try:
                box["value"] = fn(*args, **kwargs)
            except BaseException as e:  # handed to the caller
                box["error"] = e
        t = threading.Thread(target=work, daemon=True)
        t.start()
        t.join(max(0.0, self.deadline - time.time()))
        if t.is_alive():
            with self.lock:
                for c in list(self.open):
                    try:
                        c.sock.shutdown(socket.SHUT_RDWR)
                    except (OSError, AttributeError):
                        pass
            raise Deadline("deadline reached during a request")
        if "error" in box:
            raise box["error"]
        return box["value"]

    def request(self, method, path, body=None, headers=None):
        """-> (connection, response, t0); the caller ends it with finish(connection)."""
        c = http.client.HTTPConnection(self.host, self.port, timeout=self.budget())
        with self.lock:
            self.open.add(c)
        try:
            data = None if body is None else json.dumps(body).encode()
            h = {"Content-Type": "application/json"} if data is not None else {}
            h.update(headers or {})
            t0 = time.monotonic()
            c.request(method, path, body=data, headers=h)
            r = c.getresponse()
        except BaseException:
            self.finish(c)
            raise
        return c, r, t0

    def finish(self, c):
        with self.lock:
            self.open.discard(c)
        c.close()

    def _get(self, path):
        c, r, _ = self.request("GET", path)
        try:
            return r.status, dict(r.getheaders()), r.read().decode()
        finally:
            self.finish(c)

    def get(self, path):
        return self.bounded(self._get, path)


def body_for(prompt, stream, max_tokens=MAX_TOKENS):
    b = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "return_token_ids": True}
    if stream:
        b["stream"] = True
        b["stream_options"] = {"include_usage": True}
    return b


def non_stream(client, prompt, rid, problems, require_echo=True, max_tokens=MAX_TOKENS):
    body = body_for(prompt, False, max_tokens)
    c, r, t0 = client.request("POST", "/v1/chat/completions", body, {"X-Request-Id": rid})
    try:
        raw = r.read().decode()
        wall = time.monotonic() - t0
        echo = r.getheader("X-Request-Id")
    finally:
        client.finish(c)
    if r.status != 200:
        problems.append(f"{rid}: HTTP {r.status} {raw[:200]}")
        return {"body": canonical(body), "status": r.status}
    j = json.loads(raw)
    ids = (j.get("strix") or {}).get("token_ids")
    usage = j.get("usage") or {}
    if (require_echo or echo is not None) and echo != rid:
        problems.append(f"{rid}: echo {echo!r}")
    if ids is None:
        problems.append(f"{rid}: no strix.token_ids")
    elif len(ids) != usage.get("completion_tokens"):
        problems.append(f"{rid}: {len(ids)} ids vs completion_tokens {usage.get('completion_tokens')}")
    choice = (j.get("choices") or [{}])[0]
    return {"body": canonical(body), "status": 200, "echo": echo, "ids": ids, "finish": choice.get("finish_reason"),
            "text": (choice.get("message") or {}).get("content"), "usage": usage, "timings": j.get("timings"), "wall_s": wall}


def stream(client, prompt, rid, problems):
    body = body_for(prompt, True)
    c, r, t0 = client.request("POST", "/v1/chat/completions", body, {"X-Request-Id": rid})
    echo = r.getheader("X-Request-Id")
    events, first_content, buf = [], None, b""
    try:
        while True:
            line = r.readline()
            if not line:
                break
            buf = line.strip()
            if not buf.startswith(b"data: "):
                continue
            payload = buf[6:].decode()
            if payload == "[DONE]":
                events.append(None)
                break
            e = json.loads(payload)
            if first_content is None and e.get("choices") and ((e["choices"][0].get("delta") or {}).get("content")):
                first_content = time.monotonic() - t0
            events.append(e)
        wall = time.monotonic() - t0
    finally:
        client.finish(c)
    out = {"body": canonical(body), "status": r.status, "echo": echo, "client_first_content_s": first_content, "wall_s": wall}
    if r.status != 200:
        problems.append(f"{rid}: HTTP {r.status}")
        return out
    if echo != rid:
        problems.append(f"{rid}: stream echo {echo!r}")
    if len(events) < 3 or events[-1] is not None:
        problems.append(f"{rid}: stream did not end with [DONE]")
        return out
    usage_ev, final_ev = events[-2], events[-3]
    if usage_ev.get("choices") != [] or "usage" not in usage_ev:
        problems.append(f"{rid}: no usage chunk before [DONE]")
    fin = (final_ev.get("choices") or [{}])[0].get("finish_reason")
    if fin is None:
        problems.append(f"{rid}: final chunk has no finish_reason")
    content_ok = all(e.get("choices") and e["choices"][0].get("finish_reason") is None for e in events[:-3])
    if not content_ok:
        problems.append(f"{rid}: finish_reason before the final chunk")
    ids_final = (final_ev.get("strix") or {}).get("token_ids")
    ids_usage = (usage_ev.get("strix") or {}).get("token_ids")
    if ids_final is None or ids_final != ids_usage:
        problems.append(f"{rid}: final-chunk ids != usage-chunk ids")
    if ids_final is not None and len(ids_final) != (usage_ev.get("usage") or {}).get("completion_tokens"):
        problems.append(f"{rid}: stream ids length != completion_tokens")
    text = "".join(((e["choices"][0].get("delta") or {}).get("content") or "") for e in events[:-3] if e.get("choices"))
    out.update({"ids": ids_final, "finish": fin, "usage": usage_ev.get("usage"), "text": text})
    return out


def run(args):
    try:
        return run_set(args)
    except (Deadline, OSError, http.client.HTTPException, ValueError) as e:  # the deadline or a dead server: the set fails (partial results dropped)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"label": args.label, "mode": args.mode, "url": args.url, "started": time.time(), "cases": [],
                       "problems": [f"aborted: {e}"]}, f)
        print(f"{args.label}: aborted: {e}")
        return 1


def run_set(args):
    client = Client(args.url, deadline=args.deadline)
    problems = []
    started = time.time()
    st, _, raw = client.get("/health")
    if st != 200:
        sys.exit(f"health {st}")
    st, _, raw = client.get("/api/status")
    status0 = json.loads(raw) if st == 200 else None
    if args.mode == "unarmed" and status0 and status0.get("armed"):
        problems.append("server already armed at the start of an unarmed set")
    if args.mode == "armed":
        client.get("/metrics")
    prefix = hashlib.sha256(args.label.encode()).hexdigest()[:16]
    seq = [0]

    def rid():
        seq[0] += 1
        return f"{prefix}.{seq[0]}"

    if args.warmup:
        client.bounded(non_stream, client, WARMUP[1], rid(), [], require_echo=False, max_tokens=WARMUP_MAX_TOKENS)
    cases = []
    last_metrics = last_status = None
    for name, prompt in PROMPTS:
        modes = ["non-stream"] if args.mode == "ids-only" else ["non-stream", "stream"]
        for mode in modes:
            r = rid()
            if mode == "non-stream":
                res = client.bounded(non_stream, client, prompt, r, problems, require_echo=args.mode != "ids-only")
            else:
                res = client.bounded(stream, client, prompt, r, problems)
            res.update({"case": name, "mode": mode, "request_id": r})
            cases.append(res)
            if args.mode == "armed":
                last_metrics = client.get("/metrics")[2]
                last_status = json.loads(client.get("/api/status?detail=1")[2])
            elif args.mode == "unarmed":
                s = client.get("/api/status")
                if s[0] == 200 and json.loads(s[2]).get("armed"):
                    problems.append("server armed during an unarmed set")
    if args.mode == "armed" and last_status is not None:
        recs = {x.get("id"): x for x in last_status.get("requests", [])}
        for c in cases:
            rec = recs.get(c["request_id"])
            if rec is None:
                problems.append(f"{c['request_id']}: missing from the status ring")
                continue
            if rec.get("first_token_ms") is None:
                problems.append(f"{c['request_id']}: armed record without first_token_ms")
            if c.get("ids") is not None and rec.get("output_tokens") != len(c["ids"]):
                problems.append(f"{c['request_id']}: ring output_tokens {rec.get('output_tokens')} != {len(c['ids'])}")
            c["engine_first_token_ms"] = rec.get("first_token_ms")
            c["engine_record"] = rec
            # the engine's submit-to-first-publish lies inside the client's request-to-first-content time
            client_ms = (c.get("client_first_content_s") or 0.0) * 1000.0
            if c["mode"] == "stream" and rec.get("first_token_ms") is not None and client_ms and rec["first_token_ms"] > client_ms + 5.0:
                problems.append(f"{c['request_id']}: engine first_token_ms {rec['first_token_ms']:.1f} > client {client_ms:.1f}")
        for fam in ("strix_ttft_seconds_count", "strix_generated_tokens_total", "mm_completed", "strix_engine_info"):
            if fam not in (last_metrics or ""):
                problems.append(f"metrics without {fam}")
    out = {"label": args.label, "mode": args.mode, "url": args.url, "started": started, "status_start": status0, "cases": cases,
           "last_metrics": last_metrics, "last_status": last_status, "problems": problems}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"{args.label}: {len(cases)} requests, {len(problems)} problems")
    for p in problems:
        print("  " + p)
    return 1 if problems else 0


def first_divergence(a, b):
    """None when equal, else the first differing index (the shorter length when one is a prefix)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def pack_ids(ids):
    """Token ids as stored in the receipt: a plain list, or gzip+base64 of little-endian uint32 when long."""
    if ids is None or len(ids) <= 256:
        return ids
    raw = b"".join(int(i).to_bytes(4, "little") for i in ids)
    return {"gzip_base64_u32le": base64.b64encode(gzip.compress(raw, mtime=0)).decode(), "n": len(ids)}


def unpack_ids(v):
    if v is None or isinstance(v, list):
        return v
    raw = gzip.decompress(base64.b64decode(v["gzip_base64_u32le"]))
    ids = [int.from_bytes(raw[i:i + 4], "little") for i in range(0, len(raw), 4)]
    if len(ids) != v.get("n"):
        raise ValueError("packed id count mismatch")
    return ids


TOLERANCE_PCT = 1.0     # |armed - unarmed| decode rate within this: no measurable overhead
MAX_SPREAD_PCT = 2.0    # unarmed N1 vs N2 spread above this: inconclusive (noise too large to judge)
IDENTITY_FIELDS = ("build", "binary_sha256", "model", "tokenizer_sha256")
# The experiment: label -> (set mode, role); fixed here so that no input can redeclare it.
EXPECTED_ARMS = {"N1": ("unarmed", "unarmed"), "A1": ("armed", "armed"), "N2": ("unarmed", "unarmed"),
                 "A2": ("armed", "extra"), "P": ("ids-only", "extra")}
EXPECTED_ORDER = ["N1", "A1", "N2", "A2", "P"]
FINISH_REASONS = ("stop", "length")
EXIT_OK, EXIT_FAILED, EXIT_INCONCLUSIVE, EXIT_OVERHEAD = 0, 1, 2, 3


def expected_cases(mode):
    """{(case, mode): canonical body} a set of this mode must contain, exactly."""
    modes = ["non-stream"] if mode == "ids-only" else ["non-stream", "stream"]
    return {(name, m): canonical(body_for(prompt, m == "stream")) for name, prompt in PROMPTS for m in modes}


def valid_ids(ids):
    return isinstance(ids, list) and len(ids) > 0 and all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in ids)


def check_matrix(arms, order, expected_order, cases, errors):
    """The structural checks shared by compare (on the sets) and verify (on the receipt).
    arms: {label: {"mode", "role", identity fields...}}; cases: {(case, mode): {label: arm}} with arm["body"], arm["ids"],
    arm["finish"], arm["usage"]."""
    if order != EXPECTED_ORDER or (expected_order and expected_order != EXPECTED_ORDER):
        errors.append(f"trial order {order} != expected {EXPECTED_ORDER}")
    if sorted(arms) != sorted(EXPECTED_ARMS):
        errors.append(f"arms {sorted(arms)} != expected {sorted(EXPECTED_ARMS)}")
    want_total = sum(len(expected_cases(m)) for m, _ in EXPECTED_ARMS.values())
    have_total = sum(len(v) for v in cases.values())
    if have_total != want_total:
        errors.append(f"{have_total} observations, expected {want_total}")
    for label, a in arms.items():
        if label in EXPECTED_ARMS and (a.get("mode"), a.get("role")) != EXPECTED_ARMS[label]:
            errors.append(f"arm {label}: mode/role {a.get('mode')}/{a.get('role')} != expected {EXPECTED_ARMS[label]}")
        for f in IDENTITY_FIELDS:
            if not isinstance(a.get(f), str) or not a.get(f):
                errors.append(f"arm {label}: identity field {f} missing")
        want = expected_cases(a.get("mode"))
        have = {k for k, v in cases.items() if label in v}
        for k in sorted(set(want) - have):
            errors.append(f"arm {label}: missing case {k[0]}/{k[1]}")
        for k in sorted(have - set(want)):
            errors.append(f"arm {label}: unexpected case {k[0]}/{k[1]}")
        for k in sorted(have & set(want)):
            arm = cases[k][label]
            if arm.get("body") != want[k]:
                errors.append(f"arm {label}: {k[0]}/{k[1]} body differs from the corpus")
            if not valid_ids(arm.get("ids")):
                errors.append(f"arm {label}: {k[0]}/{k[1]} has no valid token-id array")
            elif (arm.get("usage") or {}).get("completion_tokens") != len(arm["ids"]):
                errors.append(f"arm {label}: {k[0]}/{k[1]} completion_tokens != number of token ids")
            if arm.get("finish") not in FINISH_REASONS:
                errors.append(f"arm {label}: {k[0]}/{k[1]} finish reason {arm.get('finish')!r}")


def equalities(ref, cases):
    """Every arm of every case against the reference arm's non-stream ids of the same prompt; also finish reasons."""
    out, identical, total = {}, 0, 0
    for (case, mode), arms in cases.items():
        base = arms.get(ref) if mode == "non-stream" else cases.get((case, "non-stream"), {}).get(ref)
        for label, arm in arms.items():
            total += 1
            if base is None or not valid_ids(base.get("ids")) or not valid_ids(arm.get("ids")):
                eq, d = False, "missing"
            else:
                d = first_divergence(base["ids"], arm["ids"])
                eq = d is None and arm.get("finish") == base.get("finish")
            out[(case, mode, label)] = (eq, d)
            identical += eq
    return out, identical, total


def compare(args):
    errors = []
    with open(args.identity, encoding="utf-8") as f:
        identity = json.load(f)  # {label: {build, binary_sha256, model, tokenizer_sha256}}
    sets, roles = {}, {}
    for role, paths in (("unarmed", args.unarmed), ("armed", args.armed), ("extra", args.extra)):
        for path in paths:
            with open(path, encoding="utf-8") as f:
                s = json.load(f)
            if s["label"] in sets:
                errors.append(f"duplicate set label {s['label']}")
                continue
            sets[s["label"]] = s
            roles[s["label"]] = role
    order = sorted(sets, key=lambda l: sets[l].get("started", 0.0))  # the order the sets ran in
    unarmed = [l for l in sets if roles[l] == "unarmed"]
    armed = [l for l in sets if roles[l] == "armed"]
    if len(unarmed) != 2 or len(armed) != 1:
        errors.append("need exactly two unarmed sets and one armed set")
    ref = unarmed[0] if unarmed else order[0]
    cases = {}
    for label in order:
        for c in sets[label]["cases"]:
            key = (c["case"], c["mode"])
            if label in cases.setdefault(key, {}):
                errors.append(f"arm {label}: duplicate case {key[0]}/{key[1]}")
                continue
            cases[key][label] = {"body": c["body"], "ids": c.get("ids"), "finish": c.get("finish"), "request_id": c.get("request_id"),
                                 "usage": c.get("usage"), "timings": c.get("timings"), "wall_s": c.get("wall_s"),
                                 "client_first_content_s": c.get("client_first_content_s"),
                                 "engine_first_token_ms": c.get("engine_first_token_ms")}
    arms = {l: {**identity.get(l, {}), "mode": sets[l]["mode"], "role": roles[l]} for l in order}
    check_matrix(arms, order, args.expect_order, cases, errors)
    eqs, identical, total = equalities(ref, cases)
    # overhead: non-stream engine decode rates, the armed set against the two unarmed sets, matched per case
    rows = []
    if len(unarmed) == 2 and len(armed) == 1:
        for case, _ in PROMPTS:
            vals = [cases.get((case, "non-stream"), {}).get(l, {}).get("timings") or {} for l in unarmed + armed]
            rates = [v.get("decode_tok_s") for v in vals]
            if any(not isinstance(x, (int, float)) or x <= 0 for x in rates):
                errors.append(f"overhead: {case} lacks a decode rate in some set")
                continue
            base = (rates[0] + rates[1]) / 2.0
            rows.append({"case": case, "unarmed_decode_tok_s": rates[:2], "armed_decode_tok_s": rates[2],
                         "delta_pct": 100.0 * (rates[2] - base) / base,
                         "unarmed_spread_pct": 100.0 * abs(rates[0] - rates[1]) / base,
                         "unarmed_prefill_ms": [vals[0].get("prefill_ms"), vals[1].get("prefill_ms")], "armed_prefill_ms": vals[2].get("prefill_ms")})
    mean = sum(r["delta_pct"] for r in rows) / len(rows) if rows else None
    spread = max((r["unarmed_spread_pct"] for r in rows), default=None)
    if len(rows) != len(PROMPTS):
        verdict = "inconclusive (cases missing)"
    elif spread > MAX_SPREAD_PCT:
        verdict = "inconclusive (unarmed spread above the limit)"
    elif abs(mean) <= TOLERANCE_PCT:
        verdict = "no measurable overhead"
    else:
        verdict = "overhead measured"
    problems = {k: v["problems"] for k, v in sets.items() if v["problems"]}
    receipt_cases = []
    for (case, mode), per_arm in sorted(cases.items()):
        entry = {"case": case, "mode": mode, "body": expected_cases("full").get((case, mode)), "arms": {}}
        for label, arm in per_arm.items():
            eq, d = eqs[(case, mode, label)]
            entry["arms"][label] = {**{k: v for k, v in arm.items() if k != "ids"}, "token_ids": pack_ids(arm["ids"]),
                                    "equal": eq, "first_divergence": d}
        receipt_cases.append(entry)
    bodies = sorted({c["body"] for s in sets.values() for c in s["cases"]})
    receipt = {
        "schema": "hyperloom/ab-receipt-v1",
        "subject": "strix-minimax-serve instrumentation: previous build vs new build unarmed vs armed",
        "arms": [{"label": l, "url": sets[l]["url"], "router_policy": None, "recipe_digest": None, **arms[l]} for l in order],
        "corpus_digest": hashlib.sha256("\n".join(bodies).encode()).hexdigest(), "corpus": bodies,
        "trial_order": order, "expected_order": args.expect_order, "reference_arm": ref,
        "stop_policy": "greedy, max_tokens per body, the engine's stop token", "cases": receipt_cases,
        "summary": {"identical": identical, "total": total, "token_identical": identical == total and total > 0,
                    "structure_errors": errors,
                    "overhead": {"metric": "engine decode tok/s (non-stream timings), armed - mean(two unarmed) per case",
                                 "tolerance_pct": TOLERANCE_PCT, "max_spread_pct": MAX_SPREAD_PCT, "cases": rows,
                                 "mean_delta_pct": mean, "max_unarmed_spread_pct": spread, "verdict": verdict},
                    "problems": problems}}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=1, ensure_ascii=False)
    for e in errors:
        print("  error: " + e)
    print(f"token identity {identical}/{total}; structure errors {len(errors)}; sets with problems {len(problems)}; "
          f"overhead mean {None if mean is None else round(mean, 3)} % (max unarmed spread "
          f"{None if spread is None else round(spread, 3)} %): {verdict}")
    if errors or problems or identical != total or total == 0:
        return EXIT_FAILED
    return EXIT_OK if verdict == "no measurable overhead" else EXIT_OVERHEAD if verdict == "overhead measured" else EXIT_INCONCLUSIVE


def verify(args):
    """Recomputes, offline, the structure, the corpus digest and every equality of a receipt from its arrays."""
    with open(args.receipt, encoding="utf-8") as f:
        r = json.load(f)
    errors = []
    if r.get("schema") != "hyperloom/ab-receipt-v1":
        errors.append("schema")
    labels = [a["label"] for a in r["arms"]]
    if len(set(labels)) != len(labels):
        errors.append("duplicate arm labels")
    arms = {a["label"]: a for a in r["arms"]}
    if labels != r["trial_order"]:
        errors.append("arms not listed in trial order")
    cases = {}
    for c in r["cases"]:
        key = (c["case"], c["mode"])
        if key in cases:
            errors.append(f"duplicate case {key}")
        cases[key] = {}
        for label, arm in c["arms"].items():
            if label not in arms:
                errors.append(f"case {key} has an unknown arm {label}")
            if arm.get("body") != c.get("body"):
                errors.append(f"case {key} arm {label}: body differs from the case body")
            try:
                ids = unpack_ids(arm.get("token_ids"))
            except (ValueError, KeyError, TypeError):
                ids = None
            cases[key][label] = {"body": arm.get("body"), "ids": ids, "finish": arm.get("finish"), "usage": arm.get("usage")}
    check_matrix({l: arms[l] for l in labels}, r["trial_order"], r.get("expected_order"), cases, errors)
    corpus = sorted({c["body"] for c in r["cases"]})
    if corpus != sorted(r["corpus"]) or hashlib.sha256("\n".join(corpus).encode()).hexdigest() != r["corpus_digest"]:
        errors.append("corpus digest")
    eqs, identical, total = equalities(r["reference_arm"], cases)
    for c in r["cases"]:
        for label, arm in c["arms"].items():
            eq, d = eqs[(c["case"], c["mode"], label)]
            if eq != arm["equal"] or d != arm["first_divergence"]:
                errors.append(f"{c['case']}/{c['mode']} {label}: recorded {arm['equal']}/{arm['first_divergence']} recomputed {eq}/{d}")
    if identical != r["summary"]["identical"] or total != r["summary"]["total"]:
        errors.append("summary counts")
    for e in errors:
        print("  error: " + e)
    ok = not errors and identical == total and total > 0
    print(f"verify: {'OK' if ok else 'FAILED'}; identical {identical}/{total}; errors {len(errors)}")
    return EXIT_OK if ok else EXIT_FAILED


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("url")
    r.add_argument("--label", required=True)
    r.add_argument("--mode", choices=("unarmed", "armed", "ids-only"), required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--warmup", action="store_true")
    r.add_argument("--deadline", type=float, default=0.0, help="unix time after which no request is started")
    c = sub.add_parser("compare")
    c.add_argument("--unarmed", nargs="+", required=True)
    c.add_argument("--armed", nargs="+", required=True)
    c.add_argument("--extra", nargs="*", default=[])
    c.add_argument("--identity", required=True, help="JSON {label: {build, binary_sha256, model, tokenizer_sha256}}")
    c.add_argument("--expect-order", nargs="*", default=[], help="labels in the order the sets must have run")
    c.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("receipt")
    a = p.parse_args()
    return {"run": run, "compare": compare, "verify": verify}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
