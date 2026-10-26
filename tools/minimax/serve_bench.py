#!/usr/bin/env python3
"""Serve benchmark for strix-minimax-serve (stdlib only; qualification notes).

Runs a SCRIPT (JSON list of ops) against one server and appends one JSON line per run to --out:
  {"op": "policy", "policy": POLICY}                               POSTs POLICY to /admin/sched (e.g. before serial)
  {"op": "cold", "name": N, "max_tokens": M}                       one request (first request of a fresh server)
  {"op": "serial", "names": [...], "max_tokens": M, "ref": true}     each prompt alone, in order (the exactness reference)
  {"op": "cell", "cell": ID, "kind": "burst" | "closed" | "inject", "names": [...], "max_tokens": M,
   "concurrency": C, "late": NAME, "delay": S, "arms": {ARM: POLICY, ...}, "order": [ARM, ...]}
      burst  = every name at t = 0;  closed = C clients, the names dealt round-robin, each client sequential;
      inject = the names at t = 0 plus LATE at t = DELAY (a long prompt arriving while the others decode).
      Before each run the arm's POLICY is POSTed to /admin/sched (counters reset).
  {"op": "cancel", "name": N, "after": K, "then": [...], "max_tokens": M}  a stream dropped after K chunks (during
      its output), then the names of "then" (exactness after slot reuse)
  {"op": "cancel_prefill", "policy": POLICY, "during": [...], "name": N, "start": S, "after_s": T, "then": [...]}
      the names of "during" at t = 0, N at t = S whose connection is closed T seconds later (before its first token:
      its prefill is open); passes if the server counted >= 1 cancelled prefill and "then" (run concurrently, so the
      cancelled slot is reused) is exact
Per request: client TTFT (first content chunk), chunk gaps (ITL), the server's timings (ttft_ms, max_gap_ms, stall_ms,
ids_fnv, completion_tokens) and text_fnv; exact = ids_fnv (else text_fnv) equal to the serial reference of that name and
max_tokens. --strict (fail closed): every planned request must complete with its final timings chunk and [DONE], carry
ids_fnv, have a reference (cells), every arm's policy POST must succeed; any failure is recorded in the run's "failures"
and the exit status is 1. --deadline EPOCH_S: nothing starts after it (recorded as a failure), reads time out at it.

usage: serve_bench.py URL --prompts P.json --script S.json --out results.jsonl [--ref ref.json] [--strict] [--deadline T]
"""
import argparse
import json
import socket
import threading
import time
import urllib.error
import urllib.request

DEADLINE = None
LEGACY = False


def remaining():
    return 3600.0 if DEADLINE is None else DEADLINE - time.time()


def fnv(data):
    h = 14695981039346656037
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return "%016x" % h


def pct(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def admin(url, policy=None):
    try:
        if policy is None:
            req = urllib.request.Request(url + "/admin/sched")
        else:
            body = dict(policy)
            body["reset_counters"] = True
            req = urllib.request.Request(url + "/admin/sched", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=max(1.0, min(30.0, remaining()))) as r:
            return json.loads(r.read())
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError):
        return None


def one(url, name, text, max_tokens, cancel_after=None, close_after_s=None):
    body = json.dumps({"model": "minimax", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
                       "stream": True}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    times, parts, timings, done, err = [], [], {}, False, None
    try:
        timeout = close_after_s if close_after_s is not None else max(1.0, remaining())
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    done = True
                    break
                j = json.loads(data)
                if "timings" in j:
                    timings = j["timings"]
                if DEADLINE is not None and time.time() > DEADLINE:
                    err = "deadline"  # absolute: a stream that keeps going past it is cut
                    break
                delta = j["choices"][0].get("delta", {}).get("content", "")
                if delta:
                    times.append(time.perf_counter())
                    parts.append(delta)
                    if cancel_after is not None and len(times) >= cancel_after:
                        break
    except (socket.timeout, TimeoutError) as e:
        err = "timeout" if close_after_s is None else None
        if close_after_s is not None:
            err = None  # the intended close
    except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as e:
        err = "%s: %s" % (type(e).__name__, e)
    t1 = time.perf_counter()
    text_out = "".join(parts)
    gaps = [b - a for a, b in zip(times, times[1:])]
    return {"name": name, "t0": t0, "ttft": (times[0] if times else t1) - t0, "total": t1 - t0, "chunks": len(times),
            "gaps": gaps, "text_fnv": fnv(text_out.encode()), "text": text_out[:200], "timings": timings, "done": done,
            "error": err, "tokens": timings.get("completion_tokens", len(times))}


def run_requests(url, prompts, plan, max_tokens):
    """plan: list of (client_id, delay_s, [names]); each client runs its names sequentially after its delay."""
    out, lock = [], threading.Lock()
    start = time.perf_counter()

    def client(delay, names):
        time.sleep(max(0.0, start + delay - time.perf_counter()))
        for n in names:
            try:
                r = one(url, n, prompts[n], max_tokens)
            except Exception as e:  # never lose a planned request
                r = {"name": n, "t0": time.perf_counter(), "ttft": 0.0, "total": 0.0, "chunks": 0, "gaps": [], "text_fnv": "",
                     "text": "", "timings": {}, "done": False, "error": "%s: %s" % (type(e).__name__, e), "tokens": 0}
            with lock:
                out.append(r)

    th = [threading.Thread(target=client, args=(d, ns), daemon=True) for _, d, ns in plan]
    for t in th:
        t.start()
    for t in th:
        t.join(max(1.0, remaining()))
    planned = sum(len(ns) for _, _, ns in plan)
    end = max((r["t0"] + r["total"] for r in out), default=time.perf_counter())
    return out, end - start, planned


def check(rs, planned, ref, max_tokens, strict, need_ref):
    failures = []
    if len(rs) != planned:
        failures.append("%d of %d planned requests returned" % (len(rs), planned))
    for r in rs:
        if r.get("error"):
            failures.append("%s: %s" % (r["name"], r["error"]))
        elif (strict or LEGACY) and not r["done"]:
            failures.append("%s: no [DONE]" % r["name"])
        elif strict and not r["timings"].get("ids_fnv"):
            failures.append("%s: no ids_fnv" % r["name"])
        if need_ref and strict and ("%s:%d" % (r["name"], max_tokens)) not in ref:
            failures.append("%s: no reference" % r["name"])
    return failures


def summarize(rs, wall, ref, max_tokens):
    toks = sum(r["tokens"] for r in rs)
    gaps = [g for r in rs for g in r["gaps"]]
    exact = 0
    checked = 0
    for r in rs:
        k = ref.get("%s:%d" % (r["name"], max_tokens))
        if not k:
            continue
        checked += 1
        ids = r["timings"].get("ids_fnv")
        if ids and k.get("ids_fnv"):
            exact += ids == k["ids_fnv"]
        else:
            exact += r["text_fnv"] == k["text_fnv"]
    return {"requests": len(rs), "wall_s": round(wall, 3), "tokens": toks, "aggregate_tok_s": round(toks / wall, 3) if wall > 0 else 0,
            "ttft_p50_s": round(pct([r["ttft"] for r in rs], 0.5), 3), "ttft_p90_s": round(pct([r["ttft"] for r in rs], 0.9), 3),
            "ttft_max_s": round(max((r["ttft"] for r in rs), default=0), 3),
            "itl_p50_ms": round(1e3 * pct(gaps, 0.5), 1), "itl_p90_ms": round(1e3 * pct(gaps, 0.9), 1),
            "itl_max_ms": round(1e3 * max(gaps), 1) if gaps else 0,
            "server_max_gap_ms": round(max((r["timings"].get("max_gap_ms", 0) for r in rs), default=0), 1),
            "server_stall_ms": round(sum(r["timings"].get("stall_ms", 0) for r in rs), 1),
            "exact": exact, "checked": checked}


def per(rs):
    return [{k: r.get(k) for k in ("name", "ttft", "total", "tokens", "text_fnv", "timings", "error")} for r in rs]


def main():
    global DEADLINE, LEGACY
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--script", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", help="reference JSON (read if present; written by serial ops with ref: true)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--legacy", action="store_true", help="a server without timings / ids_fnv (the tip): completion with [DONE] required")
    ap.add_argument("--deadline", type=float, help="epoch seconds")
    a = ap.parse_args()
    DEADLINE = a.deadline
    LEGACY = a.legacy
    if a.legacy:
        a.strict = False
    prompts = {p["name"]: p["text"] for p in json.load(open(a.prompts, encoding="utf-8"))}
    script = json.load(open(a.script, encoding="utf-8"))
    ref = {}
    if a.ref:
        try:
            ref = json.load(open(a.ref, encoding="utf-8"))
        except OSError:
            ref = {}
    outf = open(a.out, "a", encoding="utf-8")
    failed = 0

    def emit(rec, failures):
        nonlocal failed
        rec["tag"] = a.tag
        rec["failures"] = failures
        if a.strict and rec.get("checked") is not None and rec.get("exact") != rec.get("checked"):
            failures.append("exact %s of %s" % (rec.get("exact"), rec.get("checked")))
        failed += 1 if failures else 0
        outf.write(json.dumps(rec) + "\n")
        outf.flush()
        print(json.dumps({k: v for k, v in rec.items() if k != "per_request"}), flush=True)

    for op in script:
        kind = op["op"]
        mt = op.get("max_tokens", 64)
        if remaining() < 15:
            emit({"op": kind, "skipped": "deadline"}, ["deadline: op not started"])
            break
        if kind == "policy":  # {"op": "policy", "policy": POLICY}: set before serial / cold ops
            pol = admin(a.url, op["policy"])
            emit({"op": "policy", "policy": pol and {k: pol[k] for k in pol if k != "counters"}}, [] if pol else ["policy POST failed"])
        elif kind == "cold":
            r = one(a.url, op["name"], prompts[op["name"]], mt)
            emit({"op": "cold", "name": op["name"], "ttft_s": round(r["ttft"], 3), "total_s": round(r["total"], 3),
                  "timings": r["timings"], "text_fnv": r["text_fnv"]}, check([r], 1, ref, mt, a.strict, False))
        elif kind == "serial":
            rs = []
            t0 = time.perf_counter()
            for n in op["names"]:
                if remaining() < 5:
                    break  # the missing requests fail the run (check: fewer returned than planned)
                r = one(a.url, n, prompts[n], mt)
                rs.append(r)
                if op.get("ref") and not r.get("error") and r["done"]:
                    ref["%s:%d" % (n, mt)] = {"ids_fnv": r["timings"].get("ids_fnv"), "text_fnv": r["text_fnv"], "text": r["text"]}
            if op.get("ref") and a.ref:
                json.dump(ref, open(a.ref, "w", encoding="utf-8"), indent=1)
            s = summarize(rs, time.perf_counter() - t0, ref, mt)
            s.update({"op": "serial", "per_request": per(rs)})
            emit(s, check(rs, len(op["names"]), ref, mt, a.strict, True))
        elif kind == "cancel":
            r = one(a.url, op["name"], prompts[op["name"]], mt, cancel_after=op.get("after", 4))
            time.sleep(1.0)
            rs = [one(a.url, n, prompts[n], mt) for n in op.get("then", [])]
            s = summarize(rs, sum(x["total"] for x in rs) or 1.0, ref, mt)
            f = check(rs, len(op.get("then", [])), ref, mt, a.strict, True)
            if r["chunks"] < op.get("after", 4):
                f.append("cancelled stream got %d chunks" % r["chunks"])
            s.update({"op": "cancel", "cancelled": op["name"], "per_request": per(rs)})
            emit(s, f)
        elif kind == "cancel_prefill":
            f = []
            pol = admin(a.url, op["policy"])
            if pol is None:
                f.append("policy POST failed")
            plan = [(i, 0.0, [n]) for i, n in enumerate(op["during"])]
            hold = {}

            def late():
                time.sleep(op["start"])
                hold["r"] = one(a.url, op["name"], prompts[op["name"]], mt, close_after_s=op["after_s"])

            th = threading.Thread(target=late, daemon=True)
            th.start()
            rs, wall, planned = run_requests(a.url, prompts, plan, mt)
            th.join(max(1.0, remaining()))
            after = admin(a.url) or {}
            cancels = (after.get("counters") or {}).get("cancels", 0)
            if hold.get("r", {}).get("chunks", 1) != 0:
                f.append("the closed request got a token before closing (prefill not open at the close)")
            if cancels < 1:
                f.append("server counted %s cancelled prefills" % cancels)
            time.sleep(1.0)
            # "then" concurrently, so the slots up to the cancelled one's are all reused
            rs2, _, planned2 = run_requests(a.url, prompts, [(i, 0.0, [n]) for i, n in enumerate(op.get("then", []))], mt)
            f += check(rs + rs2, planned + planned2, ref, mt, a.strict, True)
            s = summarize(rs + rs2, wall, ref, mt)
            s.update({"op": "cancel_prefill", "cancels": cancels, "counters": after.get("counters"), "per_request": per(rs + rs2)})
            emit(s, f)
        elif kind == "cell":
            names = op["names"]
            arms = op.get("arms", {"default": None})
            for arm in op.get("order", list(arms)):
                if remaining() < 15:
                    emit({"op": "cell", "cell": op["cell"], "arm": arm, "skipped": "deadline"}, ["deadline: run not started"])
                    break
                f = []
                if arms.get(arm) is not None:
                    pol = admin(a.url, arms[arm])
                    if pol is None:
                        f.append("policy POST failed")
                else:
                    pol = admin(a.url)
                if op["kind"] == "burst":
                    plan = [(i, 0.0, [n]) for i, n in enumerate(names)]
                elif op["kind"] == "closed":
                    c = op["concurrency"]
                    plan = [(i, 0.0, names[i::c]) for i in range(c)]
                elif op["kind"] == "inject":
                    plan = [(i, 0.0, [n]) for i, n in enumerate(names)] + [(len(names), op["delay"], [op["late"]])]
                else:
                    raise SystemExit("unknown cell kind " + op["kind"])
                rs, wall, planned = run_requests(a.url, prompts, plan, mt)
                f += check(rs, planned, ref, mt, a.strict, True)
                s = summarize(rs, wall, ref, mt)
                after = admin(a.url)
                late = [r for r in rs if r["name"] == op.get("late")]
                early = [r for r in rs if r["name"] != op.get("late")]
                # coverage the run must show (e.g. {"ds": {"unequal_windows": 1, "rejections": 1}}): unqualified otherwise
                for key, least in (op.get("require", {}).get(arm) or {}).items():
                    got = ((after or {}).get("counters") or {}).get(key, 0)
                    if got < least:
                        f.append("coverage: %s = %s < %s (unqualified)" % (key, got, least))
                s.update({"op": "cell", "cell": op["cell"], "arm": arm, "policy": pol and {k: pol[k] for k in pol if k != "counters"},
                          "counters": after and after.get("counters"),
                          "late_ttft_s": round(late[0]["ttft"], 3) if late else None,
                          "early_server_max_gap_ms": round(max((r["timings"].get("max_gap_ms", 0) for r in early), default=0), 1),
                          "per_request": per(rs)})
                emit(s, f)
        else:
            raise SystemExit("unknown op " + kind)
    print("serve_bench: %d failed runs" % failed, flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
