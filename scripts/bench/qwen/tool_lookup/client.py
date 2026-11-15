#!/usr/bin/env python3
"""Strict ABAB client for the tool-result lookup source on a running OpenAI-compatible strix server.

usage: client.py PORT OUT.jsonl --classes rewrite_num:320,prose:256 [--pairs 3] [--deadline SECONDS]

Round r, class c: arm A (tool_lookup=false) then arm B (tool_lookup=true), adjacent requests of one server process,
greedy, non-streaming, token ids returned. Every record carries the wall time, the strix stats of the request and
the GPU clock / busy readings (sysfs, when present) before and after it. A round is started only when the deadline
leaves room for it, so the file always holds whole pairs.
"""
import argparse, glob, hashlib, json, os, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import request  # noqa: E402


def gpu():
    out = {}
    for path in glob.glob("/sys/class/drm/card*/device/pp_dpm_sclk"):
        try:
            cur = [l for l in open(path).read().splitlines() if l.strip().endswith("*")]
            if cur: out["sclk"] = cur[0].split(":")[1].strip(" *")
        except OSError:
            pass
    for path in glob.glob("/sys/class/drm/card*/device/gpu_busy_percent"):
        try: out["busy"] = int(open(path).read().strip())
        except (OSError, ValueError): pass
    return out


def one(port, name, max_tokens, arm):
    body = dict(request(name))
    body.update({"max_tokens": max_tokens, "temperature": 0, "enable_thinking": False, "return_token_ids": True})
    if arm in ("A", "B"): body["tool_lookup"] = arm == "B"
    g0, s = gpu(), time.time()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=110) as r: j = json.loads(r.read())
    e = time.time()
    st = j.get("strix", {})
    ids = st.get("token_ids", [])
    msg = j["choices"][0].get("message") or {}
    rec = {"cls": name, "arm": arm, "wall_s": round(e - s, 3), "tokens": j["usage"]["completion_tokens"],
           "prompt": j["usage"]["prompt_tokens"], "ids_sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16],
           "finish": j["choices"][0].get("finish_reason"), "tool_calls": len(msg.get("tool_calls") or []),
           "gpu_before": g0, "gpu_after": gpu()}
    for k, v in st.items():
        if k != "token_ids" and len(json.dumps(v)) < 300: rec[k] = v
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("port", type=int)
    p.add_argument("out")
    p.add_argument("--classes", required=True)
    p.add_argument("--pairs", type=int, default=3)
    p.add_argument("--deadline", type=float, default=1e9, help="seconds from start; no round starts past the estimate")
    a = p.parse_args()
    classes = [(c.split(":")[0], int(c.split(":")[1])) for c in a.classes.split(",")]
    t0 = time.time()
    with open(a.out, "w") as fo:
        warm = one(a.port, "warm", 16, "warm")
        fo.write(json.dumps(warm) + "\n")
        round_s = 0.0
        for r in range(a.pairs):
            if r and time.time() - t0 + round_s * 1.1 > a.deadline:
                print("deadline: %d rounds" % r, flush=True)
                break
            rs = time.time()
            for name, n in classes:
                for arm in ("A", "B"):
                    try: rec = one(a.port, name, n, arm)
                    except Exception as ex: rec = {"cls": name, "arm": arm, "error": repr(ex)[:200]}
                    rec["round"] = r
                    fo.write(json.dumps(rec) + "\n"); fo.flush()
                    print(name, arm, {k: rec.get(k) for k in ("error", "tokens", "wall_s", "mtp_cycles", "ids_sha")}, flush=True)
            round_s = time.time() - rs


if __name__ == "__main__":
    main()
