"""Forced-k sweep: strix-gemma4-decode --chat --dflash on --draft-k K (MTP), 3 interleaved rounds vs OFF.
usage: sweep.py <lang> <k,k,...>   (paths from SWEEP_ROOT / SWEEP_MODELS / SWEEP_ROPE; writes sweep-<lang>.jsonl, prints a summary)"""
# Node tool (scratch-dir layout); run through a turn script: "sweep [c4] <lang> <k,k,...>".
import json
import os
import statistics
import subprocess
import sys
import time

c4 = sys.argv[1] == "c4"
if c4:
    sys.argv = sys.argv[1:]
lang, ks = sys.argv[1], [int(k) for k in sys.argv[2].split(",")]
root = os.environ.get("SWEEP_ROOT", ".")
M = os.environ.get("SWEEP_MODELS", f"{root}/models")
X = f"{root}/bench-assets"
A = f"{root}/drafter-mtp"
rope = os.environ.get("SWEEP_ROPE", f"{root}/rope")
base = ["nice", "-n", "19", "build/strix-gemma4-decode", f"{M}/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf", "--backend", "device",
        "--rope-tables", rope, "--tokenizer", f"{M}/bf16/tokenizer.json", "--chat", f"{X}/{lang}.json",
        "--max-tokens", "256", "--ignore-eos", "--output", "json", "--draft-mtp", A]
configs = [0] + ks
rows = []
if c4:
    # C4: four identical streams through the real Coordinator (decode --prompt --streams 4); forced k uses DraftMode::on.
    # --chat is C1-only, so render the chat template exactly as serve/chat.cpp render_chat does for one user message
    # (no system, thinking off): <bos><|turn>user NL {strip(content)}<turn|> NL <|turn>model NL. encode() parses specials.
    # (g4 used the raw text without the template: the target emitted <turn|> (106) 256 times; that run is invalid.)
    msgs = json.load(open(f"{X}/{lang}.json"))["messages"]
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    NL = chr(10)
    text = "<bos><|turn>user" + NL + msgs[0]["content"].strip() + "<turn|>" + NL + "<|turn>model" + NL
    base = [a for a in base if a not in ("--chat", f"{X}/{lang}.json")] + ["--prompt", text, "--streams", "4"]
    c1_off = None
    if os.path.exists(f"sweep-{lang}.jsonl"):
        c1_off = next((json.loads(l)["ids"] for l in open(f"sweep-{lang}.jsonl") if json.loads(l)["k"] == 0), None)
    rounds = int(os.environ.get("ROUNDS", "3"))
    with open(f"sweep-c4-{lang}.jsonl", "w") as out:
        for rnd in range(rounds):
            for k in (configs if rnd % 2 == 0 else list(reversed(configs))):
                args = base + (["--dflash", "off"] if k == 0 else ["--dflash", "on", "--draft-k", str(k)])
                res = subprocess.run(args, capture_output=True, text=True, check=True)
                streams = [json.loads(l) for l in res.stdout.splitlines() if l.startswith("{")]
                assert len(streams) == 4
                wall = max(r["timings"]["prefill_ms"] + r["timings"]["decode_ms"] for r in streams)
                row = {"k": k, "round": rnd, "batch_ms": wall, "ids": [r["ids"] for r in streams],
                       "accepted": sum(r["acceptance"]["accepted"] for r in streams),
                       "attempted": sum(r["acceptance"]["attempted"] for r in streams),
                       "target_calls": max(r["target_calls"] for r in streams)}
                out.write(json.dumps(row) + chr(10)); out.flush(); rows.append(row)
                from collections import Counter
                print(json.dumps({"run": len(rows), "k": k, "round": rnd, "batch_ms": round(wall, 1), "accepted": row["accepted"],
                                  "attempted": row["attempted"], "target_calls": row["target_calls"],
                                  "same_as_c1_chat_off": None if c1_off is None else all(i == c1_off for i in row["ids"]),
                                  "top_ids": Counter(row["ids"][0]).most_common(3)}), flush=True)
    ref = [r for r in rows if r["k"] == 0][0]["ids"][0]
    assert all(ids == ref for r in rows for ids in r["ids"]), "C4 ids differ from OFF"
    off = statistics.median(r["batch_ms"] for r in rows if r["k"] == 0)
    print(f"C4 {lang}: all {len(rows)} runs x 4 streams ids == OFF; OFF batch {off:.0f} ms ({4 * len(ref) * 1000 / off:.1f} agg tok/s)")
    for k in configs:
        g = [r for r in rows if r["k"] == k]
        b = statistics.median(r["batch_ms"] for r in g)
        print(json.dumps({"c4": lang, "k": k, "batch_ms": round(b, 1), "ratio_off_over_k": round(off / b, 3),
                          "agg_tok_s": round(4 * len(ref) * 1000 / b, 2), "accepted": sum(r["accepted"] for r in g),
                          "attempted": sum(r["attempted"] for r in g), "target_calls": statistics.median(r["target_calls"] for r in g)}))
    sys.exit(0)
with open(f"sweep-{lang}.jsonl", "w") as out:
    for rnd in range(3):
        order = configs if rnd % 2 == 0 else list(reversed(configs))
        for k in order:
            args = base + (["--dflash", "off"] if k == 0 else ["--dflash", "on", "--draft-k", str(k)])
            t0 = time.time()
            res = subprocess.run(args, capture_output=True, text=True, check=True)
            wall = time.time() - t0
            j = json.loads(res.stdout)
            rec = j["receipts"][-1]
            row = {"k": k, "round": rnd, "process_s": wall, "ids": j["ids"], "prefill_ms": rec["timings"]["prefill_ms"],
                   "decode_ms": rec["timings"]["decode_ms"], "draft_ms": rec["timings"]["draft_ms"],
                   "verify_ms": rec["timings"]["verify_ms"], "accepted": rec["acceptance"]["accepted"],
                   "attempted": rec["acceptance"]["attempted"], "cycles": rec["acceptance"]["cycles"],
                   "depth_eligible": rec["acceptance"]["depth_eligible"], "depth_accepted": rec["acceptance"]["depth_accepted"]}
            out.write(json.dumps(row) + "\n")
            out.flush()
            rows.append(row)
off_ids = [r["ids"] for r in rows if r["k"] == 0][0]
assert all(r["ids"] == off_ids for r in rows), "forced-k ids differ from OFF"
off_req = statistics.median(r["prefill_ms"] + r["decode_ms"] for r in rows if r["k"] == 0)
print(f"{lang}: all {len(rows)} runs ids == OFF ({len(off_ids)} tokens); OFF request {off_req:.0f} ms")
for k in configs:
    g = [r for r in rows if r["k"] == k]
    req = statistics.median(r["prefill_ms"] + r["decode_ms"] for r in g)
    dec = statistics.median(r["decode_ms"] for r in g)
    acc = sum(r["accepted"] for r in g)
    att = sum(r["attempted"] for r in g)
    cyc = sum(r["cycles"] for r in g)
    de = [sum(r["depth_eligible"][i] for r in g) for i in range(7)]
    da = [sum(r["depth_accepted"][i] for r in g) for i in range(7)]
    print(json.dumps({"lang": lang, "k": k, "request_ms": round(req, 1), "request_ratio_off_over_k": round(off_req / req, 3),
                      "decode_tok_s": round(256000 / dec, 2), "tokens_per_cycle": round(768 / cyc, 3) if cyc else None,
                      "accepted": acc, "attempted": att, "draft_ms_per_cycle": round(sum(r["draft_ms"] for r in g) / max(1, cyc), 3),
                      "verify_ms_per_cycle": round(sum(r["verify_ms"] for r in g) / max(1, cyc), 3),
                      "depth_survival": [round(a / e, 3) if e else None for a, e in zip(da, de)]}))
