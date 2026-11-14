#!/usr/bin/env python3
"""FAST-identity quality gate over teacher-forced logit dumps, with run identity.

Two subcommands (Python 3.9+, numpy):

  seal     Run on the node that wrote one rank's dumps, right after the engine exits. Checks that every expected
           dump exists, is fresh, has the declared row count and only finite values, then writes
           manifest-r<rank>.json (schema strix-fast-dump/v1): run id, arm, rank, binary sha256, the environment the
           arm was launched with, per prompt the token-file sha256 and per dump its bytes / sha256 / width.
           A missing, short, stale or already sealed dump is an error: nothing is written.

  compare  Compares a reference run directory with a candidate one (each holding one manifest per rank). Every
           dump is re-hashed against its manifest; run ids must differ between arms and agree between the ranks of
           an arm; prompts, rows and vocabulary widths must agree. Reports per prompt and pooled KL(ref || cand),
           top-1 agreement and the NLL change on the real next tokens, and, when both arms carry block-selection
           dumps, the selected ids that differ per layer and position. Exit 0 = gate passed, 1 = gate failed,
           2 = identity or input error (no verdict). The dNLL limit is absolute (--max-dnll) unless
           --dnll-rule floor is given: then it is mean + 2 sd (n - 1) of >= 3 --null-arm runs, allowed only for
           the engine / profile pairs in DNLL_POLICY (DS41 FAST prefill); the absolute verdict stays a reference.

Dump layout inside a run directory (PREFIX is the --prefix value, default "q"):
  PREFIX-<prompt>-tf-r<rank>.f32   [rows][width] little-endian FP32: the head logits of the last `rows` prompt
                                   positions, this rank's contiguous vocabulary slice (ranks concatenate in order)
  PREFIX-<prompt>-sel-r<rank>.u32  optional selection records: 8 x u32 header {"SELD", 1, mode, position low / high,
                                   rows, heads, top-k}, rows x heads counts, rows x heads x top-k block ids; the
                                   n-th record with the same first position belongs to the rank's n-th layer
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np

SCHEMA = "strix-fast-dump/v1"
SEL_MAGIC = 0x444C4553


class GateError(Exception):
    """Identity or input defect: the comparison has no verdict."""


# dNLL rules per engine / profile. "abs" (every engine, the default): pooled dNLL <= --max-dnll. "floor": pooled dNLL <=
# mean + FLOOR_SIGMAS * sd (n - 1) of at least FLOOR_MIN_NULLS layer-0 null-control arms measured on the same prompts and
# binary as the candidate; KL and top-1 limits are unchanged. Only the DeepSeek V4.1 FAST prefill profile may use "floor":
# its layer-0 chaos floor (any one-ulp change at layer 0 re-routes MoE rows) already costs dNLL ~0.012 > 0.01. Adding an
# entry here is a policy decision, not a tuning knob.
DNLL_POLICY = {
    ("ds41", "fast-prefill"): ("abs", "floor"),
}
FLOOR_MIN_NULLS = 3
FLOOR_SIGMAS = 2.0
WITNESS_SCHEMA = "strix-fast-execution/v1"
DS41_NULL_CONTROL = {"kind": "layer0-null", "layer": 0, "magnitude": 1, "unit": "ulp"}


def execution_witness(doc):
    """Validate producer evidence and its binding to this rank's sealed inputs/outputs."""
    w = doc.get("execution")
    if not isinstance(w, dict) or w.get("schema") != WITNESS_SCHEMA:
        raise GateError("missing or invalid execution witness")
    for key, expected in (("run_id", doc["run_id"]), ("rank", doc["rank"]),
                          ("binary_sha256", doc["binary"]["sha256"]), ("prompts", doc["prompts"])):
        if w.get(key) != expected:
            raise GateError(f"execution witness differs from sealed {key}")
    for key in ("model_sha256", "numerical_config_sha256"):
        value = w.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise GateError(f"execution witness: invalid {key}")
    if any(not isinstance(w.get(k), str) or not w[k] for k in ("engine", "profile")):
        raise GateError("execution witness: missing engine/profile")
    if "control" not in w or (w["control"] is not None and not isinstance(w["control"], dict)):
        raise GateError("execution witness: missing or invalid control identity")
    return w


def check_floor_execution(ref, cand, nulls, engine, profile):
    for i, (mr, mc) in enumerate(zip(ref, cand)):
        wr, wc = execution_witness(mr), execution_witness(mc)
        if (wc["engine"], wc["profile"]) != (engine, profile) or wc["control"] is not None:
            raise GateError("candidate execution witness disagrees with floor engine/profile or is a control")
        if (wr["engine"], wr["profile"], wr["control"]) != (engine, "exact", None):
            raise GateError("floor reference must witness the exact baseline without a control")
        if wr["model_sha256"] != wc["model_sha256"]:
            raise GateError("floor reference/candidate model identity differs")
        for _, mn in nulls:
            wn = execution_witness(mn[i])
            if any(wn[k] != wr[k] for k in ("engine", "profile", "model_sha256", "numerical_config_sha256")):
                raise GateError("null execution witness differs from the reference model/baseline numerical configuration")
            # JSON comparison also refuses booleans in place of integer layer/magnitude.
            if json.dumps(wn["control"], sort_keys=True) != json.dumps(DS41_NULL_CONTROL, sort_keys=True):
                raise GateError("null execution witness must attest the DS41 one-ulp layer-0 control")


def allowed_dnll_rules(engine, profile):
    return DNLL_POLICY.get((engine, profile), ("abs",))


def check_dnll_policy(rule, engine, profile, n_nulls):
    """Refuses a rule the policy table does not grant, or a floor without enough nulls (GateError = no verdict)."""
    if rule == "abs":
        if n_nulls:
            raise GateError("--null-arm is only used by --dnll-rule floor")
        return
    if not engine or not profile:
        raise GateError("--dnll-rule floor needs --engine and --profile (checked against DNLL_POLICY)")
    if rule not in allowed_dnll_rules(engine, profile):
        granted = ", ".join(f"{e}/{p}" for (e, p), rules in sorted(DNLL_POLICY.items()) if rule in rules)
        raise GateError(f"--dnll-rule {rule} is not allowed for engine {engine} profile {profile} (DNLL_POLICY grants it to: "
                        f"{granted}); use the absolute rule")
    if n_nulls < FLOOR_MIN_NULLS:
        raise GateError(f"--dnll-rule floor needs at least {FLOOR_MIN_NULLS} --null-arm layer-0 null controls, got {n_nulls}")


def floor_limit(null_dnll):
    """(mean, sd with n - 1, limit = mean + FLOOR_SIGMAS * sd) of the null-control pooled dNLL values."""
    v = np.asarray(null_dnll, dtype=np.float64)
    if v.ndim != 1 or v.size < FLOOR_MIN_NULLS:
        raise GateError(f"floor rule needs at least {FLOOR_MIN_NULLS} null dNLL values, got {v.size}")
    if not np.all(np.isfinite(v)):
        raise GateError("floor rule: a null dNLL is not finite")
    mean, sd = float(v.mean()), float(v.std(ddof=1))
    return mean, sd, mean + FLOOR_SIGMAS * sd


def dnll_verdict(dnll, rule, max_dnll, null_dnll=()):
    """The dNLL part of the gate: {rule, limit, pass, abs_limit, abs_pass[, mean, sd, sigmas, n_nulls]}."""
    out = {"rule": rule, "abs_limit": max_dnll, "abs_pass": bool(dnll <= max_dnll)}
    if rule == "floor":
        mean, sd, limit = floor_limit(null_dnll)
        out.update(mean=mean, sd=sd, sigmas=FLOOR_SIGMAS, n_nulls=len(null_dnll), limit=limit)
    else:
        out["limit"] = max_dnll
    out["pass"] = bool(dnll <= out["limit"])
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_pairs(items, what):
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key or key in out:
            raise GateError(f"{what}: expected unique NAME=VALUE, got {item!r}")
        out[key] = value
    return out


def read_selection(path):
    """Returns [(position, rows, heads, topk, mode, counts[rows, heads], ids[rows, heads, topk])] in file order."""
    raw = np.fromfile(path, "<u4")
    out, o = [], 0
    while o < raw.size:
        if raw.size - o < 8 or int(raw[o]) != SEL_MAGIC or int(raw[o + 1]) != 1:
            raise GateError(f"{path}: bad selection record header at word {o}")
        mode, pos = int(raw[o + 2]), int(raw[o + 3]) | (int(raw[o + 4]) << 32)
        rows, heads, topk = int(raw[o + 5]), int(raw[o + 6]), int(raw[o + 7])
        n = rows * heads
        if n == 0 or topk == 0 or raw.size - o - 8 < n * (1 + topk):
            raise GateError(f"{path}: short selection record at word {o}")
        cnt = raw[o + 8:o + 8 + n].reshape(rows, heads)
        ids = raw[o + 8 + n:o + 8 + n * (1 + topk)].reshape(rows, heads, topk)
        if int(cnt.max()) > topk:
            raise GateError(f"{path}: selection count above top-k at word {o}")
        out.append((pos, rows, heads, topk, mode, cnt, ids))
        o += 8 + n * (1 + topk)
    if not out:
        raise GateError(f"{path}: no selection records")
    return out


# ---------------------------------------------------------------------------------------------------------- seal
def seal(a):
    d = pathlib.Path(a.dir)
    manifest = d / f"manifest-r{a.rank}.json"
    if manifest.exists():
        raise GateError(f"{manifest}: already sealed (run directories are never reused)")
    if not a.run_id or not a.arm:
        raise GateError("seal: empty --run-id or --arm")
    binary = pathlib.Path(a.binary)
    if not binary.is_file():
        raise GateError(f"seal: no binary {binary}")
    env = parse_pairs(a.env, "--env")
    attest = parse_pairs(a.attest, "--attest")
    prompts = {}
    for name, tok_path in parse_pairs(a.prompt, "--prompt").items():
        tok = pathlib.Path(tok_path)
        if not tok.is_file() or tok.stat().st_size == 0 or tok.stat().st_size % 4:
            raise GateError(f"seal: bad token file {tok}")
        files = {}
        kinds = [("tf", "f32", True)] + ([("sel", "u32", True)] if a.select else [])
        for kind, ext, required in kinds:
            f = d / f"{a.prefix}-{name}-{kind}-r{a.rank}.{ext}"
            if not f.is_file():
                if required:
                    raise GateError(f"seal: missing dump {f}")
                continue
            st = f.stat()
            if a.not_before is not None and st.st_mtime < a.not_before:
                raise GateError(f"seal: stale dump {f} (written before this run started)")
            entry = {"name": f.name, "bytes": st.st_size, "sha256": sha256_file(f)}
            if kind == "tf":
                if st.st_size == 0 or st.st_size % (a.rows * 4):
                    raise GateError(f"seal: {f} has {st.st_size} bytes, not {a.rows} rows of FP32")
                entry["width"] = st.st_size // (a.rows * 4)
                z = np.memmap(f, "<f4", "r")
                if not np.isfinite(z).all():
                    raise GateError(f"seal: nonfinite logits in {f}")
            else:
                recs = read_selection(f)
                modes = sorted({r[4] for r in recs})
                if len(modes) != 1:
                    raise GateError(f"seal: {f} mixes selection modes {modes}")
                entry["records"], entry["mode"] = len(recs), modes[0]
                if "sel_mode" in attest and str(modes[0]) != attest["sel_mode"]:
                    raise GateError(f"seal: {f} was written by selection mode {modes[0]}, arm declares {attest['sel_mode']}")
            files[kind] = entry
        prompts[name] = {"tokens": tok.stat().st_size // 4, "tokens_sha256": sha256_file(tok), "rows": a.rows, "files": files}
    if not prompts:
        raise GateError("seal: no --prompt")
    doc = {"schema": SCHEMA, "run_id": a.run_id, "arm": a.arm, "rank": a.rank, "prefix": a.prefix,
           "binary": {"name": binary.name, "sha256": sha256_file(binary)}, "env": env, "attest": attest,
           "sealed_unix": int(time.time()), "prompts": prompts}
    if a.execution_witness:
        try:
            doc["execution"] = json.loads(pathlib.Path(a.execution_witness).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise GateError(f"cannot read execution witness: {e}") from e
        execution_witness(doc)
    tmp = manifest.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, manifest)
    print(f"sealed {manifest.name}: run {a.run_id} arm {a.arm} rank {a.rank} binary {doc['binary']['sha256'][:12]} "
          f"prompts {' '.join(prompts)}")
    return 0


# ------------------------------------------------------------------------------------------------------- compare
def load_arm(d, ranks):
    d = pathlib.Path(d)
    docs = []
    for r in ranks:
        p = d / f"manifest-r{r}.json"
        if not p.is_file():
            raise GateError(f"{p}: missing manifest (the arm was not sealed)")
        doc = json.loads(p.read_text(encoding="utf-8"))
        if doc.get("schema") != SCHEMA or doc.get("rank") != r:
            raise GateError(f"{p}: not a {SCHEMA} manifest of rank {r}")
        docs.append(doc)
    first = docs[0]
    for doc in docs[1:]:
        for key in ("run_id", "arm", "env", "attest", "prefix"):
            if doc[key] != first[key]:
                raise GateError(f"{d}: ranks disagree on {key}")
        if sorted(doc["prompts"]) != sorted(first["prompts"]):
            raise GateError(f"{d}: ranks disagree on the prompt set")
    if any("execution" in doc for doc in docs):
        witnesses = [execution_witness(doc) for doc in docs]
        for w in witnesses[1:]:
            for key in ("engine", "profile", "model_sha256", "numerical_config_sha256", "control"):
                if w[key] != witnesses[0][key]:
                    raise GateError(f"{d}: ranks disagree on execution {key}")
    return d, docs


def checked_file(d, entry):
    f = d / entry["name"]
    if not f.is_file() or f.stat().st_size != entry["bytes"]:
        raise GateError(f"{f}: missing or not the sealed size")
    if sha256_file(f) != entry["sha256"]:
        raise GateError(f"{f}: content differs from its manifest")
    return f


def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def arm_logits(d, docs, name):
    shards, rows = [], docs[0]["prompts"][name]["rows"]
    for doc in docs:
        p = doc["prompts"][name]
        if p["rows"] != rows:
            raise GateError(f"{d}: ranks disagree on the rows of {name}")
        e = p["files"]["tf"]
        shards.append(np.memmap(checked_file(d, e), "<f4", "r", shape=(rows, e["width"])))
    return shards, rows


def compare_logits(da, a, db, b, name, tokens, chunk=16):
    sa, rows = arm_logits(da, a, name)
    sb, rows_b = arm_logits(db, b, name)
    if rows != rows_b or [s.shape for s in sa] != [s.shape for s in sb]:
        raise GateError(f"{name}: arms disagree on rows or vocabulary widths")
    total = len(tokens)
    if rows > total:
        raise GateError(f"{name}: {rows} rows scored but the prompt has {total} tokens")
    kl, agree, nla, nlb = np.empty(rows), np.empty(rows, bool), [], []
    identical = True
    for i in range(0, rows, chunk):
        za = np.concatenate([s[i:i + chunk] for s in sa], axis=1).astype(np.float64)
        zb = np.concatenate([s[i:i + chunk] for s in sb], axis=1).astype(np.float64)
        if not (np.isfinite(za).all() and np.isfinite(zb).all()):
            raise GateError(f"{name}: nonfinite logits")
        identical = identical and bool((za == zb).all())
        la, lb = log_softmax(za), log_softmax(zb)
        kl[i:i + chunk] = (np.exp(la) * (la - lb)).sum(axis=1)
        agree[i:i + chunk] = za.argmax(axis=1) == zb.argmax(axis=1)
        for j in range(za.shape[0]):
            pos = total - rows + i + j  # the row at prompt position pos predicts token pos + 1
            if pos + 1 < total:
                nla.append(-la[j, tokens[pos + 1]])
                nlb.append(-lb[j, tokens[pos + 1]])
    return {"rows": rows, "kl": kl, "agree": agree, "nll_ref": np.array(nla), "nll_cand": np.array(nlb), "identical": identical}


def compare_selection(da, a, db, b, name):
    """Per global layer (ranks in order, each rank's records in layer order): slots and block ids that differ."""
    layers, base = [], 0
    for doc_a, doc_b in zip(a, b):
        ea, eb = doc_a["prompts"][name]["files"].get("sel"), doc_b["prompts"][name]["files"].get("sel")
        if ea is None or eb is None:
            return None
        ra, rb = read_selection(checked_file(da, ea)), read_selection(checked_file(db, eb))
        if len(ra) != len(rb):
            raise GateError(f"{name}: arms wrote a different number of selection records")
        seen, depth = {}, 0
        for x, y in zip(ra, rb):
            if x[:4] != y[:4]:
                raise GateError(f"{name}: selection records disagree on position / rows / heads / top-k")
            pos, rows, heads, topk = x[:4]
            layer = seen.get(pos, 0)
            seen[pos] = layer + 1
            depth = max(depth, layer + 1)
            while len(layers) <= base + layer:
                layers.append({"slots": 0, "slot_diff": 0, "id_diff": 0, "first_pos": None, "modes": [x[4], y[4]]})
            acc = layers[base + layer]
            ca, cb = x[5].astype(np.int64), y[5].astype(np.int64)
            slot = np.arange(topk)[None, None, :]
            big = np.int64(1) << 40
            ia = np.sort(np.where(slot < ca[:, :, None], x[6].astype(np.int64), big), axis=2)
            ib = np.sort(np.where(slot < cb[:, :, None], y[6].astype(np.int64), big), axis=2)
            neq = ia != ib
            lost = np.zeros((rows, heads), np.int64)  # ids of the reference set missing from the candidate set
            for r, h in zip(*np.nonzero(neq.any(axis=2))):
                lost[r, h] = len(set(ia[r, h][ia[r, h] < big].tolist()) - set(ib[r, h][ib[r, h] < big].tolist()))
            bad = neq.any(axis=2)
            acc["slots"] += rows * heads
            acc["slot_diff"] += int(bad.sum())
            acc["id_diff"] += int(lost.sum())
            if bad.any():
                p = pos + int(np.nonzero(bad.any(axis=1))[0][0])
                acc["first_pos"] = p if acc["first_pos"] is None else min(acc["first_pos"], p)
        base += depth
    return layers


def compare(a):
    ranks = sorted(set(a.ranks))
    null_dirs = a.null_arm or []
    if a.expect_identical and (a.dnll_rule != "abs" or null_dirs):
        raise GateError("--expect-identical has no dNLL limit: drop --dnll-rule / --null-arm")
    check_dnll_policy(a.dnll_rule, a.engine, a.profile, len(null_dirs))
    da, ma = load_arm(a.ref, ranks)
    db, mb = load_arm(a.cand, ranks)
    if ma[0]["run_id"] == mb[0]["run_id"]:
        raise GateError("both directories hold the same run id: one run compared with itself")
    tokens = parse_pairs(a.tokens, "--tokens")
    names = sorted(ma[0]["prompts"])
    if names != sorted(mb[0]["prompts"]) or names != sorted(tokens):
        raise GateError("arms and --tokens disagree on the prompt set")
    nulls, seen_ids = [], {ma[0]["run_id"], mb[0]["run_id"]}
    for nd in null_dirs:  # floor rule: every null is a sealed run on the candidate's binary and prompts
        dn, mn = load_arm(nd, ranks)
        if mn[0]["run_id"] in seen_ids:
            raise GateError(f"{nd}: null arm run id {mn[0]['run_id']} repeats the reference, the candidate or another null")
        seen_ids.add(mn[0]["run_id"])
        if [m["binary"]["sha256"] for m in mn] != [m["binary"]["sha256"] for m in mb]:
            raise GateError(f"{nd}: null arm ran another binary than the candidate (the floor must be measured on the same binary)")
        if sorted(mn[0]["prompts"]) != names:
            raise GateError(f"{nd}: null arm prompt set differs from the reference / candidate")
        nulls.append((dn, mn))
    if a.dnll_rule == "floor":
        check_floor_execution(ma, mb, nulls, a.engine, a.profile)
    report = {"ref": {k: ma[0][k] for k in ("run_id", "arm", "env", "attest")}, "cand": {k: mb[0][k] for k in ("run_id", "arm", "env", "attest")},
              "ref_binary": [m["binary"]["sha256"] for m in ma], "cand_binary": [m["binary"]["sha256"] for m in mb], "prompts": {}}
    if a.dnll_rule == "floor":
        report["ref"]["execution"] = ma[0]["execution"]
        report["cand"]["execution"] = mb[0]["execution"]
    print(f"ref  run {ma[0]['run_id']} arm {ma[0]['arm']} env {ma[0]['env']} binary {report['ref_binary'][0][:12]}")
    print(f"cand run {mb[0]['run_id']} arm {mb[0]['arm']} env {mb[0]['env']} binary {report['cand_binary'][0][:12]}")
    pooled = {"kl": [], "agree": [], "nll_ref": [], "nll_cand": []}
    identical, sel_slots, sel_slot_diff, sel_id_diff, sel_seen = True, 0, 0, 0, False
    for name in names:
        tok_sha = sha256_file(tokens[name])
        for m in ma + mb + [m for _, mn in nulls for m in mn]:
            if m["prompts"][name]["tokens_sha256"] != tok_sha:
                raise GateError(f"{name}: a sealed run used a different token file than --tokens")
        tok = np.fromfile(tokens[name], "<u4").astype(np.int64)
        r = compare_logits(da, ma, db, mb, name, tok)
        identical = identical and r["identical"]
        for k in pooled:
            pooled[k].append(r[k])
        row = {"rows": r["rows"], "kl_mean": float(r["kl"].mean()), "kl_max": float(r["kl"].max()),
               "kl_median": float(np.median(r["kl"])), "top1": float(r["agree"].mean()),
               "nll_ref": float(r["nll_ref"].mean()), "dnll": float(r["nll_cand"].mean() - r["nll_ref"].mean()),
               "identical": r["identical"]}
        print(f"{name}: rows {row['rows']} KL mean {row['kl_mean']:.5f} median {row['kl_median']:.5f} max {row['kl_max']:.4f}; "
              f"top-1 {row['top1']:.4f}; NLL ref {row['nll_ref']:.4f} dNLL {row['dnll']:+.5f}; logits identical {row['identical']}")
        layers = compare_selection(da, ma, db, mb, name)
        if layers is not None:
            sel_seen = True
            slots, sd, idd = (sum(x[k] for x in layers) for k in ("slots", "slot_diff", "id_diff"))
            sel_slots, sel_slot_diff, sel_id_diff = sel_slots + slots, sel_slot_diff + sd, sel_id_diff + idd
            first = next((i for i, x in enumerate(layers) if x["slot_diff"]), None)
            row["selection"] = {"layers": len(layers), "slots": slots, "slot_diff": sd, "id_diff": idd, "first_layer": first,
                                "first_pos": None if first is None else layers[first]["first_pos"],
                                "modes": layers[0]["modes"], "per_layer": [[x["slot_diff"], x["id_diff"]] for x in layers]}
            print(f"{name}: selection modes ref/cand {layers[0]['modes']}: {len(layers)} layers, {slots} (row, head) slots, "
                  f"{sd} with a different set ({sd / slots:.3e}), {idd} ids replaced; first at layer {first}"
                  f"{'' if first is None else ' position ' + str(layers[first]['first_pos'])}")
            if first is not None:
                print(f"{name}: differing slots per layer: {' '.join(str(x['slot_diff']) for x in layers)}")
        report["prompts"][name] = row
    kl = float(np.concatenate(pooled["kl"]).mean())
    top1 = float(np.concatenate(pooled["agree"]).mean())
    dnll = float(np.concatenate(pooled["nll_cand"]).mean() - np.concatenate(pooled["nll_ref"]).mean())
    null_rows = []
    for dn, mn in nulls:  # pooled exactly like the candidate: same reference, same prompts, same labels
        acc = {"kl": [], "agree": [], "nll_ref": [], "nll_cand": []}
        for name in names:
            r = compare_logits(da, ma, dn, mn, name, np.fromfile(tokens[name], "<u4").astype(np.int64))
            for k in acc:
                acc[k].append(r[k])
        nd_ = float(np.concatenate(acc["nll_cand"]).mean() - np.concatenate(acc["nll_ref"]).mean())
        null_rows.append({"dir": str(dn), "run_id": mn[0]["run_id"], "arm": mn[0]["arm"], "env": mn[0]["env"],
                          "execution": mn[0]["execution"],
                          "rows": int(sum(len(k) for k in acc["kl"])), "kl": float(np.concatenate(acc["kl"]).mean()),
                          "top1": float(np.concatenate(acc["agree"]).mean()), "dnll": nd_})
        print(f"null {mn[0]['arm']} run {mn[0]['run_id']} env {mn[0]['env']}: KL {null_rows[-1]['kl']:.5f}, "
              f"top-1 {null_rows[-1]['top1']:.4f}, dNLL {nd_:+.5f}")
    dv = None
    if a.expect_identical:
        ok = identical and sel_slot_diff == 0
        rule = "identical logits and selections"
    else:
        dv = dnll_verdict(dnll, a.dnll_rule, a.max_dnll, [x["dnll"] for x in null_rows])
        dv.update(engine=a.engine, profile=a.profile, nulls=null_rows)
        ok = kl <= a.max_kl and top1 >= a.min_top1 and dv["pass"]
        if a.dnll_rule == "floor":
            rule = (f"KL <= {a.max_kl}, top-1 >= {a.min_top1}, dNLL <= {dv['limit']:+.5f} (floor {a.engine}/{a.profile}: "
                    f"{dv['n_nulls']} nulls mean {dv['mean']:+.5f} + {FLOOR_SIGMAS:g} sd {dv['sd']:.5f})")
        else:
            rule = f"KL <= {a.max_kl}, top-1 >= {a.min_top1}, dNLL <= {a.max_dnll}"
    report["pooled"] = {"rows": int(sum(len(k) for k in pooled["kl"])), "kl": kl, "top1": top1, "dnll": dnll, "identical": identical,
                        "selection": {"slots": sel_slots, "slot_diff": sel_slot_diff, "id_diff": sel_id_diff} if sel_seen else None,
                        "rule": rule, "dnll_rule": dv, "pass": ok}
    if dv is not None and a.dnll_rule == "floor":
        print(f"REFERENCE absolute dNLL <= {a.max_dnll}: {'PASS' if dv['abs_pass'] else 'FAIL'} (not the verdict under the floor rule)")
    sel = f"; selection {sel_slot_diff}/{sel_slots} slots differ, {sel_id_diff} ids" if sel_seen else ""
    print(f"POOLED rows {report['pooled']['rows']}: KL {kl:.5f}, top-1 {top1:.4f}, dNLL {dnll:+.5f}, logits identical {identical}{sel} "
          f"[{rule}] -> {'PASS' if ok else 'FAIL'}")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seal")
    s.add_argument("--dir", required=True)
    s.add_argument("--rank", type=int, required=True)
    s.add_argument("--arm", required=True)
    s.add_argument("--run-id", required=True)
    s.add_argument("--binary", required=True)
    s.add_argument("--prefix", default="q")
    s.add_argument("--rows", type=int, required=True)
    s.add_argument("--prompt", action="append", metavar="NAME=TOKENS.u32")
    s.add_argument("--env", action="append", metavar="NAME=VALUE", help="environment the arm was launched with")
    s.add_argument("--attest", action="append", metavar="KEY=VALUE", help="sel_mode=N must equal the selection dump's mode")
    s.add_argument("--execution-witness", metavar="JSON", help="launcher/engine execution evidence; required on every floor arm")
    s.add_argument("--select", action="store_true", help="selection dumps are required")
    s.add_argument("--not-before", type=float, help="unix time the run started: older dumps are stale")
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("cand")
    c.add_argument("--ranks", type=int, nargs="+", default=[0])
    c.add_argument("--tokens", action="append", metavar="NAME=TOKENS.u32")
    c.add_argument("--max-kl", type=float, default=0.02)
    c.add_argument("--min-top1", type=float, default=0.93)
    c.add_argument("--max-dnll", type=float, default=0.01)
    c.add_argument("--dnll-rule", choices=("abs", "floor"), default="abs",
                   help="abs: dNLL <= --max-dnll (default, every engine); floor: dNLL <= mean + 2 sd of the --null-arm runs "
                        "(only engine/profile pairs in DNLL_POLICY, today ds41/fast-prefill)")
    c.add_argument("--null-arm", action="append", metavar="DIR",
                   help="sealed layer-0 null-control run on the candidate's binary and prompts (repeat, >= 3 for floor)")
    c.add_argument("--engine", help="engine of the candidate (floor rule policy check), e.g. ds41")
    c.add_argument("--profile", help="profile of the candidate (floor rule policy check), e.g. fast-prefill")
    c.add_argument("--expect-identical", action="store_true", help="repeat of one identity: any difference fails")
    c.add_argument("--json")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "seal" and a.rows <= 0:
            raise GateError("seal: --rows must be positive")
        return seal(a) if a.cmd == "seal" else compare(a)
    except GateError as e:
        print(f"fast_logit_gate: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
