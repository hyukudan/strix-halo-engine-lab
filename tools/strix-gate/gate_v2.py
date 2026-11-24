#!/usr/bin/env python3
"""FAST gate v2: one statistics implementation for every engine (README "FAST gate v2").

Limits are the FAST gate's (KL <= 0.02, top-1 >= 0.93, dNLL <= 0.01), applied pooled AND per window, measured so
that the reference's own chaos is not charged to the lever:
  R1 MoE engines: the lever arm replays the reference's expert routes (ids and weights); the route flip rate of the
     free-running lever arm, over the free-running null arm's, must be <= 1.5 (reported per window and pooled).
  R2 per window: KL <= max(0.02, 1.5 x the null arm's KL on that window).
  Null floor (pooled bars only): a pooled bar whose pooled null (same windows, row-weighted mean of the null arms) fails
     it, or lies within 2 of its own row se of failing it, is judged relative to that null and only relaxed:
     KL <= max(0.02, null + 0.02), top-1 >= min(0.93, null - 0.01), dNLL <= max(0.01, null + 0.01). Printed on its own
     NULL-FLOOR line (JSON pooled.null_floor), with the bars that pass only under the floor.
     Top-1 and dNLL per window are three-state (one-sided, z = 2): PASS at disagreement <= L, FAIL above
     L + z x sqrt(L (1 - L) / rows), UNRESOLVED between, with L = max(0.07, 1.5 x the null arm's disagreement); dNLL
     PASS at <= 0.01, FAIL above 0.01 + z x se, UNRESOLVED between, se capped at 1.5 x the null's se on that window.
     A 63-row window has dNLL se ~0.02, twice the limit: the absolute form failed the exact one-ulp null itself on 7 of
     16 DS41 windows, and a two-state "significant only" form passed a real loss on one content class 5 times in 6.
     An UNRESOLVED window is cleared only by its content class: the windows of a class (declared before scoring;
     default one class, or the prompt-length band with a manifest) are pooled and judged by the same rule; a class
     verdict is final on >= 800 rows. The pooled top-1 and dNLL limits stay absolute (>= 800 rows pooled).
  R3 dNLL on held-out ids (labels that neither arm generated), at least 800 scored rows pooled, standard error shown.
  R4 reference check: a window where the candidate beats the exact reference by more than 3 se (and, with a null, the
     candidate-minus-null difference is also below -3 se) is "reference suspect"; a window whose reference NLL is far
     above the others (> max(3 x median, median + 2 nats), >= 3 windows) is "off-distribution" (control or
     chat-template ids, a timing prompt). Either makes the verdict HOLD, never PASS.
  Nulls are evaluation-order (one-ulp class) runs; a null equal to the reference (KL 0) is refused (no verdict).

  python gate_v2.py qual OUT --ref REF --cand FAST --null NULL [--cand-free FASTF --null-free NULLF] [--json F]
      option_qual runner output: OUT/logits/<case>.<arm>.bin, OUT/log/<case>.<arm>.stderr, OUT/routes/<case>.<arm>.rte
  python gate_v2.py qual OUT --format head --manifest M --ref exact --cand FAST --null NULL [--trace-diag LOG ...]
      prefill quality tool head dumps OUT/<case>-<arm>.f32 (64 x V rows, labels = the manifest's prompt IDs); an engine
      without a route tap runs every arm free and reports R1 flips from its trace diag lines
  fast_logit_gate.py compare REF CAND --rule v2 --null-arm NULL ...   (sealed strix-fast-dump arms)
  qual --classes C.json: {"case": "class", ...} declares the content class of each window before scoring.
  qual --r4-frozen R.json --r4-sha256 SHA --reference-identity ID.json: use the pinned, reference-only R4 record.
      r4-freeze freezes the original windows of a preregistered plan; r4-substitute writes a new version with one
      independently approved change. The complete live set must match that version. See the README for intake,
      reference evidence and retrospective diagnostics.
  qual --tier turbo: the same rules with the TURBO limits (TURBO_LIMITS: KL <= 0.05, top-1 >= 0.90, dNLL <= 0.03,
      route flip ratio <= 2.0; the per-window disagreement floor follows as 1 - 0.90 = 0.10). TURBO levers trade
      quality for speed and are explicit opt-in, never a default; a TURBO verdict never qualifies a FAST lever. The tier
      is printed on its own TIER line and stored in the JSON ("tier", "limits"). Default --tier fast: output unchanged.
Exit 0 = PASS, 1 = FAIL, 2 = no verdict (bookkeeping defect, too few rows, labels not held out, missing null),
     3 = UNRESOLVED (provisional: the lever stays opt-in for the unresolved classes), 4 = HOLD (reference check).
"""
import argparse
import glob
import hashlib
import json
import math
import os
import pathlib
import re
import struct
import sys
import tempfile

import numpy as np

LIMITS = {"max_kl": 0.02, "min_top1": 0.93, "max_dnll": 0.01, "null_factor": 1.5, "max_flip_ratio": 1.5, "min_rows": 800,
          "window_z": 2.0, "suspect_z": 3.0, "se_null_cap": 1.5, "ref_nll_factor": 3.0, "ref_nll_margin": 2.0,
          "floor_z": 2.0, "floor_kl_margin": 0.02, "floor_top1_margin": 0.01, "floor_dnll_margin": 0.01}
# TURBO tier: looser acceptance bars, every other rule (nulls, replay, classes, >= 800 rows, reference check) unchanged.
TURBO_LIMITS = dict(LIMITS, max_kl=0.05, min_top1=0.90, max_dnll=0.03, max_flip_ratio=2.0)
TIERS = {"fast": LIMITS, "turbo": TURBO_LIMITS}
EXIT = {"PASS": 0, "FAIL": 1, "UNRESOLVED": 3, "HOLD": 4}
ROUTE_MAGIC = b"STXROUT1"
RECEIPT = re.compile(rb"^strix-route-file/v1 (\{.*\})\s*$", re.M)
MIN_LABEL_ROWS = 32  # below this the held-out check cannot tell greedy ids from text


class GateError(Exception):
    """A defect of the evidence: no verdict (exit 2)."""


def mean_se(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        raise GateError("no scored rows")
    se = float(x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else float("inf")
    return float(x.mean()), se


def _judge(kl, agree, d, null_kl, null_dis, null_d, limits):
    """R2 + R4 on one window or one pooled content class: KL two-state, top-1 and dNLL three-state, reference check.

    null_d: per-row dNLL of the null arm on the same labels (None: no se cap, the reference check has no null term)."""
    z = limits.get("window_z", LIMITS["window_z"])
    kl_limit = max(limits["max_kl"], limits["null_factor"] * null_kl)
    dnll, se = mean_se(d)
    dis = 1.0 - float(agree.mean())
    dis_pass = min(1.0, max(1.0 - limits["min_top1"], limits["null_factor"] * null_dis))
    # one-sided binomial score test of "disagreement <= dis_pass" on these rows
    dis_fail = min(1.0, dis_pass + z * math.sqrt(dis_pass * (1.0 - dis_pass) / max(int(agree.size), 1)))
    j = {"kl": float(kl.mean()), "top1": 1.0 - dis, "dnll": dnll, "null_kl": null_kl, "kl_limit": kl_limit,
         "top1_pass": 1.0 - dis_pass, "top1_limit": 1.0 - dis_fail}
    se_used = se
    if null_d is not None and null_d.size > 1:
        j["null_dnll"], j["null_dnll_se"] = mean_se(null_d)
        cap = limits.get("se_null_cap", LIMITS["se_null_cap"]) * j["null_dnll_se"]
        if math.isfinite(cap) and 0.0 < cap < se:
            se_used, j["dnll_se_rows"] = cap, se
    j["dnll_se"] = se_used
    j["dnll_limit"] = limits["max_dnll"] + (z * se_used if math.isfinite(se_used) else 0.0)
    j["states"] = {"kl": "pass" if j["kl"] <= kl_limit else "fail",
                   "top1": "pass" if dis <= dis_pass else ("fail" if dis > dis_fail else "unresolved"),
                   "dnll": "pass" if dnll <= limits["max_dnll"] else ("fail" if dnll > j["dnll_limit"] else "unresolved")}
    s = j["states"].values()
    j["state"] = "fail" if "fail" in s else ("unresolved" if "unresolved" in s else "pass")
    # R4: the lever beats the exact reference on the true text by more than suspect_z se, and the null does not move
    # the same way (paired candidate-minus-null not below -suspect_z se as well)
    sz = limits.get("suspect_z", LIMITS["suspect_z"])
    suspect = math.isfinite(se_used) and dnll < -sz * se_used
    if suspect and null_d is not None and null_d.size == d.size and d.size > 1:
        j["cand_minus_null"], cn_se = mean_se(d - null_d)
        suspect = j["cand_minus_null"] < -sz * cn_se
    j["reference_suspect"] = bool(suspect)
    return j


def window_stats(name, kl, agree, nll_ref, nll_cand, null_kls, hit_ref=None, hit_cand=None, limits=LIMITS, null_top1s=None,
                 null_nlls=None, cls=None, null_kl_rows=None):
    """One window: R2 KL limit relative to this window's null; top-1 and dNLL PASS / UNRESOLVED / FAIL on its rows; R4.

    null_top1s: the null arms' top-1 agreement on this window (None: the null-relative top-1 term is absent).
    null_nlls: the null arms' per-row NLL on the same labels (the first is used: se cap and the paired reference check).
    cls: the window's content class, declared before scoring (None: one class for the whole gate set).
    null_kl_rows: the first null arm's per-row KL on this window (None: the pooled null-floor KL has no se term)."""
    kl, agree = np.asarray(kl, np.float64), np.asarray(agree, bool)
    nll_ref = np.asarray(nll_ref, np.float64)
    nll_cand = np.asarray(nll_cand, np.float64)
    if nll_ref.ndim != 1 or nll_cand.shape != nll_ref.shape or kl.ndim != 1 or agree.shape != kl.shape or not kl.size:
        raise GateError(f"{name}: unpaired or empty score rows")
    if not all(np.isfinite(v).all() for v in (kl, nll_ref, nll_cand, null_kls)):
        raise GateError(f"{name}: non-finite score or null evidence")
    d = nll_cand - nll_ref
    if not null_kls:
        raise GateError(f"{name}: no null arm (v2 needs the one-ulp / order-only null of every window)")
    null_kl = float(np.mean(null_kls))
    if null_top1s and not all(0.0 <= x <= 1.0 for x in null_top1s):
        raise GateError(f"{name}: null top-1 agreement is outside [0, 1]")
    null_dis = 1.0 - float(np.mean(null_top1s)) if null_top1s else 0.0
    null_d = None
    if null_nlls:
        null_nll = np.asarray(null_nlls[0], np.float64)
        if null_nll.shape != nll_ref.shape or not np.isfinite(null_nll).all():
            raise GateError(f"{name}: invalid or unpaired null NLL rows")
        null_d = null_nll - nll_ref
    if null_kl_rows is not None:
        null_kl_rows = np.asarray(null_kl_rows, np.float64)
        if null_kl_rows.shape != kl.shape or not np.isfinite(null_kl_rows).all():
            raise GateError(f"{name}: invalid or unpaired null KL rows")
    w = {"name": name, "class": cls or "all", "rows": int(kl.size), "dnll_rows": int(d.size),
         "ref_nll": float(nll_ref.mean())}
    w.update(_judge(kl, agree, d, null_kl, null_dis, null_d, limits))
    if null_top1s:
        w["null_top1"] = 1.0 - null_dis
    if hit_ref is not None:
        w["label_is_ref_top1"] = float(np.mean(hit_ref))
        w["label_is_cand_top1"] = float(np.mean(hit_cand))
    w["checks"] = {k: v != "fail" for k, v in w["states"].items()}   # not failed (UNRESOLVED is not a pass)
    w["pass"] = w["state"] == "pass"
    w["_rows"] = {"kl": kl, "agree": agree, "dnll": d}
    if null_d is not None:
        w["_rows"]["null_dnll"] = null_d
    if null_kl_rows is not None:
        w["_rows"]["null_kl"] = null_kl_rows
    if null_nlls:   # every null's mean dNLL (the pooled null floor uses their mean, like the null KL)
        w["null_dnll_mean"] = float(np.mean([np.mean(np.asarray(x, np.float64) - nll_ref) for x in null_nlls]))
    return w


def check_held_out(windows):
    """R3: labels must not be an arm's own greedy ids (both arms agreeing with every label means no held-out text)."""
    rows = sum(w["dnll_rows"] for w in windows if "label_is_ref_top1" in w)
    if rows < MIN_LABEL_ROWS:
        return None
    for key in ("label_is_ref_top1", "label_is_cand_top1"):
        frac = sum(w[key] * w["dnll_rows"] for w in windows) / rows
        if frac >= 1.0:
            raise GateError(f"every label equals the {'reference' if 'ref' in key else 'candidate'}'s top-1: the labels are "
                            "that arm's greedy ids, not held-out text (R3)")
    return {k: sum(w[k] * w["dnll_rows"] for w in windows) / rows for k in ("label_is_ref_top1", "label_is_cand_top1")}


def read_routes(path):
    """strix-route-file/v1 -> list of (layer, ids[rows, k] uint32, weights[rows, k] float32)."""
    return _parse_routes(pathlib.Path(path).read_bytes(), path)


def _parse_routes(data, path):
    if len(data) < 16 or data[:8] != ROUTE_MAGIC or int.from_bytes(data[8:12], "little") != 1:
        raise GateError(f"{path}: not a strix-route-file/v1")
    out, at = [], 16
    while at < len(data):
        if at + 16 > len(data):
            raise GateError(f"{path}: truncated record header")
        layer, rows, k, _ = np.frombuffer(data, "<u4", 4, at).tolist()
        n = rows * k
        if k == 0 or k > 64 or at + 16 + 8 * n > len(data):
            raise GateError(f"{path}: corrupt or truncated record {len(out)}")
        ids = np.frombuffer(data, "<u4", n, at + 16).reshape(rows, k)
        w = np.frombuffer(data, "<f4", n, at + 16 + 4 * n).reshape(rows, k)
        out.append((layer, ids, w))
        at += 16 + 8 * n
    return out


def route_flips(ref_path, arm_path):
    """Ids of the reference's selected sets missing from the arm's, per layer (order-free). Same call sequence required."""
    return _route_flips(read_routes(ref_path), read_routes(arm_path), arm_path)


def _route_flips(a, b, arm_path):
    if len(a) != len(b):
        raise GateError(f"{arm_path}: {len(b)} router calls, the reference has {len(a)}")
    per = {}
    for i, ((la, ia, _), (lb, ib, _)) in enumerate(zip(a, b)):
        if la != lb or ia.shape != ib.shape:
            raise GateError(f"{arm_path}: call {i} is layer {lb} shape {ib.shape}, the reference has layer {la} shape {ia.shape}")
        k = ia.shape[1]
        missing = k - np.array([np.intersect1d(x, y).size for x, y in zip(ia, ib)], dtype=np.int64)
        s = per.setdefault(int(la), [0, 0])
        s[0] += ia.size
        s[1] += int(missing.sum())
    slots, ids = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
    return {"slots": slots, "ids": ids, "rate": ids / slots if slots else 0.0, "per_layer": per}


def flip_ratio(cand, null):
    if null["rate"] == 0.0:
        return 1.0 if cand["rate"] == 0.0 else float("inf")
    return cand["rate"] / null["rate"]


def route_receipt(stderr_bytes, what):
    found = RECEIPT.findall(stderr_bytes)
    if len(found) != 1:
        raise GateError(f"{what}: expected one strix-route-file/v1 receipt line, found {len(found)}")
    return json.loads(found[0])


def check_replay(ref_receipt, arm_receipt, ref_file, what, null=False):
    """R1 bookkeeping: each arm replayed every route and the null exercised its declared control."""
    out = ref_receipt.get("out")
    if not out or "in" in ref_receipt or "null" in ref_receipt:
        raise GateError(f"{what}: the reference must record its routes free-running (out only)")
    rin = arm_receipt.get("in")
    if not rin:
        raise GateError(f"{what}: the arm did not replay the reference's routes")
    if pathlib.PurePath(rin["path"]).name != pathlib.PurePath(ref_file).name:
        raise GateError(f"{what}: replayed {rin['path']}, not the reference file {ref_file}")
    if not rin["complete"] or rin["consumed"] != out["records"] or rin["records"] != out["records"]:
        raise GateError(f"{what}: replay consumed {rin['consumed']} of {rin['records']} records, the reference wrote {out['records']}")
    evidence = arm_receipt.get("null", {})
    if null and not (evidence.get("nudged") or (evidence.get("order") and evidence.get("applications", 0) > 0)):
        raise GateError(f"{what}: no applied null control in the route receipt")
    return {"records": rin["consumed"], "rows": rin["rows"], "local_flips": arm_receipt.get("local_flips"),
            "null": arm_receipt.get("null")}


def verdict(windows, flips=None, limits=LIMITS, allow_null_floor=False, r4_frozen=None):
    """Pooled + per-window verdict. windows: window_stats dicts with their per-row arrays under '_rows'.
    r4_frozen: verified frozen R4 record (load_r4_frozen) or None (R4 bar from these windows, as before)."""
    if len({w["name"] for w in windows}) != len(windows):
        raise GateError("duplicate window in pooled evidence")
    if r4_frozen is not None:
        rec, expected = r4_frozen
        _r4_inventory((w["name"] for w in windows), expected)
        for w in windows:
            slot, ref = expected[w["name"]]
            if w["rows"] != slot["rows"] or w["dnll_rows"] != slot["rows"]:
                raise GateError(f"{w['name']}: R4 scored row count differs")
            _r4_check_reference(w["name"], {**w["r4_reference"], "ref_nll": w["ref_nll"]}, ref)
            w["r4_content_class"] = slot["class"]
            w["class"] = slot["band"] if rec["plan"]["partition"] == "band" else slot["class"]
    held = check_held_out(windows)
    kl = np.concatenate([w["_rows"]["kl"] for w in windows])
    agree = np.concatenate([w["_rows"]["agree"] for w in windows])
    d = np.concatenate([w["_rows"]["dnll"] for w in windows])
    dnll, se = mean_se(d)
    pooled = {"rows": int(kl.size), "dnll_rows": int(d.size), "kl": float(kl.mean()), "top1": float(agree.mean()),
              "dnll": dnll, "dnll_se": se, "held_out": held}
    if d.size < limits["min_rows"]:
        raise GateError(f"{d.size} scored dNLL rows pooled, v2 needs at least {limits['min_rows']} (se {se:.4f})")
    floor = null_floor(windows, pooled, limits)
    pooled["null_floor"] = floor
    floor["enabled"] = bool(allow_null_floor)
    checks = {b: floor[b]["pass"] if allow_null_floor else floor[b]["absolute_pass"] for b in FLOOR_BARS}
    checks["windows"] = all(w["state"] != "fail" for w in windows)
    pooled["classes"] = class_verdicts(windows, limits)
    final_pass = {c["name"] for c in pooled["classes"] if c["final"] and c["state"] == "pass"}
    for w in windows:
        if w["state"] == "unresolved":
            w["cleared_by_class"] = w["class"] in final_pass
    pooled["off_distribution"] = off_distribution(windows, limits, rec if r4_frozen is not None else None)
    pooled["reference_suspect"] = [w["name"] for w in windows if w["reference_suspect"]]
    pooled["unresolved"] = sorted({w["class"] for w in windows if w["state"] == "unresolved" and not w["cleared_by_class"]} |
                                  {c["name"] for c in pooled["classes"] if c["state"] == "unresolved" or not c["final"]})
    if r4_frozen is not None:
        pooled["r4_frozen"] = _r4_receipt(windows, rec)
        sparse = [c["name"] for c in pooled["r4_frozen"]["classes"] if c["bar"] is None]
        if sparse:
            pooled["unresolved"] = sorted(set(pooled["unresolved"]) | set(sparse))
    if flips is not None:
        cand = {"slots": sum(f["cand"]["slots"] for f in flips), "ids": sum(f["cand"]["ids"] for f in flips)}
        null = {"slots": sum(f["null"]["slots"] for f in flips), "ids": sum(f["null"]["ids"] for f in flips)}
        for x in (cand, null):
            x["rate"] = x["ids"] / x["slots"] if x["slots"] else 0.0
        pooled["flips"] = {"cand": cand, "null": null, "ratio": flip_ratio(cand, null)}
        checks["flip_ratio"] = pooled["flips"]["ratio"] <= limits["max_flip_ratio"]
    checks["classes"] = all(c["state"] != "fail" for c in pooled["classes"])
    pooled["checks"] = checks
    if not all(checks.values()):
        pooled["state"] = "FAIL"
    elif pooled["reference_suspect"] or pooled["off_distribution"]:
        pooled["state"] = "HOLD"     # never PASS: the exact path is examined on those windows first
    elif pooled["unresolved"]:
        pooled["state"] = "UNRESOLVED"   # provisional: the lever stays opt-in for these classes
    else:
        pooled["state"] = "PASS"
    pooled["pass"] = pooled["state"] == "PASS"
    return pooled


FLOOR_BARS = ("kl", "top1", "dnll")


def null_floor(windows, pooled, limits=LIMITS):
    """Null-floor rule for the POOLED bars, judged against the pooled null of the same windows (row-weighted).

    A bar is engaged when the pooled null fails it or lies within floor_z of its own row se of failing it (the null is
    not measurably inside the bar). An engaged bar only relaxes: KL <= max(0.02, null + 0.02), top-1 >= min(0.93,
    null - 0.01), dNLL <= max(0.01, null + 0.01). Without the null's evidence for a bar (no per-window null top-1 / null
    NLL rows) that bar stays absolute. The pooled null is the mean of the null arms; its se comes from the first null's
    rows (KL without per-row null KL: se 0)."""
    rows = np.array([w["rows"] for w in windows], np.float64)
    z = limits["floor_z"]
    out = {"z": z}
    nk = float(np.dot(rows, [w["null_kl"] for w in windows]) / rows.sum())
    kr = [w["_rows"].get("null_kl") for w in windows]
    nk_se = mean_se(np.concatenate(kr))[1] if all(x is not None for x in kr) else None
    nt = nt_se = None
    if all("null_top1" in w for w in windows):
        nt = float(np.dot(rows, [w["null_top1"] for w in windows]) / rows.sum())
        nt_se = math.sqrt(nt * (1.0 - nt) / rows.sum())
    nd = nd_se = None
    if all("null_dnll" in w["_rows"] for w in windows):
        nd_se = mean_se(np.concatenate([w["_rows"]["null_dnll"] for w in windows]))[1]
        drows = np.array([w["dnll_rows"] for w in windows], np.float64)
        nd = float(np.dot(drows, [w["null_dnll_mean"] for w in windows]) / drows.sum())
    for b, value, absl, null, se, margin, sign in (
            ("kl", pooled["kl"], limits["max_kl"], nk, nk_se, limits["floor_kl_margin"], 1.0),
            ("top1", pooled["top1"], limits["min_top1"], nt, nt_se, limits["floor_top1_margin"], -1.0),
            ("dnll", pooled["dnll"], limits["max_dnll"], nd, nd_se, limits["floor_dnll_margin"], 1.0)):
        # sign +1: lower is better (value <= limit); -1: higher is better (value >= limit)
        engaged = null is not None and sign * (null + sign * z * (se or 0.0) - absl) > 0.0
        limit = absl
        if engaged:
            limit = max(absl, null + margin) if sign > 0 else min(absl, null - margin)
        out[b] = {"value": value, "absolute": absl, "null": null, "null_se": se, "margin": margin, "engaged": engaged,
                  "limit": limit, "relaxed": limit != absl, "pass": bool(sign * (limit - value) >= 0.0),
                  "absolute_pass": bool(sign * (absl - value) >= 0.0)}
    out["applied"] = [b for b in FLOOR_BARS if out[b]["relaxed"]]
    out["decisive"] = [b for b in out["applied"] if out[b]["pass"] and not out[b]["absolute_pass"]]
    return out


def floor_line(p):
    """The separate NULL-FLOOR verdict line: every pooled bar, its pooled null, engaged or not, the limit used."""
    f = p["null_floor"]
    parts = []
    for b in FLOOR_BARS:
        x = f[b]
        nul = "null n/a" if x["null"] is None else (
            f"null {x['null']:.5f}" + (f" se {x['null_se']:.5f}" if x["null_se"] is not None else " se n/a"))
        op = ">=" if b == "top1" else "<="
        lim = (f"limit {x['limit']:.5f} = null {'-' if b == 'top1' else '+'} {x['margin']:g}" if x["relaxed"] else
               f"limit {x['absolute']:g} absolute{' (engaged, relax-only)' if x['engaged'] else ''}")
        parts.append(f"{b} {x['value']:.5f} {op} {lim}, {nul} -> {'pass' if x['pass'] else 'FAIL'}")
    tail = ("not applied (every pooled null measurably inside its bar)" if not f["applied"] else
            f"applied to {','.join(f['applied'])}" + (f"; these pooled bars pass ONLY under the floor: {','.join(f['decisive'])}"
                                                      if f["decisive"] else ""))
    policy = "enabled by explicit opt-in" if f["enabled"] else "ADVISORY ONLY; absolute pooled limits govern PASS/FAIL"
    return f"NULL-FLOOR ({policy}; pooled null of the same windows, z {f['z']:g}): " + "; ".join(parts) + f" -> {tail}"


def class_verdicts(windows, limits=LIMITS):
    """The windows of each declared content class pooled and judged by the same rule (final on >= min_rows rows)."""
    groups = {}
    for w in windows:
        groups.setdefault(w["class"], []).append(w)
    out = []
    for name, ws in groups.items():
        cat = lambda k: np.concatenate([w["_rows"][k] for w in ws])  # noqa: E731
        rows = np.array([w["dnll_rows"] for w in ws], np.float64)
        null_kl = float(np.dot(rows, [w["null_kl"] for w in ws]) / rows.sum())
        null_dis = (float(np.dot(rows, [1.0 - w["null_top1"] for w in ws]) / rows.sum())
                    if all("null_top1" in w for w in ws) else 0.0)
        null_d = cat("null_dnll") if all("null_dnll" in w["_rows"] for w in ws) else None
        c = {"name": name, "windows": [w["name"] for w in ws], "rows": int(rows.sum())}
        c.update(_judge(cat("kl"), cat("agree"), cat("dnll"), null_kl, null_dis, null_d, limits))
        c["final"] = c["rows"] >= limits["min_rows"]
        out.append(c)
    return out


def _r4_bar(nlls, limits=LIMITS):
    """R4 off-distribution statistics of a set of reference NLLs: (median, max(factor x median, median + margin))."""
    med = float(np.median([float(x) for x in nlls]))
    return med, max(limits["ref_nll_factor"] * med, med + limits["ref_nll_margin"])


R4_PLAN_SCHEMA = "strix-gate-r4-plan/v2"
R4_SCHEMA = "strix-gate-r4-frozen/v3"
R4_LEGACY_PLAN = "strix-gate-r4-plan/v1"
R4_LEGACY_RECORD = "strix-gate-r4-frozen/v2"
R4_INTAKE_SCHEMA = "strix-gate-r4-intake/v1"
R4_RUNS_SCHEMA = "strix-gate-r4-runs/v1"
R4_REF_RUNS_SCHEMA = "strix-gate-r4-ref-runs/v1"
R4_CLASS_MIN = 3
R4_SUBSTITUTE_DIVISOR = 10
R4_NLL_TOL = 1e-9


def _r4_digest(doc):
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()


def _r4_json(path):
    def unique(pairs):
        doc = {}
        for key, value in pairs:
            if key in doc:
                raise GateError(f"{path}: duplicate JSON key {key}")
            doc[key] = value
        return doc

    return json.loads(pathlib.Path(path).read_bytes(), object_pairs_hook=unique)


def _r4_fields(doc, fields, what):
    if not isinstance(doc, dict) or set(doc) != set(fields.split()):
        raise GateError(f"{what}: expected fields {fields}")


def _r4_text(value):
    return isinstance(value, str) and bool(value.strip())


def _r4_name(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", value))


def _r4_sha(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _r4_timestamp(value):
    return type(value) is int and value > 0


def _r4_pinned(path, pin, what):
    if not path or not _r4_sha(pin):
        raise GateError(f"R4 requires independently pinned {what}")
    doc = _r4_json(path)
    if _r4_digest(doc) != pin:
        raise GateError(f"R4 {what} differs from the independently pinned digest")
    return doc


def _r4_identity(identity):
    fields = "model checkpoint_sha256 tokenizer_sha256 tier profile device_class regime"
    if isinstance(identity, dict) and "build_sha256" in identity:
        fields += " build_sha256"
        if not _r4_sha(identity["build_sha256"]):
            raise GateError("R4 reference identity: invalid build digest")
    _r4_fields(identity, fields, "R4 reference identity")
    if (identity["tier"] not in ("exact", "base") or
            not all(_r4_sha(identity[k]) for k in ("checkpoint_sha256", "tokenizer_sha256")) or
            not all(_r4_text(identity[k]) for k in ("model", "profile", "device_class", "regime"))):
        raise GateError("R4 reference identity: invalid model/checkpoint, tokenizer, tier/profile, device class or regime")
    return identity


def _r4_slot(slot, fmt, name, routes=None):
    fields = "class band tokens ids_sha256 rows labels_sha256"
    if routes is not None and fmt == "steps":
        fields += " first_position"
    replay = routes is not None and routes["mode"] == "replay"
    if replay:
        fields += " route_rows"
    _r4_fields(slot, fields, f"R4 slot {name}")
    if (not all(_r4_text(slot[k]) for k in ("class", "band")) or
            not all(type(slot[k]) is int and slot[k] > 0 for k in ("tokens", "rows")) or
            not all(_r4_sha(slot[k]) for k in ("ids_sha256", "labels_sha256")) or
            ((fmt == "head" or routes is not None) and slot["rows"] >= slot["tokens"])):
        raise GateError(f"R4 slot {name}: invalid class, band, token length, scored rows or hash")
    if "first_position" in slot and (type(slot["first_position"]) is not int or slot["first_position"] < 0):
        raise GateError(f"R4 slot {name}: invalid first scored position")
    if replay:
        _r4_fields(slot["route_rows"], " ".join(map(str, routes["layers"])), f"R4 slot {name} route rows")
        if any(type(n) is not int or n <= 0 for n in slot["route_rows"].values()):
            raise GateError(f"R4 slot {name}: invalid router row count")


def _r4_cell(slot):
    return slot["class"], slot["band"]


def _r4_strict(plan):
    return plan["schema"] == R4_PLAN_SCHEMA


def _r4_plan(plan):
    fields = "schema set_id reference format partition originals reserves extensions"
    strict = isinstance(plan, dict) and plan.get("schema") == R4_PLAN_SCHEMA
    if strict:
        fields += " vocab registered_at_ns routes"
    _r4_fields(plan, fields, "R4 plan")
    fmt = plan["format"]
    if (plan["schema"] not in (R4_PLAN_SCHEMA, R4_LEGACY_PLAN) or not _r4_text(plan["set_id"]) or
            fmt not in ("head", "steps", "spool") or
            plan["partition"] not in ("content", "band")):
        raise GateError("R4 plan: invalid schema, set ID, format or partition")
    _r4_identity(plan["reference"])
    routes = None
    if strict:
        if ("build_sha256" not in plan["reference"] or type(plan["vocab"]) is not int or
                plan["vocab"] < 2 or not _r4_timestamp(plan["registered_at_ns"])):
            raise GateError("R4 plan: reference build, positive registration time and output vocabulary are required")
        routes = plan["routes"]
        _r4_fields(routes, "mode layers top_k", "R4 route policy")
        mode, layers, top_k = (routes[k] for k in ("mode", "layers", "top_k"))
        if (mode not in ("none", "trace", "replay") or not isinstance(layers, list) or
                any(type(i) is not int or i < 0 for i in layers) or layers != sorted(set(layers)) or
                type(top_k) is not int or
                (mode == "none" and (layers or top_k != 0)) or
                (mode != "none" and (not layers or not 1 <= top_k <= 64))):
            raise GateError("R4 plan: invalid expected route mode, layers or top-k")
    originals, reserves, extensions = (plan[k] for k in ("originals", "reserves", "extensions"))
    if not all(isinstance(x, dict) for x in (originals, reserves, extensions)) or len(originals) < R4_CLASS_MIN:
        raise GateError("R4 plan: expected at least three originals and maps for reserves and extensions")
    if (set(reserves) | set(extensions)) - set(originals):
        raise GateError("R4 plan: reserve or extension names an unknown original")
    seen = set(originals)
    prompt_hashes = []
    for name, slot in originals.items():
        if not _r4_name(name):
            raise GateError(f"R4 plan: invalid window name {name!r}")
        _r4_slot(slot, fmt, name, routes)
        prompt_hashes.append(slot["ids_sha256"])
    for original, options in reserves.items():
        if not isinstance(options, list) or not options:
            raise GateError(f"R4 plan: {original} needs an ordered list of reserves")
        for option in options:
            _r4_fields(option, "case slot", f"R4 reserve for {original}")
            case, slot = option["case"], option["slot"]
            if not _r4_name(case) or case in seen:
                raise GateError(f"R4 plan: invalid or repeated reserve {case!r}")
            _r4_slot(slot, fmt, case, routes)
            if _r4_cell(slot) != _r4_cell(originals[original]):
                raise GateError(f"R4 plan: reserve {case} differs in class or band from {original}")
            seen.add(case)
            prompt_hashes.append(slot["ids_sha256"])
    if strict and len(set(prompt_hashes)) != len(prompt_hashes):
        raise GateError("R4 plan: prompts cannot be reused across originals and reserves")
    for name, extension in extensions.items():
        _r4_fields(extension, "rows labels_sha256 reason" if strict else "rows labels_sha256", f"R4 extension {name}")
        if strict and not _r4_text(extension["reason"]):
            raise GateError(f"R4 extension {name}: a preregistered reason is required")
        slot = {**originals[name], "rows": extension["rows"], "labels_sha256": extension["labels_sha256"]}
        _r4_slot(slot, fmt, name, routes)
        if fmt != "head" or slot["rows"] <= originals[name]["rows"]:
            raise GateError(f"R4 extension {name}: expected a longer, nested head slice")
    return plan


def _r4_nll(nll, what):
    if type(nll) not in (int, float) or not 0 <= nll <= sys.float_info.max:
        raise GateError(f"{what}: reference NLL must be finite and nonnegative")
    return nll


def _r4_stats(plan, originals, limits):
    if set(originals) != set(plan["originals"]):
        raise GateError("R4 record: original reference evidence differs from the plan")
    groups = {}
    for name, evidence in originals.items():
        _r4_evidence(evidence, name, plan)
        groups.setdefault(plan["originals"][name]["class"], []).append(evidence["ref_nll"])

    def stats(nlls):
        median, bar = _r4_bar(nlls, limits) if len(nlls) >= R4_CLASS_MIN else (None, None)
        if bar is not None and not math.isfinite(bar):
            raise GateError("R4 record: derived bar must be finite")
        return {"n": len(nlls), "median": median, "bar": bar}

    return {"pooled": stats([w["ref_nll"] for w in originals.values()]),
            "classes": {cls: stats(nlls) for cls, nlls in sorted(groups.items())}}


def _r4_evidence(evidence, name, plan):
    fields = "ref_nll ref_sha256 routes_sha256"
    _r4_fields(evidence, fields + (" stderr_sha256" if _r4_strict(plan) else ""), f"R4 evidence {name}")
    _r4_nll(evidence["ref_nll"], name)
    if not _r4_sha(evidence["ref_sha256"]) or (evidence["routes_sha256"] is not None and
                                             not _r4_sha(evidence["routes_sha256"])):
        raise GateError(f"R4 evidence {name}: invalid reference artifact hash")
    if _r4_strict(plan):
        stderr = evidence["stderr_sha256"]
        if plan["routes"]["mode"] == "replay" and not _r4_sha(stderr):
            raise GateError(f"R4 evidence {name}: reference route receipt digest required")
        if plan["routes"]["mode"] != "replay" and (stderr is not None or evidence["routes_sha256"] is not None):
            raise GateError(f"R4 evidence {name}: unexpected reference route evidence")
        if plan["routes"]["mode"] == "replay" and evidence["routes_sha256"] is None:
            raise GateError(f"R4 evidence {name}: reference routes digest required")


def _r4_alternative(rec, replaced, by):
    plan = rec["plan"]
    original = plan["originals"].get(replaced)
    if original is None:
        raise GateError(f"R4 substitution: {replaced} must be an original slot")
    if by == replaced:
        extension = plan["extensions"].get(replaced)
        if extension is None:
            raise GateError(f"R4 substitution: {replaced} has no preregistered row extension")
        slot = {**original, "rows": extension["rows"], "labels_sha256": extension["labels_sha256"]}
        return slot, {"kind": "preregistered-extension", "original_rows": original["rows"]}
    reserves = plan["reserves"].get(replaced)
    if not reserves or reserves[0]["case"] != by:
        raise GateError(f"R4 substitution: {by} is not the first preregistered reserve for {replaced}")
    # Retrospective receipts can reproduce the historical, pooled reference-only trigger.
    stats = rec["stats"]
    ref_nll = rec["originals"][replaced]["ref_nll"]
    bar = stats["pooled"]["bar"] if rec["retrospective"] else stats["classes"][original["class"]]["bar"]
    if bar is None or ref_nll <= bar:
        raise GateError(f"R4 substitution: {replaced} has no reference-only outlier trigger")
    return reserves[0]["slot"], {"kind": "off-distribution", "original_nll": ref_nll, "bar": bar}


def _r4_expected(rec):
    """Verify the complete ledger and return the exact effective window inventory."""
    ledger = rec["substitutions"]
    plan = rec["plan"]
    expected = {name: (slot, rec["originals"][name]) for name, slot in plan["originals"].items()}
    if not isinstance(ledger, list) or len(ledger) > len(expected) // R4_SUBSTITUTE_DIVISOR:
        raise GateError("R4 substitution cap: at most 10% of original windows")
    cells, replaced_slots = set(), set()
    strict = _r4_strict(plan)
    run_ids = set()
    if strict and rec["reference_runs"] is not None:
        _r4_capture_record(rec["reference_runs"], run_ids)
    if strict and not rec["retrospective"] and not run_ids:
        raise GateError("R4 record requires independent original REF and REF repeat receipts")
    for index, entry in enumerate(ledger):
        fields = "previous_sha256 replaced by reason approval slot reference selection"
        _r4_fields(entry, fields + (" reference_runs" if strict else ""), "R4 substitution")
        replaced, by = entry["replaced"], entry["by"]
        if not _r4_name(replaced) or not _r4_name(by) or replaced in replaced_slots:
            raise GateError("R4 substitution: invalid or repeated original slot")
        if not _r4_text(entry["reason"]) or not _r4_text(entry["approval"]):
            raise GateError("R4 substitution: reason and independent approval are required")
        prior = _r4_digest({**rec, "substitutions": ledger[:index]})
        if entry["previous_sha256"] != prior:
            raise GateError("R4 substitution: previous record digest differs")
        slot, trigger = _r4_alternative(rec, replaced, by)
        extension = trigger["kind"] == "preregistered-extension"
        if strict and extension and entry["reason"] != plan["extensions"][replaced]["reason"]:
            raise GateError("R4 substitution: extension reason differs from the preregistered reason")
        if strict:
            if entry["reference_runs"] is not None:
                _r4_capture_record(entry["reference_runs"], run_ids)
            elif not rec["retrospective"]:
                raise GateError("R4 substitution requires independent reference run receipts")
        if entry["slot"] != slot:
            raise GateError("R4 substitution: replacement hash, class, band or scored slice differs from the plan")
        cell = _r4_cell(slot)
        if cell in cells:
            raise GateError("R4 substitution cap: at most one change per class-band cell")
        cells.add(cell)
        replaced_slots.add(replaced)
        evidence = entry["reference"]
        _r4_evidence(evidence, by, plan)
        proof = {"ref_repeat_sha256": evidence["ref_sha256"], "routes_repeat_sha256": evidence["routes_sha256"]}
        if strict and extension:
            trigger["overlap_sha256"] = rec["originals"][replaced]["ref_sha256"]
        historical_extension = rec["retrospective"] and extension
        selection = entry["selection"]
        if selection != {**trigger, **proof} and not (historical_extension and
                selection == {**trigger, "ref_repeat_sha256": None, "routes_repeat_sha256": None}):
            raise GateError("R4 substitution: missing or inconsistent reference-only selection evidence")
        del expected[replaced]
        expected[by] = slot, evidence
    return expected


def load_r4_frozen(path, pin, limits=LIMITS):
    """Verify against the independently recorded digest, including the complete substitution ledger."""
    doc = _r4_json(path)
    strict = isinstance(doc, dict) and doc.get("schema") == R4_SCHEMA
    fields = "schema plan_sha256 plan retrospective originals stats substitutions record_sha256"
    _r4_fields(doc, fields + (" reference_runs" if strict else ""), "R4 record")
    rec = {k: v for k, v in doc.items() if k != "record_sha256"}
    if not _r4_sha(pin):
        raise GateError("R4 requires the independently pinned record digest (--r4-sha256)")
    if doc["schema"] not in (R4_SCHEMA, R4_LEGACY_RECORD) or type(rec["retrospective"]) is not bool:
        raise GateError(f"{path}: invalid R4 record schema")
    if not isinstance(rec["originals"], dict):
        raise GateError("R4 record: invalid original reference evidence")
    plan = _r4_plan(rec["plan"])
    if strict != _r4_strict(plan):
        raise GateError("R4 record and plan schemas differ")
    if _r4_digest(plan) != rec["plan_sha256"]:
        raise GateError("R4 record: preregistered plan digest differs")
    if _r4_stats(plan, rec["originals"], limits) != rec["stats"]:
        raise GateError("R4 record: per-class or pooled n/median/bar differs from the original windows")
    if doc["record_sha256"] != pin or _r4_digest(rec) != pin:
        raise GateError("R4 record differs from the independently pinned record digest")
    return rec, _r4_expected(rec)


def _r4_token_sha(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def _r4_inputs(case, slot, ids, head, plan):
    if ids is None or len(ids) != slot["tokens"] or _r4_token_sha(ids) != slot["ids_sha256"]:
        raise GateError(f"{case}: R4 input content hash or token length differs")
    rows = slot["rows"]
    if _r4_strict(plan) and ((ids < 0).any() or (ids >= plan["vocab"]).any()):
        raise GateError(f"{case}: R4 prompt ID outside the pinned vocabulary")
    if head or _r4_strict(plan):
        labels = ids[1:rows + 1] if plan["format"] == "spool" else ids[-rows:]
        if _r4_token_sha(labels) != slot["labels_sha256"]:
            raise GateError(f"{case}: R4 scored labels or row selection differs")
    return {"ids": ids, "rows": rows + 1} if head else {}


def _r4_artifact(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _r4_dump(path, plan, slot, ids):
    """Validate a dump against the plan, returning the digest of the bytes validated."""
    fmt, vocab, rows = plan["format"], plan["vocab"], slot["rows"]
    if fmt == "head":
        if path.stat().st_size != (rows + 1) * vocab * 4:
            raise GateError(f"{path}: R4 head shape differs from the pinned rows and vocabulary")
        return _r4_artifact(path)
    from tools.qualification.option_qual.records import SPOOL_MAGIC, reader

    digest = hashlib.sha256()
    if fmt == "spool":
        header = SPOOL_MAGIC + struct.pack("<QQQ", vocab, rows, len(ids)) + ids.astype("<u4").tobytes()

        class SpoolHash:
            def update(self, block):
                at = self.offset
                end = min(at + len(block), len(header))
                if at < end and block[:end - at] != header[at:end]:
                    raise GateError(f"{path}: R4 spool prompt IDs or shape differ from the pinned plan")
                self.offset += len(block)
                digest.update(block)

            offset = 0

        hasher = SpoolHash()
        first, labels = 0, ids[1:rows + 1]
    else:
        hasher = digest
        first, labels = slot["first_position"], ids[-rows:]
    count = 0
    for i, row in enumerate(reader(fmt)(path, hasher)):
        if (i >= rows or row.step != i or row.position != first + i or
                row.label != labels[i] or row.logits.size != vocab):
            raise GateError(f"{path}: R4 row {i} differs from the pinned position, label or vocabulary")
        count += 1
    if count != rows:
        raise GateError(f"{path}: R4 scored row count differs")
    return digest.hexdigest()


def _r4_overlap(path, slot, original, frozen, info):
    """Hash the nested original selection, including the final unscored row."""
    skip = (slot["rows"] - original["rows"]) * info["vocab"] * 4
    whole, overlap = hashlib.sha256(), hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            whole.update(block)
            overlap.update(block[skip:])
            skip = max(0, skip - len(block))
    if whole.hexdigest() != info["ref_sha256"] or overlap.hexdigest() != frozen["ref_sha256"]:
        raise GateError(f"{path}: R4 extension changes the frozen reference overlap")
    return overlap.hexdigest()


def _r4_checked_bytes(path, pin):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != pin:
        raise GateError(f"{path}: R4 run artifact hash differs")
    return data


def _r4_capture_record(capture, seen):
    _r4_fields(capture, "sha256 run_ids finished_at_ns", "R4 reference run evidence")
    ids = capture["run_ids"]
    if (not _r4_sha(capture["sha256"]) or not _r4_timestamp(capture["finished_at_ns"]) or
            not isinstance(ids, list) or len(ids) != 2 or not all(_r4_name(r) for r in ids) or
            len(set(ids)) != 2 or seen.intersection(ids)):
        raise GateError("R4 reference requires distinct independent run identities")
    seen.update(ids)


def _r4_run_arms(doc, roles, slots, plan, after, seen):
    """Validate the role and per-window assertions of a pinned executor receipt."""
    arms = doc["arms"]
    _r4_fields(arms, " ".join(roles), "R4 run receipt arms")
    mode = plan["routes"]["mode"]
    for arm, (role, identity) in roles.items():
        run = arms[arm]
        _r4_fields(run, "role identity run_id started_at_ns finished_at_ns windows", f"R4 run {arm}")
        run_id, start, finish = (run[k] for k in ("run_id", "started_at_ns", "finished_at_ns"))
        if run["role"] != role or run["identity"] != identity:
            raise GateError(f"{arm}: R4 run role or numerical identity differs")
        if not _r4_name(run_id) or run_id in seen:
            raise GateError(f"{arm}: R4 requires distinct independent run identities")
        seen.add(run_id)
        if not _r4_timestamp(start) or not _r4_timestamp(finish) or not after < start <= finish:
            raise GateError(f"{arm}: R4 run chronology differs from the approved intake or plan")
        _r4_fields(run["windows"], " ".join(slots), f"R4 run {arm} window inventory")
        for case, window in run["windows"].items():
            _r4_fields(window, "slot vocab logits_sha256 routes_sha256 stderr_sha256 replay_sha256", f"R4 run {arm}/{case}")
            if (_r4_digest(window["slot"]) != _r4_digest(slots[case]) or type(window["vocab"]) is not int or
                    window["vocab"] != plan["vocab"]):
                raise GateError(f"{arm}/{case}: R4 run window inventory or vocabulary differs")
            route_file = mode == "replay" and role in ("reference", "reference-repeat", "candidate-free", "null-free")
            stderr = mode == "replay" or (mode == "trace" and role in ("candidate", "null"))
            replay = mode == "replay" and role in ("candidate", "null")
            for field, required in (("logits_sha256", True), ("routes_sha256", route_file),
                                    ("stderr_sha256", stderr), ("replay_sha256", replay)):
                if not (_r4_sha(window[field]) if required else window[field] is None):
                    raise GateError(f"{arm}/{case}: R4 run requires the declared {field} evidence")
    return arms


def _r4_distinct(paths, seen):
    for path in paths:
        st = path.stat()
        key = st.st_dev, st.st_ino
        if key in seen:
            raise GateError(f"{path}: R4 requires separate artifacts for each role and window")
        seen.add(key)


def _r4_capture_runs(a, plan, slots, previous):
    doc = _r4_pinned(a.r4_runs, a.r4_runs_sha256, "reference run receipts (--r4-runs and --r4-runs-sha256)")
    _r4_fields(doc, "schema plan_sha256 previous_sha256 arms", "R4 reference run receipts")
    if (doc["schema"] != R4_REF_RUNS_SCHEMA or doc["plan_sha256"] != _r4_digest(plan) or
            doc["previous_sha256"] != previous):
        raise GateError("R4 reference run receipts belong to another plan or record version")
    _r4_arm_names((a.ref, a.ref_repeat))
    roles = {a.ref: ("reference", plan["reference"]), a.ref_repeat: ("reference-repeat", plan["reference"])}
    seen = set()
    arms = _r4_run_arms(doc, roles, slots, plan, plan["registered_at_ns"], seen)
    return arms, {"sha256": a.r4_runs_sha256, "run_ids": sorted(seen),
                  "finished_at_ns": max(run["finished_at_ns"] for run in arms.values())}


def _r4_arm_names(arms):
    if not all(_r4_name(arm) for arm in arms) or len(set(arms)) != len(arms):
        raise GateError("R4 requires distinct arm names for separate roles")


def _r4_qualification(a, rec, expected):
    plan = rec["plan"]
    intake = _r4_pinned(a.r4_intake, a.r4_intake_sha256, "intake (--r4-intake and --r4-intake-sha256)")
    _r4_fields(intake, "schema qualification_id plan_sha256 record_sha256 approval approved_at_ns "
               "reuse_approval candidate tier allow_null_floor null_order null_controls", "R4 intake")
    if (intake["schema"] != R4_INTAKE_SCHEMA or intake["plan_sha256"] != rec["plan_sha256"] or
            intake["record_sha256"] != a.r4_sha256):
        raise GateError("R4 intake authorizes a different plan or final record")
    qid = intake["qualification_id"]
    reuse = intake["reuse_approval"]
    if (not _r4_text(qid) or not _r4_text(intake["approval"]) or
            not (reuse is None if qid == plan["set_id"] else _r4_text(reuse))):
        raise GateError("R4 intake requires approval and explicit authorization for reference reuse")
    if (intake["tier"] != a.tier or intake["allow_null_floor"] is not a.allow_null_floor or
            intake["null_order"] != a.null):
        raise GateError("R4 requested scoring policy differs from the intake")
    candidate = intake["candidate"]
    _r4_fields(candidate, "lever profile build_sha256", "R4 candidate identity")
    if not _r4_text(candidate["lever"]) or not _r4_text(candidate["profile"]) or not _r4_sha(candidate["build_sha256"]):
        raise GateError("R4 intake: invalid candidate lever, profile or build digest")
    mode = plan["routes"]["mode"]
    free = (a.cand_free, a.null_free) if mode == "replay" else ()
    _r4_arm_names((a.ref, a.cand, *a.null, *free))
    if a.trace_diag or (mode != "replay" and (a.cand_free is not None or a.null_free is not None)):
        raise GateError("R4 route evidence must use the approved receipt mode")
    controls = intake["null_controls"]
    _r4_fields(controls, " ".join(a.null), "R4 intake null controls")
    if not all(_r4_text(control) for control in controls.values()):
        raise GateError("R4 intake requires an evaluation-order identity for every null")
    roles = {a.cand: ("candidate", candidate)}
    roles.update({n: ("null", {"reference": plan["reference"], "control": controls[n]}) for n in a.null})
    if mode == "replay":
        roles[a.cand_free] = "candidate-free", candidate
        roles[a.null_free] = "null-free", roles[a.null[0]][1]
    runs = _r4_pinned(a.r4_runs, a.r4_runs_sha256, "scored run receipts (--r4-runs and --r4-runs-sha256)")
    _r4_fields(runs, "schema intake_sha256 record_sha256 null_order arms", "R4 scored run receipts")
    if (runs["schema"] != R4_RUNS_SCHEMA or runs["intake_sha256"] != a.r4_intake_sha256 or
            runs["record_sha256"] != a.r4_sha256 or runs["null_order"] != a.null):
        raise GateError("R4 scored run receipts belong to another qualification, record or null order")
    captures = [rec["reference_runs"], *(e["reference_runs"] for e in rec["substitutions"])]
    approval = intake["approved_at_ns"]
    if not _r4_timestamp(approval) or approval <= max(c["finished_at_ns"] for c in captures):
        raise GateError("R4 intake must approve the final record after reference capture and before scoring")
    seen = {r for c in captures for r in c["run_ids"]}
    arms = _r4_run_arms(runs, roles, {c: slot for c, (slot, _) in expected.items()}, plan, approval, seen)
    return intake, {arm: run["windows"] for arm, run in arms.items()}


def _r4_routes(out, case, arm):
    path = out / "routes" / f"{case}.{arm}.rte"
    return _r4_artifact(path) if path.exists() else None


def _r4_route_file(path, pin, policy, slot):
    calls = _parse_routes(_r4_checked_bytes(path, pin), path)
    counts = {layer: 0 for layer in policy["layers"]}
    for layer, ids, weights in calls:
        if layer not in counts or ids.shape[1] != policy["top_k"] or not len(ids) or not np.isfinite(weights).all():
            raise GateError(f"{path}: R4 route layer, top-k or weights differ from the declared coverage")
        counts[layer] += len(ids)
    if any(n != slot["route_rows"][str(layer)] for layer, n in counts.items()):
        raise GateError(f"{path}: R4 incomplete route row coverage")
    return calls


def _r4_route_log(path, pin):
    receipt = route_receipt(_r4_checked_bytes(path, pin), path)
    if not isinstance(receipt, dict):
        raise GateError(f"{path}: invalid route receipt")
    return receipt


def _r4_free_route(receipt, calls, path):
    output = receipt.get("out")
    if (not isinstance(output, dict) or "in" in receipt or
            output.get("records") != len(calls) or output.get("rows") != sum(len(ids) for _, ids, _ in calls) or
            pathlib.PurePath(str(output.get("path"))).name != path.name):
        raise GateError(f"{path}: R4 free-running route receipt differs from the captured routes")


def _r4_null_route(receipt, control, what):
    null = receipt.get("null")
    if isinstance(null, dict):
        if null.get("order") == control and type(null.get("applications")) is int and null["applications"] > 0:
            return
        if ("order" not in null and all(type(null.get(k)) is int for k in ("layer", "seed", "nudged")) and
                null["nudged"] > 0 and control == f"route-nudge/{null['layer']}/{null['seed']}"):
            return
    raise GateError(f"{what}: R4 null route receipt does not show the approved control")


def _r4_trace(path, pin, case, arm, policy, slot):
    traces = []
    for line in _r4_checked_bytes(path, pin).decode("utf-8", errors="replace").splitlines():
        match = TRACE_DIAG.match(line)
        if match and match.groups() == (case, arm):
            traces.append(json.loads(line))
    if len(traces) != 1 or not isinstance(traces[0].get("layers"), list):
        raise GateError(f"{path}: R4 requires one complete trace for {case}.{arm}")
    layers = traces[0]["layers"]
    cols = ("l", "router_ids")
    if (any(not isinstance(layer, dict) or any(type(layer.get(k)) is not int for k in cols) for layer in layers) or
            sorted(layer["l"] for layer in layers) != policy["layers"] or
            any(not 0 <= layer["router_ids"] <= policy["top_k"] * slot["tokens"] for layer in layers)):
        raise GateError(f"{path}: R4 incomplete trace layer coverage or invalid flip counts")
    slots = slot["tokens"] * len(layers)
    ids = sum(layer["router_ids"] for layer in layers)
    return {"slots": slots, "ids": ids, "rate": ids / slots}


def _r4_case_routes(a, out, case, plan, slot, pinned, intake, artifacts, seen):
    policy = plan["routes"]
    mode = policy["mode"]
    route = lambda arm: out / "routes" / f"{case}.{arm}.rte"  # noqa: E731
    err = lambda arm: out / "log" / f"{case}.{arm}.stderr"  # noqa: E731
    allowed = {a.ref, a.cand_free, a.null_free} if mode == "replay" else set()
    for arm in {a.ref, *artifacts} - allowed:
        if route(arm).exists():
            raise GateError(f"{case}.{arm}: R4 unexpected route artifact for the approved evidence mode")
    if mode == "none":
        return None, None
    if mode == "trace":
        _r4_distinct((err(arm) for arm in (a.cand, *a.null)), seen)
        traces = {arm: _r4_trace(err(arm), artifacts[arm]["stderr_sha256"], case, arm, policy, slot)
                  for arm in (a.cand, *a.null)}
        return {"cand": traces[a.cand], "null": traces[a.null[0]]}, None
    _r4_distinct((route(arm) for arm in allowed), seen)
    _r4_distinct((err(arm) for arm in (a.ref, *artifacts)), seen)
    ref = _r4_route_file(route(a.ref), pinned["routes_sha256"], policy, slot)
    ref_log = _r4_route_log(err(a.ref), pinned["stderr_sha256"])
    _r4_free_route(ref_log, ref, route(a.ref))
    controls = intake["null_controls"]
    replay = {}
    for arm in (a.cand, *a.null):
        log = _r4_route_log(err(arm), artifacts[arm]["stderr_sha256"])
        if artifacts[arm]["replay_sha256"] != pinned["routes_sha256"]:
            raise GateError(f"{case}.{arm}: R4 replay receipt binds a different reference route artifact")
        if arm in controls:
            _r4_null_route(log, controls[arm], err(arm))
        elif "null" in log:
            raise GateError(f"{case}.{arm}: R4 candidate unexpectedly applied a null control")
        source = log.get("in")
        if (not isinstance(source, dict) or not {"path", "complete", "consumed", "records", "rows"} <= source.keys() or
                type(source["path"]) is not str or source["complete"] is not True or
                source["rows"] != ref_log["out"]["rows"] or "out" in log):
            raise GateError(f"{case}.{arm}: R4 incomplete replay route coverage")
        replay[arm] = check_replay(ref_log, log, route(a.ref), f"{case}.{arm}", null=arm in controls)
    flips = {}
    for role, arm in (("cand", a.cand_free), ("null", a.null_free)):
        calls = _r4_route_file(route(arm), artifacts[arm]["routes_sha256"], policy, slot)
        log = _r4_route_log(err(arm), artifacts[arm]["stderr_sha256"])
        _r4_free_route(log, calls, route(arm))
        if role == "null":
            _r4_null_route(log, controls[a.null[0]], err(arm))
        elif "null" in log:
            raise GateError(f"{case}.{arm}: R4 candidate unexpectedly applied a null control")
        flips[role] = _route_flips(ref, calls, route(arm))
    return flips, {"cand": replay[a.cand], "null": [replay[n] for n in a.null]}


def _r4_reference(case, slot, arrays, info, routes, plan, stderr=None):
    if info["rows"] != slot["rows"] or _r4_token_sha(arrays["label"]) != slot["labels_sha256"]:
        raise GateError(f"{case}: R4 scored labels or row count differs")
    nll = arrays["nll_r"]
    if (nll < 0).any() or not np.isfinite(nll).all():
        raise GateError(f"{case}: reference NLL must be finite and nonnegative on every row")
    if _r4_strict(plan) and info["vocab"] != plan["vocab"]:
        raise GateError(f"{case}: R4 reduced vocabulary differs from the pinned plan")
    evidence = {"ref_nll": _r4_nll(float(nll.mean()), case), "ref_sha256": info["ref_sha256"], "routes_sha256": routes}
    if _r4_strict(plan):
        evidence["stderr_sha256"] = stderr
    return evidence


def _r4_inventory(live, expected):
    live, expected = set(live), set(expected)
    missing, extra = sorted(expected - live), sorted(live - expected)
    if missing or extra:
        raise GateError(f"R4 window inventory differs: missing {missing}; unexpected {extra}")


def _r4_discover(out, pattern, arm):
    # Discover the live reference set independently of --cases and the prompt manifest.
    template = (out / pattern.format(case="\0", arm=arm)).as_posix()
    if template.count("\0") != 1:
        raise GateError("R4 logit path must contain one {case}")
    prefix, suffix = template.split("\0")
    wildcard = glob.escape(prefix) + "*" + glob.escape(suffix)
    return {path[len(prefix):len(path) - len(suffix) if suffix else None]
            for path in (pathlib.Path(p).as_posix() for p in glob.iglob(wildcard))}


def _r4_run(a, manifest, out, pattern, limits):
    rec, expected = load_r4_frozen(a.r4_frozen, getattr(a, "r4_sha256", None), limits)
    plan = rec["plan"]
    _r4_run_identity(a, plan)
    if getattr(a, "r4_regime", None) not in (None, plan["reference"]["regime"]):
        raise GateError("R4 regime differs from the pinned reference identity")
    if a.format != plan["format"]:
        raise GateError("R4 logit format differs from the pinned plan")
    if getattr(a, "classes", None) is not None:
        raise GateError("R4 class partition comes from the pinned plan; omit --classes")
    if (rec["retrospective"] or not _r4_strict(plan)) and not getattr(a, "r4_diagnostic", False):
        raise GateError("R4 retrospective or legacy record requires --r4-diagnostic")
    if manifest is None:
        raise GateError("R4 requires --manifest with the complete effective window set")
    _r4_inventory(manifest, expected)
    for arm in (a.ref, a.cand, *a.null):
        _r4_inventory(_r4_discover(out, pattern, arm), expected)
    if a.cases is not None:
        _r4_inventory(a.cases, expected)
        if len(a.cases) != len(expected):
            raise GateError("R4 --cases repeats a window")
    return rec, expected


def _r4_check_reference(case, actual, expected):
    ref = expected["ref_nll"]
    if (actual["ref_sha256"] != expected["ref_sha256"] or
            actual["routes_sha256"] != expected["routes_sha256"] or
            actual.get("stderr_sha256") != expected.get("stderr_sha256")):
        raise GateError(f"{case}: R4 reference artifact hash differs")
    if abs(actual["ref_nll"] - ref) > R4_NLL_TOL * max(1.0, ref):
        raise GateError(f"{case}: R4 reference NLL differs from the pinned value")


def _r4_receipt(windows, rec):
    plan, stats = rec["plan"], rec["stats"]
    classes = []
    live_median, live_bar = _r4_bar([w["ref_nll"] for w in windows])
    for cls, st in sorted(stats["classes"].items()):
        live = sorted((w for w in windows if w["r4_content_class"] == cls), key=lambda w: (-w["ref_nll"], w["name"]))
        classes.append({"name": cls, **st, "live": len(live), "highest": [live[0]["name"], live[0]["ref_nll"]],
                        "flagged": [w["name"] for w in live if w["off_distribution"]]})
    return {"set_id": plan["set_id"], "reference": plan["reference"], "record_sha256": _r4_digest(rec),
            "plan_sha256": rec["plan_sha256"], "retrospective": rec["retrospective"], "pooled_diagnostic": stats["pooled"],
            "live_median": live_median, "live_bar": live_bar, "classes": classes, "substitutions": rec["substitutions"]}


def _r4_lines(r):
    pooled = r["pooled_diagnostic"]
    lines = [f"R4 frozen {r['set_id']} {r['reference']['regime']}: record sha256 {r['record_sha256']}; "
             f"plan sha256 {r['plan_sha256']}",
             f"R4 pooled diagnostic: {pooled['n']} original windows median {pooled['median']:.3f} bar {pooled['bar']:.3f}; "
             f"live median {r['live_median']:.3f} bar {r['live_bar']:.3f}"]
    for c in r["classes"]:
        bar = (f"bar {c['bar']:.3f} (median {c['median']:.3f})" if c["bar"] is not None
               else f"provisional (requires {R4_CLASS_MIN} original windows)")
        lines.append(f"R4 class {c['name']}: {c['n']} original windows, {bar}; live {c['live']}, "
                     f"highest {c['highest'][0]} {c['highest'][1]:.3f}, flagged {','.join(c['flagged']) or 'none'}")
    lines.extend(f"R4 substitution: {s['replaced']} -> {s['by']} [{s['slot']['class']}, {s['slot']['band']}] "
                 f"approval {s['approval']}, {s['selection']['kind']}: {s['reason']}" for s in r["substitutions"])
    return lines


def off_distribution(windows, limits=LIMITS, frozen=None):
    """Windows whose reference NLL is far above the gate set's (control / chat-template ids, a timing prompt).
    A verified record supplies the original per-class bars."""
    if frozen is not None:
        bars = {c: st["bar"] for c, st in frozen["stats"]["classes"].items() if st["bar"] is not None}
        flagged = [w["name"] for w in windows if w["ref_nll"] > bars.get(w["r4_content_class"], math.inf)]
    elif len(windows) < 3:
        return []
    else:
        bar = _r4_bar([w["ref_nll"] for w in windows], limits)[1]
        flagged = [w["name"] for w in windows if w["ref_nll"] > bar]
    for w in windows:
        w["off_distribution"] = w["name"] in flagged
    return flagged


def window_line(w, flips=None, limits=LIMITS):
    fl = ""
    if flips:
        fl = (f"; flips free cand {flips['cand']['rate']:.3e} null {flips['null']['rate']:.3e} ratio "
              f"{flip_ratio(flips['cand'], flips['null']):.2f}")
    nt = f", null {w['null_top1']:.4f}" if "null_top1" in w else ""
    nd = (f"; null dNLL {w['null_dnll']:+.5f} se {w['null_dnll_se']:.5f}" if "null_dnll" in w else "; null dNLL n/a")
    cap = f" (rows se {w['dnll_se_rows']:.5f}, capped 1.5 x null)" if "dnll_se_rows" in w else ""
    flags = "".join(f" [{f}]" for f in ("reference_suspect", "off_distribution") if w.get(f))
    if w.get("cleared_by_class"):
        flags += f" [cleared by class {w['class']}]"
    return (f"window {w['name']} [{w['class']}]: rows {w['rows']} ref NLL {w['ref_nll']:.3f} KL {w['kl']:.5f} (limit "
            f"{w['kl_limit']:.5f} = max({limits['max_kl']:g}, 1.5 x null {w['null_kl']:.5f})) top-1 {w['top1']:.4f} "
            f"(limit "
            f"{w['top1_limit']:.4f}{nt}; pass >= {w['top1_pass']:.4f}) dNLL {w['dnll']:+.5f} se {w['dnll_se']:.5f}{cap} "
            f"(pass <= {limits['max_dnll']:+g}, fail > {w['dnll_limit']:+.5f}){nd}{fl} -> {_state_text(w)}{flags}")


def _state_text(x):
    st = x["state"]
    return st.upper() + ("" if st == "pass" else " " + ",".join(k for k, v in x["states"].items() if v == st))


def class_line(c):
    return (f"class {c['name']}: windows {len(c['windows'])} rows {c['rows']} ({'final' if c['final'] else 'provisional'}, "
            f">= {LIMITS['min_rows']}) KL {c['kl']:.5f} top-1 {c['top1']:.4f} (fail < {c['top1_limit']:.4f}) dNLL "
            f"{c['dnll']:+.5f} se {c['dnll_se']:.5f} (fail > {c['dnll_limit']:+.5f}) -> {_state_text(c)}")


def pooled_line(p, limits=LIMITS):
    fl = ""
    if "flips" in p:
        f = p["flips"]
        fl = (f"; route flips free cand {f['cand']['rate']:.3e} / null {f['null']['rate']:.3e} = {f['ratio']:.2f} "
              f"(<= {limits['max_flip_ratio']:g})")
    bad = [k for k, ok in p["checks"].items() if not ok]
    why = {"FAIL": ",".join(bad), "UNRESOLVED": "classes " + ",".join(p.get("unresolved", [])) + " (provisional: opt-in there)",
           "HOLD": "; ".join(x for x in (
               ("reference suspect " + ",".join(p.get("reference_suspect", []))) if p.get("reference_suspect") else "",
               ("off-distribution reference " + ",".join(p.get("off_distribution", []))) if p.get("off_distribution") else "")
               if x) + " (examine the exact path on those windows, then re-run)", "PASS": ""}[p["state"]]
    lines = [class_line(c) for c in p.get("classes", [])]
    lines.append(floor_line(p))
    lim = {b: p["null_floor"][b]["limit" if p["null_floor"]["enabled"] else "absolute"] for b in FLOOR_BARS}
    lines.append(f"POOLED v2 rows {p['rows']} (dNLL rows {p['dnll_rows']}, >= {LIMITS['min_rows']}): KL {p['kl']:.5f} "
                 f"(<= {lim['kl']:.5g}) top-1 {p['top1']:.4f} (>= {lim['top1']:.5g}) dNLL {p['dnll']:+.5f} (<= "
                 f"{lim['dnll']:+.5g}) se {p['dnll_se']:.5f}{fl} -> {p['state']}"
                 + (f" {why}" if why else ""))
    return "\n".join(lines)


def _option_qual_reduce():
    root = pathlib.Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tools.qualification.option_qual.reduce import reduce_pair  # noqa: E402
    return reduce_pair


def read_manifest(path):
    """Prompt manifest rows NAME TOKENS IDS_PATH [ARMS] -> {name: ids}; the IDs file is whitespace-separated (a relative
    path is next to the manifest)."""
    path, rows = pathlib.Path(path), {}
    for line in path.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if not f or f[0].startswith("#"):
            continue
        if len(f) < 3:
            raise GateError(f"{path}: manifest row {line!r} needs NAME TOKENS IDS_PATH")
        if f[0] in rows:
            raise GateError(f"{path}: case {f[0]} twice")
        ids_path = pathlib.Path(f[2]) if pathlib.Path(f[2]).is_absolute() else path.parent / f[2]
        ids = np.array(ids_path.read_text(encoding="utf-8").split(), dtype=np.int64)
        if ids.size != int(f[1]):
            raise GateError(f"{path}: {f[0]} has {ids.size} IDs, the manifest says {f[1]}")
        rows[f[0]] = ids
    return rows


TRACE_DIAG = re.compile(r'^\{"diag":1,"case":"([^"]+)","arm":"([^"]+)"')


def read_trace_diag(paths):
    """Free-running route flips from an engine's per-layer trace, for engines without a route tap: lines
    {"diag":1,"case":C,"arm":A,...,"layers":[{"l":0,...,"router_ids":I,...},...]} where router_ids counts the
    reference's selected experts missing from the arm's selection on that layer (order-free, over the traced rows).
    Returns {(case, arm): (ids, layers)}; a diag line without layers (trace off) is skipped."""
    out = {}
    for p in paths or []:
        for line in pathlib.Path(p).read_text(encoding="utf-8", errors="replace").splitlines():
            m = TRACE_DIAG.match(line)
            if not m:
                continue
            ids = [int(x) for x in re.findall(r'"router_ids":(\d+)', line)]
            if not ids:
                continue
            if m.groups() in out:
                raise GateError(f"{p}: a second trace of {m.group(1)}.{m.group(2)}")
            out[m.groups()] = (sum(ids), len(ids))
    return out


def null_moved(name, null_kls):
    """A null arm equal to the reference (KL exactly 0 on every row) measured nothing: its perturbation was absorbed
    (e.g. one f32 ulp under an FP16 input rounding). Refuse it (no verdict) instead of flooring the null-relative limit."""
    for i, k in enumerate(null_kls):
        if float(k) == 0.0:
            raise GateError(f"{name}: null arm {i} equals the reference (KL 0 on every row): the perturbation was absorbed, "
                            "not a null (re-run it with a step that survives the consumer's input rounding)")


def qual(a):
    """option_qual adapter: reduce every case's REF vs CAND / NULL logits, check replay receipts, count free flips."""
    frozen_path = getattr(a, "r4_frozen", None)
    r4_opts = ("r4_sha256", "r4_regime", "reference_identity", "r4_diagnostic",
               "r4_intake", "r4_intake_sha256", "r4_runs", "r4_runs_sha256")
    if frozen_path is None and any(getattr(a, opt, None) not in (None, False) for opt in r4_opts):
        raise GateError("frozen R4 options require --r4-frozen")
    out = pathlib.Path(a.out)
    tier = getattr(a, "tier", "fast") or "fast"
    if tier not in TIERS:
        raise GateError(f"unknown tier {tier}; tiers {sorted(TIERS)}")
    limits = TIERS[tier]
    if tier != "fast":
        print(f"TIER {tier}: KL <= {limits['max_kl']:g}, top-1 >= {limits['min_top1']:g}, dNLL <= "
              f"{limits['max_dnll']:g}, route flip ratio <= {limits['max_flip_ratio']:g} (explicit opt-in tier; never "
              f"qualifies a FAST lever)")
    reduce_pair = _option_qual_reduce()
    head = a.format == "head"
    manifest = read_manifest(a.manifest) if a.manifest else None
    classes = json.loads(pathlib.Path(a.classes).read_text(encoding="utf-8")) if getattr(a, "classes", None) else None
    if head and manifest is None:
        raise GateError("--format head needs --manifest (the prompt IDs are the held-out labels)")
    pattern = a.logits or ("{case}-{arm}.f32" if head else "logits/{case}.{arm}.bin")
    path_of = lambda case, arm: out / pattern.format(case=case, arm=arm)  # noqa: E731
    r4_frozen = _r4_run(a, manifest, out, pattern, limits) if frozen_path is not None else None
    diagnostic = r4_frozen is not None and getattr(a, "r4_diagnostic", False)
    intake, runs, seen = None, {}, set()
    plan, extensions = None, {}
    if r4_frozen is not None:
        cases = sorted(r4_frozen[1])
        rec, expected = r4_frozen
        plan = rec["plan"]
        extensions = {s["by"]: s for s in rec["substitutions"] if s["replaced"] == s["by"]}
        if not diagnostic:
            intake, runs = _r4_qualification(a, rec, expected)
            for arm in runs:
                _r4_inventory(_r4_discover(out, pattern, arm), expected)
    elif a.cases:
        cases = a.cases
    elif manifest is not None:
        cases = [c for c in manifest if path_of(c, a.ref).exists()]
    else:
        cases = sorted(p.name[:-len(f".{a.ref}.bin")] for p in (out / "logits").glob(f"*.{a.ref}.bin"))
    if not cases:
        raise GateError(f"{out} holds no {a.ref} logits")
    diag = read_trace_diag(a.trace_diag)
    if diag and manifest is None:
        raise GateError("--trace-diag needs --manifest (flip slots = prompt rows x traced layers)")
    windows, flips, replay, traced = [], [], {}, 0
    for case in cases:
        if manifest is not None and case not in manifest:
            raise GateError(f"{case}: not in the manifest")
        if r4_frozen is not None:
            slot, pinned = r4_frozen[1][case]
            kw = _r4_inputs(case, slot, manifest[case], head, plan)
        else:
            kw = {"ids": manifest[case], "rows": a.head_rows} if head else {}
        logit = lambda arm: path_of(case, arm)  # noqa: E731
        route = lambda arm: out / "routes" / f"{case}.{arm}.rte"  # noqa: E731
        err = lambda arm: (out / "log" / f"{case}.{arm}.stderr").read_bytes()  # noqa: E731
        validated, artifacts = {}, {arm: windows[case] for arm, windows in runs.items()}
        if plan is not None and _r4_strict(plan):
            checked = (a.ref, *artifacts) if intake else (a.ref, a.cand, *a.null)
            if intake:
                _r4_distinct((logit(arm) for arm in checked), seen)
            for arm in checked:
                validated[arm] = _r4_dump(logit(arm), plan, slot, manifest[case])
                if arm in artifacts and validated[arm] != artifacts[arm]["logits_sha256"]:
                    raise GateError(f"{case}.{arm}: R4 run logit artifact hash differs")
        ra, ref_info = reduce_pair(logit(a.ref), logit(a.cand), a.format, **kw)
        null_evidence = [reduce_pair(logit(a.ref), logit(n), a.format, **kw) for n in a.null]
        nulls = [rows for rows, _ in null_evidence]
        null_moved(case, [float(x["kl_rc"].mean()) for x in nulls])
        if r4_frozen is not None:
            if validated:
                for arm, info in zip((a.cand, *a.null), (ref_info, *(meta for _, meta in null_evidence))):
                    if (info["ref_sha256"] != validated[a.ref] or info["cand_sha256"] != validated[arm] or
                            info["vocab"] != plan["vocab"]):
                        raise GateError(f"{case}.{arm}: R4 logits changed during validation or reduction")
            if case in extensions:
                original = plan["originals"][case]
                _r4_inputs(case, original, manifest[case], head, plan)
                _r4_overlap(logit(a.ref), slot, original, rec["originals"][case], ref_info)
            if intake:
                # These digests are checked against immutable bytes by _r4_case_routes below.
                routes, stderr = pinned["routes_sha256"], pinned["stderr_sha256"]
            else:
                routes, stderr = _r4_routes(out, case, a.ref), None
                if pinned.get("stderr_sha256") is not None:
                    stderr = hashlib.sha256(err(a.ref)).hexdigest()
            reference = _r4_reference(case, slot, ra, ref_info, routes, plan, stderr)
            _r4_check_reference(case, reference, pinned)
            if any(info["ref_sha256"] != pinned["ref_sha256"] for _, info in null_evidence):
                raise GateError(f"{case}: R4 reference changed between candidate and null reduction")
            partition = r4_frozen[0]["plan"]["partition"]
            cls = slot["band"] if partition == "band" else slot["class"]
        else:
            cls = window_class(case, classes, manifest)
        w = window_stats(case, ra["kl_rc"], ra["agree"].astype(bool), ra["nll_r"], ra["nll_c"],
                         [float(x["kl_rc"].mean()) for x in nulls], ra["top1_r_is_label"], ra["top1_c_is_label"],
                         null_top1s=[float(x["agree"].mean()) for x in nulls], null_nlls=[x["nll_c"] for x in nulls],
                         cls=cls, null_kl_rows=nulls[0]["kl_rc"], limits=limits)
        if r4_frozen is not None:
            w["r4_reference"] = reference
        moe = route(a.ref).exists()
        f = None
        if intake:
            f, replay_case = _r4_case_routes(a, out, case, plan, slot, pinned, intake, artifacts, seen)
            moe = replay_case is not None
            if moe:
                replay[case] = w["replay"] = replay_case
            if f is not None:
                flips.append(f)
                w["flips"] = {k: {x: v[x] for x in ("slots", "ids", "rate")} for k, v in f.items()}
                traced += not moe
        elif moe:
            ref_r = route_receipt(err(a.ref), f"{case}.{a.ref}")
            replay[case] = {"cand": check_replay(ref_r, route_receipt(err(a.cand), f"{case}.{a.cand}"), route(a.ref), f"{case}.{a.cand}"),
                            "null": [check_replay(ref_r, route_receipt(err(n), f"{case}.{n}"), route(a.ref), f"{case}.{n}", null=True)
                                     for n in a.null]}
            w["replay"] = replay[case]
            if not (a.cand_free and a.null_free):
                raise GateError(f"{case}: MoE routes present: --cand-free and --null-free arms are required (R1 flip ratio)")
            f = {"cand": route_flips(route(a.ref), route(a.cand_free)), "null": route_flips(route(a.ref), route(a.null_free))}
            flips.append(f)
            w["flips"] = {k: {x: v[x] for x in ("slots", "ids", "rate")} for k, v in f.items()}
        elif diag:
            # no route tap: every arm ran free, so the flips come from the engine trace of the candidate and the null
            fc, fn = diag.get((case, a.cand_free or a.cand)), diag.get((case, a.null_free or a.null[0]))
            if (fc is None) != (fn is None):
                raise GateError(f"{case}: trace of only one of the free-running candidate / null arms (R1 needs both)")
            if fc is not None:
                if fc[1] != fn[1]:
                    raise GateError(f"{case}: candidate trace has {fc[1]} layers, the null's {fn[1]}")
                slots = int(manifest[case].size) * fc[1]
                f = {k: {"slots": slots, "ids": v[0], "rate": v[0] / slots} for k, v in (("cand", fc), ("null", fn))}
                flips.append(f)
                w["flips"] = f
                traced += 1
        windows.append(w)
        print(window_line(w, f, limits))
        for c in ([replay[case]["cand"]] + replay[case]["null"]) if moe else []:
            lf = c["local_flips"] or {}
            evidence = c["null"]
            suffix = ""
            if evidence and "order" in evidence:
                suffix = f"; null order {evidence['order']} applications {evidence['applications']}"
            elif evidence:
                suffix = f"; null layer {evidence['layer']} seed {evidence['seed']} nudged {evidence['nudged']}"
            print(f"  replay {case}: {c['records']} router calls, {c['rows']} rows replayed; local flips "
                  f"{lf.get('ids', 0)}/{lf.get('slots', 0)}{suffix}")
    r1 = "replay" if replay else (f"free-running, trace flips on {traced} of {len(windows)} windows" if traced else "none")
    if traced and not replay:
        print(f"R1: no route tap: every arm ran free, so KL / top-1 / dNLL include route flips; flips from the engine trace "
              f"on {traced} of {len(windows)} windows (slots = prompt rows x traced layers)")
    p = verdict(windows, flips if flips else None, limits=limits,
                allow_null_floor=getattr(a, 'allow_null_floor', False), r4_frozen=r4_frozen)
    if intake:
        p["r4_frozen"].update(qualification_id=intake["qualification_id"], candidate=intake["candidate"],
                              intake_sha256=a.r4_intake_sha256, runs_sha256=a.r4_runs_sha256)
        print(f"R4 qualification {intake['qualification_id']}: intake sha256 {a.r4_intake_sha256}; "
              f"scored runs sha256 {a.r4_runs_sha256}")
    for line in _r4_lines(p["r4_frozen"]) if r4_frozen is not None else []:
        print(line)
    summary = pooled_line(p, limits)
    if diagnostic:
        print("R4 DIAGNOSTIC ONLY (no qualification verdict)")
        summary = "\n".join(f"DIAGNOSTIC {line}" for line in summary.splitlines())
        p.update(diagnostic_state=p["state"], state="NO VERDICT", **{"pass": False})
    print(summary)
    if a.json:
        doc = {"schema": "strix-fast-gate-v2/v1", "limits": limits, "ref": a.ref, "cand": a.cand, "null": a.null,
               "format": a.format, "r1": r1,
               "windows": [{k: v for k, v in w.items() if k != "_rows"} for w in windows], "pooled": p}
        if tier != "fast":
            doc["tier"] = tier
        pathlib.Path(a.json).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return 2 if diagnostic else EXIT[p["state"]]


def window_class(case, classes, manifest):
    """The declared content class (--classes), else the prompt-length band with a manifest, else one class."""
    if classes:
        if case not in classes:
            raise GateError(f"{case}: --classes declares no class for this window (classes are declared before scoring)")
        return str(classes[case])
    if manifest is not None and case in manifest:
        n = int(manifest[case].size)
        for band in (4096, 16384, 65536):
            if n <= band:
                return f"prompt<={band // 1024}K"
        return "prompt>64K"
    return "all"


def _r4_run_identity(a, plan):
    identity_path = getattr(a, "reference_identity", None)
    if identity_path is None:
        raise GateError("R4 requires a reference run receipt (--reference-identity)")
    if _r4_identity(_r4_json(identity_path)) != plan["reference"]:
        raise GateError("R4 reference identity differs from the pinned plan")


def _r4_capture(a, plan, slots, previous=None, retrospective=False):
    """Measure only reference arms. The reducer hashes the actual input bytes it reads."""
    _r4_run_identity(a, plan)
    manifest = read_manifest(a.manifest)
    out, fmt = pathlib.Path(a.out), plan["format"]
    head = fmt == "head"
    pattern = a.logits or ("{case}-{arm}.f32" if head else "logits/{case}.{arm}.bin")
    reduce_pair = _option_qual_reduce()
    strict = _r4_strict(plan)
    repeat = a.ref_repeat is not None
    runs, capture, evidence, overlaps, seen = None, None, {}, {}, set()
    if strict and (not retrospective or a.r4_runs is not None or a.r4_runs_sha256 is not None):
        if not repeat:
            raise GateError("R4 requires an independent reference repeat (--ref-repeat)")
        runs, capture = _r4_capture_runs(a, plan, slots, _r4_digest(previous) if previous is not None else None)
    elif a.r4_runs is not None or a.r4_runs_sha256 is not None:
        raise GateError("R4 reference run receipts require a v2 plan")
    for case, slot in slots.items():
        ids = manifest.get(case)
        kw = _r4_inputs(case, slot, ids, head, plan)
        original = plan["originals"].get(case)
        extended = original is not None and slot["rows"] != original["rows"]
        if extended:
            _r4_inputs(case, original, ids, head, plan)
        ref = out / pattern.format(case=case, arm=a.ref)
        second = out / pattern.format(case=case, arm=a.ref_repeat) if repeat else ref
        if repeat and ref.samefile(second):
            raise GateError(f"{case}: R4 selection evidence requires a separate reference repeat")
        paths = {a.ref: ref}
        if repeat:
            paths[a.ref_repeat] = second
        validated = {}
        if strict:
            _r4_distinct(paths.values(), seen)
            for arm, path in paths.items():
                validated[arm] = _r4_dump(path, plan, slot, ids)
        arrays, info = reduce_pair(ref, second, fmt, **kw)
        if strict and (info["ref_sha256"] != validated[a.ref] or
                       info["cand_sha256"] != validated[a.ref_repeat if repeat else a.ref]):
            raise GateError(f"{case}: R4 reference bytes changed during validation or reduction")
        routes = _r4_routes(out, case, a.ref)
        if info["ref_sha256"] != info["cand_sha256"] or (repeat and routes != _r4_routes(out, case, a.ref_repeat)):
            raise GateError(f"{case}: reference repeat is not bitwise identical")
        if extended:
            overlaps[case] = _r4_overlap(ref, slot, original, previous["originals"][case], info)
        stderr = None
        if runs:
            policy = plan["routes"]
            for arm, path in paths.items():
                window = runs[arm]["windows"][case]
                if window["logits_sha256"] != validated[arm] or window["routes_sha256"] != routes:
                    raise GateError(f"{case}.{arm}: R4 reference run artifact hash differs")
                if policy["mode"] == "replay":
                    route = out / "routes" / f"{case}.{arm}.rte"
                    err = out / "log" / f"{case}.{arm}.stderr"
                    _r4_distinct((route, err), seen)
                    calls = _r4_route_file(route, routes, policy, slot)
                    log = _r4_route_log(err, window["stderr_sha256"])
                    _r4_free_route(log, calls, route)
                    if "null" in log:
                        raise GateError(f"{case}.{arm}: R4 reference unexpectedly applied a null control")
            stderr = runs[a.ref]["windows"][case]["stderr_sha256"]
        elif strict and plan["routes"]["mode"] == "replay":
            stderr = _r4_artifact(out / "log" / f"{case}.{a.ref}.stderr")
        evidence[case] = _r4_reference(case, slot, arrays, info, routes, plan, stderr)
    return evidence, capture, overlaps


def _r4_write(path, rec):
    """Publish one complete, immutable version. A concurrent publisher of this path must fail."""
    path = pathlib.Path(path)
    if path.exists():
        raise GateError(f"{path} exists: a frozen R4 version is written once")
    pin = _r4_digest(rec)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as tmp:
        temporary = pathlib.Path(tmp.name)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(dict(rec, record_sha256=pin), f, indent=1, sort_keys=True, allow_nan=False)
            f.write("\n")
        os.link(temporary, path)
    finally:
        temporary.unlink()
    print(f"R4 record sha256 {pin}")


def r4_freeze_cmd(a):
    plan = _r4_plan(_r4_json(a.r4_plan))
    if _r4_digest(plan) != a.r4_plan_sha256:
        raise GateError("R4 plan differs from the independently pinned plan digest")
    if pathlib.Path(a.record).exists():
        raise GateError(f"{a.record} exists: a frozen R4 version is written once")
    strict = _r4_strict(plan)
    if not strict and not a.retrospective:
        raise GateError("R4 legacy plans require --retrospective")
    originals, capture, _ = _r4_capture(a, plan, plan["originals"], retrospective=a.retrospective)
    rec = {"schema": R4_SCHEMA if strict else R4_LEGACY_RECORD,
           "plan_sha256": a.r4_plan_sha256, "plan": plan, "retrospective": a.retrospective,
           "originals": originals, "stats": _r4_stats(plan, originals, LIMITS), "substitutions": []}
    if strict:
        rec["reference_runs"] = capture
    _r4_expected(rec)
    _r4_write(a.record, rec)
    return 0


def r4_substitute_cmd(a):
    rec, _ = load_r4_frozen(a.record, a.r4_sha256)
    slot, trigger = _r4_alternative(rec, a.replaced, a.by)
    repeat = a.ref_repeat is not None
    extension = trigger["kind"] == "preregistered-extension"
    if not repeat and not (rec["retrospective"] and extension):
        raise GateError("R4 substitution requires an independent reference repeat (--ref-repeat)")
    plan = rec["plan"]
    strict = _r4_strict(plan)
    if strict and extension and a.reason != plan["extensions"][a.replaced]["reason"]:
        raise GateError("R4 substitution: extension reason differs from the preregistered reason")
    captured, capture, overlaps = _r4_capture(a, plan, {a.by: slot}, rec, retrospective=rec["retrospective"])
    reference = captured[a.by]
    if strict and extension:
        trigger["overlap_sha256"] = overlaps[a.by]
    entry = {"previous_sha256": a.r4_sha256, "replaced": a.replaced, "by": a.by, "reason": a.reason,
             "approval": a.approval, "slot": slot, "reference": reference,
             "selection": {**trigger, "ref_repeat_sha256": reference["ref_sha256"] if repeat else None,
                           "routes_repeat_sha256": reference["routes_sha256"] if repeat else None}}
    if strict:
        entry["reference_runs"] = capture
    rec["substitutions"].append(entry)
    _r4_expected(rec)
    _r4_write(a.new_record, rec)
    return 0


def _r4_capture_args(parser):
    parser.add_argument("--out", required=True, help="directory of the reference run")
    parser.add_argument("--manifest", required=True, help="NAME TOKENS IDS_PATH [ARMS] rows")
    parser.add_argument("--reference-identity", required=True, help="reference run identity JSON receipt")
    parser.add_argument("--ref", required=True, help="reference arm name")
    parser.add_argument("--ref-repeat", help="independently repeated reference arm name (required for qualification)")
    parser.add_argument("--logits", help="logit path under OUT, {case} {arm}")
    _r4_runs_args(parser)


def _r4_runs_args(parser):
    parser.add_argument("--r4-runs", help="independently attested run receipt bundle JSON")
    parser.add_argument("--r4-runs-sha256", help="canonical receipt bundle digest from independent intake")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("qual", help="gate option_qual runner output")
    q.add_argument("out")
    q.add_argument("--ref", required=True)
    q.add_argument("--cand", required=True)
    q.add_argument("--null", action="append", required=True)
    q.add_argument("--cand-free")
    q.add_argument("--null-free")
    q.add_argument("--cases", nargs="*")
    q.add_argument("--format", default="steps", choices=["steps", "spool", "head"])
    q.add_argument("--manifest", help="NAME TOKENS IDS_PATH [ARMS] rows (needed by --format head and --trace-diag)")
    q.add_argument("--head-rows", type=int, default=64, help="rows per head dump (--format head)")
    q.add_argument("--logits", help="logit path under OUT, {case} {arm} (default logits/{case}.{arm}.bin; head {case}-{arm}.f32)")
    q.add_argument("--trace-diag", action="append", help="engine log with per-layer trace diag lines (no route tap)")
    q.add_argument("--classes", help="JSON {case: content class} declared before scoring (default: prompt-length band "
                   "with --manifest, else one class)")
    q.add_argument("--json")
    q.add_argument("--tier", default="fast", choices=sorted(TIERS), help="acceptance limits: fast (default, "
                   "the FAST gate v2 bars) or turbo (TURBO_LIMITS, looser; explicit opt-in levers only)")
    q.add_argument("--allow-null-floor", action="store_true", help="explicitly opt in to relaxed pooled acceptance bars; default keeps absolute FAST limits")
    q.add_argument("--r4-frozen", help="frozen R4 record of the qualification set (r4-freeze): the off-distribution bar "
                   "of the ORIGINAL windows; without it the bar comes from these windows")
    q.add_argument("--r4-sha256", help="independently pinned R4 record digest from qualification intake")
    q.add_argument("--r4-regime", help="optionally assert the regime of the reference run")
    q.add_argument("--reference-identity", help="reference run identity JSON receipt for frozen R4")
    q.add_argument("--r4-diagnostic", action="store_true", help="score frozen R4 for diagnosis only (exit 2)")
    q.add_argument("--r4-intake", help="independent pre-score approval of the final record and candidate")
    q.add_argument("--r4-intake-sha256", help="canonical approval digest from independent intake")
    _r4_runs_args(q)
    pd = sub.add_parser("r4-plan-digest", help="validate and print the canonical digest of a preregistration plan")
    pd.add_argument("plan")
    fz = sub.add_parser("r4-freeze", help="write frozen R4 from the pinned original set's reference artifacts")
    fz.add_argument("record")
    fz.add_argument("--r4-plan", required=True, help="preregistered window set plan JSON")
    fz.add_argument("--r4-plan-sha256", required=True, help="independently pinned canonical plan digest")
    fz.add_argument("--retrospective", action="store_true", help="mark an archived reconstruction diagnostic-only")
    _r4_capture_args(fz)
    sb = sub.add_parser("r4-substitute", help="publish a new frozen record version with one approved change")
    sb.add_argument("record")
    sb.add_argument("new_record")
    sb.add_argument("--r4-sha256", required=True, help="independently pinned previous record digest")
    sb.add_argument("--replaced", required=True)
    sb.add_argument("--by", required=True)
    sb.add_argument("--reason", required=True)
    sb.add_argument("--approval", required=True, help="independent review receipt or decision ID")
    _r4_capture_args(sb)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "r4-plan-digest":
            print(_r4_digest(_r4_plan(_r4_json(a.plan))))
            return 0
        if a.cmd == "r4-freeze":
            return r4_freeze_cmd(a)
        if a.cmd == "r4-substitute":
            return r4_substitute_cmd(a)
        return qual(a)
    except GateError as e:
        print(f"NO VERDICT: {e}")
        return 2
    except (OSError, ValueError) as e:  # a missing / malformed arm file is a defect of the evidence, never a FAIL (exit 1)
        print(f"NO VERDICT: {type(e).__name__}: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
