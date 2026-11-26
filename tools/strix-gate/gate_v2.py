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
  R4 reference check: a window or final class where the candidate beats the exact reference by more than 3 se (and, with a null, the
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
  qual --tier turbo: the same rules with the TURBO limits (TURBO_LIMITS: KL <= 0.05, top-1 >= 0.90, dNLL <= 0.03,
      route flip ratio <= 2.0; the per-window disagreement floor follows as 1 - 0.90 = 0.10). TURBO levers trade
      quality for speed and are explicit opt-in, never a default; a TURBO verdict never qualifies a FAST lever. The tier
      is printed on its own TIER line and stored in the JSON ("tier", "limits"). Default --tier fast: output unchanged.
Exit 0 = PASS, 1 = FAIL, 2 = no verdict (bookkeeping defect, too few rows, labels not held out, missing null),
     3 = UNRESOLVED (provisional: the lever stays opt-in for the unresolved classes), 4 = HOLD (reference check).
"""
import argparse
import contextlib
import io
import json
import math
import os
import pathlib
import re
import stat
import statistics
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import Future

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


def null_row_mean(name, metric, values, shape):
    rows = np.asarray(values, np.float64)
    if rows.shape == shape:
        rows = rows[np.newaxis, :]
    if rows.ndim != 2 or not rows.shape[0] or rows.shape[1:] != shape or not np.isfinite(rows).all():
        raise GateError(f"{name}: invalid or unpaired null {metric} rows")
    return np.sort(rows, axis=0).mean(axis=0)


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
        if math.isfinite(cap) and cap < se:
            se_used, j["dnll_se_rows"] = cap, se
    j["dnll_se"] = se_used
    j["dnll_limit"] = limits["max_dnll"] + (z * se_used if math.isfinite(se_used) else 0.0)
    j["states"] = {"kl": "pass" if j["kl"] <= kl_limit else "fail",
                   "top1": "pass" if dis <= dis_pass else ("fail" if dis > dis_fail else "unresolved"),
                   "dnll": "pass" if dnll <= limits["max_dnll"] else ("fail" if dnll > j["dnll_limit"] else "unresolved")}
    s = j["states"].values()
    j["state"] = "fail" if "fail" in s else ("unresolved" if "unresolved" in s else "pass")
    # R4: the lever beats the exact reference on the true text by more than suspect_z se, and the null does not move
    # the same way (paired candidate-minus-null is also below -suspect_z se).
    sz = limits.get("suspect_z", LIMITS["suspect_z"])
    suspect = math.isfinite(se) and dnll < -sz * se
    if suspect and null_d is not None and null_d.size == d.size and d.size > 1:
        j["cand_minus_null"], cn_se = mean_se(d - null_d)
        suspect = j["cand_minus_null"] < -sz * cn_se
    j["reference_suspect"] = bool(suspect)
    return j


def window_stats(name, kl, agree, nll_ref, nll_cand, null_kls, hit_ref=None, hit_cand=None, limits=LIMITS, null_top1s=None,
                 null_nlls=None, cls=None, null_kl_rows=None, null_top1_rows=None):
    """One window: R2 KL limit relative to this window's null; top-1 and dNLL PASS / UNRESOLVED / FAIL on its rows; R4.

    null_top1s: the null arms' top-1 agreement on this window (None: the null-relative top-1 term is absent).
    null_nlls: the null arms' per-row NLL on the same labels (mean per row for the se cap and paired reference check).
    cls: the window's content class, declared before scoring (None: one class for the whole gate set).
    null_kl_rows and null_top1_rows: per-row KL and agreement of the null arms, for the pooled null-floor se."""
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
    null_kl = float(np.mean(sorted(null_kls)))
    if null_top1s and not all(0.0 <= x <= 1.0 for x in null_top1s):
        raise GateError(f"{name}: null top-1 agreement is outside [0, 1]")
    null_dis = 1.0 - float(np.mean(sorted(null_top1s))) if null_top1s else 0.0
    null_d = None
    if null_nlls:
        null_nll = null_row_mean(name, "NLL", null_nlls, nll_ref.shape)
        null_d = null_nll - nll_ref
    if null_kl_rows is not None:
        null_kl_rows = null_row_mean(name, "KL", null_kl_rows, kl.shape)
    if null_top1_rows is not None:
        null_top1_rows = null_row_mean(name, "top-1", null_top1_rows, agree.shape)
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
    if null_top1_rows is not None:
        w["_rows"]["null_top1"] = null_top1_rows
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


def read_routes(path, opener=None):
    """strix-route-file/v1 -> list of (layer, ids[rows, k] uint32, weights[rows, k] float32)."""
    with open(path, "rb", opener=opener) as f:
        data = f.read()
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


def route_flips(ref_path, arm_path, opener=None):
    """Ids of the reference's selected sets missing from the arm's, per layer (order-free). Same call sequence required."""
    a, b = read_routes(ref_path, opener), read_routes(arm_path, opener)
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


def verdict(windows, flips=None, limits=LIMITS, allow_null_floor=False):
    """Pooled + per-window verdict. windows: window_stats dicts with their per-row arrays under '_rows'."""
    if len({w["name"] for w in windows}) != len(windows):
        raise GateError("duplicate window in pooled evidence")
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
    pooled["off_distribution"] = off_distribution(windows, limits)
    pooled["reference_suspect"] = [w["name"] for w in windows if w["reference_suspect"]]
    pooled["reference_suspect_classes"] = [c["name"] for c in pooled["classes"] if c["final"] and c["reference_suspect"]]
    pooled["unresolved"] = sorted({w["class"] for w in windows if w["state"] == "unresolved" and not w["cleared_by_class"]} |
                                  {c["name"] for c in pooled["classes"] if c["state"] == "unresolved" or not c["final"]})
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
    elif pooled["reference_suspect"] or pooled["reference_suspect_classes"] or pooled["off_distribution"]:
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
    NLL rows) that bar stays absolute. The pooled null and its se use the paired per-row averages of all null arms
    (KL without per-row evidence: se 0, agreement without per-row evidence: binomial se)."""
    rows = np.array([w["rows"] for w in windows], np.float64)
    z = limits["floor_z"]
    out = {"z": z}
    nk = float(np.dot(rows, [w["null_kl"] for w in windows]) / rows.sum())
    kr = [w["_rows"].get("null_kl") for w in windows]
    nk_se = mean_se(np.concatenate(kr))[1] if all(x is not None for x in kr) else None
    nt = nt_se = None
    if all("null_top1" in w for w in windows):
        nt = float(np.dot(rows, [w["null_top1"] for w in windows]) / rows.sum())
        tr = [w["_rows"].get("null_top1") for w in windows]
        nt_se = (mean_se(np.concatenate(tr))[1] if all(x is not None for x in tr)
                 else math.sqrt(nt * (1.0 - nt) / rows.sum()))
    nd = nd_se = None
    if all("null_dnll" in w["_rows"] for w in windows):
        nd, nd_se = mean_se(np.concatenate([w["_rows"]["null_dnll"] for w in windows]))
    for b, value, absl, null, se, margin, sign in (
            ("kl", pooled["kl"], limits["max_kl"], nk, nk_se, limits["floor_kl_margin"], 1.0),
            ("top1", pooled["top1"], limits["min_top1"], nt, nt_se, limits["floor_top1_margin"], -1.0),
            ("dnll", pooled["dnll"], limits["max_dnll"], nd, nd_se, limits["floor_dnll_margin"], 1.0)):
        # sign +1: lower is better (value <= limit); -1: higher is better (value >= limit)
        engaged = null is not None and sign * (null - absl) + z * (se or 0.0) >= 0.0
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
        rows = np.array([w["rows"] for w in ws], np.float64)
        null_kl = float(np.dot(rows, [w["null_kl"] for w in ws]) / rows.sum())
        null_dis = (float(np.dot(rows, [1.0 - w["null_top1"] for w in ws]) / rows.sum())
                    if all("null_top1" in w for w in ws) else 0.0)
        null_d = cat("null_dnll") if all("null_dnll" in w["_rows"] for w in ws) else None
        c = {"name": name, "windows": [w["name"] for w in ws], "rows": sum(w["dnll_rows"] for w in ws)}
        c.update(_judge(cat("kl"), cat("agree"), cat("dnll"), null_kl, null_dis, null_d, limits))
        c["final"] = c["rows"] >= limits["min_rows"]
        out.append(c)
    return out


def off_distribution(windows, limits=LIMITS):
    """Windows whose reference NLL is far above the gate set's (control / chat-template ids, a timing prompt)."""
    if len(windows) < 3:
        return []
    med = float(np.median([w["ref_nll"] for w in windows]))
    bar = max(limits["ref_nll_factor"] * med, med + limits["ref_nll_margin"])
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
    flag = " [reference_suspect]" if c["reference_suspect"] else ""
    return (f"class {c['name']}: windows {len(c['windows'])} rows {c['rows']} ({'final' if c['final'] else 'provisional'}, "
            f">= {LIMITS['min_rows']}) KL {c['kl']:.5f} top-1 {c['top1']:.4f} (fail < {c['top1_limit']:.4f}) dNLL "
            f"{c['dnll']:+.5f} se {c['dnll_se']:.5f} (fail > {c['dnll_limit']:+.5f}) -> {_state_text(c)}{flag}")


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
               ("reference suspect class " + ",".join(p["reference_suspect_classes"])) if p.get("reference_suspect_classes") else "",
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


def read_manifest(path, opener=None):
    """Prompt manifest rows NAME TOKENS IDS_PATH [ARMS] -> {name: ids}; the IDs file is whitespace-separated (a relative
    path is next to the manifest)."""
    path, rows = pathlib.Path(path), {}
    with open(path, encoding="utf-8", opener=opener) as f:
        lines = f.read().splitlines()
    for line in lines:
        f = line.split()
        if not f or f[0].startswith("#"):
            continue
        if len(f) < 3:
            raise GateError(f"{path}: manifest row {line!r} needs NAME TOKENS IDS_PATH")
        if f[0] in rows:
            raise GateError(f"{path}: case {f[0]} twice")
        ids_path = pathlib.Path(f[2]) if pathlib.Path(f[2]).is_absolute() else path.parent / f[2]
        with open(ids_path, encoding="utf-8", opener=opener) as ids_file:
            ids = np.array(ids_file.read().split(), dtype=np.int64)
        if ids.size != int(f[1]):
            raise GateError(f"{path}: {f[0]} has {ids.size} IDs, the manifest says {f[1]}")
        rows[f[0]] = ids
    return rows


TRACE_DIAG = re.compile(r'^\{"diag":1,"case":"([^"]+)","arm":"([^"]+)"')


def read_trace_diag(paths, opener=None):
    """Free-running route flips from an engine's per-layer trace, for engines without a route tap: lines
    {"diag":1,"case":C,"arm":A,...,"layers":[{"l":0,...,"router_ids":I,...},...]} where router_ids counts the
    reference's selected experts missing from the arm's selection on that layer (order-free, over the traced rows).
    Returns {(case, arm): (ids, layers)}; a diag line without layers (trace off) is skipped."""
    out = {}
    for p in paths or []:
        with open(p, encoding="utf-8", errors="replace", opener=opener) as f:
            lines = f.read().splitlines()
        for line in lines:
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
    out = pathlib.Path(a.out)
    r4_input = getattr(a, "r4_assess", None)
    protected = set()

    def record_open(path, flags):
        fd = os.open(path, flags)
        try:
            protected.add(_r4_identity(fd))
        except OSError:
            protected.add(None)
        return fd

    opener = record_open if r4_input is not None else None

    def reduce(ref, cand, **kw):
        if opener is None:
            return reduce_pair(ref, cand, a.format, **kw)
        return _r4_reduce_pair(reduce_pair, ref, cand, a.format, opener, **kw)

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
    manifest = read_manifest(a.manifest, opener) if a.manifest else None
    classes = None
    if getattr(a, "classes", None):
        with open(a.classes, encoding="utf-8", opener=opener) as f:
            classes = json.load(f)
    if head and manifest is None:
        raise GateError("--format head needs --manifest (the prompt IDs are the held-out labels)")
    pattern = a.logits or ("{case}-{arm}.f32" if head else "logits/{case}.{arm}.bin")
    path_of = lambda case, arm: out / pattern.format(case=case, arm=arm)  # noqa: E731
    if a.cases:
        cases = a.cases
    elif manifest is not None:
        cases = [c for c in manifest if path_of(c, a.ref).exists()]
    else:
        cases = sorted(p.name[:-len(f".{a.ref}.bin")] for p in (out / "logits").glob(f"*.{a.ref}.bin"))
    if not cases:
        raise GateError(f"{out} holds no {a.ref} logits")
    diag = read_trace_diag(a.trace_diag, opener)
    if diag and manifest is None:
        raise GateError("--trace-diag needs --manifest (flip slots = prompt rows x traced layers)")
    windows, flips, replay, traced = [], [], {}, 0
    for case in cases:
        if manifest is not None and case not in manifest:
            raise GateError(f"{case}: not in the manifest")
        kw = {"ids": manifest[case], "rows": a.head_rows} if head else {}
        logit = lambda arm: path_of(case, arm)  # noqa: E731
        route = lambda arm: out / "routes" / f"{case}.{arm}.rte"  # noqa: E731
        def err(arm):
            with open(out / "log" / f"{case}.{arm}.stderr", "rb", opener=opener) as f:
                return f.read()

        ra, _ = reduce(logit(a.ref), logit(a.cand), **kw)
        nulls = [reduce(logit(a.ref), logit(n), **kw)[0] for n in a.null]
        null_moved(case, [float(x["kl_rc"].mean()) for x in nulls])
        w = window_stats(case, ra["kl_rc"], ra["agree"].astype(bool), ra["nll_r"], ra["nll_c"],
                         [float(x["kl_rc"].mean()) for x in nulls], ra["top1_r_is_label"], ra["top1_c_is_label"],
                         null_top1s=[float(x["agree"].mean()) for x in nulls], null_nlls=[x["nll_c"] for x in nulls],
                         cls=window_class(case, classes, manifest), null_kl_rows=[x["kl_rc"] for x in nulls],
                         null_top1_rows=[x["agree"] for x in nulls], limits=limits)
        ref_route = route(a.ref)
        moe = ref_route.exists()
        f = None
        if moe:
            ref_r = route_receipt(err(a.ref), f"{case}.{a.ref}")
            replay[case] = {"cand": check_replay(ref_r, route_receipt(err(a.cand), f"{case}.{a.cand}"), ref_route, f"{case}.{a.cand}"),
                            "null": [check_replay(ref_r, route_receipt(err(n), f"{case}.{n}"), ref_route, f"{case}.{n}", null=True)
                                     for n in a.null]}
            w["replay"] = replay[case]
            if not (a.cand_free and a.null_free):
                raise GateError(f"{case}: MoE routes present: --cand-free and --null-free arms are required (R1 flip ratio)")
            f = {"cand": route_flips(ref_route, route(a.cand_free), opener),
                 "null": route_flips(ref_route, route(a.null_free), opener)}
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
                allow_null_floor=getattr(a, 'allow_null_floor', False))
    exit_code = EXIT[p["state"]]
    print(pooled_line(p, limits))
    if a.json:
        doc = {"schema": "strix-fast-gate-v2/v1", "limits": limits, "ref": a.ref, "cand": a.cand, "null": a.null,
               "format": a.format, "r1": r1,
               "windows": [{k: v for k, v in w.items() if k != "_rows"} for w in windows], "pooled": p}
        if tier != "fast":
            doc["tier"] = tier
        with open(a.json, "w", encoding="utf-8", opener=opener) as f:
            f.write(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    stdout = sys.stdout
    # Ordinary stdout must finish before optional output, including in the plain CLI.
    # Flush real Python stdio here so a daemon never owns its buffer lock at shutdown.
    stdout_flushed = type(stdout) is io.TextIOWrapper
    if stdout_flushed:
        try:
            stdout.flush()
        except (OSError, ValueError):
            return exit_code
    if r4_input is not None:
        try:
            _run_r4_diagnostic(r4_input, windows, p, classes, manifest, limits, protected,
                               stdout, sys.stderr, stdout_flushed)
        except BaseException:
            pass  # The ordinary verdict and receipt are already final.
    return exit_code


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


R4_CLASS_MIN = 3
R4_MAX_INPUT_BYTES = 1024 * 1024
R4_MAX_OUTPUT_BYTES = 4096


def _r4_class_stats(nlls, limits):
    n = len(nlls)
    med = float(statistics.median(nlls)) if n >= R4_CLASS_MIN else None
    bar = max(limits["ref_nll_factor"] * med, med + limits["ref_nll_margin"]) if med is not None else None
    if bar is not None and not math.isfinite(bar):
        raise GateError("derived bar must be finite")
    return {"n": n, "median": med, "bar": bar}


def _r4_label(text):
    # Keep labels on one line and reserve the uppercase verdict token for "would be PASS".
    return json.dumps(text).replace("PASS", r"\u0050ASS")


def _open_r4_original(path):
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOCTTY", 0) | getattr(os, "O_BINARY", 0)
    return os.fdopen(os.open(path, flags), "rb")


def _read_r4_original(source):
    with (contextlib.nullcontext(source) if isinstance(source, io.IOBase) else _open_r4_original(source)) as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            raise GateError("original NLL JSON must be a regular file")
        data = f.read(R4_MAX_INPUT_BYTES + 1)
    if len(data) > R4_MAX_INPUT_BYTES:
        raise GateError(f"original NLL JSON exceeds size limit ({R4_MAX_INPUT_BYTES} bytes)")
    source = data.decode("utf-8")
    # Bound nesting before handing the text to the recursive JSON decoder. The schema needs two levels.
    depth = 0
    quoted = escaped = False
    for char in source:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > 2:
                raise GateError("original NLL JSON exceeds nesting limit (2 levels)")
        elif char in "]}":
            depth = max(0, depth - 1)
    return json.loads(source)


def r4_assessment(path, windows, pooled, manifest, limits):
    original = _read_r4_original(path)
    if not isinstance(original, dict) or not original:
        raise GateError("expected a JSON object mapping content classes to original reference NLLs")
    for nlls in original.values():
        if not isinstance(nlls, list) or any(
                type(nll) not in (int, float) or not math.isfinite(nll) or nll < 0 for nll in nlls):
            raise GateError("original reference NLLs must be arrays of finite, nonnegative numbers")

    lines, flagged = [], []
    content = {c["name"]: c for c in class_verdicts(windows, limits)}
    band_windows = [dict(w, **{"class": window_class(w["name"], None, manifest)}) for w in windows]
    bands = {c["name"]: c for c in class_verdicts(band_windows, limits)}
    unresolved = False
    for cls in sorted(original.keys() | content.keys()):
        stats = _r4_class_stats(original.get(cls, []), limits)
        n, med, bar = stats["n"], stats["median"], stats["bar"]
        detail = f"median {med:.3f}, bar {bar:.3f}" if bar is not None else f"bar unavailable (need {R4_CLASS_MIN} originals)"
        outliers = [w["name"] for w in windows if w["class"] == cls and bar is not None and w["ref_nll"] > bar]
        flagged.extend(outliers)
        rows = content.get(cls, {}).get("rows", 0)
        provisional = bar is None or rows < limits["min_rows"] or content[cls]["state"] == "unresolved"
        unresolved |= provisional
        lines.append(f"content class {_r4_label(cls)}: {n} original windows, {detail}; {rows} scored rows"
                     + (" (provisional; UNRESOLVED)" if bar is None else " (provisional)" if provisional else "")
                     + (f"; off-distribution {_r4_label(', '.join(outliers))}" if outliers else ""))
    for band in sorted({window_class(case, None, manifest) for case in manifest}):
        rows = bands.get(band, {}).get("rows", 0)
        provisional = rows < limits["min_rows"] or bands[band]["state"] == "unresolved"
        unresolved |= provisional
        lines.append(f"prompt band {_r4_label(band)}: {rows} scored rows" + (" (provisional)" if provisional else ""))

    failed = any(not ok for k, ok in pooled["checks"].items() if k != "classes") or any(
        c["state"] == "fail" for c in (*content.values(), *bands.values()))
    suspect_class = any(c["final"] and c["reference_suspect"] for c in (*content.values(), *bands.values()))
    state = ("FAIL" if failed else "HOLD" if flagged or pooled["reference_suspect"] or suspect_class
             else "UNRESOLVED" if unresolved else "PASS")
    return [*lines, f"frozen-bar assessment would be {state}"]


def _r4_identity(fd):
    info = os.fstat(fd)
    return (info.st_dev, info.st_ino) if info.st_ino else None


def _r4_reduce_pair(reduce_pair, ref, cand, fmt, opener, **kw):
    # The shared readers accept descriptors and close them after reading.
    # Reclaim descriptors that an early validation error left unopened.
    pending = {}
    try:
        for path in (ref, cand):
            fd = opener(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
            pending[fd] = _r4_identity(fd)
        return reduce_pair(*pending, fmt, **kw)
    except BaseException as exc:
        # Release any suspended reader generators before reclaiming unused descriptors.
        traceback.clear_frames(exc.__traceback__)
        for fd, identity in pending.items():
            with contextlib.suppress(OSError):
                if _r4_identity(fd) == identity:
                    os.close(fd)
        raise


def _r4_alias(fd, protected):
    info = os.fstat(fd)
    if (stat.S_ISREG(info.st_mode) and not info.st_ino
            or (stat.S_ISREG(info.st_mode) or stat.S_ISFIFO(info.st_mode)) and None in protected):
        return True  # An unidentified regular file cannot safely receive the report.
    return bool(info.st_ino and (info.st_dev, info.st_ino) in protected)


def _r4_refusal(protected):
    # A redirected stderr cannot carry its own refusal. Use a separate terminal if present.
    tty = "CONOUT$" if os.name == "nt" else "/dev/tty"
    fd = os.open(tty, os.O_WRONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        if os.isatty(fd) and not _r4_alias(fd, protected):
            return _start_r4_writer(fd, "diagnostic unavailable: stderr aliases a gate input or output\n")
    finally:
        os.close(fd)


def _start_r4_writer(fd, text):
    # The helper waits for authorization before writing, even if launching it stalls.
    script = ("import os, sys\n"
              "if os.read(0, 1):\n"
              "    try: os.write(2, sys.argv[1].encode('ascii'))\n"
              "    except OSError: pass\n")
    return subprocess.Popen([sys.executable, "-I", "-S", "-c", script, text],
                            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=fd, bufsize=0)


def _write_r4_diagnostic(fd, stderr, text, protected):
    if fd is None:
        stderr.write(text)  # In-memory capture in embedded calls.
        return
    # Hold a duplicate only for the handoff, and check the handle the helper will inherit.
    fd = os.dup(fd)
    try:
        if _r4_alias(fd, protected):
            return _r4_refusal(protected)
        return _start_r4_writer(fd, text)
    finally:
        os.close(fd)


def _run_r4_diagnostic(*args):
    result = Future()
    deadline = time.monotonic() + 0.5
    threading.Thread(target=_r4_diagnostic, daemon=True, args=(*args, result)).start()
    try:
        writer = result.result(timeout=max(0, deadline - time.monotonic()))
    except BaseException:
        if result.cancel():
            return
        writer = result.result()  # The worker has claimed the handoff and is publishing it.
    if writer is not None:
        with writer:
            try:
                if time.monotonic() < deadline:
                    writer.stdin.write(b"1")  # One byte fits in the empty control pipe.
                    writer.wait(timeout=max(0, deadline - time.monotonic()))
            finally:
                if writer.poll() is None:
                    writer.kill()


def _r4_diagnostic(path, windows, pooled, classes, manifest, limits, protected, stdout, stderr,
                   stdout_flushed, result):
    fd, writer = None, None
    try:
        if stdout is stderr:
            return
        if not stdout_flushed:
            stdout.flush()  # An embedded stream may block; keep it inside the deadline.
        if result.cancelled():
            return
        if not isinstance(stderr, io.StringIO):
            fd = stderr.fileno()
            mode = os.fstat(fd).st_mode
            if stat.S_ISREG(mode) or stat.S_ISFIFO(mode):
                with contextlib.suppress(AttributeError, OSError, ValueError):
                    identity = _r4_identity(stdout.fileno())
                    if identity is not None:
                        protected.add(identity)
            if _r4_alias(fd, protected):
                writer = _r4_refusal(protected)
                return
        try:
            with _open_r4_original(path) as source:
                try:
                    protected.add(_r4_identity(source.fileno()))
                except OSError:
                    protected.add(None)
                if fd is not None and _r4_alias(fd, protected):
                    if not result.cancelled():
                        writer = _r4_refusal(protected)
                    return
                if not classes or manifest is None:
                    raise GateError("requires --classes and --manifest for content and prompt-band checks")
                lines = r4_assessment(source, windows, pooled, manifest, limits)
            text = "".join(f"R4 DIAGNOSTIC ONLY: {line}\n" for line in lines)
            if len(text) > R4_MAX_OUTPUT_BYTES:  # Assessment lines are ASCII.
                raise GateError(f"diagnostic output exceeds size limit ({R4_MAX_OUTPUT_BYTES} bytes)")
            if not result.cancelled():
                writer = _write_r4_diagnostic(fd, stderr, text, protected)
        except BaseException as e:
            if not result.cancelled():
                if fd is not None and isinstance(e, OSError) and not isinstance(e, FileNotFoundError):
                    protected.add(None)
                if fd is not None and _r4_alias(fd, protected):
                    writer = _r4_refusal(protected)
                else:
                    text = f"diagnostic unavailable: {type(e).__name__} {_r4_label(str(e)[:256])}\n"
                    writer = _write_r4_diagnostic(fd, stderr, text, protected)
    except BaseException:
        pass
    finally:
        if result.set_running_or_notify_cancel():
            result.set_result(writer)
        elif writer is not None:
            with writer:
                writer.kill()


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
    q.add_argument("--r4-assess", metavar="ORIGINAL_NLLS", help="diagnostic only on stderr: JSON {content class: "
                   "[original reference NLLs]} (requires --classes and --manifest)")
    q.add_argument("--json")
    q.add_argument("--tier", default="fast", choices=sorted(TIERS), help="acceptance limits: fast (default, "
                   "the FAST gate v2 bars) or turbo (TURBO_LIMITS, looser; explicit opt-in levers only)")
    q.add_argument("--allow-null-floor", action="store_true", help="explicitly opt in to relaxed pooled acceptance bars; default keeps absolute FAST limits")
    a = ap.parse_args(argv)
    try:
        return qual(a)
    except GateError as e:
        print(f"NO VERDICT: {e}")
        return 2
    except (OSError, ValueError) as e:  # a missing / malformed arm file is a defect of the evidence, never a FAIL (exit 1)
        print(f"NO VERDICT: {type(e).__name__}: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
