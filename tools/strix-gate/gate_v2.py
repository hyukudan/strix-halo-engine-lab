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
  qual --r4-frozen R.json [--r4-regime decode]: the R4 off-distribution bar is the one FROZEN from the ORIGINAL windows
      of the qualification set (written once by `gate_v2.py r4-freeze R.json --set-id ID --regime NAME=GATE_JSON ...`
      from the gate JSON of the first run), so replacing a window (`gate_v2.py r4-substitute R.json --regime NAME
      --replaced A --by B --reason TEXT`; the substitute is chosen from the reference only) cannot move the bar. Every
      window, substitutes included, is checked against that bar; an unchanged window must reproduce its frozen reference
      NLL; an altered record, an unknown window or a missing reason is refused (no verdict). Without it: unchanged.
  qual --tier turbo: the same rules with the TURBO limits (TURBO_LIMITS: KL <= 0.05, top-1 >= 0.90, dNLL <= 0.03,
      route flip ratio <= 2.0; the per-window disagreement floor follows as 1 - 0.90 = 0.10). TURBO levers trade
      quality for speed and are explicit opt-in, never a default; a TURBO verdict never qualifies a FAST lever. The tier
      is printed on its own TIER line and stored in the JSON ("tier", "limits"). Default --tier fast: output unchanged.
Exit 0 = PASS, 1 = FAIL, 2 = no verdict (bookkeeping defect, too few rows, labels not held out, missing null),
     3 = UNRESOLVED (provisional: the lever stays opt-in for the unresolved classes), 4 = HOLD (reference check).
"""
import argparse
import hashlib
import json
import math
import pathlib
import re
import sys

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
    data = pathlib.Path(path).read_bytes()
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
    a, b = read_routes(ref_path), read_routes(arm_path)
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
    if r4_frozen is not None:
        pooled["r4_frozen"] = r4_frozen_check(windows, r4_frozen)
    pooled["off_distribution"] = off_distribution(windows, limits, r4_frozen)
    pooled["reference_suspect"] = [w["name"] for w in windows if w["reference_suspect"]]
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


R4_SCHEMA = "strix-gate-r4-frozen/v1"
R4_NLL_TOL = 1e-9   # an unchanged window's reference NLL must reproduce (the reference is bitwise reproducible)


def _r4_digest(frozen):
    return hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def r4_freeze(set_id, regimes, sources=None, limits=LIMITS):
    """Frozen R4 record of a qualification set, written once when the set is built: regimes {regime: {window: reference
    NLL}} of the ORIGINAL windows. The frozen block (and its sha256) never changes; substitutions are appended beside it."""
    if not set_id:
        raise GateError("r4 freeze: a set id is required")
    frozen = {"set_id": str(set_id), "regimes": {}}
    for regime, ws in regimes.items():
        if len(ws) < 3:
            raise GateError(f"r4 freeze: regime {regime} has {len(ws)} windows, R4 needs at least 3")
        med, bar = _r4_bar(ws.values(), limits)
        frozen["regimes"][regime] = {"windows": {k: float(v) for k, v in ws.items()}, "median": med, "bar": bar,
                                     "ref_nll_factor": limits["ref_nll_factor"], "ref_nll_margin": limits["ref_nll_margin"]}
        if sources and regime in sources:
            frozen["regimes"][regime]["source_sha256"] = sources[regime]
    return {"schema": R4_SCHEMA, "frozen": frozen, "frozen_sha256": _r4_digest(frozen), "substitutions": []}


def r4_expected(rec, regime):
    """{window: None (unchanged original) | substitution dict} of a regime after the record's substitutions, in order."""
    exp = {k: None for k in rec["frozen"]["regimes"][regime]["windows"]}
    for s in rec.get("substitutions", []):
        if s.get("regime") != regime:
            continue
        if not isinstance(s.get("reason"), str) or not s["reason"].strip():
            raise GateError(f"r4 record: substitution {s.get('replaced')} -> {s.get('by')} states no reason")
        if s.get("replaced") not in exp:
            raise GateError(f"r4 record: substitution replaces {s.get('replaced')!r}, not a window of regime {regime} "
                            "at that point")
        del exp[s["replaced"]]
        if not s.get("by") or s["by"] in exp:
            raise GateError(f"r4 record: substitute {s.get('by')!r} is empty or already a window of regime {regime}")
        exp[s["by"]] = s
    return exp


def load_r4_frozen(path, regime=None, limits=LIMITS):
    """Read and verify a frozen R4 record; any inconsistency is a defect of the evidence (no verdict)."""
    rec = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(rec, dict) or rec.get("schema") != R4_SCHEMA:
        raise GateError(f"{path}: not a {R4_SCHEMA} record")
    frozen = rec.get("frozen")
    if not isinstance(frozen, dict) or _r4_digest(frozen) != rec.get("frozen_sha256"):
        raise GateError(f"{path}: frozen block does not match its sha256 (record altered after it was frozen): refused")
    regs = frozen.get("regimes") or {}
    if regime is None:
        if len(regs) != 1:
            raise GateError(f"{path}: record holds regimes {sorted(regs)}: name one with --r4-regime")
        regime = next(iter(regs))
    if regime not in regs:
        raise GateError(f"{path}: no frozen regime {regime} (record holds {sorted(regs)})")
    g = regs[regime]
    if (g.get("ref_nll_factor"), g.get("ref_nll_margin")) != (limits["ref_nll_factor"], limits["ref_nll_margin"]):
        raise GateError(f"{path}: regime {regime} was frozen with factor {g.get('ref_nll_factor')} margin "
                        f"{g.get('ref_nll_margin')}, the gate's R4 is {limits['ref_nll_factor']} / {limits['ref_nll_margin']}")
    if len(g.get("windows") or {}) < 3:
        raise GateError(f"{path}: regime {regime} holds fewer than 3 original windows")
    med, bar = _r4_bar(g["windows"].values(), limits)
    if (med, bar) != (g.get("median"), g.get("bar")):
        raise GateError(f"{path}: regime {regime} median/bar {g.get('median')}/{g.get('bar')} are not those of its "
                        f"{len(g['windows'])} original windows ({med}/{bar}): refused")
    return {"set_id": frozen["set_id"], "regime": regime, "record_sha256": rec["frozen_sha256"], "median": med, "bar": bar,
            "windows": g["windows"], "expected": r4_expected(rec, regime)}


def r4_frozen_check(windows, fr):
    """Every live window is an unchanged original (same reference NLL) or a recorded substitute; returns the receipt."""
    for w in windows:
        if w["name"] not in fr["expected"]:
            raise GateError(f"{w['name']}: neither an original window of R4 set {fr['set_id']} ({fr['regime']}) nor a "
                            "recorded substitute")
        if fr["expected"][w["name"]] is None:
            ref = fr["windows"][w["name"]]
            if abs(float(w["ref_nll"]) - ref) > R4_NLL_TOL * max(1.0, abs(ref)):
                raise GateError(f"{w['name']}: reference NLL {w['ref_nll']!r} differs from the frozen {ref!r} of set "
                                f"{fr['set_id']} ({fr['regime']}) but no substitution is recorded")
    live = {w["name"] for w in windows}
    live_med, live_bar = _r4_bar([w["ref_nll"] for w in windows]) if len(windows) >= 3 else (None, None)
    return {"set_id": fr["set_id"], "regime": fr["regime"], "record_sha256": fr["record_sha256"], "median": fr["median"],
            "bar": fr["bar"], "originals": len(fr["windows"]), "live_median": live_med, "live_bar": live_bar,
            "substitutions": [{k: s[k] for k in ("replaced", "by", "reason")} for s in fr["expected"].values() if s],
            "missing": sorted(n for n in fr["expected"] if n not in live)}


def r4_frozen_lines(r):
    lines = [f"R4 bar frozen at {r['bar']:.3f} from {r['originals']} original windows (set id {r['set_id']}, regime "
             f"{r['regime']}, median {r['median']:.3f}, record sha256 {r['record_sha256'][:16]})"]
    lines += [f"R4 substitution: {s['replaced']} -> {s['by']} ({s['reason']})" for s in r["substitutions"]]
    if r["live_bar"] is not None:
        lines.append(f"R4 live set median {r['live_median']:.3f} bar {r['live_bar']:.3f} (not used: the bar is frozen)")
    if r["missing"]:
        lines.append(f"R4 windows of the set not in this evidence: {','.join(r['missing'])}")
    return lines


def off_distribution(windows, limits=LIMITS, frozen=None):
    """Windows whose reference NLL is far above the gate set's (control / chat-template ids, a timing prompt).
    frozen: a verified R4 record (load_r4_frozen): the bar of the ORIGINAL set, so a substitution cannot move it."""
    if frozen is not None:
        bar = frozen["bar"]
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
    out = pathlib.Path(a.out)
    tier = getattr(a, "tier", "fast") or "fast"
    if tier not in TIERS:
        raise GateError(f"unknown tier {tier}; tiers {sorted(TIERS)}")
    limits = TIERS[tier]
    if tier != "fast":
        print(f"TIER {tier}: KL <= {limits['max_kl']:g}, top-1 >= {limits['min_top1']:g}, dNLL <= "
              f"{limits['max_dnll']:g}, route flip ratio <= {limits['max_flip_ratio']:g} (explicit opt-in tier; never "
              f"qualifies a FAST lever)")
    r4_frozen = (load_r4_frozen(a.r4_frozen, getattr(a, "r4_regime", None), limits)
                 if getattr(a, "r4_frozen", None) else None)
    reduce_pair = _option_qual_reduce()
    head = a.format == "head"
    manifest = read_manifest(a.manifest) if a.manifest else None
    classes = json.loads(pathlib.Path(a.classes).read_text(encoding="utf-8")) if getattr(a, "classes", None) else None
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
    diag = read_trace_diag(a.trace_diag)
    if diag and manifest is None:
        raise GateError("--trace-diag needs --manifest (flip slots = prompt rows x traced layers)")
    windows, flips, replay, traced = [], [], {}, 0
    for case in cases:
        if manifest is not None and case not in manifest:
            raise GateError(f"{case}: not in the manifest")
        kw = {"ids": manifest[case], "rows": a.head_rows} if head else {}
        logit = lambda arm: path_of(case, arm)  # noqa: E731
        route = lambda arm: out / "routes" / f"{case}.{arm}.rte"  # noqa: E731
        err = lambda arm: (out / "log" / f"{case}.{arm}.stderr").read_bytes()  # noqa: E731
        ra, _ = reduce_pair(logit(a.ref), logit(a.cand), a.format, **kw)
        nulls = [reduce_pair(logit(a.ref), logit(n), a.format, **kw)[0] for n in a.null]
        null_moved(case, [float(x["kl_rc"].mean()) for x in nulls])
        w = window_stats(case, ra["kl_rc"], ra["agree"].astype(bool), ra["nll_r"], ra["nll_c"],
                         [float(x["kl_rc"].mean()) for x in nulls], ra["top1_r_is_label"], ra["top1_c_is_label"],
                         null_top1s=[float(x["agree"].mean()) for x in nulls], null_nlls=[x["nll_c"] for x in nulls],
                         cls=window_class(case, classes, manifest), null_kl_rows=nulls[0]["kl_rc"], limits=limits)
        moe = route(a.ref).exists()
        f = None
        if moe:
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
    for line in r4_frozen_lines(p["r4_frozen"]) if r4_frozen is not None else []:
        print(line)
    print(pooled_line(p, limits))
    if a.json:
        doc = {"schema": "strix-fast-gate-v2/v1", "limits": limits, "ref": a.ref, "cand": a.cand, "null": a.null,
               "format": a.format, "r1": r1,
               "windows": [{k: v for k, v in w.items() if k != "_rows"} for w in windows], "pooled": p}
        if tier != "fast":
            doc["tier"] = tier
        pathlib.Path(a.json).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return EXIT[p["state"]]


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


def r4_freeze_cmd(a):
    path = pathlib.Path(a.record)
    if path.exists():
        raise GateError(f"{path} exists: a frozen R4 record is written once, when the set is built")
    regimes, sources = {}, {}
    for spec in a.regime:
        name, sep, src = spec.partition("=")
        if not sep or not name or name in regimes:
            raise GateError(f"--regime {spec!r}: expected NAME=GATE_JSON, each regime once")
        raw = pathlib.Path(src).read_bytes()
        ws = json.loads(raw).get("windows") or []
        regimes[name] = {w["name"]: float(w["ref_nll"]) for w in ws}
        if len(regimes[name]) != len(ws):
            raise GateError(f"{src}: duplicate window")
        sources[name] = hashlib.sha256(raw).hexdigest()
    rec = r4_freeze(a.set_id, regimes, sources)
    path.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for name, g in rec["frozen"]["regimes"].items():
        print(f"R4 frozen {a.set_id} {name}: {len(g['windows'])} windows median {g['median']:.3f} bar {g['bar']:.3f}")
    print(f"record sha256 {rec['frozen_sha256']}")
    return 0


def r4_substitute_cmd(a):
    path = pathlib.Path(a.record)
    load_r4_frozen(path, a.regime)                       # verified before it is extended
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec.setdefault("substitutions", []).append({"regime": a.regime, "replaced": a.replaced, "by": a.by, "reason": a.reason})
    load = r4_expected(rec, a.regime)                    # the chain stays consistent
    path.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"R4 substitution recorded ({a.regime}): {a.replaced} -> {a.by}; {len(load)} windows expected")
    return 0


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
    q.add_argument("--r4-regime", help="regime of the frozen record to use (needed when it holds more than one)")
    fz = sub.add_parser("r4-freeze", help="write the frozen R4 record of a qualification set from its first gate JSONs")
    fz.add_argument("record")
    fz.add_argument("--set-id", required=True)
    fz.add_argument("--regime", action="append", required=True, metavar="NAME=GATE_JSON",
                    help="the gate JSON (qual --json) of the ORIGINAL windows of this regime")
    sb = sub.add_parser("r4-substitute", help="record that a window of a frozen set was replaced, and why")
    sb.add_argument("record")
    sb.add_argument("--regime", required=True)
    sb.add_argument("--replaced", required=True)
    sb.add_argument("--by", required=True)
    sb.add_argument("--reason", required=True)
    a = ap.parse_args(argv)
    try:
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
