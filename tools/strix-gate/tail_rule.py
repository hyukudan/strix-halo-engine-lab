#!/usr/bin/env python3
"""Tail event rule of the FAST / TURBO logit gate: confident departures from the reference, counted against the null.

A pooled mean (KL, top-1, dNLL) cannot tell a rare real defect from noise: one window that fails by a few rows looks
the same as rounding. The event rule looks only at rows where the reference is confident:
  margin  = reference top-1 log-probability minus its top-2 log-probability on that row (nats)
  event   = the arm's top-1 differs from the reference top-1 on a row with margin >= MARGIN (2 nats: the reference
            top-1 is at least e^2 = 7.4 times as likely as the runner-up)
A one-ulp / evaluation-order null flips near ties, almost never at that margin. The null's event rate
p0 = (null events + 1) / (null rows + 2) (no null: p0 = 1 / (rows + 2)) is the rate a rounding-class arm shows; the
arm FAILS the rule when its event count is too high for that rate: exact binomial upper tail P(X >= events) < ALPHA.
The verdict lists the first events (prompt, row, margin) so a failing window is classified by its rows, not its mean.
"""
import numpy as np

TAIL = {"policy": "tail-event-v1", "margin": 2.0, "alpha": 0.01, "show": 8}


def ref_margin(log_probs):
    """Per row: top-1 minus top-2 of a [rows, vocab] array of log-probabilities (or logits; the gap is the same)."""
    z = np.asarray(log_probs)
    if z.ndim != 2 or z.shape[1] < 2:
        raise ValueError("ref_margin needs [rows, vocab >= 2]")
    top = np.partition(z, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def count_events(results, margin=TAIL["margin"]):
    """(rows, events, [(prompt index, row, margin)]) over compare_logits results that carry 'agree' and 'margin_ref'."""
    rows, found = 0, []
    for p, r in enumerate(results):
        agree, m = np.asarray(r["agree"], bool), np.asarray(r["margin_ref"], float)
        if agree.shape != m.shape:
            raise ValueError("agree and margin_ref differ in length")
        rows += int(agree.size)
        found.extend((p, int(i), float(m[i])) for i in np.flatnonzero(~agree & (m >= margin)))
    return rows, len(found), found


def tail_verdict(results, null_results=(), policy=TAIL, binom_sf=None):
    """The arm's events against the pooled null rate. null_results: one results list per null arm (same prompts)."""
    if binom_sf is None:
        from fast_logit_gate import binom_sf  # one exact binomial tail for every gate rule
    rows, events, found = count_events(results, policy["margin"])
    null_rows = null_events = 0
    for rs in null_results:
        nr, ne, _ = count_events(rs, policy["margin"])
        null_rows, null_events = null_rows + nr, null_events + ne
    p0 = (null_events + 1) / (null_rows + 2) if null_rows else 1.0 / (rows + 2)
    p = binom_sf(events, rows, p0) if rows else 1.0
    return {"policy": policy["policy"], "margin": policy["margin"], "alpha": policy["alpha"], "rows": rows,
            "events": events, "null_arms": len(null_results), "null_rows": null_rows, "null_events": null_events,
            "p0": p0, "p": p, "pass": bool(p >= policy["alpha"]),
            "first": [{"prompt": a, "row": b, "margin": c} for a, b, c in found[:policy["show"]]]}


def describe(v, names=None):
    first = ", ".join(f"{names[e['prompt']] if names else e['prompt']}:{e['row']} ({e['margin']:.2f})" for e in v["first"])
    return (f"tail events (top-1 flips at reference margin >= {v['margin']:.1f} nats): {v['events']} of {v['rows']} rows; "
            f"null {v['null_events']} of {v['null_rows']} ({v['null_arms']} arms), p0 {v['p0']:.2e}, "
            f"P(X >= {v['events']}) {v['p']:.3g} -> {'PASS' if v['pass'] else 'FAIL'} (alpha {v['alpha']})"
            + (f"; first {first}" if first else ""))
