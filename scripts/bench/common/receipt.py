#!/usr/bin/env python3
"""Run receipts shared by every kit under scripts/bench/ (standard library only).

  receipt.py meta DIR --rc N --wall-ms N --mem TEXT [--host-tag T] [--bin-sha S]
      written by runlib.sh kit_run after the process exits.
  receipt.py summarize DIR --tag TAG [--arm ARM] [--round N] [--tokens N] [--metric NAME=REGEX] ...
      parses one run directory into DIR/receipt.json and prints one line.

Timing. Prefill wall is the LAST match of --prefill-regex (default: the engines' `prefill-wall-ms=` and the
pair `prefill-handoff-seconds=` lines). With two or more prefills in one process the FIRST is reported as cold
and the LAST as warm (methodology: never compare a warm figure with a cold one). --metric adds any other number
(last match of a regex with one group), e.g. decode tok/s.

Power, clock and GPU busy are integrated over the warm prefill window [anchor - prefill, anchor], anchor = the
time stamp of the line that carried the figure (stderr.ts). GPU clocks per token = busy x sclk x window / tokens.

Validity (a receipt with valid=false is excluded by abab.py and its reasons are printed):
  rc != 0; kfd evicted_ms grew inside the timed window (memory pressure moved our buffers while timing; growth
  during model load is a warning, --evict-whole-run makes it fatal); MemAvailable fell below --min-avail-gb;
  kernel log events (amdgpu / kfd / thunderbolt / oom / reset) in the run are a warning (--strict-klog: fatal);
  --require REGEX lines missing from stderr; --reject REGEX lines present.
"""
import argparse
import hashlib
import json
import os
import re
import sys

KIT_EVICT_PAD_S = 0.25  # one sampler period after the end line; the first in-window sample is the baseline
DEFAULT_PREFILL = [r"prefill-handoff-seconds=([\d.]+)@s", r"prefill-wall-ms=([\d.]+)"]
DEFAULT_REJECT = [r"what\(\)", r"\btimed out\b", r"^(?!.*cache policy).*\bdeclined\b"]


def read_text(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as stream:
            return stream.read()
    except OSError:
        return ""


def sha_file(path, n=16):
    try:
        with open(path, "rb") as stream:
            return hashlib.sha256(stream.read()).hexdigest()[:n]
    except OSError:
        return "-"


def stdout_json(text):
    """Last JSON object printed on stdout (the engines print one result object)."""
    start = text.rfind("\n{")
    for begin in ([start + 1] if start >= 0 else []) + [text.find("{")]:
        if begin < 0:
            continue
        try:
            return json.loads(text[begin:])
        except ValueError:
            continue
    return {}


def samples(path):
    rows = []
    for line in read_text(path).splitlines():
        part = line.split()
        if len(part) >= 9:
            try:
                rows.append([float(x) for x in part[:9]])
            except ValueError:
                pass
    return rows


def stamps(path):
    out = []
    for line in read_text(path).splitlines():
        part = line.split(" ", 1)
        if len(part) == 2:
            try:
                out.append((float(part[0]), part[1]))
            except ValueError:
                pass
    return out


def prefill_values(err, patterns):
    """All prefill figures in ms, in stderr order, from the first pattern that matches at all."""
    for pat in patterns:
        scale = 1.0
        if pat.endswith("@s"):
            pat, scale = pat[:-2], 1e3
        found = [float(v) * scale for v in re.findall(pat, err, re.M)]
        if found:
            return pat, found
    return None, []


def growth(values):
    """Sum of increments: evicted_ms is a per-process counter, it reads 0 once the process has exited."""
    return sum(max(0.0, b - a) for a, b in zip(values, values[1:]))


def integrate(rows, t0, t1):
    joules = cycles = busy = span = temp = 0.0
    for a, b in zip(rows, rows[1:]):
        lo, hi = max(a[0], t0), min(b[0], t1)
        if hi > lo:
            dt = hi - lo
            joules += a[1] / 1e6 * dt
            cycles += a[2] * dt
            busy += a[4] * dt
            span += dt
            temp = max(temp, a[3] / 1e3)
    return joules, cycles, busy, span, temp


def cmd_meta(args):
    meta = dict(rc=args.rc, wall_ms=args.wall_ms, mem=args.mem.strip(), host=args.host_tag, bin_sha=args.bin_sha)
    with open(os.path.join(args.dir, "meta.json"), "w") as stream:
        json.dump(meta, stream)


def cmd_summarize(args):
    d = args.dir
    meta = json.loads(read_text(os.path.join(d, "meta.json")) or "{}")
    err = read_text(os.path.join(d, "stderr"))
    out_json = stdout_json(read_text(os.path.join(d, "stdout")))
    pat, pf = prefill_values(err, args.prefill_regex or DEFAULT_PREFILL)
    if not pf and out_json.get("generation_request_elapsed_ms"):
        pf = [float(out_json["generation_request_elapsed_ms"])]
    warm = pf[-1] if pf else 0.0
    cold = pf[0] if pf else 0.0
    tl = stamps(os.path.join(d, "stderr.ts"))
    anchor = 0.0
    if pat:
        # stderr.ts holds one stamp per stderr line (same order), so the line index gives the arrival time.
        rx = re.compile(pat)
        idx = [i for i, line in enumerate(err.splitlines()) if rx.search(line)]
        if idx and idx[-1] < len(tl):
            anchor = tl[idx[-1]][0]
    if args.anchor_regex:
        # The figure came from a JSON field (or a line without a usable stamp): anchor on the named end-of-prefill line.
        rx = re.compile(args.anchor_regex)
        idx = [i for i, line in enumerate(err.splitlines()) if rx.search(line)]
        if idx and idx[-1] < len(tl):
            anchor = tl[idx[-1]][0]
    if not anchor and tl:
        anchor = tl[-1][0]
    rows = samples(os.path.join(d, "samples.tsv"))
    joules, cyc, busy, span, temp = integrate(rows, anchor - warm / 1e3, anchor)
    sp = span or 1.0
    ev = [r[8] for r in rows]
    t0w = anchor - warm / 1e3
    evw = [r[8] for r in rows if t0w <= r[0] <= anchor + KIT_EVICT_PAD_S]
    avail = [r[6] for r in rows]
    klog = [l for l in read_text(os.path.join(d, "klog.txt")).splitlines() if l.strip()]
    metrics = {}
    for item in args.metric or []:
        name, _, rx = item.partition("=")
        found = re.findall(rx, err) or re.findall(rx, read_text(os.path.join(d, "stdout")))
        if found:
            last = found[-1]
            metrics[name] = float(last[0] if isinstance(last, tuple) else last)
    gen = out_json.get("generated_tokens")
    tokens = args.tokens
    rec = dict(
        tag=args.tag, arm=args.arm or args.tag, round=args.round, host=meta.get("host"), bin_sha=meta.get("bin_sha"),
        rc=meta.get("rc"), wall_ms=meta.get("wall_ms"), mem_start=meta.get("mem"),
        prefills=len(pf), prefill_ms_cold=cold, prefill_ms_warm=warm, tokens=tokens,
        tok_s_cold=(tokens / (cold / 1e3) if tokens and cold else 0.0),
        tok_s_warm=(tokens / (warm / 1e3) if tokens and warm else 0.0),
        window_s=round(span, 3), joules=round(joules, 1), watts=round(joules / sp, 1),
        sclk_mhz=round(cyc / sp / 1e6, 0), busy_pct=round(busy / sp, 1), temp_max_c=round(temp, 0),
        gpu_clocks_per_token=(round((busy / sp / 100.0) * (cyc / sp) * span / tokens, 1) if tokens and span else None),
        logits_sha=sha_file(os.path.join(d, "logits.f32")),
        gen_sha=(hashlib.sha256(json.dumps(gen).encode()).hexdigest()[:12] if gen is not None else "-"),
        identity=out_json.get("prefill_identity"), metrics=metrics,
        evicted_ms_run=(growth(ev) if ev else -1), evicted_ms_window=(growth(evw) if len(evw) > 1 else -1),
        avail_min_gb=(round(min(avail) / 2 ** 20, 1) if avail else -1),
        klog_events=len(klog))
    reasons = []
    if rec["rc"] not in (0, None):
        reasons.append("rc=%s" % rec["rc"])
    if rec["evicted_ms_window"] > 0 or (args.evict_whole_run and rec["evicted_ms_run"] > 0):
        reasons.append("kfd evicted_ms +%d in the timed window" % max(rec["evicted_ms_window"], 0))
    if avail and max(avail) > 0 and rec["avail_min_gb"] < args.min_avail_gb:
        reasons.append("MemAvailable %.1f GiB < %.1f" % (rec["avail_min_gb"], args.min_avail_gb))
    warnings = []
    if rec["evicted_ms_run"] > 0 and rec["evicted_ms_window"] <= 0:
        warnings.append("kfd evicted_ms +%d outside the timed window (load)" % rec["evicted_ms_run"])
    if klog:
        (reasons if args.strict_klog else warnings).append("klog %d events: %s" % (len(klog), klog[0][-80:]))
    if not rows:
        reasons.append("no samples")
    elif not any(r[6] for r in rows):
        warnings.append("host memory and kfd evictions not sampled on this host")
    if args.need_prefill and not pf:
        reasons.append("no prefill figure")
    for rx in args.require or []:
        if not re.search(rx, err, re.M):
            reasons.append("missing /%s/" % rx[:40])
    for rx in (args.reject if args.reject is not None else DEFAULT_REJECT):
        m = re.search(rx, err, re.M)
        if m:
            reasons.append("stderr /%s/" % rx[:30])
    rec["valid"] = not reasons
    rec["invalid_reasons"] = reasons
    rec["warnings"] = warnings
    with open(os.path.join(d, "receipt.json"), "w") as stream:
        json.dump(rec, stream, indent=1)
    extra = " ".join("%s=%g" % kv for kv in metrics.items())
    print("== %s %s rc=%s pf_ms warm=%.1f cold=%.1f (%d) tok/s=%.0f sclk=%.0f busy=%.0f W=%.0f Gclk/tok=%s logits=%s gen=%s %s %s" % (
        rec["tag"], "VALID" if rec["valid"] else "INVALID[" + "; ".join(reasons) + "]", rec["rc"], warm, cold, len(pf),
        rec["tok_s_warm"], rec["sclk_mhz"], rec["busy_pct"], rec["watts"],
        ("%.0f" % rec["gpu_clocks_per_token"]) if rec["gpu_clocks_per_token"] else "-", rec["logits_sha"], rec["gen_sha"],
        extra, rec["mem_start"] or ""))
    for w in warnings:
        print("   warn: " + w[:200])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("meta")
    m.add_argument("dir")
    m.add_argument("--rc", type=int, required=True)
    m.add_argument("--wall-ms", type=int, required=True)
    m.add_argument("--mem", default="")
    m.add_argument("--host-tag", default="local")
    m.add_argument("--bin-sha", default="-")
    s = sub.add_parser("summarize")
    s.add_argument("dir")
    s.add_argument("--tag", required=True)
    s.add_argument("--arm")
    s.add_argument("--round", type=int, default=0, help="ABAB round index (abab.py pairs arms by round)")
    s.add_argument("--tokens", type=int, default=0, help="prompt tokens, for prefill tok/s and clocks per token")
    s.add_argument("--prefill-regex", action="append", help="regex with one float group; suffix @s = seconds")
    s.add_argument("--anchor-regex", help="stderr line that ends the timed prefill (default: the figure's own line)")
    s.add_argument("--metric", action="append", help="NAME=REGEX (one float group, last match)")
    s.add_argument("--min-avail-gb", type=float, default=float(os.environ.get("KIT_MIN_AVAIL_GB", "2")))
    s.add_argument("--require", action="append")
    s.add_argument("--reject", action="append")
    s.add_argument("--need-prefill", action="store_true")
    s.add_argument("--evict-whole-run", action="store_true", help="any eviction in the run invalidates (decode kits)")
    s.add_argument("--strict-klog", action="store_true", help="kernel-log events invalidate instead of warning")
    args = p.parse_args(argv)
    {"meta": cmd_meta, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
