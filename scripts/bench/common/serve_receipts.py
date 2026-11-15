#!/usr/bin/env python3
"""serve_receipts.py DIR --tag TAG --arm ARM --server-up 0|1

Turns the serve_bench.py results of one served run (DIR/<class>-<round>.json, server data in DIR/server/) into
one receipt per (class, round) at DIR/<class>-<round>/receipt.json, in the same schema as receipt.py, so
abab.py compares served decode the same way it compares prefill (e.g. abab.py summary OUT --metric e2e_C8).
Metrics per concurrency n: agg_C<n> (steady aggregate tok/s from server cycle receipts, when present),
e2e_C<n> (client end-to-end tok/s), req_C<n> (per-request p50), ttft_C<n> (s), fail_C<n> (failed requests).
Validity: the server came up, no failed request, and the server process passed the receipt.py checks
(kfd evicted_ms flat, MemAvailable floor, no kernel-log events).
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import receipt  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir")
    p.add_argument("--tag", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--server-up", type=int, default=1)
    p.add_argument("--round-base", type=int, default=0, help="ABAB round of this server start (added to the in-run round)")
    p.add_argument("--client", type=int, default=0, help="1: a --client-cmd run; one server-validity receipt")
    p.add_argument("--min-avail-gb", type=float, default=float(os.environ.get("KIT_MIN_AVAIL_GB", "2")))
    args = p.parse_args(argv)
    sdir = os.path.join(args.dir, "server")
    rows = receipt.samples(os.path.join(sdir, "samples.tsv"))
    ev = [r[8] for r in rows]
    avail = [r[6] for r in rows]
    klog = [l for l in receipt.read_text(os.path.join(sdir, "klog.txt")).splitlines() if l.strip()]
    meta = json.loads(receipt.read_text(os.path.join(sdir, "meta.json")) or "{}")
    base = []
    if not args.server_up:
        base.append("server not ready")
    if ev and receipt.growth(ev) > 0:
        base.append("kfd evicted_ms +%d" % receipt.growth(ev))
    if avail and max(avail) > 0 and min(avail) / 2 ** 20 < args.min_avail_gb:
        base.append("MemAvailable %.1f GiB" % (min(avail) / 2 ** 20))
    if klog:
        base.append("klog %d events" % len(klog))
    busy = sum(r[4] for r in rows) / len(rows) if rows else 0.0
    sclk = sum(r[2] for r in rows) / len(rows) / 1e6 if rows else 0.0
    files = sorted(glob.glob(os.path.join(args.dir, "*-*.json")))
    if args.client:
        files = []
        rec = dict(tag=args.tag + "-client", arm=args.arm, round=args.round_base, host=meta.get("host"), bin_sha=meta.get("bin_sha"),
                   rc=0, wall_ms=meta.get("wall_ms"), mem_start=meta.get("mem"), prefills=0, prefill_ms_cold=0.0,
                   prefill_ms_warm=0.0, tokens=0, tok_s_cold=0.0, tok_s_warm=0.0, sclk_mhz=round(sclk), busy_pct=round(busy, 1),
                   gpu_clocks_per_token=None, logits_sha="-", gen_sha="-", identity=None, metrics={},
                   evicted_ms_run=(receipt.growth(ev) if ev else -1), avail_min_gb=(round(min(avail) / 2 ** 20, 1) if avail else -1),
                   klog_events=len(klog), valid=not base, invalid_reasons=base)
        os.makedirs(os.path.join(args.dir, "client"), exist_ok=True)
        with open(os.path.join(args.dir, "client", "receipt.json"), "w") as stream:
            json.dump(rec, stream, indent=1)
        print("== %s %s" % (rec["tag"], "VALID" if rec["valid"] else "INVALID[" + "; ".join(base) + "]"))
    elif not files:
        print("== %s INVALID[%s] no serve_bench results" % (args.tag, "; ".join(base) or "no bursts"))
    for path in files:
        m = re.match(r"(.+)-(\d+)\.json$", os.path.basename(path))
        if not m:
            continue
        cls, rnd = m.group(1), args.round_base + int(m.group(2))
        try:
            with open(path) as stream:
                result = json.load(stream)
        except (OSError, ValueError):
            continue
        metrics, reasons = {}, list(base)
        for b in result.get("bursts", []):
            n, c, s = b["concurrency"], b.get("client", {}), b.get("steady", {})
            for key, val in (("agg", s.get("aggregate_tok_s")), ("e2e", c.get("end_to_end_tok_s")),
                             ("req", c.get("per_request_tok_s_p50")), ("ttft", c.get("ttft_p50_s")), ("fail", c.get("failed"))):
                if val is not None:
                    metrics["%s_C%d" % (key, n)] = round(float(val), 3)
            if c.get("failed"):
                reasons.append("C%d failed %d" % (n, c["failed"]))
        rec = dict(tag="%s-%s-%d" % (args.tag, cls, rnd), arm="%s-%s" % (args.arm, cls), round=rnd, host=meta.get("host"),
                   bin_sha=meta.get("bin_sha"), rc=0, wall_ms=meta.get("wall_ms"), mem_start=meta.get("mem"), prefills=0,
                   prefill_ms_cold=0.0, prefill_ms_warm=0.0, tokens=0, tok_s_cold=0.0, tok_s_warm=0.0,
                   sclk_mhz=round(sclk), busy_pct=round(busy, 1), gpu_clocks_per_token=None, logits_sha="-",
                   gen_sha="-", identity=None, metrics=metrics, evicted_ms_run=(receipt.growth(ev) if ev else -1),
                   avail_min_gb=(round(min(avail) / 2 ** 20, 1) if avail else -1), klog_events=len(klog),
                   valid=not reasons, invalid_reasons=reasons)
        d = os.path.join(args.dir, "%s-%s" % (cls, m.group(2)))
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "receipt.json"), "w") as stream:
            json.dump(rec, stream, indent=1)
        print("== %s %s %s" % (rec["tag"], "VALID" if rec["valid"] else "INVALID[" + "; ".join(reasons) + "]",
                               " ".join("%s=%g" % kv for kv in sorted(metrics.items()) if not kv[0].startswith("fail"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
