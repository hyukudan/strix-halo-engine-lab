"""Summarize existing launcher queue spans; never execute a model or probe."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

MAX_BYTES = 16 * 1024 * 1024
MAX_SPANS = 32768


def unique(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def integer(value):
    if type(value) is not int or not 0 <= value <= 0xffffffffffffffff:
        raise ValueError("expected uint64 metadata")
    return value


def flag(value):
    if type(value) is not bool:
        raise ValueError("expected boolean metadata")
    return value


def timestamp(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or value > 0xffffffffffffffff:
        raise ValueError("expected finite nonnegative device timestamp")
    return value


def merge(intervals):
    out = []
    for begin, end in sorted(intervals):
        if out and begin <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([begin, end])
    return out


def summary(spans):
    queues = {}
    for row in spans:
        queues.setdefault(row["queue"], []).append((row["begin_ns"], row["end_ns"]))
    per_queue = {queue: merge(intervals) for queue, intervals in queues.items()}
    merged = merge([interval for intervals in per_queue.values() for interval in intervals])
    union = sum(end - begin for begin, end in merged)
    envelope = merged[-1][1] - merged[0][0] if merged else 0
    # Merge within each queue FIRST: nested launcher spans cannot inflate the
    # distinct-queue concurrency count. Equal-time endpoints have zero measure.
    edges = {}
    for intervals in per_queue.values():
        for begin, end in intervals:
            edges[begin] = edges.get(begin, 0) + 1
            edges[end] = edges.get(end, 0) - 1
    active, previous, overlap = 0, None, 0
    for point, delta in sorted(edges.items()):
        if previous is not None and active >= 2:
            overlap += point - previous
        active += delta
        previous = point
    return {"observed_spans": len(spans), "observed_queues": len(queues),
            "observed_queue_span_union_ns": union,
            "observed_queue_span_envelope_ns": envelope,
            "observed_envelope_without_recorded_span_ns": envelope - union,
            "observed_distinct_queue_span_overlap_ns": overlap,
            "per_queue_union_ns": [{"queue": queue, "union_ns": sum(end - begin for begin, end in intervals)}
                                   for queue, intervals in sorted(per_queue.items())]}


def collect(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("queue trace exceeds bounded input size")
    value = json.loads(raw, object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if not isinstance(value, dict) or not isinstance(value.get("device_queue_trace"), dict):
        raise ValueError("expected one exported device_queue_trace object")
    trace = value["device_queue_trace"]
    if trace.get("scope") != "launcher-queue-spans-including-gaps-and-waits" or not flag(trace.get("profiling_only")):
        raise ValueError("unsupported timing scope")
    if trace.get("common_origin") != "same-device-HIP-timing-event":
        raise ValueError("queue spans require a declared common device clock origin")
    for field in ("clock_domain", "device_binding", "timing_resolution_approx_ns"):
        if not integer(trace.get(field)):
            raise ValueError("device clock/binding/resolution must be explicit and nonzero")
    rows = trace.get("spans")
    if not isinstance(rows, list) or len(rows) > MAX_SPANS:
        raise ValueError("invalid bounded span array")
    known, groups, tickets = [], {}, set()
    complete = flag(trace.get("all_recorded_events_completed")) and not flag(trace.get("observer_failed"))
    for field in ("dropped", "capture_skipped", "dependency_dropped", "api_errors"):
        complete = not integer(trace.get(field)) and complete
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid span record")
        ticket = integer(row.get("ticket"))
        if not ticket or ticket in tickets or not integer(row.get("queue")):
            raise ValueError("duplicate/invalid span ticket or queue")
        tickets.add(ticket)
        key = tuple(integer(row.get(field)) for field in
                    ("request_id", "session_epoch", "execution_identity", "position"))
        for field in ("step_id", "layer", "logical_rows"):
            integer(row.get(field))
        timed = flag(row.get("timing_known"))
        done = flag(row.get("gpu_event_completed"))
        returned = flag(row.get("launcher_returned"))
        complete = complete and timed and done and returned
        if not timed:
            if row.get("begin_ns") is not None or row.get("end_ns") is not None:
                raise ValueError("unknown timing cannot carry timestamp values")
            continue
        if not done:
            raise ValueError("known timing requires completed GPU events")
        begin, end = timestamp(row.get("begin_ns")), timestamp(row.get("end_ns"))
        if end < begin:
            raise ValueError("reversed device interval")
        known.append(row)
        groups.setdefault(key, []).append(row)
    result = summary(known)
    return {"schema_version": 1, "evidence_kind": "existing-launcher-queue-spans-only",
            "input_sha256": hashlib.sha256(raw).hexdigest(), "input_bytes": len(raw),
            "clock_domain_declared": trace["clock_domain"], "device_binding_declared": trace["device_binding"],
            "timing_resolution_approx_ns": trace["timing_resolution_approx_ns"],
            "scope": trace["scope"], "all_reported_spans_timed_and_returned": bool(rows) and complete,
            "unknown_timing_spans": len(rows) - len(known),
            "kernel_busy_ns": None, "overlap_of_kernels_ns": None, "device_critical_path_ns": None,
            "native_execution_independently_verified": None, "qualification": None,
            "observed": result, "per_frontier": [dict(zip(
                ("request_id", "session_epoch", "execution_identity", "position"), key), observed=summary(group))
                for key, group in sorted(groups.items())]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = collect(args.input)
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if receipt["all_reported_spans_timed_and_returned"] else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, OverflowError, RecursionError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
