#!/usr/bin/env python3
"""Compare campaign throughput with public best-on-record measurements."""

import argparse
from contextlib import contextmanager, nullcontext
from datetime import date
from decimal import Decimal
from functools import lru_cache
from html import escape
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT / "docs/records/best_on_record.json"
VIEW = RECORDS.with_suffix(".md")
CATALOG = ROOT / "docs/optimizations.json"
TIERS = ("exact", "fast", "turbo")
METRICS = ("prefill-2k", "prefill-8k", "prefill-32k", "decode-b1", "aggregate-c8")
BANDS = {"prefill-2k": (1536, 3072), "prefill-8k": (6144, 10240), "prefill-32k": (24576, 40960)}
KEY = re.compile(
    rf"[a-z0-9][a-z0-9.-]*/(?:one|pair)/(?:{'|'.join(TIERS)})/"
    rf"(?:cli|served)/(?:{'|'.join(METRICS)})"
)
CAMPAIGN_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*")
PUBLIC_RECEIPT = re.compile(r"(?:README\.md|docs/(?:[\w.-]+/)*[\w.-]+\.md)(?:#[\w.-]+)?")
CATALOG_ALIASES = {
    "deepseek": {"deepseek-v4.1"}, "deepseek-v4.1-flash": {"deepseek-v4.1"},
    "deepseek41": {"deepseek-v4.1"}, "hy4": {"hy4-flash"},
    "glm": {"glm-5.3"}, "glm-5.3-flash": {"glm-5.3"},
    "minimax": {"minimax-m3.1"}, "minimax-m3.1-flash": {"minimax-m3.1"},
    "qwen": {"qwen3.8", "qwen3.8-27b"}, "qwen3.8-flash-next": {"qwen3.8"},
    "gemma4": {"gemma4", "nemotron-3.5-bolt-e4b"},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(content):
    return json.loads(content, object_pairs_hook=unique_keys)


def load(path):
    content = path.read_text(encoding="utf-8")
    return json.loads(content) if path == CATALOG else read_json(content)


def fields(value, required, optional=()):
    require(isinstance(value, dict) and set(required) <= value.keys() <= set(required) | set(optional),
            f"expected fields {', '.join(required)}; optional: {', '.join(optional)}")


def nonempty(value):
    return isinstance(value, str) and bool(value.strip()) and all(c.isprintable() for c in value)


def valid_key(key):
    require(isinstance(key, str) and KEY.fullmatch(key), f"invalid row ID: {key!r}")


def number(value):
    require(type(value) in (int, float), f"expected a number: {value!r}")
    value = Decimal(str(value))
    require(value.is_finite(), "number must be finite")
    return value


def validate_workload(workload, key):
    fields(workload, ("prompt_tokens", "generated_tokens", "requests", "timing"), ("notes",))
    metric = key.split("/")[-1]
    for token_field in ("prompt_tokens", "generated_tokens"):
        tokens = workload[token_field]
        minimum = 1 if token_field == "prompt_tokens" else 0
        counts = tokens if isinstance(tokens, list) else [tokens]
        require((tokens is None or all(type(n) is int and n >= minimum for n in counts))
                and (not isinstance(tokens, list) or len(tokens) == 2 and tokens[0] < tokens[1]),
                f"{key}: invalid {token_field}")
    expected = 8 if metric == "aggregate-c8" else 1
    require(type(workload["requests"]) is int and workload["requests"] == expected,
            f"{key}: requests must be {expected}")
    timing = {"decode-b1": ("decode", "wall"), "aggregate-c8": ("decode", "wall")}
    require(workload["timing"] in timing.get(metric, ("prefill", "ttft", "estimated")),
            f"{key}: invalid timing")
    if metric in BANDS:
        lower, upper = BANDS[metric]
        tokens = workload["prompt_tokens"]
        require(type(tokens) is int and lower <= tokens <= upper, f"{key}: prompt is outside its band")
    if "notes" in workload:
        require(nonempty(workload["notes"]), f"{key}: invalid workload notes")


@lru_cache(maxsize=None)
def anchors(path):
    content = path.read_text(encoding="utf-8")
    headings = re.findall(r"^#{1,6} (.+)$", content, re.M)
    return set(re.findall(r'<a id="([^"]+)">', content)) | {
        re.sub(r"[^\w -]", "", title.lower()).replace(" ", "-") for title in headings
    }


def validate_measurement(row, key, *, public=False, failed=False, root=ROOT):
    evidence = ("commit", "date", "receipt")
    if failed:
        fields(row, (*evidence, "failure"))
        require(nonempty(row["failure"]), f"{key}: missing failure description")
    else:
        fields(row, (*evidence, "value", "workload"), ("receipt_revision",))
        require(number(row["value"]) > 0, f"{key}: throughput must be positive")
        validate_workload(row["workload"], key)
        if "receipt_revision" in row:
            require(row["receipt_revision"] is True, f"{key}: receipt_revision must be true")
    commit, day, receipt = (row[k] for k in evidence)
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{7,40}", commit),
            f"{key}: invalid commit")
    require(isinstance(day, str) and re.fullmatch(r"\d{4}-\d\d-\d\d", day), f"{key}: invalid date")
    date.fromisoformat(day)
    require(nonempty(receipt), f"{key}: missing receipt")
    if public:
        file, _, anchor = receipt.partition("#")
        path = root / file
        require(PUBLIC_RECEIPT.fullmatch(receipt) and path.is_file()
                and path.resolve().is_relative_to(root.resolve()), f"{key}: invalid public receipt {receipt!r}")
        require(not anchor or anchor in anchors(path), f"{key}: missing receipt anchor {receipt!r}")


def validate_ledger(ledger, root=ROOT):
    fields(ledger, ("version", "records", "acceptances"))
    require(type(ledger["version"]) is int and ledger["version"] == 1, "unsupported ledger version")
    require(isinstance(ledger["records"], dict) and ledger["records"], "ledger has no measurements")
    for key, row in ledger["records"].items():
        valid_key(key)
        validate_measurement(row, key, public=True, root=root)
    require(isinstance(ledger["acceptances"], list), "acceptances must be a list")
    last = {}
    for entry in ledger["acceptances"]:
        fields(entry, ("row", "reason", "previous", "replacement"))
        key = entry["row"]
        valid_key(key)
        require(key in ledger["records"] and nonempty(entry["reason"]), f"{key}: invalid acceptance")
        for name in ("previous", "replacement"):
            validate_measurement(entry[name], key, public=True, root=root)
        previous, replacement = (entry[n]["value"] for n in ("previous", "replacement"))
        require(previous > replacement and previous >= last.get(key, 0), f"{key}: invalid acceptance history")
        require(entry["previous"]["workload"]["timing"] == entry["replacement"]["workload"]["timing"],
                f"{key}: acceptance changes the timing boundary")
        last[key] = replacement
    for key, value in last.items():
        require(ledger["records"][key]["value"] >= value, f"{key}: record below latest acceptance")


def validate_campaign(campaign):
    fields(campaign, ("version", "runs"))
    require(type(campaign["version"]) is int and campaign["version"] == 1, "unsupported campaign version")
    require(isinstance(campaign["runs"], dict) and campaign["runs"], "campaign has no runs")
    for run, rows in campaign["runs"].items():
        require(CAMPAIGN_ID.fullmatch(run), f"invalid run ID: {run!r}")
        require(isinstance(rows, dict) and rows, f"{run}: no measurements")
        commits = set()
        for key, row in rows.items():
            valid_key(key)
            validate_measurement(row, key, failed=isinstance(row, dict) and "failure" in row)
            commits.add(row["commit"])
        require(len(commits) == 1, f"{run}: mixed revisions")


def check_catalog(catalog, records, previous=()):
    old = {(row.get("engine"), row.get("name")): row for row in previous}
    for row in catalog:
        engine, name = row.get("engine"), row.get("name")
        links = row.get("record_rows")
        if links is not None:
            require(isinstance(links, list) and links and all(isinstance(key, str) for key in links)
                    and len(set(links)) == len(links), f"{name}: expected distinct record row IDs")
            for key in links:
                require(key in records, f"{name}: unknown record {key!r}")
                require(engine in ("shared", "platform") or key.split("/")[0] in CATALOG_ALIASES.get(engine, {engine}),
                        f"{name}: record belongs to a different engine: {key}")
        exclusion = row.get("record_exclusion")
        require(exclusion is None or links is None and nonempty(exclusion),
                f"{name}: expected a reason for exclusion or record links")
        prior = old.get((engine, name))
        if prior and prior.get("record_rows"):
            require(links or exclusion, f"{name}: removed record coverage")
        gain = row.get("gain", "").strip().lower()
        default = row.get("default", "").strip().lower()
        if not gain or not default or re.match(r"^(off\b|0(?:\s|$)|opt[- ]in\b|explicit\b|disabled\b|superseded\b)", default):
            continue
        if gain.rstrip(".") in ("-", "none", "unmeasured", "not measured", "no isolated measurement",
                                "no throughput claim", "no measured gain", "pending qualification"):
            continue
        # Historical prose includes numeric default modes as well as ON / AUTO.
        # Review new or edited prose even when it has no recognizable timing units.
        changed = previous and (prior is None or any(row.get(field) != prior.get(field)
                                                     for field in ("default", "gain")))
        if changed or measured_claim(gain):
            require(links or exclusion, f"{name}: classify the default-on claim with record_rows or record_exclusion")


def measured_claim(gain):
    if re.match(r"^(no |none claimed|neutral|observability|measurement, not)", gain):
        return False
    return bool(re.search(r"\d.*(?:\bms\b|tok/s|%|×|\bx[0-9]|->|→)|\b(?:faster|speedup)\b", gain))


def check_history(ledger, previous):
    past = previous["acceptances"]
    require(ledger["acceptances"][:len(past)] == past, "regression acceptances are append-only")
    floor = {key: row["value"] for key, row in previous["records"].items()}
    for entry in ledger["acceptances"][len(past):]:
        key = entry["row"]
        require(key in floor and entry["previous"]["value"] >= floor[key], f"{key}: acceptance skips the prior floor")
        floor[key] = entry["replacement"]["value"]
    for key, value in floor.items():
        require(key in ledger["records"], f"{key}: record removed")
        require(ledger["records"][key]["value"] >= value, f"{key}: record lowered without an acceptance")


def view(ledger):
    columns = "| Row | tok/s | Prompt tokens | Generated tokens | Requests | Timing | Commit | Date | Receipt | Conditions |"
    lines = ["# Best on record", "", "Generated from [the ledger](best_on_record.json). See the [integration guide](README.md).",
             "", columns, "|---|---:|---:|---:|---:|---|---|---|---|---|"]
    for key, row in sorted(ledger["records"].items()):
        workload = row["workload"]

        def counts(field):
            value = workload[field]
            if isinstance(value, list):
                return "–".join(map(str, value))
            return "—" if value is None else str(value)

        commit = row["commit"] + (" (receipt)" if row.get("receipt_revision") else "")
        notes = escape(workload.get("notes", "")).replace("|", "&#124;")
        lines.append(f"| {key} | {row['value']:g} | {counts('prompt_tokens')} | {counts('generated_tokens')} | "
                     f"{workload['requests']} | {workload['timing']} | {commit} | {row['date']} | "
                     f"[Evidence](../../{row['receipt']}) | {notes} |")
    if ledger["acceptances"]:
        lines.extend(("", "## Accepted regressions", ""))
        for entry in ledger["acceptances"]:
            reason = escape(entry["reason"])
            lines.append(f"- {entry['row']}, {entry['previous']['value']:g} → {entry['replacement']['value']:g} tok/s. {reason}")
    return "\n".join(lines) + "\n"


def percent(value, reference):
    return (number(value) - number(reference)) * 100 / number(reference)


def workload_compatible(first, second, *, cross_route=False):
    prompt_timing = {"prefill", "ttft", "estimated"}
    both_prefill = first["timing"] in prompt_timing and second["timing"] in prompt_timing
    if (first["timing"] != second["timing"] and not (cross_route and both_prefill)
            or first["requests"] != second["requests"]):
        return False
    left, right = first["prompt_tokens"], second["prompt_tokens"]
    if cross_route:
        if type(left) is not int or type(right) is not int or abs(percent(left, right)) > 10:
            return False
    elif left != right:
        return False
    return both_prefill or first["generated_tokens"] == second["generated_tokens"]


def compare(ledger, campaign, tolerance, required):
    results, diagnostics = [], []
    for run, rows in sorted(campaign["runs"].items()):
        for key in sorted(required - rows.keys()):
            results.append(dict(run=run, row=key, status="MISSING"))
        for key, row in sorted(rows.items()):
            record = ledger["records"].get(key)
            result = dict(run=run, row=key, status="NO RECORD", measurement=row, record=record)
            if "failure" in row:
                result["status"] = "FAILED MEASUREMENT"
            elif record:
                change = percent(row["value"], record["value"])
                result.update(delta_pct=float(change),
                              status="ABOVE RECORD" if change > tolerance else
                              "BELOW RECORD" if change < -tolerance else "WITHIN NOISE")
                if row["workload"]["timing"] != record["workload"]["timing"]:
                    # Still surface the numeric gap. A different clock cannot replace the record.
                    result["timing_differs"] = True
                if any(row["workload"][field] != record["workload"][field]
                       for field in ("prompt_tokens", "generated_tokens")):
                    result["workload_differs"] = True
            results.append(result)
            if "value" not in row:
                continue
            engine, placement, tier, route, metric = key.split("/")
            index = TIERS.index(tier)
            peers = []
            if index:
                peers.append(("/".join((engine, placement, TIERS[index - 1], route, metric)),
                              0, "tier slower than the tier below"))
            if route == "served":
                peers.append(("/".join((engine, placement, tier, "cli", metric)), 2, "served more than 2% below the CLI"))
            for peer, threshold, label in peers:
                other = rows.get(peer, {})
                if "value" not in other or not workload_compatible(row["workload"], other["workload"],
                                                                    cross_route=route == "served" and threshold == 2):
                    continue
                change = percent(row["value"], other["value"])
                if change < -threshold:
                    diagnostics.append(dict(run=run, row=key, peer=peer, status=label, delta_pct=float(change),
                                            timing_differs=row["workload"]["timing"] != other["workload"]["timing"]))
    return results, diagnostics


def propose(ledger, results, reasons):
    accepted = {}
    for key, reason in reasons.items():
        require(nonempty(reason), f"{key}: acceptance needs a written reason")
        losses = [r for r in results if r["row"] == key and r["status"] == "BELOW RECORD"]
        require(len(losses) == 1 and not losses[0].get("timing_differs"),
                f"{key}: acceptance needs exactly one below-record measurement with matching timing")
        result = losses[0]
        accepted[key] = result["measurement"]
        ledger["acceptances"].append(dict(row=key, reason=reason, previous=result["record"],
                                           replacement=result["measurement"]))
        result["status"] = "ACCEPTED REGRESSION"
    updates = dict(accepted)
    for result in results:
        key = result["row"]
        if result["status"] in ("ABOVE RECORD", "NO RECORD") and not result.get("timing_differs") and key not in accepted:
            row = result["measurement"]
            if key not in updates or row["value"] > updates[key]["value"]:
                updates[key] = row
    ledger["records"].update(updates)


def atomic_write(path, content):
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
        try:
            tmp.write(content.encode("utf-8"))
            tmp.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


@contextmanager
def write_lock(path):
    lock = path.with_name(path.name + ".lock")
    with lock.open("x", encoding="utf-8"):
        pass
    try:
        yield
    finally:
        lock.unlink()


def base_file(revision, path, fallback):
    relative = path.relative_to(ROOT).as_posix()
    git = ["git", "-c", "safe.directory=*", "-C", str(ROOT)]
    if not subprocess.check_output([*git, "ls-tree", revision, "--", relative]):
        return fallback
    content = subprocess.check_output([*git, "show", f"{revision}:{relative}"], encoding="utf-8")
    return json.loads(content) if path == CATALOG else read_json(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path, nargs="?")
    parser.add_argument("--ledger", type=Path, default=RECORDS)
    parser.add_argument("--tolerance", type=Decimal, default=Decimal(3), help="noise tolerance in percent")
    parser.add_argument("--require", action="append", default=[], metavar="ROW", help="require this row in each run")
    parser.add_argument("--complete", action="store_true", help="require every recorded row in each run")
    parser.add_argument("--report-json", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--accept-regression", nargs=2, action="append", default=[], metavar=("ROW", "REASON"))
    parser.add_argument("--base", help="check the ledger ratchet and catalog coverage against this revision")
    args = parser.parse_args()
    try:
        with write_lock(args.ledger.resolve()) if args.write else nullcontext():
            return run(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"best on record: {error}", file=sys.stderr)
        return 2


def run(args):
    require(args.tolerance.is_finite() and 0 <= args.tolerance < 100, "tolerance must be in [0, 100)")
    reasons = dict(args.accept_regression)
    require(len(reasons) == len(args.accept_regression), "duplicate regression acceptance")
    require(not reasons or args.write and args.campaign, "acceptance requires --write and a campaign")
    require(args.campaign or not (args.require or args.complete or args.report_json), "report options need a campaign")
    ledger = load(args.ledger)
    validate_ledger(ledger)
    if args.ledger.resolve() == RECORDS:
        previous = base_file(args.base, CATALOG, []) if args.base else []
        check_catalog(load(CATALOG), ledger["records"], previous)
    if args.base:
        require(args.ledger.resolve() == RECORDS, "--base requires the repository ledger")
        prior = base_file(args.base, RECORDS, None)
        if prior is not None:
            check_history(ledger, prior)

    if args.campaign:
        required = set(args.require)
        for key in required:
            valid_key(key)
        if args.complete:
            required.update(ledger["records"])
        campaign = load(args.campaign)
        validate_campaign(campaign)
        results, diagnostics = compare(ledger, campaign, args.tolerance, required)
        if args.write:
            propose(ledger, results, reasons)
        failed = any(r["status"] in ("BELOW RECORD", "FAILED MEASUREMENT", "MISSING") for r in results)
        if args.report_json:
            print(json.dumps(dict(tolerance_pct=float(args.tolerance), results=results, diagnostics=diagnostics)))
        else:
            print(f"Record comparison tolerance: {args.tolerance}%")
            for r in results:
                detail = f" {r['delta_pct']:+.2f}%" if "delta_pct" in r else ""
                if "measurement" in r:
                    row = r["measurement"]
                    if "value" in row:
                        baseline = f"{r['record']['value']:g} -> " if r["record"] else ""
                        detail += f" ({baseline}{row['value']:g} tok/s)"
                    else:
                        detail += f" ({row['failure']})"
                if r.get("timing_differs"):
                    detail += " [different timing]"
                if r.get("workload_differs"):
                    detail += " [different token counts]"
                print(f"{r['status']}\t{r['run']}\t{r['row']}{detail}")
            for d in diagnostics:
                detail = " [different timing]" if d["timing_differs"] else ""
                print(f"{d['status']}\t{d['run']}\t{d['row']}\t{d['peer']} {d['delta_pct']:+.2f}%{detail}")
        if failed:
            return 1

    view_path = args.ledger.with_suffix(".md")
    if args.write:
        validate_ledger(ledger)
        if args.campaign:
            ledger["records"] = dict(sorted(ledger["records"].items()))
            atomic_write(args.ledger, json.dumps(ledger, indent=2, ensure_ascii=False) + "\n")
        atomic_write(view_path, view(ledger))
    elif not args.campaign:
        require(view_path.read_text(encoding="utf-8") == view(ledger), "stale ledger view; regenerate with --write")
        print(f"Validated {len(ledger['records'])} records and their catalog links.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
