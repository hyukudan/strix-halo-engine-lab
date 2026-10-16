"""Collect existing CTest JUnit evidence; never run tests, builds or a decoder.
A caller supplies the exact predeclared test names. Missing, extra and skipped
cases block the suite gate. This receipt does not qualify hardware or a model.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

MAX_XML_BYTES = 32 * 1024 * 1024
MAX_NAMES_BYTES = 2 * 1024 * 1024
MAX_CASES = 100000


def read_bounded(path, maximum):
    path = Path(path)
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    if not data or len(data) > maximum:
        raise ValueError("evidence is empty or exceeds its byte bound")
    return data, {"path": str(path.resolve()), "bytes": len(data),
                  "sha256": hashlib.sha256(data).hexdigest()}


def name(value):
    if not isinstance(value, str) or not value or len(value) > 4096 or "\0" in value:
        raise ValueError("invalid test name")
    return value


def declared_count(suite, key):
    value = suite.get(key)
    if value is None:
        return None
    if not value.isascii() or not value.isdigit() or len(value) > 12:
        raise ValueError("invalid JUnit declared count")
    return int(value)


def parse_case(case):
    identifier = name(case.get("name"))
    status = case.get("status", "run")
    if status not in ("run", "notrun", "disabled", "passed", "failed", "skipped"):
        raise ValueError("unknown JUnit case status")
    children = list(case)
    failure = any(child.tag == "failure" for child in children)
    error = any(child.tag == "error" for child in children)
    skipped = any(child.tag == "skipped" for child in children) or status in ("notrun", "disabled", "skipped")
    if skipped and (failure or error or status == "failed"):
        raise ValueError("JUnit case claims both failed and skipped")
    duration = case.get("time")
    if duration is not None:
        try:
            duration = float(duration)
        except ValueError as exc:
            raise ValueError("invalid JUnit case duration") from exc
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("invalid JUnit case duration")
    outcome = "skipped" if skipped else "failed" if (failure or error or status == "failed") else "passed"
    diagnostics = []
    for child in children:
        if child.tag in ("failure", "error", "skipped"):
            diagnostics.append({"kind": child.tag, "message": child.get("message", "")[:4096],
                                "text": "".join(child.itertext())[:4096]})
    return {"name": identifier, "outcome": outcome, "seconds_reported": duration,
            "diagnostics": diagnostics}, failure or (status == "failed" and not error), error


def collect(expected_path, junit_path, stage):
    if stage not in ("host", "native", "model"):
        raise ValueError("unknown declared stage")
    names_raw, names_pin = read_bounded(expected_path, MAX_NAMES_BYTES)
    expected = json.loads(names_raw)
    if not isinstance(expected, list) or not expected or len(expected) > MAX_CASES:
        raise ValueError("expected names must be a bounded nonempty array")
    expected = [name(value) for value in expected]
    if len(set(expected)) != len(expected):
        raise ValueError("duplicate expected test name")
    raw, xml_pin = read_bounded(junit_path, MAX_XML_BYTES)
    xml_text = raw.decode("utf-8-sig")
    if "\0" in xml_text or "<!DOCTYPE" in xml_text.upper() or "<!ENTITY" in xml_text.upper():
        raise ValueError("only UTF-8 XML without DTD/entity declarations is supported")
    root = ET.fromstring(xml_text)
    if root.tag not in ("testsuite", "testsuites"):
        raise ValueError("unsupported JUnit root")
    if sum(element.tag in ("testsuite", "testsuites") for element in root.iter()) > 256:
        raise ValueError("too many nested suites")
    pending = [(root, 0, False)]
    while pending:
        element, depth, in_case = pending.pop()
        if depth > 32 or (in_case and element.tag == "testcase"):
            raise ValueError("nested testcase or excessive XML depth")
        for child in element:
            if child.tag == "testcase" and element.tag != "testsuite":
                raise ValueError("testcase must belong directly to a suite")
            pending.append((child, depth + 1, in_case or element.tag == "testcase"))
    cases = list(root.iter("testcase"))
    if len(cases) > MAX_CASES:
        raise ValueError("too many reported cases")
    parsed = {}
    for case in cases:
        result, failure, error = parse_case(case)
        if result["name"] in parsed:
            raise ValueError("duplicate reported test name")
        parsed[result["name"]] = (result, failure, error)
    # Suite counts are cross-checks, never a substitute for individual results.
    for suite in root.iter():
        if suite.tag not in ("testsuite", "testsuites"):
            continue
        entries = [parsed[name(case.get("name"))] for case in suite.iter("testcase")]
        for key, actual in (("tests", len(entries)), ("failures", sum(bool(row[1]) for row in entries)),
                            ("errors", sum(bool(row[2]) for row in entries))):
            count = declared_count(suite, key)
            if count is not None and count != actual:
                raise ValueError("JUnit declared counts disagree with individual cases")
        skip = declared_count(suite, "skipped")
        disabled = declared_count(suite, "disabled")
        if skip is not None or disabled is not None:
            if (skip or 0) + (disabled or 0) != sum(row[0]["outcome"] == "skipped" for row in entries):
                raise ValueError("JUnit omitted/skipped counts disagree with individual cases")
    actual_names = set(parsed)
    missing, extra = sorted(set(expected) - actual_names), sorted(actual_names - set(expected))
    results = [row[0] for row in parsed.values()]
    counts = {kind: sum(row["outcome"] == kind for row in results) for kind in ("passed", "failed", "skipped")}
    coverage = not missing and not extra
    gate = coverage and counts["passed"] == len(expected)
    return {"schema_version": 1, "evidence_kind": "existing-ctest-junit-only", "stage_declared": stage,
            "inputs": {"expected_names": names_pin, "junit": xml_pin},
            "expected_tests": len(expected), "reported_tests": len(results), "counts": counts,
            "missing": missing, "extra": extra, "complete_coverage": coverage,
            "all_requested_tests_passed": gate, "native_execution_observed": None, "qualification": None,
            "results": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-names", required=True)
    parser.add_argument("--junit", required=True)
    parser.add_argument("--stage", required=True, choices=("host", "native", "model"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = collect(args.expected_names, args.junit, args.stage)
    # Never overwrite source evidence or a previous receipt.
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if receipt["all_requested_tests_passed"] else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, ET.ParseError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
