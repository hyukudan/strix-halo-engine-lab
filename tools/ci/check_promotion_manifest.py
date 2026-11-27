"""Validate versioned promotion evidence and generate its profile table."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/profile/promotion-receipts.json"
HEADER = ROOT / "include/strix/profile/promotion_receipts.inc"
LEGACY = ROOT / "tools/ci/profile_legacy_receipts.json"
SHA = r"[0-9a-f]{64}"
IDENTITY = r"[^\s{},;|=]+"
SCOPE = re.compile(r"ladder-arm/v4;exec/v1 backend=(hip|cuda) device=([^;]+?) prompt=(\d+)"
                   r" context=([1-9]\d*) lanes=(\d+) chunk=([1-9]\d*) prefill_rows=\d+ decode_rows=\d+"
                   r" verify_rows=\d+ generated=\d+ settings=([^;]+);([^;]+);candidate=([^;]+);tier=(fast|turbo)")
LEGACY_FORMS = {
    "band": re.compile(r"\.band_bound\s*=\s*true\b"),
    "incumbent": re.compile(r"\bincumbent_cell\("),
    "legacy-reference": re.compile(r"\.over\s*=(?!=)"),
    "window-band": re.compile(r"\bband_row\("),
    "unversioned-table-row": None,
}
GATE_FIELDS = ("plan_sha256", "witness_sha256", "reference_binary_sha256", "candidate_binary_sha256",
               "reference_pack_sha256", "candidate_pack_sha256", "reference_config_sha256", "candidate_config_sha256")
ROW_FIELDS = {"id", "source", "lever", "value", "identity", "tier", "phase", "device", "contexts", "lanes",
              "reference", "noise", "revision", "gate", "timing"}
TABLE = re.compile(r"\bLadderRow\s+\w+\s*\[[^]]*\]\s*=\s*\{")
BINDING = re.compile(r'\bpromotion_band\s*\(\s*"([a-z0-9-]+)"\s*\)')
LITERAL = re.compile(r"""R"([^ ()\\]{0,16})\(.*?\)\1"|/\*.*?\*/|//[^\n]*|"(?:[^"\\]|\\.)*"|(?<!\w)'(?:[^'\\]|\\.)+'""", re.S)


def check(valid, message):
    if not valid:
        raise ValueError(message)


def load(path, payload=None):
    def unique(pairs):
        doc = dict(pairs)
        check(len(doc) == len(pairs), f"{path}: duplicate JSON field")
        return doc
    data = path.read_bytes() if payload is None else payload
    return json.loads(data, object_pairs_hook=unique)


def sha(value):
    return isinstance(value, str) and re.fullmatch(SHA, value)


def tag(value, pattern):
    return isinstance(value, str) and re.fullmatch(pattern, value)


def within(value, lo, hi):
    return type(value) in (int, float) and math.isfinite(value) and lo <= value <= hi


def band(value, max_value):
    return (isinstance(value, list) and len(value) == 2 and all(type(x) is int for x in value)
            and 1 <= value[0] <= value[1] <= max_value)


def settings_fields(text):
    result = {}
    for field in re.split(r"[|,]", text):
        name, sep, value = field.partition("=")
        if sep and name in {"pack", "route", "ranks"}:
            check(name not in result and value, f"ambiguous captured {name}")
            result[name] = value
    return result


def validate(entry):
    check(isinstance(entry, dict) and set(entry) == ROW_FIELDS, "unexpected promotion entry fields")
    key = entry["id"]
    check(tag(key, r"[a-z0-9]+(?:-[a-z0-9]+)*"), "invalid promotion ID")
    check(tag(entry["source"], r"(?:include/strix|src|apps)/[\w./-]+\.(?:hpp|cpp|inl|inc|cu|hip)")
          and ".." not in entry["source"].split("/"), f"{key}: invalid source")
    check(all(tag(entry[k], IDENTITY) for k in ("lever", "value", "identity", "device")), f"{key}: invalid row identity")
    check(entry["tier"] in ("fast", "turbo") and entry["phase"] in ("prefill", "decode"), f"{key}: invalid tier or phase")
    check(tag(entry["revision"], r"[0-9a-f]{40}"), f"{key}: invalid source revision")
    contexts, lanes = entry["contexts"], entry["lanes"]
    check(band(contexts, 2**64 - 1) and band(lanes, 2**32 - 1), f"{key}: invalid band")
    noise = entry["noise"]
    check(within(noise, 0, 1) and noise < 1, f"{key}: invalid timing noise")
    reference = entry["reference"]
    match = SCOPE.fullmatch(reference) if isinstance(reference, str) else None
    check(match, f"{key}: incomplete execution reference")
    backend, product, prompt, _, lane, _, settings, denominator, candidate, tier = match.groups()
    check(tag(product, re.escape(entry["device"]) + rf"(?::[^|]+)?\|[^|]+\|cu[1-9]\d*\|mem[1-9]\d*\|clk[1-9]\d*\|{backend}\d+\|drv\d+"),
          f"{key}: expected a full device product identity")
    arm_token = rf"({IDENTITY})=(?:{IDENTITY})?{{{IDENTITY}}}"
    levers = [re.fullmatch(arm_token, token) for token in denominator.split(",")]
    check(all(levers), f"{key}: malformed denominator lever chain")
    names = [item[1] for item in levers]
    check(len(names) == len(set(names)) and entry["lever"] in names, f"{key}: missing or repeated denominator lever")
    check(candidate == f"{entry['lever']}={entry['value']}{{{entry['identity']}}}" and tier == entry["tier"],
          f"{key}: reference candidate or tier differs from the row")
    check(contexts[0] <= int(prompt) <= contexts[1] and lanes[0] <= int(lane) <= lanes[1],
          f"{key}: captured reference is outside its band")
    execution = settings_fields(settings)
    check(set(execution) == {"pack", "route", "ranks"} and tag(execution["ranks"], r"[1-9]\d*"),
          f"{key}: captured settings require pack, route and rank count")
    gate = entry["gate"]
    check(isinstance(gate, dict) and set(gate) == {"receipt_sha256", *GATE_FIELDS} and all(map(sha, gate.values())),
          f"{key}: requires hashed v2 promotion evidence")
    check(gate["reference_binary_sha256"] == gate["candidate_binary_sha256"], f"{key}: gate must use the measured binary")
    check(execution["pack"] == gate["candidate_pack_sha256"], f"{key}: gate pack differs from captured pack")
    if tier == "fast":
        check(gate["reference_pack_sha256"] == gate["candidate_pack_sha256"], f"{key}: FAST must compare the same pack")
    timing = entry["timing"]
    check(isinstance(timing, dict) and set(timing) == {"receipt_sha256", "samples"} and sha(timing["receipt_sha256"])
          and isinstance(timing["samples"], list), f"{key}: requires paired timing evidence")
    measured = {}
    for sample in timing["samples"]:
        check(isinstance(sample, dict) and set(sample) == {"prompt", "lanes", "abba_ms"}, f"{key}: invalid timing point")
        point = sample["prompt"], sample["lanes"]
        check(all(type(n) is int for n in point) and contexts[0] <= point[0] <= contexts[1]
              and lanes[0] <= point[1] <= lanes[1] and point not in measured, f"{key}: repeated or out-of-band timing")
        durations = sample["abba_ms"]
        check(isinstance(durations, list) and len(durations) == 4 and all(within(x, 1e-9, 1e12) for x in durations),
              f"{key}: invalid ABBA duration")
        a, b, c, d = durations
        measured[point] = (a + d) / (b + c)
    required = {(p, n) for p in contexts for n in lanes} | {(int(prompt), int(lane))}
    check(required <= measured.keys() and min(measured.values()) > 1 + noise,
          f"{key}: gain beyond noise is required at both band ends and the recorded execution")
    return min(measured.values())


def seal(draft, gate_path, timing_path):
    gate_bytes, timing_bytes = gate_path.read_bytes(), timing_path.read_bytes()
    gate, timing = load(gate_path, gate_bytes), load(timing_path, timing_bytes)
    proof, pooled = gate.get("promotion", {}), gate.get("pooled", {})
    check(gate.get("schema") == "strix-fast-gate-v2/v1" and proof.get("schema") == "strix-gate-promotion/v1"
          and all(sha(proof.get(field)) for field in GATE_FIELDS)
          and gate.get("tier", "fast") == draft["tier"] and pooled.get("state") == "PASS"
          and pooled.get("pass") is True and pooled.get("frozen_qualification", {}).get("sha256") == proof["plan_sha256"],
          "gate did not issue a matching held-out promotion PASS")
    check(timing.get("schema") == "strix-profile-abba/v1"
          and timing["candidate_pack_sha256"] == proof["candidate_pack_sha256"]
          and timing["binary_sha256"] == proof["candidate_binary_sha256"]
          and timing["reference_config_sha256"] == proof["reference_config_sha256"]
          and timing["candidate_config_sha256"] == proof["candidate_config_sha256"]
          and timing["reference"] == draft["reference"] and timing["revision"] == draft["revision"],
          "timing capture differs from the draft or the quality evidence")
    entry = {**draft, "gate": {"receipt_sha256": hashlib.sha256(gate_bytes).hexdigest(),
                               **{field: proof[field] for field in GATE_FIELDS}},
             "timing": {"receipt_sha256": hashlib.sha256(timing_bytes).hexdigest(), "samples": timing["samples"]}}
    validate(entry)
    return entry


def header(entries):
    lines = ["// Generated from the versioned promotion manifest.",
             "[[nodiscard]] consteval LadderRow promotion_band(std::string_view id) {"]
    for entry in entries:
        speedup = validate(entry)
        tier, phase = entry["tier"], entry["phase"]
        a, b = entry["contexts"]
        x, y = entry["lanes"]
        literal = json.dumps
        lines.extend([
            f'    if (id == {literal(entry["id"])}) return {{',
            f'        .lever = {literal(entry["lever"])}, .value = {literal(entry["value"])},',
            f'        .tier = Name::{tier}, .phase = Phase::{phase}, .device = {literal(entry["device"])},',
            f'        .min_context = {a}ULL, .max_context = {b}ULL, .min_lanes = {x}U, .max_lanes = {y}U,',
            f'        .speedup = {speedup!r}, .noise = {entry["noise"]!r}, .receipt = id,',
            f'        .over = {literal(entry["reference"])},',
            f'        .identity = {literal(entry["identity"])}, .band_bound = true,',
            f'        .band_evidence = {{{{{a}ULL, {b}ULL}}, {{{a}ULL, {b}ULL}}, id}}, .promotion = id}};'])
    lines.extend(['    if (id.empty()) return {};', '    throw id;', '}', ''])
    return "\n".join(lines)


def unversioned_table_rows(code):
    count = 0
    for match in TABLE.finditer(code):
        depth, start = 0, match.end()
        for i in range(start, len(code)):
            token = code[i]
            if token == "}" and depth == 0 or token == "," and depth == 0:
                row = code[start:i].strip()
                count += bool(row and not row.startswith("promotion_band"))
                start = i + 1
                if token == "}":
                    break
            elif token in "{([":
                depth += 1
            elif token in "})]":
                depth -= 1
    return count


def source_inventory():
    bindings, historical = {}, {}
    for tree in ("include/strix", "src", "apps"):
        for path in sorted((ROOT / tree).rglob("*")):
            if path.suffix not in (".hpp", ".cpp", ".inc", ".inl", ".hip", ".cu"):
                continue
            if path in (HEADER, ROOT / "include/strix/engine_profile.hpp"):
                continue
            source = path.read_text(encoding="utf-8", errors="surrogateescape")
            name = path.relative_to(ROOT).as_posix()
            without_comments = LITERAL.sub(lambda m: m[0] if m[0].startswith('"') else " ", source)
            for key in BINDING.findall(without_comments):
                check(key not in bindings, f"{key}: promotion registered more than once")
                bindings[key] = name
            code = LITERAL.sub(" ", source)
            check(len(BINDING.findall(without_comments)) == len(re.findall(r"\bpromotion_band\s*\(", code))
                  and not re.search(r"\.promotion\s*=(?!=)", code), f"{name}: use a literal promotion manifest ID")
            counts = {key: len(pattern.findall(code)) for key, pattern in LEGACY_FORMS.items() if pattern is not None}
            counts["unversioned-table-row"] = unversioned_table_rows(code)
            if any(counts.values()):
                historical[name] = counts
    return bindings, historical


def validate_sources(entries, bindings, historical, limits):
    check(bindings == {row["id"]: row["source"] for row in entries}, "promotion manifest entries and source bindings differ")
    for source, counts in historical.items():
        for key, count in counts.items():
            check(count <= limits.get(source, {}).get(key, 0),
                  f"{source}: new {key} receipt requires a versioned promotion entry")


def validate_legacy_limits(limits, previous):
    for source, counts in limits.items():
        for key, count in counts.items():
            check(key in LEGACY_FORMS and type(count) is int and 0 <= count <= previous.get(source, {}).get(key, 0),
                  f"{source}: historical {key} receipt allowance may only shrink")


def base_legacy_limits(revision):
    git = ["git", "-c", "safe.directory=*", "-C", str(ROOT)]
    subprocess.run([*git, "rev-parse", "--verify", revision], check=True, capture_output=True)
    path = LEGACY.relative_to(ROOT).as_posix()
    base = subprocess.run([*git, "show", f"{revision}:{path}"], capture_output=True)
    if base.returncode == 0:
        return load(LEGACY, base.stdout)
    # The first integration creates the grandfathered inventory.
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--write-header", action="store_true")
    modes.add_argument("--seal", nargs=3, type=Path, metavar=("DRAFT", "GATE", "TIMING"))
    parser.add_argument("--base", help="revision for the shrink-only historical allowance")
    args = parser.parse_args()
    try:
        if args.seal:
            draft, gate, timing = args.seal
            print(json.dumps(seal(load(draft), gate, timing), sort_keys=True, indent=2))
            return
        doc = load(MANIFEST)
        check(isinstance(doc, dict) and set(doc) == {"schema", "promotions"}
              and doc["schema"] == "strix-profile-promotion-manifest/v1"
              and isinstance(doc["promotions"], list), "invalid promotion manifest schema")
        entries = doc["promotions"]
        generated = header(entries)
        keys = [row["id"] for row in entries]
        check(keys == sorted(set(keys)), "promotion IDs must be unique and sorted")
        bindings, historical = source_inventory()
        limits = load(LEGACY)
        if args.base:
            previous = base_legacy_limits(args.base)
            if previous is not None:
                validate_legacy_limits(limits, previous)
        validate_sources(entries, bindings, historical, limits)
        if args.write_header:
            HEADER.write_text(generated, encoding="utf-8", newline="\n")
        else:
            check(HEADER.read_text(encoding="utf-8") == generated, "generated promotion table differs from manifest")
        print(f"profile promotion manifest: {len(entries)} qualified bands")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"profile promotion manifest: {error}\n")


if __name__ == "__main__":
    main()
