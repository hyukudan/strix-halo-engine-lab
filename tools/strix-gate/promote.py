#!/usr/bin/env python3
"""Tier promotion of one ladder band in two commands: the reference stage and the candidate stage.

    promote.py band SPEC.json --stage reference   capture E, N (and NF) -> freeze the plan; publish its SHA-256
    promote.py band SPEC.json --stage candidate   capture F (and FF) -> witness -> gate v2 --promotion -> ABBA ->
                                                  draft -> sealed manifest entry (data/profile/promotion-receipts.json)

Each step is also a subcommand (capture, plan, witness, gate, abba, draft, seal). The band spec is data (schema
strix-promotion-band/v1): the band (id, source, lever, tier, phase, device class, contexts, lanes, noise, revision),
the capture (format, cases, classes, arm names, one command per arm), the execution identities (binary, pack, the
numerical configuration of the reference and the candidate, the null controls) and the timing commands. No engine
knowledge lives here: an engine reaches the band through the commands of its spec and the "promotion_reference" its
profile receipt reports (profile::LadderArms::promotion_over()).

The plan is frozen from the reference-only captures before any candidate capture exists (the plan step refuses
otherwise): window IDs, content classes, the per-window reference mean NLL per class and the null envelope (worst
control of every window, replayed and free-running). The witness binds every executed arm of every case to the plan,
the consumed logit bytes, the binary, the pack content identity and the configuration. The ABBA samples cover both
band ends of every lane bound and the recorded execution (check_promotion_manifest.validate()).
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import shlex
import struct
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (HERE, ROOT / "tools" / "ci"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import gate_v2 as v2  # noqa: E402
import check_promotion_manifest as cpm  # noqa: E402

SPEC_SCHEMA = "strix-promotion-band/v1"
PLAN_SCHEMA = "strix-gate-frozen-qualification/v1"
WITNESS_SCHEMA = "strix-gate-held-out-witness/v1"
ABBA_SCHEMA = "strix-profile-abba/v1"
HEX64 = re.compile(r"[0-9a-f]{64}")
CANDIDATE = re.compile(r";candidate=([^=;{}]+)=([^;{}]*)\{([^;{}]+)\};tier=(fast|turbo)$")


class PromotionError(RuntimeError):
    pass


def check(valid, message):
    if not valid:
        raise PromotionError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def pack_files(pack):
    """strix::pack_content_files(): a file by its name; a directory by every regular file's relative generic path,
    skipping any path component that starts with '.'."""
    pack = pathlib.Path(pack)
    if pack.is_file():
        return [(pack.name, pack)]
    check(pack.is_dir(), f"pack is neither a file nor a directory: {pack}")
    files = []
    for dirpath, dirnames, filenames in os.walk(pack, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in filenames:
            path = pathlib.Path(dirpath) / name
            if not name.startswith(".") and path.is_file():
                files.append((path.relative_to(pack).as_posix(), path))
    check(files, f"pack directory holds no file: {pack}")
    return sorted(files)


def pack_identity(pack):
    """strix::pack_content_identity(): STRIX-MODEL-CONTENT-V1 over the pack files (full read)."""
    h = hashlib.sha256(b"STRIX-MODEL-CONTENT-V1")
    for name, path in pack_files(pack):
        encoded = name.encode("utf-8")
        h.update(struct.pack("<Q", len(encoded)) + encoded + struct.pack("<Q", path.stat().st_size))
        h.update(bytes.fromhex(sha256_file(path)))
    return h.hexdigest()


def write_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=1, sort_keys=True) + "\n").encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return hashlib.sha256(data).hexdigest()


class Band:
    """The band spec with its derived paths and readers."""

    def __init__(self, path):
        self.path = pathlib.Path(path)
        spec = json.loads(self.path.read_text(encoding="utf-8"))
        check(isinstance(spec, dict) and spec.get("schema") == SPEC_SCHEMA, f"{path}: not a {SPEC_SCHEMA} spec")
        self.spec = spec
        for key in ("id", "source", "lever", "tier", "phase", "device", "contexts", "lanes", "noise", "revision",
                    "out", "capture", "execution"):
            check(key in spec, f"{path}: missing {key}")
        check(spec["tier"] in ("fast", "turbo"), "tier must be fast or turbo")
        check(re.fullmatch(r"[0-9a-f]{40}", spec["revision"]), "revision must be a full 40-hex commit")
        base = self.path.parent
        self.out = pathlib.Path(spec["out"]) if pathlib.Path(spec["out"]).is_absolute() else base / spec["out"]
        cap = spec["capture"]
        arms = cap["arms"]
        self.ref, self.cand, self.null = arms["ref"], arms["cand"], list(arms["null"])
        self.null_free, self.cand_free = arms.get("null_free"), arms.get("cand_free")
        check(bool(self.null_free) == bool(self.cand_free), "free-running arms come in pairs (cand_free, null_free)")
        names = [self.ref, self.cand, *self.null, *(a for a in (self.cand_free, self.null_free) if a)]
        check(len(names) == len(set(names)) and all(names), "arm names must be distinct and non-empty")
        self.format = cap.get("format", "steps")
        self.head_rows = int(cap.get("head_rows", 64))
        self.manifest_path = self._rel(cap.get("manifest"))
        self.manifest = v2.read_manifest(self.manifest_path) if self.manifest_path else None
        check(self.format != "head" or self.manifest is not None, "head captures need a manifest")
        self.pattern = cap.get("logits") or ("{case}-{arm}.f32" if self.format == "head" else "logits/{case}.{arm}.bin")
        self.cases = list(cap["cases"])
        check(self.cases and len(self.cases) == len(set(self.cases)), "cases must be listed once each")
        classes = cap["classes"]
        self.classes = json.loads(self._rel(classes).read_text(encoding="utf-8")) if isinstance(classes, str) else classes
        check(isinstance(self.classes, dict) and all(isinstance(self.classes.get(c), str) and self.classes[c].strip()
                                                     for c in self.cases), "every case needs a content class")
        self.documents = cap.get("documents") or {}
        self.commands = cap.get("commands", {})
        self.plan_path, self.witness_path = self.out / "promotion" / "plan.json", self.out / "promotion" / "witness.json"
        self.gate_path, self.abba_path = self.out / "promotion" / "gate.json", self.out / "promotion" / "abba.json"
        self.draft_path, self.entry_path = self.out / "promotion" / "draft.json", self.out / "promotion" / "entry.json"
        self.reduce_pair = v2._option_qual_reduce()

    def _rel(self, value):
        if not value:
            return None
        p = pathlib.Path(value)
        return p if p.is_absolute() else self.path.parent / p

    def logit(self, case, arm):
        return self.out / self.pattern.format(case=case, arm=arm)

    def reduce(self, case, arm):
        kw = ({"ids": self.manifest[case], "rows": self.head_rows, "include_unscored": True}
              if self.format == "head" else {})
        return self.reduce_pair(self.logit(case, self.ref), self.logit(case, arm), self.format, **kw)

    def labels_sha(self, case, rows):
        return v2.ids_digest(self.manifest[case] if self.format == "head" else rows["label"])

    def reference_arms(self):
        return [self.ref, *self.null, *([self.null_free] if self.null_free else [])]

    def candidate_arms(self):
        return [self.cand, *([self.cand_free] if self.cand_free else [])]


def capture(band, arms):
    """Run the spec's command of every arm for every case (stderr to OUT/log/CASE.ARM.stderr, the gate reads it)."""
    (band.out / "log").mkdir(parents=True, exist_ok=True)
    for arm in arms:
        command = band.commands.get(arm)
        check(isinstance(command, str) and command.strip(), f"no capture command for arm {arm}")
        for case in band.cases:
            line = command.format(case=case, arm=arm, out=shlex.quote(str(band.out)))
            err = band.out / "log" / f"{case}.{arm}.stderr"
            with open(err, "wb") as stderr:
                rc = subprocess.run(line, shell=True, stderr=stderr, check=False).returncode
            check(rc == 0, f"{case}.{arm}: capture command failed rc {rc} (stderr {err})")
            check(band.logit(case, arm).is_file(), f"{case}.{arm}: the capture wrote no logits")


def plan(band, headroom=1.0):
    """Freeze the plan from the reference-only arms; refuses once any candidate capture exists."""
    check(headroom >= 1.0, "the envelope headroom cannot tighten the observed controls")
    for case in band.cases:
        for arm in band.candidate_arms():
            check(not band.logit(case, arm).exists(), f"{case}.{arm}: candidate captured before the plan was frozen")
    windows, nll, worst = {}, {}, {"max_kl": 0.0, "min_top1": 1.0, "max_abs_dnll": 0.0}
    for case in band.cases:
        labels = None
        for arm in [*band.null, *([band.null_free] if band.null_free else [])]:
            rows, _ = band.reduce(case, arm)
            check(not rows["nonfinite_r"].any() and not rows["nonfinite_c"].any(), f"{case}.{arm}: non-finite logits")
            labels = labels or band.labels_sha(case, rows)
            check(band.labels_sha(case, rows) == labels, f"{case}: the reference labels differ between controls")
            worst["max_kl"] = max(worst["max_kl"], float(rows["kl_rc"].mean()))
            worst["min_top1"] = min(worst["min_top1"], float(rows["agree"].mean()))
            worst["max_abs_dnll"] = max(worst["max_abs_dnll"], abs(float((rows["nll_c"] - rows["nll_r"]).mean())))
            mean_nll = float(rows["nll_r"].mean())
        cls = band.classes[case]
        windows[case] = {"class": cls, "ids_sha256": labels}
        if case in band.documents:
            windows[case]["document"] = band.documents[case]
        nll.setdefault(cls, []).append(mean_nll)
    for cls, values in nll.items():
        check(len(values) >= v2.R4_CLASS_MIN, f"class {cls}: {len(values)} windows, the plan needs "
              f"{v2.R4_CLASS_MIN} or more per class")
    envelope = {"max_kl": worst["max_kl"] * headroom, "min_top1": max(0.0, 1.0 - (1.0 - worst["min_top1"]) * headroom),
                "max_abs_dnll": worst["max_abs_dnll"] * headroom}
    doc = {"schema": PLAN_SCHEMA, "band": band.spec["id"], "windows": windows, "reference_nll": nll,
           "null_envelope": envelope}
    return write_json(band.plan_path, doc)


def arm_control(band, case, arm, control):
    """The order-only control a null arm applied: declared in the spec, or "stderr" (its strix-route-file/v1 receipt)."""
    if control == "stderr":
        receipt = v2.route_receipt((band.out / "log" / f"{case}.{arm}.stderr").read_bytes(), f"{case}.{arm}")
        control = receipt.get("null", {})
        control = {"order": control.get("order"), "applications": control.get("applications")}
    check(isinstance(control, dict) and isinstance(control.get("order"), str) and control["order"].strip()
          and type(control.get("applications")) is int and control["applications"] > 0,
          f"{case}.{arm}: the null control needs an order and a positive application count")
    return control


def execution_identities(band):
    ex = band.spec["execution"]
    binary = ex.get("binary_sha256") or sha256_file(band._rel(ex["binary"]))
    pack = ex.get("pack_sha256") or pack_identity(band._rel(ex["pack"]))
    check(HEX64.fullmatch(binary) and HEX64.fullmatch(pack), "binary and pack identities must be SHA-256")
    configs = ex["configs"]
    return binary, pack, canonical_sha(configs["reference"]), canonical_sha(configs["candidate"])


def witness(band):
    plan_bytes = band.plan_path.read_bytes()
    frozen = json.loads(plan_bytes)["windows"]
    binary, pack, ref_config, cand_config = execution_identities(band)
    controls = band.spec["execution"].get("controls", {})
    cases = {}
    for case in band.cases:
        labels = frozen[case]["ids_sha256"]
        records = {}
        for arm in [band.ref, *band.null, band.cand, *(a for a in (band.cand_free, band.null_free) if a)]:
            _, info = band.reduce(case, arm)
            candidate_side = arm in band.candidate_arms()
            record = {"binary_sha256": binary, "model_sha256": pack,
                      "numerical_config_sha256": cand_config if candidate_side else ref_config,
                      "logits_sha256": info["ref_sha256"] if arm == band.ref else info["cand_sha256"],
                      "ids_sha256": labels}
            if band.format == "head":
                record["head_ids_sha256"] = labels
            if arm == band.ref:
                record["control"] = None
            elif arm in band.null:
                record["control"] = arm_control(band, case, arm, controls.get(arm))
            records[arm] = record
        cases[case] = records
    doc = {"schema": WITNESS_SCHEMA, "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(), "cases": cases}
    return write_json(band.witness_path, doc)


def gate(band, extra=()):
    plan_sha = sha256_file(band.plan_path)
    args = ["qual", str(band.out), "--ref", band.ref, "--cand", band.cand, "--format", band.format,
            "--logits", band.pattern, "--cases", *band.cases, "--tier", band.spec["tier"],
            "--promotion", "--frozen-qualification", str(band.plan_path), plan_sha,
            "--held-out-witness", str(band.witness_path), "--json", str(band.gate_path)]
    for arm in band.null:
        args += ["--null", arm]
    if band.cand_free:
        args += ["--cand-free", band.cand_free, "--null-free", band.null_free]
    if band.manifest_path:
        args += ["--manifest", str(band.manifest_path), "--head-rows", str(band.head_rows)]
    rc = v2.main(args + list(extra))
    receipt = json.loads(band.gate_path.read_text(encoding="utf-8")) if band.gate_path.exists() else {}
    state = receipt.get("pooled", {}).get("state", "NO RECEIPT")
    return rc, state


def required_samples(band, recorded=None):
    contexts, lanes = band.spec["contexts"], band.spec["lanes"]
    points = {(p, n) for p in contexts for n in lanes}
    if recorded:
        points.add(recorded)
    return sorted(points)


def recorded_point(reference):
    m = re.search(r" prompt=(\d+) context=\d+ lanes=(\d+) ", reference)
    check(m, "the promotion reference has no prompt / lanes")
    return int(m[1]), int(m[2])


def time_once(command, prompt, lanes, pattern):
    started = time.perf_counter()
    done = subprocess.run(command.format(prompt=prompt, lanes=lanes), shell=True, capture_output=True, check=False)
    wall_ms = (time.perf_counter() - started) * 1000.0
    check(done.returncode == 0, f"timing command failed rc {done.returncode}: {done.stderr[-400:]!r}")
    if not pattern:
        return wall_ms
    found = re.findall(pattern, done.stdout.decode("utf-8", "replace") + done.stderr.decode("utf-8", "replace"))
    check(len(found) == 1, f"timing pattern matched {len(found)} times (one expected)")
    return float(found[0])


def abba(band, samples_path=None):
    draft = json.loads(band.draft_path.read_text(encoding="utf-8"))
    binary, pack, ref_config, cand_config = execution_identities(band)
    points = required_samples(band, recorded_point(draft["reference"]))
    if samples_path:
        samples = json.loads(pathlib.Path(samples_path).read_text(encoding="utf-8"))
    else:
        timing = band.spec["timing"]
        samples = []
        for prompt, lanes in points:
            a = time_once(timing["reference"], prompt, lanes, timing.get("ms"))
            b = time_once(timing["candidate"], prompt, lanes, timing.get("ms"))
            c = time_once(timing["candidate"], prompt, lanes, timing.get("ms"))
            d = time_once(timing["reference"], prompt, lanes, timing.get("ms"))
            samples.append({"prompt": prompt, "lanes": lanes, "abba_ms": [a, b, c, d]})
    have = {(s["prompt"], s["lanes"]) for s in samples}
    check(set(points) <= have, f"ABBA samples miss {sorted(set(points) - have)}")
    doc = {"schema": ABBA_SCHEMA, "binary_sha256": binary, "candidate_pack_sha256": pack,
           "reference_config_sha256": ref_config, "candidate_config_sha256": cand_config,
           "reference": draft["reference"], "revision": draft["revision"], "samples": samples}
    write_json(band.abba_path, doc)
    return {(s["prompt"], s["lanes"]): (s["abba_ms"][0] + s["abba_ms"][3]) / (s["abba_ms"][1] + s["abba_ms"][2])
            for s in samples}


def draft(band, receipt_path):
    """The manifest draft from the candidate run's profile receipt (levers.LEVER.promotion_reference)."""
    text = pathlib.Path(receipt_path).read_text(encoding="utf-8")
    lever, reference, start = band.spec["lever"], None, text.find('{"profile":')
    check(start >= 0, f"{receipt_path}: no profile receipt")
    while start >= 0:  # the last receipt that reports the lever's key (a startup receipt precedes the prompt's)
        receipt, _ = json.JSONDecoder().raw_decode(text[start:])
        reference = receipt.get("levers", {}).get(lever, {}).get("promotion_reference") or reference
        start = text.find('{"profile":', start + 1)
    check(isinstance(reference, str) and reference, f"{receipt_path}: lever {lever} reports no promotion_reference "
          "(the engine did not spell Regime::promotion_execution)")
    m = CANDIDATE.search(reference)
    check(m and m[1] == lever and m[4] == band.spec["tier"], "the promotion reference names another lever or tier")
    _, pack, _, _ = execution_identities(band)
    check(f"pack={pack}" in reference, "the run's pack identity differs from the spec's pack")
    s = band.spec
    doc = {"id": s["id"], "source": s["source"], "lever": lever, "value": m[2], "identity": m[3], "tier": s["tier"],
           "phase": s["phase"], "device": s["device"], "contexts": s["contexts"], "lanes": s["lanes"],
           "noise": s["noise"], "reference": reference, "revision": s["revision"]}
    write_json(band.draft_path, doc)
    return doc


def seal(band, manifest=None):
    entry = cpm.seal(json.loads(band.draft_path.read_text(encoding="utf-8")), band.gate_path, band.abba_path)
    write_json(band.entry_path, entry)
    if manifest:
        manifest = pathlib.Path(manifest)
        doc = json.loads(manifest.read_text(encoding="utf-8"))
        check(doc.get("schema") == "strix-profile-promotion-manifest/v1", f"{manifest}: not the promotion manifest")
        check(all(e["id"] != entry["id"] for e in doc["promotions"]), f"{entry['id']} is already sealed")
        doc["promotions"] = sorted([*doc["promotions"], entry], key=lambda e: e["id"])
        manifest.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return entry


def band_stage(band, stage, receipt=None, samples=None, manifest=None, headroom=1.0, run_capture=True):
    if stage == "reference":
        if run_capture:
            capture(band, band.reference_arms())
        sha = plan(band, headroom)
        print(f"PROMOTION PLAN {band.spec['id']} sha256 {sha} ({band.plan_path}): publish before the candidate stage")
        return 0
    if run_capture:
        capture(band, band.candidate_arms())
    witness(band)
    rc, state = gate(band)
    print(f"PROMOTION GATE {band.spec['id']} {state} rc {rc}")
    if state != "PASS":
        return rc or 1
    check(receipt, "the candidate stage needs --receipt (the candidate run's profile receipt)")
    draft(band, receipt)
    ratios = abba(band, samples)
    print("PROMOTION ABBA " + " ".join(f"{p}x{n}:{r:.3f}" for (p, n), r in sorted(ratios.items())))
    entry = seal(band, manifest)
    print(f"PROMOTION SEALED {entry['id']} min ratio {min(ratios.values()):.3f} noise {band.spec['noise']} "
          f"gate {entry['gate']['receipt_sha256'][:16]} timing {entry['timing']['receipt_sha256'][:16]}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("band", "capture", "plan", "witness", "gate", "abba", "draft", "seal", "pack"):
        p = sub.add_parser(name)
        p.add_argument("spec" if name != "pack" else "path")
        if name == "band":
            p.add_argument("--stage", required=True, choices=("reference", "candidate"))
            p.add_argument("--no-capture", action="store_true", help="the captures already exist under OUT")
        if name == "capture":
            p.add_argument("--arms", required=True, help="comma-separated arm names")
        if name in ("band", "plan"):
            p.add_argument("--headroom", type=float, default=1.0, help="null envelope factor over the worst control")
        if name in ("band", "draft"):
            p.add_argument("--receipt", help="log or JSON with the candidate run's profile receipt")
        if name in ("band", "abba"):
            p.add_argument("--samples", help="JSON list of {prompt, lanes, abba_ms} measured by an external harness")
        if name in ("band", "seal"):
            p.add_argument("--manifest", help="append the sealed entry to this promotion manifest")
    a = parser.parse_args(argv)
    try:
        if a.cmd == "pack":
            print(pack_identity(a.path))
            return 0
        band = Band(a.spec)
        if a.cmd == "band":
            return band_stage(band, a.stage, a.receipt, a.samples, a.manifest, a.headroom, not a.no_capture)
        if a.cmd == "capture":
            capture(band, [x for x in a.arms.split(",") if x])
        elif a.cmd == "plan":
            print(f"PROMOTION PLAN sha256 {plan(band, a.headroom)}")
        elif a.cmd == "witness":
            print(f"PROMOTION WITNESS sha256 {witness(band)}")
        elif a.cmd == "gate":
            rc, state = gate(band)
            print(f"PROMOTION GATE {state} rc {rc}")
            return rc
        elif a.cmd == "abba":
            ratios = abba(band, a.samples)
            print("PROMOTION ABBA " + " ".join(f"{p}x{n}:{r:.3f}" for (p, n), r in sorted(ratios.items())))
        elif a.cmd == "draft":
            print(json.dumps(draft(band, a.receipt), sort_keys=True))
        elif a.cmd == "seal":
            print(json.dumps(seal(band, a.manifest), sort_keys=True))
        return 0
    except (PromotionError, v2.GateError, ValueError, KeyError, OSError) as error:
        print(f"promote: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
