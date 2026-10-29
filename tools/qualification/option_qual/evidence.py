"""One definition of valid run evidence, shared by the runner's resume repair and the analysis.

The run journal (runs.jsonl) is append-only and is never edited, reordered or pruned; it holds engine
run records and failure events. When a case's evidence turns out to be inconsistent (a malformed or
contradictory record, a marker that cannot be trusted), the runner appends a reset to resets.jsonl:
every journal line of that case written so far stops counting as a result, the case's files are moved
to quarantine, and the case is produced again. Because nothing is ever removed from the journal, an
older result can never become "the newest run" again. Failures stay counted across resets.
"""
import hashlib
import math
import re

RUN_IDENTITY = ("plan_sha256", "binary_sha256", "arms_sha256", "model_receipt_sha256")
EVENTS = ("reduction_failed",)


def _str(x):
    return isinstance(x, str) and bool(x)


def _int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _num(x):
    """A finite number of sane size; never raises (huge integers are simply not numbers here)."""
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return abs(x) <= 10 ** 12
    return isinstance(x, float) and math.isfinite(x)


RUN_FIELDS = ("case_id", "arm", "run_id", "route", "rc", "timeout", "complete", "shortened", "rows",
              "forced_steps", "problem", "logits_sha256", "nominal_length", "wall_seconds") + RUN_IDENTITY


def run_record(rec):
    """A well-formed engine run record: every field read downstream present, with its type."""
    return (isinstance(rec, dict) and "event" not in rec and all(k in rec for k in RUN_FIELDS)
            and all(_str(rec.get(k)) for k in ("case_id", "arm", "run_id") + RUN_IDENTITY)
            and isinstance(rec.get("route"), list) and all(isinstance(x, str) for x in rec["route"])
            and (rec.get("rc") is None or _int(rec.get("rc")))
            and all(isinstance(rec.get(k), bool) for k in ("timeout", "complete", "shortened"))
            and _int(rec.get("rows")) and (rec.get("forced_steps") is None or _int(rec.get("forced_steps")))
            and (rec.get("problem") is None or isinstance(rec.get("problem"), str))
            and (rec.get("logits_sha256") is None or _str(rec.get("logits_sha256")))
            and isinstance(rec.get("interrupted", False), bool)
            and _int(rec.get("nominal_length")) and _num(rec.get("wall_seconds")) and rec["wall_seconds"] >= 0
            and ("exec_seconds" not in rec or (_num(rec["exec_seconds"]) and 0 <= rec["exec_seconds"]
                                               <= rec["wall_seconds"])))


def event_record(rec):
    return (isinstance(rec, dict) and rec.get("event") in EVENTS and "error" in rec
            and all(_str(rec.get(k)) for k in ("case_id", "arm", "run_id", "of_run") + RUN_IDENTITY)
            and isinstance(rec.get("error"), str))


def run_ok(rec, identity):
    """A well-formed, successful, complete run of this identity."""
    return (run_record(rec) and all(rec[k] == identity.get(k) for k in RUN_IDENTITY)
            and rec["complete"] is True and rec["rc"] == 0 and rec["timeout"] is False and rec["problem"] is None
            and rec["rows"] == rec["forced_steps"] and bool(rec["logits_sha256"]))


CASE_IN_TEXT = re.compile(r'"case_id"\s*:\s*"([^"\\]+)"')


def _case_of(line):
    """The case a (possibly malformed) journal line belongs to, or None if it cannot be told."""
    if isinstance(line, dict):
        return line.get("case_id") if isinstance(line.get("case_id"), str) else None
    if isinstance(line, str):
        found = set(CASE_IN_TEXT.findall(line))
        return found.pop() if len(found) == 1 else None
    return None


def prefix_sha(raws, upto):
    """Digest of the first ``upto`` journal lines (exact bytes, newline-joined)."""
    h = hashlib.sha256()
    for raw in raws[:upto]:
        h.update(raw)
        h.update(b"\n")
    return h.hexdigest()


class Ledger:
    """The run journal as evidence: lines, resets, and what still counts.

    ``lines`` are the parsed journal lines in order (anything, including malformed ones); ``raws`` their
    exact bytes. ``resets`` are reset records {"case_id": ID or "*", "upto": number of journal lines
    covered, "prefix_sha256": digest of those lines, plus the identity}. A reset that does not match this
    journal's prefix or identity is not honoured (``bad_resets``); it can never void later lines.
    """

    def __init__(self, lines, resets, identity, raws=None):
        self.identity = identity or {}
        self.lines = list(lines)
        raws = list(raws) if raws is not None else None
        self.resets, self.bad_resets = [], []
        for r in resets:
            ok = (isinstance(r, dict) and _int(r.get("upto")) and 1 <= r["upto"] <= len(self.lines)
                  and (r.get("case_id") == "*" or _str(r.get("case_id")))
                  and all(r.get(k) == self.identity.get(k) for k in RUN_IDENTITY)
                  and raws is not None and r.get("prefix_sha256") == prefix_sha(raws, r["upto"]))
            (self.resets if ok else self.bad_resets).append(r)
        first = {}
        targets = set()            # runs already named by a failure event
        latest_run = {}            # (case, arm) -> run id of the latest run record so far
        self.records = []          # well-formed, de-duplicated lines (runs and events) with their position
        self.bad = []              # (position, case or None, why): malformed or contradictory lines
        for pos, line in enumerate(self.lines):
            if not (run_record(line) or event_record(line)):
                self.bad.append((pos, _case_of(line), "malformed journal line"))
                continue
            if any(line[k] != self.identity.get(k) for k in RUN_IDENTITY):
                self.bad.append((pos, line["case_id"], "journal line of another identity"))
                continue
            rid = line["run_id"]
            if rid in first:
                if first[rid] != line:
                    self.bad.append((pos, line["case_id"], f"contradictory records for run id {rid}"))
                continue           # an identical repeat of a line adds nothing
            if "event" in line:
                target = first.get(line["of_run"])
                if not (run_record(target) and (target["case_id"], target["arm"]) == (line["case_id"], line["arm"])
                        and target["complete"] is True):
                    self.bad.append((pos, line["case_id"], "failure event not bound to a complete run of its pair"))
                    continue
                if line["of_run"] in targets:
                    self.bad.append((pos, line["case_id"], "two failure events name the same run"))
                    continue
                if not line.get("reconstructed") and \
                        latest_run.get((line["case_id"], line["arm"])) != line["of_run"]:
                    self.bad.append((pos, line["case_id"], "failure event does not name the latest run of its pair"))
                    continue
                targets.add(line["of_run"])
            else:
                latest_run[(line["case_id"], line["arm"])] = rid
            first[rid] = line
            self.records.append((pos, line))

    def covered(self, pos, case_id):
        return any(pos < r["upto"] and r["case_id"] in ("*", case_id) for r in self.resets)

    def uncovered_bad(self):
        return [(pos, case, why) for pos, case, why in self.bad if not self.covered(pos, case)]

    def runs(self, case_id, label, counting=True):
        """Run records of this pair in journal order; with ``counting``, only those no reset voided."""
        return [rec for pos, rec in self.records if "event" not in rec and rec["case_id"] == case_id
                and rec["arm"] == label and not (counting and self.covered(pos, case_id))]

    def by_id(self, run_id, counting_case=None):
        for pos, rec in self.records:
            if rec["run_id"] == run_id and "event" not in rec:
                if counting_case is not None and self.covered(pos, counting_case):
                    return None
                return rec
        return None

    def newest(self, case_id, label):
        runs = self.runs(case_id, label)
        return runs[-1]["run_id"] if runs else None

    def failed(self, case_id, label):
        """Run ids that failed for this pair, in journal order of the failure evidence: failed run
        records (a timeout of a run that did not get its whole cap is a budget stop, not a failure) and
        reduction-failure events naming a complete run of exactly this pair. Counted across resets."""
        ids = []
        for pos, rec in self.records:
            if rec["case_id"] != case_id or rec["arm"] != label:
                continue
            if "event" in rec:
                target = self.by_id(rec["of_run"])
                if target is not None and (target["case_id"], target["arm"]) == (case_id, label) \
                        and target["complete"] is True and rec["of_run"] not in ids:
                    ids.append(rec["of_run"])
            elif not rec["complete"] and not (rec["timeout"] and rec["shortened"]) and not rec.get("interrupted") \
                    and rec["run_id"] not in ids:
                ids.append(rec["run_id"])
        return ids

    def post(self):
        """Post-processing time seen after an engine run (validation, hashing): wall minus execution."""
        return [rec["wall_seconds"] - rec["exec_seconds"] for _, rec in self.records
                if "event" not in rec and "exec_seconds" in rec and not rec.get("interrupted")]

    def walls(self, length, label=None):
        """Execution times of completed runs (older records without a split count with their wall time)."""
        return [rec.get("exec_seconds", rec["wall_seconds"]) for _, rec in self.records if "event" not in rec
                and rec["nominal_length"] == length and rec["complete"] and (label is None or rec["arm"] == label)]

    def result_bound(self, case_id, label, sha, cont_tokens, mark, plan_sha):
        """The marker names the newest counting run of this pair, a valid complete run with this logits
        file, the continuation's counts and the same route; the marker carries this plan."""
        rec = self.by_id(mark.get("run_id"), counting_case=case_id)
        return (rec is not None and run_ok(rec, self.identity) and mark.get("plan_sha256") == plan_sha
                and (rec["case_id"], rec["arm"], rec["logits_sha256"]) == (case_id, label, sha)
                and (rec["rows"], rec["forced_steps"]) == (cont_tokens, cont_tokens)
                and rec["route"] == mark.get("route") and self.newest(case_id, label) == rec["run_id"]
                and rec["run_id"] not in self.failed(case_id, label))

    def outcome_bound(self, mark, case_id, label, ref_sha, plan_sha):
        """A FAILED/INVALID marker names the newest counting run of this pair, carries this plan, is a
        recorded failure for FAILED, and concerns the current reference for INVALID."""
        if mark.get("status") == "FAILED":
            rec = self.by_id(mark.get("run_id"))
            if rec is None or (rec["case_id"], rec["arm"]) != (case_id, label) or mark.get("plan_sha256") != plan_sha:
                return False
            failed = self.failed(case_id, label)
            if rec["run_id"] not in failed:
                return False
            newest = self.newest(case_id, label)
            # the newest counting run is this failure, or a reset voided everything and this is the
            # latest recorded failure (the durable count survives resets)
            return newest == rec["run_id"] or (newest is None and failed[-1] == rec["run_id"])
        rec = self.by_id(mark.get("run_id"), counting_case=case_id)
        if rec is None or (rec["case_id"], rec["arm"]) != (case_id, label) or mark.get("plan_sha256") != plan_sha:
            return False
        if mark.get("status") == "INVALID" and mark.get("ref_sha256") != ref_sha:
            return False
        return self.newest(case_id, label) == rec["run_id"]
