"""Shared helpers: canonical JSON, hashes, atomic writes, ID files."""
import json
import subprocess
from pathlib import Path

from tools.qualification.longctx_fidelity.common import atomic, canonical, file_sha, sha

__all__ = ["VERSION", "atomic", "canonical", "file_sha", "sha", "read_id_text", "write_id_text", "kit_commit",
           "JournalError", "read_journal"]

VERSION = "option-qual-v1"


class JournalError(ValueError):
    pass


def read_journal(path):
    """Records of a JSON-lines journal and its incomplete final append (bytes after the last newline, or
    None). Every complete line must parse; a malformed interior record is an error, never skipped."""
    path = Path(path)
    if not path.exists():
        return [], None
    data = path.read_bytes()
    body, sep, tail = data.rpartition(b"\n")
    if not sep:
        body, tail = b"", data
    records = []
    for n, line in enumerate(body.split(b"\n") if body else [], 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            raise JournalError(f"{path.name}: malformed record at line {n}") from None
    return records, (tail if tail.strip() else None)


def read_journal_lines(path):
    """Like read_journal, but tolerant: returns (lines, tail, raws). A line that does not parse is
    returned as its raw text (the run journal keeps every line; the evidence ledger decides what a bad
    line means); ``raws`` are the exact bytes of every line, for binding resets to a journal prefix."""
    path = Path(path)
    if not path.exists():
        return [], None, []
    data = path.read_bytes()
    body, sep, tail = data.rpartition(b"\n")
    if not sep:
        body, tail = b"", data
    lines, raws = [], []
    for line in (body.split(b"\n") if body else []):
        if not line.strip():
            continue
        raws.append(line)
        try:
            lines.append(json.loads(line))
        except ValueError:
            lines.append(line.decode("utf-8", "replace"))
    return lines, (tail if tail.strip() else None), raws


def diagnostic_records(path, after_damage=False):
    """Records of a diagnostic journal and the number of damaged lines, an incomplete last line included
    (skipped, never an error). With after_damage, only the records after the last damaged line: damage
    is a durable boundary, so what was written again after it is used again."""
    lines, tail, _ = read_journal_lines(path)
    records, damaged = [], 0
    for r in lines:
        if isinstance(r, dict):
            records.append(r)
        else:
            damaged += 1
            if after_damage:
                records = []
    return records, damaged + (tail is not None)


def write_id_text(path, ids):
    atomic(path, (",".join(str(int(i)) for i in ids) + "\n").encode("ascii"))


def read_id_text(path):
    text = Path(path).read_text(encoding="ascii").strip()
    return [int(x) for x in text.replace("\n", ",").split(",") if x.strip()] if text else []


def kit_commit():
    """Commit of the checkout this package runs from; for an exported copy, the commit recorded in a
    ``COMMIT`` file next to this module at export time; else None."""
    stamp = Path(__file__).resolve().parent / "COMMIT"
    if stamp.exists():
        return stamp.read_text().strip() or None
    try:
        result = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10)
        return result.stdout.strip() or None if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None
