"""Generate, but NEVER execute, a standalone bash GPU-turn script.

Paths in the generated script are relative to the staged working directory.
The manifest hashes binary, model content receipt, list, cases and Python kit.
Model content must already be verified against that receipt before taking a turn.
"""
import argparse
import json
from pathlib import Path
import shlex
try:
    from .common import canonical, atomic, file_sha, read_ids, prompt_ids_file
except ImportError:
    from common import canonical, atomic, file_sha, read_ids, prompt_ids_file


def relative(path):
    path = Path(path)
    if path.is_absolute() or ".." in path.parts or "\n" in str(path) or "\r" in str(path):
        raise ValueError("use staged relative paths without parent traversal/newlines")
    return path.as_posix()


def make_script(binary, model, model_receipt, turn_list, receipts, script, prediction_table=None):
    binary, model, model_receipt, turn_list, receipts, script = map(
        relative, (binary, model, model_receipt, turn_list, receipts, script))
    paths = [binary, model_receipt, turn_list]
    listing = Path(turn_list)
    for line in listing.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        listed = Path(line.strip())
        if listed.is_absolute():
            raise ValueError("turn list entries must be relative to their list")
        # A list may live in a sibling directory of the case files. Resolve '..'
        # first, then verify that every staged asset stays under the launch root.
        case = relative((listing.parent / listed).resolve().relative_to(Path.cwd().resolve()))
        stem = str(Path(case).with_suffix(""))
        paths += [case, stem + ".txt", stem + ".u32"]
        metadata = json.loads(Path(case).read_text(encoding="utf-8"))
        if metadata["mode"] != "native":
            paths.append(relative(prompt_ids_file(receipts, metadata, read_ids(stem + ".u32"))))
    kit = Path("tools/qualification/longctx_fidelity")
    paths += [p.as_posix() for p in kit.glob("*.py")]
    if prediction_table:
        prediction_table = relative(prediction_table)
        paths.append(prediction_table)
    hashes = {p: file_sha(p) for p in sorted(set(paths))}
    manifest = {"schema": 1, "files": hashes, "model": model, "binary": binary,
                "model_receipt": model_receipt, "turn_list": turn_list}
    manifest_path = script + ".manifest.json"
    atomic(manifest_path, canonical(manifest))
    manifest_hash = file_sha(manifest_path)
    q = shlex.quote
    command = ["python3", "-m", "tools.qualification.longctx_fidelity.run_cli", "--binary", binary,
               "--model", model, "--model-receipt", model_receipt,
               "--turn-list", turn_list, "--out", receipts]
    if prediction_table:
        command += ["--prediction-table", prediction_table]
    preflight = r"""import hashlib, json, pathlib, subprocess, sys
manifest_path, expected, script = sys.argv[1:]
raw = pathlib.Path(manifest_path).read_bytes()
if hashlib.sha256(raw).hexdigest() != expected:
    raise SystemExit('turn manifest hash mismatch')
m = json.loads(raw)
for name, expected_hash in m['files'].items():
    h = hashlib.sha256()
    with open(name, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    if h.hexdigest() != expected_hash:
        raise SystemExit('preflight hash mismatch: ' + name)
    if name.endswith(('.py', '.sh', '.txt', '.json')) and b'\r' in pathlib.Path(name).read_bytes():
        raise SystemExit('CRLF/CR refused: ' + name)
if b'\r' in pathlib.Path(script).read_bytes():
    raise SystemExit('CRLF/CR turn script refused')
if not pathlib.Path(m['model']).is_file():
    raise SystemExit('missing model; content verification receipt required')
r = subprocess.run(['ldd', m['binary']], capture_output=True, text=True)
if r.returncode or 'not found' in r.stdout + r.stderr:
    raise SystemExit('ldd preflight failed: ' + r.stdout + r.stderr)
"""
    text = ("#!/usr/bin/env bash\nset -euo pipefail\n"
            "# A 262143-ID argv is ~3.2 MB; Linux grants argv a quarter of the stack limit\n"
            "# (at most 6 MiB), so an 8 MiB default stack (2 MiB ARG_MAX) refuses it.\n"
            "ulimit -s 65536\n"
            "turn_started=$(python3 -c 'import time; print(time.time())')\n"
            "# Stage assets first. Submit this script with the authorized GPU-turn wrapper.\n"
            'bash -n "$0"\n' + "test -x " + q(binary) + "\n" +
            "python3 - " + q(manifest_path) + " " + q(manifest_hash) + " \"$0\" <<'PY_PREFLIGHT'\n" +
            preflight + "PY_PREFLIGHT\n" + "mkdir -p " + q(receipts) + "\n" +
            "status=0\n" + " ".join(map(q, command)) + ' --turn-start "$turn_started" || status=$?\n' +
            "tar -czf " + q(receipts + ".tar.gz") + " -C " + q(receipts) + " .\n" +
            'exit "$status"\n')
    atomic(script, text.encode())
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "model", "model-receipt", "turn-list", "receipts", "script"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--prediction-table")
    a = p.parse_args()
    make_script(a.binary, a.model, a.model_receipt, a.turn_list, a.receipts, a.script, a.prediction_table)
    print(a.script + " (generated only; not executed)")


if __name__ == "__main__":
    main()
