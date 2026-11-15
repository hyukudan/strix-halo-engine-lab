"""Small shared helpers: hashing, canonical JSON, id text files."""
import hashlib
import json
import re
from pathlib import Path


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canon_sha(obj):
    return sha256_bytes(canon(obj).encode("utf-8"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def parse_ids(text):
    return [int(x) for x in re.split(r"[,\s]+", text.strip()) if x]


def read_ids(path):
    return parse_ids(Path(path).read_text(encoding="utf-8"))


def safe_name(arm):
    return re.sub(r"[^A-Za-z0-9._-]", "_", arm)


def tokenize(cmd, text):
    """Run an external tokenizer: shlex-split CMD plus the path of a temp UTF-8 text file as the LAST argument
    (same convention as strix-tokenize MODEL encode FILE); parse whitespace/comma separated ids from stdout."""
    import os
    import shlex
    import subprocess
    import tempfile
    argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="turbo-tok-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(text.encode("utf-8"))
        r = subprocess.run(argv + [path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    finally:
        os.unlink(path)
    return parse_ids(r.stdout.decode("utf-8"))
