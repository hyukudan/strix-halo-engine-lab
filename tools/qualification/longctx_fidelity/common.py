"""Portable canonical serialization and content receipts."""
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

VERSION = "longctx-fidelity-v1"
NATIVE_CAP = 262144


def case_id(family, depth, position, seed, stratum="prose", variant="normal", mode="native"):
    return f"{mode}.{depth:06d}.{family}.s{seed}.p{float(position)!r}.{stratum}.{variant}"


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def ids_bytes(ids):
    return struct.pack("<" + "I" * len(ids), *ids)


def read_ids(path):
    data = Path(path).read_bytes()
    if len(data) % 4:
        raise ValueError("truncated u32le IDs")
    return list(struct.unpack("<" + "I" * (len(data) // 4), data))
