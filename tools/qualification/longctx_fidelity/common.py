"""Portable canonical serialization and content receipts."""
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

VERSION = "longctx-fidelity-v3"
NATIVE_CAP = 262144
MODE_FACTORS = {"native": 1, "yarn2": 2, "yarn4": 4}


def mode_budget(mode, depth, generate, context=None):
    if mode not in MODE_FACTORS:
        raise ValueError("mode must be native, yarn2 or yarn4")
    cap = NATIVE_CAP * MODE_FACTORS[mode]
    if type(depth) is not int or type(generate) is not int or min(depth, generate) < 1:
        raise ValueError("D and G must be positive integers")
    if depth + generate > cap:
        raise ValueError(f"{mode} budget requires D + G <= {cap}")
    if mode == "native":
        if context is not None:
            raise ValueError("--context requires a yarn mode; native context remains auto")
        return cap, "auto"
    if context is None:
        context = max(300000, ((depth + generate + 511) // 512) * 512)
    if type(context) is not int or not NATIVE_CAP < context <= cap or context < depth + generate:
        raise ValueError(f"{mode} context requires 262144 < C <= {cap} and D + G <= C")
    return cap, context


def prompt_ids_file(directory, case, ids):
    """Materialize the exact engine text input beside private receipts."""
    path = Path(directory) / (case["case_id"] + ".prompt-ids.txt")
    atomic(path, (" ".join(map(str, ids)) + "\n").encode("ascii"))
    return path


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
