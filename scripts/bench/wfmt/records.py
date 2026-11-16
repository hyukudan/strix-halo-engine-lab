"""Version 1 portable capture records. Hashes are receipts, never equality proofs."""
import json
import pathlib
import struct

MAGIC = b"WFMT\x00\x01\r\n"
SECTIONS = {"source", "sidecars", "activation_codes", "activation_scales",
            "prepared", "partials", "output", "logits", "state"}


def fnv1a64(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def write_record(path, metadata, sections):
    if not sections or set(sections) - SECTIONS:
        raise ValueError("unknown or empty byte sections")
    header = dict(metadata)
    header["sections"] = [{"name": key, "bytes": len(value), "fnv1a64": fnv1a64(value)}
                          for key, value in sorted(sections.items())]
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    with pathlib.Path(path).open("wb") as stream:
        stream.write(MAGIC + struct.pack("<Q", len(encoded)) + encoded)
        for key in sorted(sections):
            stream.write(sections[key])


def read_record(path):
    with pathlib.Path(path).open("rb") as stream:
        if stream.read(8) != MAGIC:
            raise ValueError(f"invalid record version: {path}")
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError("truncated header size")
        size, = struct.unpack("<Q", size_bytes)
        if size > 1024 * 1024:
            raise ValueError("oversized header")
        header = json.loads(stream.read(size))
        for key in ("case", "arithmetic", "shape"):
            if key not in header:
                raise ValueError(f"missing metadata: {key}")
        sections = {}
        for section in header.pop("sections"):
            name, count = section["name"], section["bytes"]
            if name not in SECTIONS or name in sections or not isinstance(count, int) or count < 0:
                raise ValueError("invalid section")
            data = stream.read(count)
            if len(data) != count or fnv1a64(data) != section["fnv1a64"]:
                raise ValueError(f"truncated or invalid receipt: {name}")
            sections[name] = data
        if not sections or stream.read(1):
            raise ValueError("empty sections or trailing bytes")
    return header, sections


def compare_record(old, new):
    a, ab = read_record(old)
    b, bb = read_record(new)
    # Build/source identity is provenance; arithmetic, shape and case are contracts.
    for key in ("case", "arithmetic", "shape", "format", "image"):
        if a.get(key) != b.get(key):
            raise ValueError(f"metadata differs: {key}")
    if set(ab) != set(bb):
        raise ValueError("section inventory differs")
    for key in ab:
        if len(ab[key]) != len(bb[key]) or ab[key] != bb[key]:
            raise ValueError(f"bytes differ: {key}")
    return {key: fnv1a64(value) for key, value in ab.items()}
