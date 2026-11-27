"""Verify matching CLI and HTTP profile captures before changing a default."""

import argparse
import hashlib
import json
from pathlib import Path


def read_json(path):
    def unique(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError(f"{path.name}: duplicate JSON key")
        return result

    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path.name}: expected a nonempty JSON object")
    return data


def digest(path, alignment=1):
    if not path.is_file() or not path.stat().st_size or path.stat().st_size % alignment:
        raise ValueError(f"{path.name}: expected a nonempty aligned capture")
    with path.open("rb") as capture:
        return hashlib.file_digest(capture, "sha256").hexdigest()


def capture(directory):
    state = directory / "state"
    states = {item.name: digest(item) for item in sorted(state.iterdir())}
    if not states:
        raise ValueError("state: at least one payload is required")
    return {
        "effective": read_json(directory / "effective.json"),
        "receipt": read_json(directory / "receipt.json"),
        "prompt_sha256": digest(directory / "prompt.u32", 4),
        "output_sha256": digest(directory / "output.u32", 4),
        "logits_sha256": digest(directory / "last_logits.f32", 4),
        "state_sha256": states,
    }


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def verify(cli_dir, http_dir, service_fields):
    cli, http = capture(cli_dir), capture(http_dir)
    for field in service_fields:
        a, b = cli["effective"], http["effective"]
        if (field not in a and field not in b) or (field in a and field in b and canonical(a[field]) == canonical(b[field])):
            raise ValueError(f"stale service field: {field}")
        a.pop(field, None)
        b.pop(field, None)
    if not cli["effective"] or not http["effective"]:
        raise ValueError("effective config must contain common execution fields")
    for field in cli:
        if canonical(cli[field]) != canonical(http[field]):
            raise ValueError(f"CLI/HTTP mismatch: {field}")
    return {"schema": "strix-profile-parity/v1", "matched": cli, "service_fields": service_fields}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--http", required=True, type=Path)
    parser.add_argument("--service-field", action="append", default=[], metavar="KEY=REASON")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fields = {}
    try:
        for item in args.service_field:
            key, sep, reason = item.partition("=")
            if not sep or not key.strip() or "*" in key or not reason.strip() or key in fields:
                raise ValueError("use a distinct, literal service field with its reason")
            fields[key] = reason
        verified = verify(args.cli, args.http, fields)
        with args.output.open("x", encoding="utf-8") as out:
            json.dump(verified, out, sort_keys=True, indent=2)
            out.write("\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"profile parity: {error}\n")
    print("profile parity: matched effective config, receipt, IDs, logits and state")


if __name__ == "__main__":
    main()
