"""Check that malformed canonical evidence is rejected by the CI checker."""

from copy import deepcopy
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "receipt-schema"))
from check import SCHEMA, read, unique_members, validate  # noqa: E402


def reject(document, schema, section, key, value, reason):
    altered = deepcopy(document)
    target = altered[section]
    if isinstance(target, list):
        target = target[0]
    target[key] = value
    try:
        validate(altered, schema)
    except ValueError as error:
        if reason in str(error):
            return
        raise
    raise RuntimeError(f"bad {section}.{key} was accepted")


if __name__ == "__main__":
    schema = read(SCHEMA)
    receipt = read(Path(sys.argv[1]))
    reject(receipt, schema, "identity", "invented", "unknown", "invented: unknown member")
    reject(receipt, schema, "identity", "route_fingerprint", "abc", "route_fingerprint")
    reject(receipt, schema, "step", "wall_ms", -1, "number out of range")
    reject(receipt, schema, "step", "bytes_read", True, "expected integer")
    reject(receipt, schema, "speculation", "accepted", 5, "exceeds proposed")
    wrong_id = dict(receipt["identity"]["numerical"], id="0000000000000000")
    reject(receipt, schema, "identity", "numerical", wrong_id, "does not match its numerical fields")
    try:
        unique_members([("schema", 1), ("schema", 2)])
    except ValueError:
        print("receipt_schema_rejects_invalid: PASS")
    else:
        raise RuntimeError("duplicate key was accepted")
