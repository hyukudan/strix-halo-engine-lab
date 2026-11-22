#!/usr/bin/env python3
"""Validate canonical Strix receipt JSON files with the standard library."""

import argparse
import json
from decimal import Decimal, DecimalException
from pathlib import Path
import re
import sys


SCHEMA = Path(__file__).with_name("receipt-v1.schema.json")
TYPES = {"object": (dict,), "array": (list,), "string": (str,), "null": (type(None),),
         "number": (int, Decimal), "integer": (int, Decimal)}
IDENTITY_FIELDS = ("arithmetic_contract", "model_sha256", "pack_hash", "table_hash",
                   "state_format", "prefill_lineage", "reduction_plan")


def unique_members(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key: {key!r}")
        obj[key] = value
    return obj


def reject_constant(token):
    raise ValueError(f"invalid JSON number: {token}")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_members,
                      parse_float=Decimal, parse_constant=reject_constant)


def check(value, node, root, path="$"):
    # This interpreter implements the keywords used by the bundled schema.
    if "$ref" in node:
        check(value, root["$defs"][node["$ref"].removeprefix("#/$defs/")], root, path)
    if "anyOf" in node:
        errors = []
        for arm in node["anyOf"]:
            try:
                check(value, arm, root, path)
                break
            except ValueError as error:
                errors.append(str(error))
        else:
            raise ValueError(" or ".join(errors))
    if ("const" in node and value != node["const"]) or ("enum" in node and value not in node["enum"]):
        raise ValueError(f"{path}: unexpected value")
    kind = node.get("type")
    if kind is None:
        return
    if type(value) not in TYPES[kind] or (kind == "integer" and isinstance(value, Decimal)
                                        and value != value.to_integral_value()):
        raise ValueError(f"{path}: expected {kind}")
    if kind == "object":
        members = node.get("properties", {})
        for key in node.get("required", ()):
            if key not in value:
                raise ValueError(f"{path}.{key}: missing member")
        for key, item in value.items():
            if "propertyNames" in node:
                check(key, node["propertyNames"], root, path + " (member name)")
            rule = members.get(key, node.get("additionalProperties", True))
            if rule is False:
                raise ValueError(f"{path}.{key}: unknown member")
            if rule is not True:
                check(item, rule, root, f"{path}.{key}")
    elif kind == "array":
        for index, item in enumerate(value):
            check(item, node["items"], root, f"{path}[{index}]")
        if node.get("uniqueItems", False) and len(set(value)) != len(value):
            raise ValueError(f"{path}: duplicate item")
    elif kind == "string":
        value.encode("utf-8")  # Reject lone surrogates (including JSON escapes).
        if len(value) < node.get("minLength", 0) or len(value) > node.get("maxLength", len(value)):
            raise ValueError(f"{path}: invalid string length")
        if "pattern" in node and not re.fullmatch(node["pattern"], value):
            raise ValueError(f"{path}: invalid string format")
    elif kind in ("integer", "number"):
        if value < node.get("minimum", value) or value > node.get("maximum", value):
            raise ValueError(f"{path}: number out of range")


def numerical_id(identity):
    # Wire algorithm from strix::profile::NumericalIdentity::id and host_fnv::field.
    result = 14695981039346656037
    for field in ("strix-numerical-identity/v1", *(identity[name] for name in IDENTITY_FIELDS)):
        encoded = field.encode("utf-8")
        for byte in len(encoded).to_bytes(8, "little") + encoded:
            result = ((result ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"{result:016x}"


def validate(receipt, schema):
    check(receipt, schema, schema)
    identity = receipt["identity"]["numerical"]
    if identity is not None and identity["id"] != numerical_id(identity):
        raise ValueError("$.identity.numerical.id: does not match its numerical fields")
    if receipt["resolution"] is not None:
        names = [row["name"] for row in receipt["resolution"]["levers"]]
        if len(names) != len(set(names)):
            raise ValueError("$.resolution.levers: duplicate lever name")
    for index, row in enumerate(receipt["step"] or ()):
        if row["guard_failures"] > row["guards"]:
            raise ValueError(f"$.step[{index}].guard_failures: exceeds guards")
    speculation = receipt["speculation"]
    if speculation is not None and speculation["accepted"] > speculation["proposed"]:
        raise ValueError("$.speculation.accepted: exceeds proposed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, nargs="+")
    args = parser.parse_args()
    schema = read(SCHEMA)
    failed = False
    for path in args.receipt:
        try:
            validate(read(path), schema)
        except (OSError, ValueError, DecimalException) as error:
            print(f"{path.name}: FAIL {error}", file=sys.stderr)
            failed = True
        else:
            print(f"{path.name}: PASS strix-receipt/v1")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
