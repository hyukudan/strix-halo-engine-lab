#!/usr/bin/env python3
"""Synchronize a Git source archive while preserving mtimes of unchanged files."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile


def destination(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(p in ("..", "") for p in relative.parts) or "\\" in name or ":" in name:
        raise ValueError("source archive contains an unsafe path")
    target = root.joinpath(*relative.parts)
    target.resolve().relative_to(root)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        parser.error("revision must be a full lowercase Git object ID")
    root = args.source.resolve()
    root.mkdir(parents=True, exist_ok=True)
    inventory = root / ".wfmt-source-inventory.json"
    previous = json.loads(inventory.read_text()) if inventory.exists() else {"files": []}
    files, changed = [], 0
    with tarfile.open(args.archive) as archive:
        for member in archive:
            target = destination(root, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.issym():
                # Only project-internal links belong in a source archive.
                (target.parent / member.linkname).resolve().relative_to(root)
                if not target.is_symlink() or os.readlink(target) != member.linkname:
                    if target.exists() or target.is_symlink():
                        target.unlink()
                    target.symlink_to(member.linkname)
                    changed += 1
            elif member.isfile():
                data = archive.extractfile(member).read()
                if target.is_symlink() or not target.exists() or target.read_bytes() != data:
                    if target.is_symlink():
                        target.unlink()
                    target.write_bytes(data)
                    changed += 1
                target.chmod(member.mode)
            else:
                raise ValueError("source archive contains an unsupported member")
            files.append(member.name)
    removed = 0
    for name in set(previous["files"]) - set(files):
        target = destination(root, name)
        if target.is_symlink() or target.is_file():
            target.unlink()
            removed += 1
    receipt = {"schema": "wfmt.source@1", "revision": args.revision,
               "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
               "files": sorted(files)}
    inventory.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"schema": receipt["schema"], "revision": args.revision,
                      "files": len(files), "changed": changed, "removed": removed}, sort_keys=True))


if __name__ == "__main__":
    main()
