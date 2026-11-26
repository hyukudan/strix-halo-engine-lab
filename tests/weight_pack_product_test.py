import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import tempfile


def run(*args, error=None):
    command = subprocess.run([sys.argv[1], *map(str, args)], capture_output=True, text=True)
    if error is None:
        assert command.returncode == 0, command.stderr
    else:
        assert command.returncode == 1 and error in command.stderr, command
    return command.stdout


def checkpoint(path):
    tensors = {
        "head": ("BF16", [65, 128], b"\x00\x3e\x80\xbe" * (65 * 64)),
        "experts.nv": ("U8", [2, 2, 64], bytes([0x72, 0x4c]) * 128),
        "experts.nv_scale": ("F8_E4M3", [2, 2, 8], bytes([0x40]) * 32),
        "experts.nv_amax": ("F32", [2], struct.pack("<ff", 1.0, 2.0)),
        "experts.nv_outer_scale_denominator": ("F32", [1], struct.pack("<f", 2.0)),
        "experts.mx": ("F8_E4M3", [2, 2, 128], bytes([0x3a, 0xb3]) * 256),
        "experts.mx_scale": ("U8", [2, 2, 4], bytes([125]) * 16),
        "router": ("F32", [2], struct.pack("<ff", 0.5, -0.5)),
    }
    index, payload = {}, bytearray()
    for name, (dtype, shape, data) in tensors.items():
        index[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(payload), len(payload) + len(data)]}
        payload += data
    header = json.dumps(index).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    return tensors


with tempfile.TemporaryDirectory() as tmp:
    root = pathlib.Path(tmp)
    model, description = root / "tensor.safetensors", root / "recipe.json"
    tensors = checkpoint(model)
    description.write_text(json.dumps({
        "schema": "strix-pack-convert/v1",
        "roles": {"head": "sp4", "expert_gate_up": "sp3", "expert_down": "sp2", "router": "keep"},
        "rules": [
            {"role": "head", "suffix": "head"},
            {"role": "expert_gate_up", "suffix": "experts.nv", "layout": "expert-interleaved"},
            {"role": "expert_down", "suffix": "experts.mx", "layout": "expert-contiguous"},
            {"role": "router", "suffix": "router", "reason": "retain routing precision"},
        ],
    }), encoding="utf-8")
    first, second = root / "one.pack", root / "two.pack"
    args = ("convert", "--model", model, "--description", description)
    plan = run(*args, "--plan-only")
    assert "keep router router 8 bytes: retain routing precision" in plan
    assert "planned_tensors=5" in plan
    receipt = run(*args, "--out", first, "--threads", 1)
    assert receipt.startswith(plan)
    run(*args, "--out", second, "--threads", 1)
    data = first.read_bytes()
    assert second.read_bytes() == data
    third = root / "three.pack"
    run(*args, "--out", third, "--threads", 4, "--quality")
    assert third.read_bytes() == data
    run(*args, "--out", first, error="already exists")
    assert first.read_bytes() == data
    assert data.startswith(b"strix-lowbit-pack/v3 5\n")
    assert hashlib.sha256(data[:-32]).digest() == data[-32:]
    header, _ = data.split(b"\nend\n", 1)
    lines = header.decode().splitlines()
    assert lines[1] == "checkpoint " + hashlib.sha256(model.read_bytes()).hexdigest()
    encoded = {}
    for line in lines[2:]:
        tag, name, fmt, rev, rows, columns, offset, count, source, sp, group, scale, role, coder, calibration, digest = line.split()
        offset, count = int(offset), int(count)
        assert tag == "tensor" and rev == "1" and columns == "128"
        assert offset % 4096 == 0
        assert group == "32" and scale == "e5m3+tensor-base" and len(sp) == 32
        assert coder == "rtn-max-fixed@1" and calibration == "0" * 64
        assert hashlib.sha256(data[offset:offset + count]).hexdigest() == digest
        encoded[name] = (offset, count, fmt, role, source)
    assert encoded["head"][1:] == (65 * 68, "strix.sp4", "head", hashlib.sha256(tensors["head"][2]).hexdigest())
    assert encoded["experts.nv#1"][1:4] == (104, "strix.sp3", "expert_gate_up")
    nv = hashlib.sha256(b"strix-pack-source/v1n")
    nv.update(hashlib.sha256(tensors["experts.nv"][2][128:]).digest())
    nv.update(hashlib.sha256(tensors["experts.nv_scale"][2][16:]).digest())
    nv.update(tensors["experts.nv_amax"][2][4:])
    nv.update(tensors["experts.nv_outer_scale_denominator"][2])
    assert encoded["experts.nv#1"][4] == nv.hexdigest()
    assert encoded["experts.mx#0"][1:4] == (72, "strix.sp2", "expert_down")
    run("verify", "--pack", first, "--model", model, "--expect", data[-32:].hex())
    run("verify", "--pack", first, "--expect", "0" * 64, error="does not match")

    broken = root / "broken.pack"
    corrupt = bytearray(data)
    corrupt[encoded["experts.mx#0"][0]] ^= 32
    broken.write_bytes(corrupt)
    run("probe", "--pack", broken, "--tensor", "head")
    run("probe", "--pack", broken, "--tensor", "experts.mx#0", error="payload digest mismatch")
    run("verify", "--pack", broken, error="whole-file pack digest mismatch")
    corrupt[-32:] = hashlib.sha256(corrupt[:-32]).digest()
    broken.write_bytes(corrupt)
    run("verify", "--pack", broken, error="payload digest mismatch")
    corrupt = bytearray(data)
    at, size = encoded["experts.mx#0"][:2]
    corrupt[at + size] ^= 1
    broken.write_bytes(corrupt)
    run("probe", "--pack", broken, "--tensor", "experts.mx#0")
    run("verify", "--pack", broken, error="whole-file pack digest mismatch")
    broken.write_bytes(data.replace(b"pack/v3", b"pack/v9", 1))
    run("verify", "--pack", broken, error="unsupported low-bit pack version")
    broken.write_bytes(data[:-1])
    run("verify", "--pack", broken, error="truncated")
    broken.write_bytes(data.replace(b"32 e5m3", b"64 e5m3", 1))
    run("verify", "--pack", broken, error="invalid coding metadata")

    # The historical v2 bytes are built independently of the current writer.
    legacy = root / "golden-v2.pack"
    header = (
        b"strix-lowbit-pack/v2 1\ncheckpoint " + b"ab" * 32
        + b"\ntensor golden strix.q4_g64 1 1 128 4096 68 "
        + b"cd" * 32 + b"\nend\n"
    )
    payload = bytes(range(68))
    legacy.write_bytes(header.ljust(4096, b"\n") + payload)
    digest = hashlib.sha256(header + payload).hexdigest()
    run("verify", "--pack", legacy, "--expect", digest)
    run("probe", "--pack", legacy, "--tensor", "golden")
