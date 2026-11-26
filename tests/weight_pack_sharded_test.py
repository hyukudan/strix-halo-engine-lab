"""End-to-end source, planner and per-layer pack CLI checks."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def write_shard(path, tensors):
    header = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [len(payload), len(payload) + len(data)]}
        payload.extend(data)
    text = json.dumps(header).encode()
    text += b" " * (-len(text) % 8)
    path.write_bytes(struct.pack("<Q", len(text)) + text + payload)
    return len(payload)


def write_moments(path, consumer, width, full=False):
    ids = struct.pack("<I", 42)
    tag = "v4" if full else "diagonal-v1"
    header = (f"strix-calib-moments/{tag} 1 1 0\nwindow {hashlib.sha256(ids).hexdigest()} 1\n").encode()
    diagonal = [float(1 + 15 * (i % 3 == 0)) for i in range(width)]
    # A positive diagonal plus an outer product exercises error feedback on
    # this correlated input. Importance-only captures store the diagonal.
    values = [(diagonal[i] if i == j else 0.0) + 8.0 for i in range(width)
              for j in range(i + 1)] if full else diagonal
    path.write_bytes(header + ids + f"input x {width} {width} {width} {consumer}\n".encode()
                     + struct.pack(f"<{len(values)}f", *values))


def run(packer, planner, work):
    a, b = "layer.1.expert.0.weight", "layer.1.expert.1.weight"
    mx = b"".join(hashlib.sha256(bytes([i])).digest() for i in range(4))
    scales = bytes([126, 127, 128, 129] * 2)
    fp8_weights = bytes([0x3C]) * (129 * 129)  # 1.5
    fp8_scales = [1.0 + 2**-20, 0.5, 2.0, 0.25]
    payloads = [
        {"other.weight": ("F8_E4M3", [129, 129], fp8_weights),
         a: ("I8", [2, 64], mx)},
        {a + "_scale_inv": ("F8_E8M0", [2, 4], scales),
         b: ("BF16", [2, 128], struct.pack("<H", 0x4120) * 256),
         "other.weight_scale_inv": ("F32", [2, 2], struct.pack("<4f", *fp8_scales))},
    ]
    manifest = {"metadata": {"total_size": 0}, "weight_map": {}}
    for i, tensors in enumerate(payloads):
        name = f"{i}.safetensors"
        manifest["metadata"]["total_size"] += write_shard(work / name, tensors)
        manifest["weight_map"].update(dict.fromkeys(tensors, name))
    (work / "model.safetensors.index.json").write_text(json.dumps(manifest), encoding="utf-8")

    def cli(exe, *args, exit_code=0):
        result = subprocess.run([exe, *args], cwd=work, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert result.returncode == exit_code, (result.returncode, result.stdout[-300:], result.stderr)
        return result

    def probe(name, wants):
        digest = hashlib.sha256(struct.pack(f"<{len(wants)}f", *wants)).hexdigest()
        assert f"fp32_sha256={digest}" in cli(packer, "--model", ".", "--probe-source", name).stdout

    e2m1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    probe(a, [(-1.0 if nibble & 8 else 1.0) * e2m1[nibble & 7] * 2**(scales[i // 16] - 127)
              for i, byte in enumerate(mx) for nibble in [byte & 15, byte >> 4]])
    probe("other.weight", [1.5 * fp8_scales[2 * (r // 128) + c // 128]
                          for r in range(129) for c in range(129)])

    description = {"schema": "strix-pack-model/v1", "expert_groups": [
        {"first_layer": 1, "layers": 1, "experts": 2, "projections": [
            {"name": "layer.{layer}.expert.{expert}.weight", "rows": 2, "columns": 128}
        ]}], "routes": [[1, 0, 4], [1, 1, 9]],
        "retained": [{"name": "other.weight", "bytes": len(fp8_weights) + 16}]}
    (work / "model.json").write_text(json.dumps(description), encoding="utf-8")
    plan_args = ["--model-description", "model.json", "--budget-gib", "1", "--cold", "sp2", "--hot-experts", "1"]
    plan = cli(planner, *plan_args)
    assert f"{a}\tsp2\t72\n{b}\tsp3\t104\n" in plan.stdout, plan.stdout
    assert "hot_experts=1" in plan.stderr
    blocked = cli(planner, *plan_args, "--reserve-gib", "1", exit_code=2)
    assert "hot_experts=0" in blocked.stderr
    base_only = cli(planner, *plan_args, "--budget-gib", str(16817 / (2 ** 30)))
    assert f"{b}\tsp2\t72\n" in base_only.stdout and "hot_experts=0" in base_only.stderr
    (work / "plan.tsv").write_text(plan.stdout, encoding="utf-8")

    for i, name in enumerate([a, b]):
        write_moments(work / f"{i}.moments", name, 128)
    write_moments(work / "full.moments", a, 128, full=True)
    (work / "gate.ids").write_text("17\n", encoding="ascii")
    (work / "gate.txt").write_text("gate 1 gate.ids\n", encoding="ascii")
    base = ["--model", ".", "--plan", "plan.tsv", "--threads", "1"]
    cli(packer, *base, "--technique", "rtn", "--out", "rtn.lp")
    cal = ["--moments", "0.moments", "--moments", "1.moments", "--heldout", "gate.txt"]
    importance = cli(packer, *base, *cal, "--moments-bytes", "1024", "--technique", "importance",
                     "--table", "auto", "--out", "importance.lp", "--report", "importance.tsv")
    assert importance.stderr.count("bytes 1024 proxy diagonal") == 2
    assert "memory limit" in cli(packer, *base, *cal, "--moments-bytes", "1023", "--technique", "importance",
                                 "--out", "too-small.lp", exit_code=1).stderr
    assert not (work / "too-small.lp").exists() and not (work / "too-small.lp.writing").exists()
    assert "feedback requires full" in cli(packer, *base, *cal, "--technique", "both", exit_code=1).stderr
    cli(packer, "--model", ".", "--storage", a + "=sp2", "--threads", "1",
        "--moments", "full.moments", "--heldout", "gate.txt", "--technique", "both", "--out", "full.lp")

    def read_pack(name):
        file = (work / name).read_bytes()
        header = file.split(b"\nend\n", 1)[0].decode().splitlines()
        tensors = {}
        for line in header[2:]:
            _, key, fmt, version, n, k, start, size, digest, metadata = line.split()
            start, size = int(start), int(size)
            assert version == "1" and (n, k) == ("2", "128") and start % 4096 == 0
            assert len(file[start:start + size]) == size
            assert bytes.fromhex(metadata)[1] == int(fmt[-1])
            tensors[key] = (fmt, digest, file[start:start + size])
        return header[1], tensors

    checkpoint, outputs = read_pack("rtn.lp")
    assert outputs[a][0] == "strix.sp2" and outputs[b][0] == "strix.sp3"
    first_source = outputs[a][1]
    expected = hashlib.sha256(b"strix-pack-source/v1m" + hashlib.sha256(mx).digest()
                              + hashlib.sha256(scales).digest()).hexdigest()
    assert first_source == expected
    assert outputs[b][1] == hashlib.sha256(payloads[1][b][2]).hexdigest()
    assert read_pack("importance.lp")[0] == checkpoint
    assert read_pack("full.lp")[1][a][2] != outputs[a][2], "feedback emitted the RTN bytes"

    # Change only the companion scale in a different shard.
    payloads[1][a + "_scale_inv"] = ("F8_E8M0", [2, 4], bytes([125]) + scales[1:])
    write_shard(work / "1.safetensors", payloads[1])
    cli(packer, *base, "--technique", "rtn", "--out", "changed.lp")
    changed, outputs = read_pack("changed.lp")
    assert changed != checkpoint and outputs[a][1] != first_source
    manifest["weight_map"][a] = "1.safetensors"
    (work / "model.safetensors.index.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert "shard mapping" in cli(packer, "--model", ".", "--probe-source", a, exit_code=1).stderr


def ernie(planner, description, work):
    data = json.loads(description.read_text(encoding="utf-8"))
    data["routes"] = [[1 + i // 512, i % 512, 1] for i in range(3225)]
    path = work / "ernie.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    for args, expected, code in [
        (["--cold", "sp2"], "GiB=91.5", 0),
        (["--cold", "sp2", "--hot-experts", "3225"], "GiB=96.3", 0),
        (["--cold", "sp3"], "GiB=123.0", 2),
    ]:
        result = subprocess.run([planner, "--model-description", path, "--budget-gib", "108", *args],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert result.returncode == code and expected in result.stderr, result.stderr
        assert len(result.stdout.splitlines()) == 64513


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="strixpack-sharded-", dir=Path.cwd()) as directory:
        work = Path(directory)
        packer, planner, description = [Path(arg).resolve() for arg in sys.argv[1:]]
        run(packer, planner, work)
        ernie(planner, description, work)
