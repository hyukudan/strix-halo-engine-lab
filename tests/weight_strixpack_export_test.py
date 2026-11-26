"""Exercise the calibration tool's safetensors and legacy TSV output."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def run(executable, work):
    weights = {
        'projection"quote': struct.pack("<H", 0x41C0) * 128,  # 24
        r"projection\slash": struct.pack("<H", 0xC240) * 128,  # -48
    }
    tensors = {}
    payload = bytearray()
    for name, data in weights.items():
        tensors[name] = {"dtype": "BF16", "shape": [1, 128],
                         "data_offsets": [len(payload), len(payload) + len(data)]}
        payload.extend(data)
    header = json.dumps(tensors).encode()
    header += b" " * (-len(header) % 8)
    (work / "source.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)

    ids = struct.pack("<I", 42)
    covariance = b"".join(struct.pack("<f", float(col == row))
                          for row in range(128) for col in range(row + 1))
    consumers = ",".join(weights)
    moments_header = (f"strix-calib-moments/v4 1 1 0\nwindow {hashlib.sha256(ids).hexdigest()} 1\n").encode()
    inputs = f"input activation 128 1 1 {consumers}\n".encode()
    (work / "cal.moments").write_bytes(moments_header + ids + inputs + covariance)
    (work / "gate.ids").write_text("17\n", encoding="ascii")
    (work / "gate.txt").write_text("gate 1 gate.ids\n", encoding="ascii")
    command = [str(executable), "--moments", "cal.moments", "--heldout", "gate.txt", "--technique", "rtn", "--threads", "1"]
    subprocess.run(command + ["--model", "source.safetensors", "--storage", "projection=sp4",
                              "--dequant-out", "decoded.safetensors"],
                   cwd=work, check=True)

    decoded = (work / "decoded.safetensors").read_bytes()
    header_len, = struct.unpack_from("<Q", decoded)
    assert header_len % 8 == 0
    data_offset = 8 + header_len
    reopened = json.loads(decoded[8:data_offset])
    reopened.pop("__metadata__")
    assert reopened == tensors, reopened
    assert decoded[data_offset:] == payload

    subprocess.run(command + ["--model", "decoded.safetensors", "--storage", "projection=q4",
                              "--report", "report.tsv", "--out", "pack.bin"],
                   cwd=work, check=True)
    report = (work / "report.tsv").read_text(encoding="utf-8").splitlines()
    assert report[0] == "tensor\tinput\tcal_rows\teval_rows\tformat\ttechnique\trel_error\tselected"
    assert len(report) == 3 and all(len(line.split("\t")) == 8 for line in report)
    assert {line.split("\t")[0] for line in report[1:]} == weights.keys()

    checked = command.copy()
    checked[checked.index("rtn")] = "importance"
    subprocess.run(checked + ["--model", "source.safetensors", "--storage", "projection=sp4",
                              "--out", "checked.bin", "--checked-role", "attention"],
                   cwd=work, check=True)
    pack = (work / "checked.bin").read_bytes()
    assert pack.startswith(b"strix-lowbit-pack/v3 2\n")
    assert hashlib.sha256(pack[:-32]).digest() == pack[-32:]
    fit = hashlib.sha256((work / "cal.moments").read_bytes()).digest()
    expected = hashlib.sha256(b"strix-pack-calibration@1" + fit + struct.pack("<d", 0.01)).hexdigest()
    for line in pack.split(b"\nend\n", 1)[0].decode().splitlines()[2:]:
        assert line.split()[12:15] == ["attention", "clip-importance-fixed@1", expected]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="strixpack-export-", dir=Path.cwd()) as directory:
        run(Path(sys.argv[1]).resolve(), Path(directory))
