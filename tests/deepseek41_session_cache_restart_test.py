"""Fresh processes/ranks with growing operational directories; no GPU."""
import json
import struct
import os
import pathlib
import subprocess
import sys
import tempfile

exe = str(pathlib.Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix="ds41-compat-") as tmp:
    root = pathlib.Path(tmp)
    model = root / "model.bin"
    model.write_bytes(b"immutable numerical model" * 4096)
    env = dict(os.environ)
    for name in ("STRIX_HASH_CACHE", "STRIX_PREFIX_DIR", "STRIX_OUTPUT_DIR", "STRIX_LOG_DIR"):
        path = root / name
        path.mkdir()
        env[name] = str(path)
    numerical = root / "trellis"
    numerical.mkdir()
    pack = numerical / "weights.safetensors"
    def write_pack(payload):
        header=json.dumps({'x':{'dtype':'U8','shape':[len(payload)],'data_offsets':[0,len(payload)]}}).encode()
        pack.write_bytes(struct.pack('<Q',len(header))+header+payload)
    write_pack(b"numerical pack")
    env["STRIX_DS41_TRELLIS"] = str(numerical)
    # A coincidental path-valued selector must remain a scalar, not become an asset.
    env["STRIX_TEST_PATH_SELECTOR"] = str(root / "STRIX_LOG_DIR")
    def probe(rank):
        return subprocess.check_output([exe, "--compat-probe", str(model), str(rank)], env=env, text=True).splitlines()
    first = [probe(rank) for rank in range(2)]
    for name in ("STRIX_HASH_CACHE", "STRIX_PREFIX_DIR", "STRIX_OUTPUT_DIR", "STRIX_LOG_DIR"):
        (pathlib.Path(env[name]) / "new-entry").write_text("mutable run output")
    second = [probe(rank) for rank in range(2)]
    assert first == second, "compat key changed across fresh processes/restarts"
    assert first[0][0] != first[1][0], "rank must remain bound"
    assert first[0][1] == first[1][1], "shared compatibility differs across ranks"
    write_pack(b"different numerical pack")
    assert probe(0) != first[0], "immutable numerical directory content not bound"
    write_pack(b"numerical pack")
    model.write_bytes(b"changed numerical model" * 4096)
    assert probe(0) != first[0], "immutable model content not bound"
print("PASS explicit hash-cache directory: two ranks, two fresh runs, mutable outputs excluded, model bound")

# Independent server pairs contend on one set of rank-local disk directories.
# Holding rank 0's OS lock must block publication in another process.
with tempfile.TemporaryDirectory(prefix="ds41-disk-lock-") as tmp:
    root = pathlib.Path(tmp)
    (root / "rank0").mkdir()
    (root / "rank1").mkdir()
    lock = open(root / "rank0" / ".session-rank0.lock", "w+b")
    lock.write(b"0"); lock.flush()
    if os.name == "nt":
        import msvcrt
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    workers = [subprocess.Popen([exe, "--disk-worker", tmp, name]) for name in ("aa", "bb")]
    try:
        import time
        time.sleep(0.5)
        assert all(p.poll() is None for p in workers), "publication escaped interprocess lock"
        assert not list(root.rglob("*.state")), "rank payload published without reservation lock"
    finally:
        lock.close()
    for p in workers:
        assert p.wait(timeout=60) == 0, "shared-directory transaction failed"
    assert not list(root.rglob("*.state")) and not list(root.rglob("*.group")), "delete left published generations"
print("PASS interprocess lock, concurrent quota/publication/replacement/delete")
