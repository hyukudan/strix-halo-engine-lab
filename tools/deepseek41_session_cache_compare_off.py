"""CPU-only native default/OFF qualification evidence extraction and bitwise cmp."""
import pathlib
import re
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
if len(sys.argv) > 2:
    assert sys.argv[2] == "--extract"
    arm, prompt = sys.argv[3], int(sys.argv[4])
    for rank in range(2):
        stem = root / f"rank{rank}/{arm}"
        log = stem.with_suffix(".log").read_text()
        ids = re.findall(r"^pos \d+ fed \d+ argmax (\d+) top .*", log, re.M)
        digest = re.findall(rf"^state rank {rank} position (\d+) digest ([0-9a-f]{{16}}) file .*", log, re.M)
        assert len(ids) == prompt + 64 and len(digest) == 1, (arm, rank, "incomplete evidence")
        assert int(digest[0][0]) == prompt + 64
        assert (root / f"rank{rank}/{arm}-r{rank}.f32").stat().st_size == (prompt + 64) * 129280 * 4
        stem.with_suffix(".ids").write_text("\n".join(ids) + "\n")
        stem.with_suffix(".digest").write_text(" ".join(digest[0]) + "\n")
else:
    for arm in ("old", "off", "unnamed"):
        assert (root / f"{arm}.done").is_file(), f"missing successful {arm} receipt"
    for rank in range(2):
        for suffix in (".ids", ".digest", f"-r{rank}.f32", f"-r{rank}.state"):
            baseline = root / f"rank{rank}/old{suffix}"
            assert baseline.stat().st_size > 0
            for arm in ("off", "unnamed"):
                subprocess.run(["cmp", str(baseline), str(root / f"rank{rank}/{arm}{suffix}")], check=True)
    print("PASS old (05fedd80) == off == unnamed: both ranks, ids, all logits bytes, complete state bytes and position/digest")
