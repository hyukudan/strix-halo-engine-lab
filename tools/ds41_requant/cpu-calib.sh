#!/bin/bash
# CPU activation calibration: every calibration document (BOS + 127 tokens) through the 40-layer host forward,
# per-channel sums of squares of the projection inputs, then merged into calib/cpu.imat.
cd ~/ai-cluster-lab/scratch-admin/requant
M=~/ai-cluster-lab/models/DeepSeek-V4.1-Flash-1017/DeepSeek-V4.1-Flash-1017-Q2.gguf
mkdir -p calib/cpu
while read -r cat doc start len; do
  ids=$(python3 -c "import numpy as np; s=np.fromfile('data/calib.u32','<u4'); print(','.join(map(str,s[$start:$start+$len])))")
  echo "$doc $ids"
done < data/calib.segs | xargs -P 8 -L 1 bash -c 'nice -n 19 build4/strix-deepseek41-trimmed '"$M"' --layers 40 --prompt-ids $1 --generate 0 --calib calib/cpu/$0.imat > calib/cpu/$0.log 2>&1; echo "$0 $?"'
python3 - <<'PY'
import glob, numpy as np
tot = {}
for f in sorted(glob.glob("calib/cpu/*.imat")):
    d = open(f, "rb").read(); p = 0
    while p < len(d):
        nl = d.index(b"\n", p); lab, n, c = d[p:nl].decode().split(); n, c = int(n), int(c)
        v = np.frombuffer(d, "<f8", n, nl + 1); p = nl + 1 + 8 * n
        s = tot.setdefault(lab, [np.zeros(n), 0]); s[0] += v; s[1] += c
with open("calib/cpu.imat", "wb") as f:
    for lab, (v, c) in sorted(tot.items()):
        f.write(f"{lab} {v.size} {c}\n".encode()); f.write(v.astype("<f8").tobytes())
print("merged", len(tot), "inputs")
PY
