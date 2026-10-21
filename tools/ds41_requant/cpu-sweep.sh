#!/bin/bash
# CPU (host forward) gate while the two-node path is unavailable: cpu-sweep.sh N VARIANT... (base = no overlay)
# Every eval document's first N tokens (from its BOS) through the 40-layer host forward, full logits per step.
cd ~/ai-cluster-lab/scratch-admin/requant
M=~/ai-cluster-lab/models/DeepSeek-V4.1-Flash-1017/DeepSeek-V4.1-Flash-1017-Q2.gguf
N=$1; shift
jobs=()
while read -r cat doc start len; do
  ids=$(python3 -c "import numpy as np; s=np.fromfile('data/eval.u32','<u4'); print(','.join(map(str,s[$start:$start+$N])))")
  for v in "$@"; do jobs+=("$v $doc $ids"); done
done < data/eval.segs
printf "%s\n" "${jobs[@]}" | xargs -P ${PAR:-12} -L 1 bash -c '
  v=$0; doc=$1; ids=$2; mkdir -p cpu/$v; [ -s cpu/$v/$doc.f32 ] && exit 0
  ov=""; [ "$v" != base ] && ov="--overlay ov/$v.gguf"
  nice -n 19 build3/strix-deepseek41-trimmed '"$M"' --layers 40 --prompt-ids $ids --generate 0 $ov --logits cpu/$v/$doc.tmp > cpu/$v/$doc.log 2>&1 && mv cpu/$v/$doc.tmp cpu/$v/$doc.f32; echo "$v $doc $?"'
