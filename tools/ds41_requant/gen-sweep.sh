#!/bin/bash
# generate one overlay per class at FMT (sensitivity sweep); usage: gen-sweep.sh FMT [IMATRIX] [TAGSUFFIX]
cd ~/ai-cluster-lab/scratch-admin/requant
M=~/ai-cluster-lab/models/DeepSeek-V4.1-Flash-1017/DeepSeek-V4.1-Flash-1017-Q2.gguf
PY=${PY:-python3}
fmt=$1; im=${2:-}; suf=${3:-}
for c in ${CLASSES:-q_a q_b kv wo_a wo_b sh_gu sh_down head}; do
  id=s-$c-$fmt$suf
  [ -f ov/$id.gguf ] && continue
  extra=""; [ -n "$im" ] && extra="--imatrix $im"
  nice -n 15 $PY -W ignore tools/requant.py $M ov/$id.gguf --identity $id --spec $c=$fmt --jobs ${JOBS:-12} --stats ov/$id.json $extra > ov/$id.log 2>&1
  echo "$id $? $(tail -1 ov/$id.log)"
done
