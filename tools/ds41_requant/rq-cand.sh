#!/bin/bash
# Candidate turn (run on the rank-0 node; rank 1 on $PEER): rq-cand.sh SET NAME [EVAL ARGS...] for each NAME in a
# comma-separated list: compare overlay ov/NAME.gguf (or "base") against ref/SET, then combine the ranks' statistics.
D=${RQ_DIR:-$PWD}
cd $D
set=$1; names=$2; shift 2
mkdir -p res; ssh -o BatchMode=yes "${PEER:?rank-1 host}" "mkdir -p $D/res"
for name in ${names//,/ }; do
  ov=""; [ "$name" != base ] && ov="--overlay ov/$name.gguf"
  ./rq-pair.sh c-$set-$name --seq data/$set.u32 $ov --compare-ref ref/$set --out res/$set-$name "$@"
  scp -q "$PEER":$D/res/$set-$name.r1 res/ && \
    python3 tools/gate.py res/$set-$name data/$set.u32 data/$set.segs --json res/$set-$name.json $([ $set = greedy ] && echo --greedy) > res/$set-$name.txt 2>&1
  cat res/$set-$name.txt
done
