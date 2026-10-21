#!/bin/bash
# Reference turn (base model, run on cluster1 under gpu-turn-pair.sh): eval-set slices, greedy build, greedy-set slices.
D=$HOME/ai-cluster-lab/scratch-admin/requant
cd $D
mkdir -p ref/eval ref/greedy ref/long; ssh -o BatchMode=yes cluster2 "mkdir -p $D/ref/eval $D/ref/greedy $D/ref/long"
./rq-pair.sh ref-eval --seq data/eval.u32 --save-ref ref/eval
if [ ! -f data/greedy.u32 ]; then
  ./rq-pair.sh ref-greedy-build --greedy data/greedy.prompts --out data/greedy
  scp -q data/greedy.u32 data/greedy.segs cluster2:$D/data/
fi
./rq-pair.sh ref-greedy --seq data/greedy.u32 --save-ref ref/greedy
