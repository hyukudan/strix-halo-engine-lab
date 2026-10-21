#!/bin/bash
# single-node smoke (4 layers, both ranks emulated in one process)
cd ~/ai-cluster-lab/scratch-admin/requant
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib
M=~/ai-cluster-lab/models/DeepSeek-V4.1-Flash-1017/DeepSeek-V4.1-Flash-1017-Q2.gguf
B=build/strix-deepseek41-requant-eval
mkdir -p smoke/ref
E="--transport emulate --layers 4"
timeout 280 $B $M $E --seq data/eval.u32 --limit 512 --save-ref smoke/ref > smoke/ref.log 2>&1; echo ref $?
timeout 280 $B $M $E --seq data/eval.u32 --limit 512 --compare-ref smoke/ref --out smoke/base > smoke/base.log 2>&1; echo base $?
timeout 280 $B $M $E --overlay ov/smoke-L4.gguf --seq data/eval.u32 --limit 512 --compare-ref smoke/ref --out smoke/nl4 > smoke/nl4.log 2>&1; echo nl4 $?
timeout 280 $B $M $E --seq data/eval.u32 --limit 64 --rows 1 --compare-ref smoke/ref --out smoke/r1 > smoke/r1.log 2>&1; echo r1 $?
timeout 280 $B $M $E --greedy data/greedy.prompts --out smoke/greedy > smoke/greedy.log 2>&1; echo greedy $?
timeout 280 $B $M $E --seq data/calib.u32 --limit 64 --calib smoke/calib.imat > smoke/calib.log 2>&1; echo calib $?
