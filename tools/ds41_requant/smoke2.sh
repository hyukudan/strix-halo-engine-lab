#!/bin/bash
# CB4 kernels: layer + formats tests (bitwise vs the Q8_0 form) and the 4-layer pipeline with the overlay packed vs not
cd ~/ai-cluster-lab/scratch-admin/requant
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib
M=~/ai-cluster-lab/models/DeepSeek-V4.1-Flash-1017/DeepSeek-V4.1-Flash-1017-Q2.gguf
timeout 200 build3/strix-hip-ds41-layer-tests > smoke/layer-tests.log 2>&1; echo layer-tests $?
timeout 200 build3/strix-hip-ds41-formats-tests > smoke/formats-tests.log 2>&1; echo formats-tests $?
B=build3/strix-deepseek41-requant-eval
E="--transport emulate --layers 4"
for ov in smoke-L4 smoke-L4sh; do
  timeout 200 $B $M $E --overlay ov/$ov.gguf --seq data/eval.u32 --limit 512 --compare-ref smoke/ref --out smoke/$ov-q8 > smoke/$ov-q8.log 2>&1; echo $ov-q8 $?
  timeout 200 $B $M $E --overlay ov/$ov.gguf --cb4 --seq data/eval.u32 --limit 512 --compare-ref smoke/ref --out smoke/$ov-cb4 > smoke/$ov-cb4.log 2>&1; echo $ov-cb4 $?
  cmp smoke/$ov-q8.r0 smoke/$ov-cb4.r0 && cmp smoke/$ov-q8.r1 smoke/$ov-cb4.r1 && echo "$ov CB4-PIPELINE-BITWISE-EQUAL"
done
