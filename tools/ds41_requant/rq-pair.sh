#!/bin/bash
# Two-node requantization gate run (rank 0 on cluster1, rank 1 on cluster2). Run ON CLUSTER1 under gpu-turn-pair.sh:
#   ~/ai-cluster-lab/gpu-turn-pair.sh LABEL ~/ai-cluster-lab/scratch-admin/requant/rq-pair.sh TAG ARGS...
# ARGS go to strix-deepseek41-requant-eval on both ranks (same paths on both nodes; relative to the requant dir).
# Guards (integrator's pattern): usb4stream-describe --summary on both nodes must show a cable up and no warning
# before anything opens a stream; usb4auto opens only live streams with deadlines; --connect-timeout 60 per rank;
# watchdog kills both ranks if either has not connected within 90 s or one fails; every run is limited to RUN_LIMIT.
set -u
tag=$1; shift
D=$HOME/ai-cluster-lab/scratch-admin/requant
M=$HOME/ai-cluster-lab/models/DeepSeek-V4.1-Flash-1017/DeepSeek-V4.1-Flash-1017-Q2.gguf
BIN=$D/bin/strix-deepseek41-requant-eval
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib
CONNECT_WAIT=${CONNECT_WAIT:-90}; RUN_LIMIT=${RUN_LIMIT:-300}
for h in local cluster2; do
  if [ $h = local ]; then s=$(timeout 20 usb4stream-describe --summary 2>&1); else s=$(ssh -o BatchMode=yes -o ConnectTimeout=10 cluster2 'timeout 20 usb4stream-describe --summary 2>&1'); fi
  echo "$h: $(echo "$s" | head -1)"
  if ! echo "$s" | head -1 | grep -qE ' [1-9][0-9]* up'; then echo "ABORT: no USB4 cable up on $h"; exit 4; fi
done
args="$*"
mkdir -p $D/logs; ssh -o BatchMode=yes cluster2 "mkdir -p $D/logs; rm -f $D/logs/$tag-r1.log"; rm -f $D/logs/$tag-r0.log
ssh -o BatchMode=yes cluster2 "cd $D && LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib timeout $RUN_LIMIT $BIN $M --rank 1 --transport usb4auto --connect-timeout 60 $args > logs/$tag-r1.log 2>&1; echo rank1 exit \$? >> logs/$tag-r1.log" &
peer=$!
sleep 2
( cd $D && timeout $RUN_LIMIT $BIN $M --rank 0 --transport usb4auto --connect-timeout 60 $args > logs/$tag-r0.log 2>&1; echo "rank0 exit $?" >> logs/$tag-r0.log ) &
local_rank=$!
kill_all() {
  echo "WATCHDOG: $1, killing both ranks"
  pkill -9 -f "$BIN $M --rank 0" 2>/dev/null
  ssh -o BatchMode=yes cluster2 "pkill -9 -f '$BIN $M --rank 1'" 2>/dev/null
}
start=$(date +%s); connected=0
while kill -0 $local_rank 2>/dev/null || kill -0 $peer 2>/dev/null; do
  sleep 3
  r1=$(ssh -o BatchMode=yes cluster2 "grep -c '^connected' $D/logs/$tag-r1.log 2>/dev/null; grep -c 'rank1 exit [1-9]\|^error' $D/logs/$tag-r1.log 2>/dev/null" | tr '\n' ' ')
  r1_conn=$(echo $r1 | awk '{print $1}'); r1_err=$(echo $r1 | awk '{print $2}')
  r0_conn=$(grep -c '^connected' $D/logs/$tag-r0.log 2>/dev/null); r0_err=$(grep -c 'rank0 exit [1-9]\|^error' $D/logs/$tag-r0.log 2>/dev/null)
  if [ "${r0_err:-0}" -gt 0 ] || [ "${r1_err:-0}" -gt 0 ]; then kill_all "a rank failed"; break; fi
  if [ $connected = 0 ]; then
    if [ "${r0_conn:-0}" -gt 0 ] && [ "${r1_conn:-0}" -gt 0 ]; then connected=1
    elif [ $(( $(date +%s) - start )) -gt $CONNECT_WAIT ]; then kill_all "not connected after ${CONNECT_WAIT}s"; break; fi
  fi
done
wait $local_rank 2>/dev/null; wait $peer 2>/dev/null
scp -q cluster2:$D/logs/$tag-r1.log $D/logs/ 2>/dev/null
tail -3 $D/logs/$tag-r0.log; tail -2 $D/logs/$tag-r1.log
