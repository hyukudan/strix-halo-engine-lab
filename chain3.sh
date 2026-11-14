#!/bin/bash
# chain3.sh (cluster2): after chain2: build ring app + host test -> copy -> 4 pair tickets -> sum3.out + chain3.done
L=$HOME/ai-cluster-lab; K=$L/dspf; cd $K; rm -f chain3.done; O=$K/sum3.out; : > $O
S1="ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=30 cluster1"
fin() { echo "$1" > $K/chain3.done; exit 0; }
while [ ! -e $K/chain2.done ]; do sleep 20; done
{
echo "chain3 start $(date +%H:%M)"; rm -f $K/bin/app-ring
$L/cpu-build.sh dspf-b2 --compile-only -- bash $K/b2.sh 2>&1 | grep -v "cpu-build: waiting" | tail -6
if [ ! -x $K/bin/app-ring ]; then echo "BUILD FAILED"; grep -a -m6 "error\|FAILED" $K/b2.log | cut -c1-300; fin fail; fi
tail -n 2 $K/b2.test.log | cut -c1-200
echo "build end $(date +%H:%M)"
scp -q -o BatchMode=yes $K/bin/app-ring cluster1:$K/bin/ || { echo "COPY FAILED"; fin fail; }
for t in 6 7 8 9; do
  $S1 "[ -e $L/.gpu-turn/maintenance ] && echo MAINTENANCE || GPU_TURN_MAX=120 $L/gpu-turn-pair.sh dspf-t$t bash $K/t2.sh $t" 2>&1 | grep -a "^SUM\|ABORT\|MAINT\|CLAMP\|REFUSE\|DONE\|rc_r0=[1-9]\|rc_r1=[1-9]" | cut -c1-420
  echo "ticket $t end $(date +%H:%M)"
done
} >> $O 2>&1
echo ok > $K/chain3.done
