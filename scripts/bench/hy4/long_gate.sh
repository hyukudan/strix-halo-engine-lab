#!/bin/bash
# Hy4-flash long-prompt FAST gate v2: one GPU step per call (run each under the node's GPU turn, one load, <= 150 s).
# Statistics and file layout: long_gate.py. Environment: BIN (strix-hy4-rank), PARTS ("P1.gguf P2.gguf"), DIR (work dir
# holding DOC.ids), GATE (this directory). Every step ends with a one-line summary in $DIR/sum.txt.
#   [SAVES=P,P] [SCORE=A:B,A:B] long_gate.sh ref DOC K UNTIL CTX   piece K of the reference (image ref-DOC-(K-1).img in)
#   long_gate.sh arm DOC N CTX [ROWS]      candidate: windows over [0, N) + 199 steps, routes replayed (needs pos-DOC.npz)
#   long_gate.sh null DOC N CTX            null: reference form from img-DOC-(N-1024), replay + one-ulp nudge (layer 1, seed 1)
#   long_gate.sh pos DOC                   (CPU) reference route pieces -> pos-DOC.npz
set -u
MODE=$1; DOC=$2; : "${BIN:?}" "${PARTS:?}" "${DIR:?}" "${GATE:?}"
S=$DIR/sum.txt; TO=${TO:-140}; BUDGET=${BUDGET:-95}
cd "$DIR" || exit 1
tail_err() { grep -a -E 'error|what\(\)|terminate|exception' "$1" | head -3 | cut -c1-240 >> $S; }
case $MODE in
ref)
  K=$3; UNTIL=$4; CTX=$5
  IN=""; [ "$K" -gt 1 ] && IN="--state-in ref-$DOC-$((K - 1)).img"
  A=""; for p in $(echo ${SAVES:-} | tr , " "); do A="$A --save-at $p:img-$DOC-$p.img"; done
  for r in $(echo ${SCORE:-} | tr , " "); do A="$A --score $r"; done
  STRIX_MOE_ROUTES_OUT=ref-$DOC-$K.rte timeout $TO $BIN $PARTS --ids-file $DOC.ids --gate-ref --verify-rows 8 --context $CTX \
    --until $UNTIL $IN --state-out ref-$DOC-$K.img --budget $BUDGET $A --dump-logits ref-$DOC-$K.f32 > ref-$DOC-$K.log 2> ref-$DOC-$K.err; RC=$?
  echo "ref $DOC $K rc=$RC $(grep -a -h 'gate-ref positions' ref-$DOC-$K.log | cut -c1-200) load $(grep -a -o 's=[0-9.]* GiB/s=[0-9.]*' ref-$DOC-$K.log | head -1)" >> $S
  tail_err ref-$DOC-$K.err ;;
pos)
  F=""; for k in $(ls ref-$DOC-*.log | sed -E "s/.*ref-$DOC-([0-9]+).log/\1/" | sort -n); do
    st=$(grep -a -h -o 'gate-ref positions [0-9]*' ref-$DOC-$k.log | awk '{print $3}'); F="$F ref-$DOC-$k.rte:$st"; done
  python3 -I $GATE/long_gate.py routes-pos pos-$DOC.npz $F | tail -2 >> $S; RC=${PIPESTATUS[0]} ;;
arm)
  N=$3; CTX=$4; ROWS=${5:-2048}; O=arm-$DOC-$N; [ "$ROWS" = 2048 ] || O=armr$ROWS-$DOC-$N
  head -n $N $DOC.ids > p-$DOC-$N.ids; sed -n "$((N + 1)),$((N + 200))p" $DOC.ids > t-$DOC-$N.ids
  python3 -I $GATE/long_gate.py replay-arm pos-$DOC.npz in-$O.rte $N $ROWS 199 > /dev/null || { echo "$O replay file FAILED" >> $S; exit 1; }
  STRIX_MOE_ROUTES_IN=in-$O.rte timeout $TO $BIN $PARTS --ids-file p-$DOC-$N.ids --teacher t-$DOC-$N.ids --gen 200 --top 1     --context $CTX --prompt-rows $ROWS --dump-logits $O.f32 > $O.log 2> $O.err; RC=$?
  B=""; [ "$O" != arm-$DOC-$N ] && [ -f arm-$DOC-$N.f32 ] && { cmp -s $O.f32 arm-$DOC-$N.f32 && B="bitwise==rows2048" || B="DIFFERS-from-rows2048"; }
  echo "$O rc=$RC $B $(grep -a -h '^prefill tokens' $O.log | cut -c1-120) $(grep -a -o '"local_flips":{[^]]*' $O.err | cut -c1-90)" >> $S
  tail_err $O.err ;;
null)
  N=$3; CTX=$4; P0=$((N - 1024))
  python3 -I $GATE/long_gate.py replay-chunks pos-$DOC.npz in-null-$DOC-$N.rte $P0 $((N + 199)) 8 > /dev/null || { echo "null $DOC $N replay file FAILED" >> $S; exit 1; }
  STRIX_MOE_ROUTES_IN=in-null-$DOC-$N.rte STRIX_MOE_ROUTE_NULL=1:1 timeout $TO $BIN $PARTS --ids-file $DOC.ids --gate-ref --verify-rows 8 \
    --context $CTX --state-in img-$DOC-$P0.img --until $((N + 199)) --score $((N - 1)):$((N + 199)) --dump-logits null-$DOC-$N.f32 \
    > null-$DOC-$N.log 2> null-$DOC-$N.err; RC=$?
  echo "null $DOC $N rc=$RC $(grep -a -h 'gate-ref positions' null-$DOC-$N.log | cut -c1-160) $(grep -a -o '"local_flips":{[^]]*' null-$DOC-$N.err | cut -c1-90)" >> $S
  tail_err null-$DOC-$N.err ;;
*) echo "usage: long_gate.sh ref|pos|arm|null DOC ..." >&2; exit 64 ;;
esac
exit ${RC:-0}
