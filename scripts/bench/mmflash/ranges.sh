#!/bin/bash
# ranges.sh REF ARM... : MiniMax MSA flash layer-range qualification in one detached run (marker $MMF_RUN/../ranges.done,
# summary ranges.summary next to it). REF = sealed exact run id on the same binary and prompts. ARM syntax:
#   e<tag>            exact arm (STRIX_MM_PF_MSA_FLASH=0)
#   n<L>s<seed>       one-ulp null on the attention output of layer L (the gate's onset control; diagnostic hook binary)
#   f<A>-<B>m<11|12>  candidate: MSA flash on layers A..B, mode 11 = STRIX_MM_PF_MSA_FLASH=1 (one-term), 12 = 2 (two-term)
#   d<A>-<B>s<seed>g<mag>  diagnostic multi-site null: +-mag ulps on every layer A..B (absolute comparison only, never a floor)
# Every arm is one pair ticket ($MMF_TURN LABEL CMD...), two lanes. Candidates get the floor rule (onset nulls n<A>s*,
# --dnll-rule floor --engine minimax --profile fast-prefill, needs the MiniMax onset rule in the gate tool) and the
# absolute comparison; summary.py writes the table. Environment: as seal.sh, plus MMF_TURN, MMF_SESSIONS (default
# cpros8k,ccode8k,cdata8k), MMF_CAP (8256), MMF_SECS (105), MMF_LABEL (ticket label prefix).
set -u
REF=$1; shift
: "${MMF_RUN:?}" "${MMF_TURN:?}" "${MMF_GATE:?}" "${MMF_PROMPTS:?}"
H=$(dirname "$0"); K=$(dirname "$MMF_RUN"); S=$K/ranges.summary
SESS=${MMF_SESSIONS:-cpros8k,ccode8k,cdata8k} CAP=${MMF_CAP:-8256} SECS=${MMF_SECS:-105} LBL=${MMF_LABEL:-mmflash}
mkdir -p "$K/tk" "$K/cmp" "$MMF_RUN"; rm -f "$K/ranges.done"; : > "$S"
say() { echo "$*" >> "$S"; }
env_of() {
  case $1 in
    e*) echo "STRIX_MM_PF_MSA_FLASH=0";;
    n*) local x=${1#n}; echo "STRIX_MM_PF_MSA_FLASH=0 STRIX_MM_DIAG_ATTN_ULP=${x%%s*}:${x##*s}";;
    f*) local x=${1#f}; echo "STRIX_MM_PF_MSA_FLASH=$(( ${x##*m} - 10 )) STRIX_MM_PF_MSA_FLASH_LAYERS=${x%%m*}";;
    d*) local x=${1#d} r s g; r=${x%%s*}; s=${x#*s}; g=${s##*g}; s=${s%%g*}; echo "STRIX_MM_PF_MSA_FLASH=0 STRIX_MM_DIAG_ATTN_ULP=$r:$s:$g";;
    *) return 1;;
  esac
}
for a in "$@"; do env_of "$a" > /dev/null || { say "bad arm $a"; touch "$K/ranges.done"; exit 64; }; done
arm() {
  local a=$1
  $MMF_TURN "$LBL-$a" bash "$H/arm.sh" "$a" "$SECS" "$CAP" "$SESS" $(env_of "$a") > "$K/tk/$a.out" 2>&1
  if bash "$H/seal.sh" "$a" "$SESS" >> "$K/tk/$a.out" 2>&1; then python3 "$H/summary.py" timing "$MMF_RUN/$a" >> "$K/timing.txt"
  else say "  $a: $(tail -2 "$K/tk/$a.out" | tr '\n' ' ' | cut -c1-200)"; fi
}
say "start $(date +%T) ref $REF arms $*"
L0=(); L1=(); i=0; for a in "$@"; do if [ $((i % 2)) = 0 ]; then L0+=("$a"); else L1+=("$a"); fi; i=$((i + 1)); done
( for a in "${L0[@]}"; do arm "$a"; done ) &
( sleep 20; for a in "${L1[@]}"; do arm "$a"; done ) &
wait
say "tickets done $(date +%T)"
T=""; for p in ${SESS//,/ }; do T="$T --tokens $p=$MMF_PROMPTS/$p.u32"; done
ok() { [ "$(cat "$MMF_RUN/$1/seal.verdict" 2>/dev/null)" = OK ]; }
for a in "$@"; do
  case $a in f*|d*) ;; *) continue;; esac
  ok "$a" || { say "  $a: not sealed"; continue; }
  nice python3 "$MMF_GATE" compare "$MMF_RUN/$REF" "$MMF_RUN/$a" --ranks 0 1 $T --json "$K/cmp/$a-abs.json" > "$K/cmp/$a-abs.txt" 2>&1
  [ "${a#f}" = "$a" ] && continue
  o=${a#f}; o=${o%%-*}; nl=""
  for n in "$@"; do case $n in n${o}s*) ok "$n" && nl="$nl --null-arm $MMF_RUN/$n";; esac; done
  nice python3 "$MMF_GATE" compare "$MMF_RUN/$REF" "$MMF_RUN/$a" --ranks 0 1 $T --dnll-rule floor --engine minimax \
      --profile fast-prefill $nl --json "$K/cmp/$a.json" > "$K/cmp/$a.txt" 2>&1
done
python3 "$H/summary.py" table "$K/cmp" "$K/timing.txt" >> "$S" 2>&1
say "done $(date +%T)"
touch "$K/ranges.done"
