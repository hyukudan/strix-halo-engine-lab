#!/usr/bin/env bash
# One GLM TP2 prefill process pair: rank 0 on this host, rank 1 on --peer (ssh), the same binary (hash checked).
# Usage: pf_pair_ab.sh --bin PATH --model DIR --prompts A.u32,B.u32 --out DIR --peer HOST
#                      [--mode exact|fast] [--env K=V,K=V] [--cap TOKENS] [--budget SECONDS]
#                      [--arms B:-/C:K=V+K=V [--reps N]]
# --arms (bundle tickets): one load, then per rep every arm runs every prompt (strict ABAB inside the process, the rank
# app's --pf-ttft-arms / --pf-ttft-reps, include/strix/env_arms.hpp); pf_clk.py prints the arm label per prefill.
# Give two or more prompts: the first prefill of the process is reported as cold, the last as warm. Writes OUT/r0.tlog and
# r1.tlog (epoch-stamped rank output), samp0/1.txt (20 ms samples: sclk, busy, power, kfd evicted_ms, MemFree,
# MemAvailable) on both hosts, copies rank 1's files back and prints pf_clk.py's per-prefill summary.
# GPU queueing is the caller's job: run this under the lab's ticket wrapper, one arm per call, arms alternated ABAB.
# Paths must be the same on both hosts (binary, model, prompts, OUT).
set -uo pipefail
KIT=$(cd "$(dirname "$0")" && pwd)
mode=fast env_list=- cap=40960 budget=110 bin='' model='' prompts='' out='' peer='' arms='' reps=1
while [ $# -gt 1 ]; do
    case $1 in
        --bin) bin=$2 ;; --model) model=$2 ;; --prompts) prompts=$2 ;; --out) out=$2 ;; --peer) peer=$2 ;;
        --arms) arms=$2 ;; --reps) reps=$2 ;;
        --mode) mode=$2 ;; --env) env_list=$2 ;; --cap) cap=$2 ;; --budget) budget=$2 ;;
        *) echo "pf_pair_ab: unknown option $1" >&2; exit 2 ;;
    esac
    shift 2
done
[ -n "$bin" ] && [ -n "$model" ] && [ -n "$prompts" ] && [ -n "$out" ] && [ -n "$peer" ] || { sed -n '2,4p' "$0" >&2; exit 2; }
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=5 -o ServerAliveInterval=15 -o ServerAliveCountMax=8"
t0=$(date +%s)
h0=$(sha256sum "$bin" | cut -c1-12)
h1=$($SSH "$peer" "sha256sum $bin" | cut -c1-12)
[ "$h0" = "$h1" ] || { echo "ABORT binaries differ ($h0 / $h1)"; exit 4; }
rm -rf "$out"; mkdir -p "$out"
$SSH "$peer" "rm -rf $out; mkdir -p $out"
scp -q -o BatchMode=yes "$KIT/pf_samp.py" "$peer:$out/" || { echo "ABORT copy to peer"; exit 4; }
cp "$KIT/pf_samp.py" "$out/"
for v in $(env | grep -oE '^STRIX_[A-Z0-9_]*'); do unset "$v"; done
base="STRIX_DIST_USB4_TIMEOUT_MS=120000 STRIX_DIST_WAIT_TIMEOUT_MS=120000 HSA_DISABLE_FRAGMENT_ALLOCATOR=1"
xe=''; [ "$env_list" != - ] && xe=${env_list//,/ }
prof=''; [ "$mode" = fast ] && prof='--profile fast'
args="--world 2 --transport usb4-auto --mtp 0 --slots 1 --capacity $cap --max-rows 8192 --pf-ttft $prompts --pf-ttft-mode gemv $prof"
[ -n "$arms" ] && args="$args --pf-ttft-arms $arms --pf-ttft-reps $reps"
rem=$((budget - ($(date +%s) - t0)))
python3 "$out/pf_samp.py" sample "$out/samp0.txt" $((rem - 4)) > /dev/null 2>&1 < /dev/null &
s0=$!
$SSH "$peer" "cd $out || exit 1; (python3 pf_samp.py sample samp1.txt $((rem - 4)) > /dev/null 2>&1 < /dev/null &);
    env $base $xe timeout -k 10 $((rem - 8)) stdbuf -oL -eL $bin $model --rank 1 $args 2>&1 | python3 -u pf_samp.py stamp r1.tlog" &
r1=$!
(cd "$out" && env $base $xe timeout -k 10 $((rem - 5)) stdbuf -oL -eL "$bin" "$model" --rank 0 $args 2>&1 |
    python3 -u pf_samp.py stamp r0.tlog)
rc0=$?
wait $r1
rc1=$?
kill $s0 2> /dev/null
scp -q -o BatchMode=yes "$peer:$out/r1.tlog" "$peer:$out/samp1.txt" "$out/" 2> /dev/null
echo "== bin $h0 mode $mode env $env_list rc $rc0/$rc1 t+$(($(date +%s) - t0)) s"
grep -ahiE 'load [0-9.]+ s|rror|FAIL|prefill identity|AGREE|DIFFER' "$out/r0.tlog" | cut -c16-230 | sort -u | head -8
python3 "$KIT/pf_clk.py" "$out"
