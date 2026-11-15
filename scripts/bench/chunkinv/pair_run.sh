#!/usr/bin/env bash
# Chunk-invariance pair run: one two-rank strix-minimax-rank process pair, run from rank 0's node inside the caller's
# GPU turn. Records per rank: TTFT lines, state/tokens FNV, MemAvailable at launch, kfd evicted_ms (last sample of the
# rank process), busy / sclk / temperature samples. Hosts, binaries and model come from arguments only.
# usage: pair_run.sh --bin RANK_BIN --model DIR --peer SSH_HOST --out DIR [--secs N] [ENV=VAL ...] -- [rank args]
set -u
bin='' model='' peer='' out='' secs=600
envs=()
while [[ $# -gt 0 && $1 != -- ]]; do
    case $1 in
        --bin) bin=$2; shift 2 ;; --model) model=$2; shift 2 ;; --peer) peer=$2; shift 2 ;;
        --out) out=$2; shift 2 ;; --secs) secs=$2; shift 2 ;;
        *=*) envs+=("$1"); shift ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done
shift
[[ -n $bin && -n $model && -n $peer && -n $out ]] || { sed -n 5p "$0" >&2; exit 2; }
ssh_peer() { ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$peer" "$@"; }
rm -rf "$out"; mkdir -p "$out" || exit 2; ssh_peer "rm -rf '$out'; mkdir -p '$out'" || exit 2
memavail='awk "/MemAvailable/{printf \"%.1f\", \$2/1048576}" /proc/meminfo'
echo "memavail_gib r0 $(bash -c "$memavail") r1 $(ssh_peer "$memavail")" > "$out/mem.txt"
# Monitor: GPU busy/sclk/temp and the rank's kfd evicted_ms (summed over its GPUs), once a second.
mon='d=/sys/class/drm/card1/device; [ -d $d ] || d=/sys/class/drm/card0/device
while :; do ev=0; for p in $(pgrep -f "[s]trix-minimax-rank"); do for f in /sys/class/kfd/kfd/proc/$p/stats_*/evicted_ms; do
[ -r "$f" ] && ev=$((ev + $(cat "$f"))); done; done
echo "$(date +%s.%N) busy=$(cat $d/gpu_busy_percent) sclk=$(grep "\*" $d/pp_dpm_sclk | tr -d "\n") temp=$(cat $d/hwmon/hwmon*/temp1_input) evicted_ms=$ev"
sleep 1; done'
bash -c "$mon" > "$out/mon0.txt" 2>/dev/null & m0=$!
ssh_peer "nohup bash -c '$mon' > '$out/mon1.txt' 2>/dev/null & echo \$!" > "$out/m1.pid"
cleanup() { kill "$m0" 2>/dev/null; ssh_peer "kill $(cat "$out/m1.pid" 2>/dev/null) 2>/dev/null"; }
trap cleanup EXIT
ssh_peer "cd '$out' && exec env ${envs[*]:-} setsid timeout -k 2 $secs '$bin' '$model' 1 usb4-auto '$out/r1' $* > '$out/r1.log' 2>&1" & peer_pid=$!
(cd "$out" && exec env "${envs[@]}" setsid timeout -k 2 "$secs" "$bin" "$model" 0 usb4-auto "$out/r0" "$@" > "$out/r0.log" 2>&1); rc0=$?
wait "$peer_pid"; rc1=$?
scp -q "$peer:$out/mon1.txt" "$peer:$out/r1.log" "$out/" 2>/dev/null
ev() { grep -o 'evicted_ms=[0-9]*' "$1" 2>/dev/null | awk -F= '{ if ($2 + 0 > m) m = $2 + 0 } END { print m + 0 }'; }
{
    echo "rc $rc0/$rc1 $(cat "$out/mem.txt") evicted_ms r0 $(ev "$out/mon0.txt") r1 $(ev "$out/mon1.txt")"
    grep -a -h -o -E 'prompt [0-9]+ tokens, TTFT [0-9.]+ s, prefill [0-9.]+ tok/s' "$out/r0.log"
    grep -a -h -o -E '(state|tokens) fnv [0-9a-f]+' "$out/r0.log" "$out/r1.log" | sort | uniq -c
    for r in 0 1; do
        awk -v r=$r '{for(i=2;i<=NF;i++){split($i,a,"=");if(a[1]=="busy"){b+=a[2];n++} } for(i=2;i<=NF;i++) if($i ~ /^[0-9]+Mhz$/){s+=$i+0; break}}
            END{if(n)printf "r%d mean busy %.0f%% sclk %.0f MHz over %d s\n", r, b/n, s/n, n}' "$out/mon$r.txt"
    done
} > "$out/summary.txt"
cat "$out/summary.txt"
