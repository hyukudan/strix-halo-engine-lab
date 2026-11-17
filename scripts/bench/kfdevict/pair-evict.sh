#!/usr/bin/env bash
# pair-evict.sh OUT BIN MODEL_DIR ARM ENV SESSIONS: ONE MiniMax pair ticket (one arm, one process per rank) that
# reports the KFD queue evictions per prefill window and the page-compaction work of both nodes during the ticket.
#
#   <queue>/gpu-turn-pair.sh LABEL <repo>/scripts/bench/kfdevict/pair-evict.sh OUT BIN MODEL_DIR base '' a=A.u32
#   <queue>/gpu-turn-pair.sh LABEL <repo>/scripts/bench/kfdevict/pair-evict.sh OUT BIN MODEL_DIR off \
#       'STRIX_HOST_DRIVER_PAGES=0' a=A.u32,b=B.u32,c=C.u32
#
# ENV: space-separated NAME=VALUE given to both ranks ('' = none). SESSIONS: name=ids.u32[,...], run in order in one
# process (first = cold, right after load; a 3-session arm reaches ~35 s after load). KIT_PEER as for every pair kit.
# The run itself is the minimax/fast-gate timing route (--rows 0): usual receipts and VALID / INVALID verdict in
# OUT/summary.txt. Added, in OUT/evict.txt (also on stdout), then OUT/evict.done:
#   verdict <the kit's "== ARM ..." lines>
#   mode ARM r0 <"runtime host memory: ..." receipt | runtime default> | shmem maps N, device maps N
#   ev ARM rK load +MS | pf1 +MS (T0-T1 s) | pf2 +MS (...) | ... | after +MS | total MS
#   vm ARM rK compact_stall +N compact_fail +N compact_success +N compact_daemon_wake +N pgmigrate_success +N ...
# A prefill window is the kit's: last sample at or before its start .. first sample at or after its end.
# Reads only what an unprivileged user can read; changes no setting; never builds.
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ $# -eq 6 ]] || { echo 'usage: pair-evict.sh OUT BIN MODEL_DIR ARM ENV SESSIONS' >&2; exit 64; }
out=$1 bin=$2 model=$3 arm=$4 envs=$5 sessions=$6
[[ -n ${KIT_PEER:-} ]] || { echo 'pair-evict: KIT_PEER is not set (ssh alias of the second node)' >&2; exit 64; }
peer_out=${KIT_PEER_OUT:-$out}
# Rank 1 prints its ready line only once rank 0 exists: do not spend the ticket waiting for it (the kit's own knob).
export KIT_PEER_WAIT=${KIT_PEER_WAIT:-8}
peer() { ssh -o BatchMode=yes -o ServerAliveInterval=15 "$KIT_PEER" "$@"; }
keys='compact_stall compact_fail compact_success compact_daemon_wake compact_daemon_migrate_scanned compact_migrate_scanned compact_isolated pgmigrate_success pgmigrate_fail'
vm() { awk -v keys="$keys" 'BEGIN { n = split(keys, k, " "); for (i = 1; i <= n; i++) want[k[i]] = 1 } ($1 in want) { print $1, $2 }' /proc/vmstat; }
vm_delta() {  # BEFORE AFTER -> "key +delta ..." in the order of $keys
    awk -v keys="$keys" 'FNR == NR { a[$1] = $2; next } { b[$1] = $2 }
        END { n = split(keys, k, " "); for (i = 1; i <= n; i++) if (k[i] in b) printf " %s +%d", k[i], b[k[i]] - a[k[i]] }' "$1" "$2"
}
# Per-window evictions of one rank directory (samples.tsv column 9 = evicted_ms of the process tree; TTFT lines in
# stdout.ts give each window's end and length). Runs through "bash -s" locally and on the peer.
windows() {
    cat <<'EOS'
export LC_ALL=C
d=$1
[ -s "$d/samples.tsv" ] && [ -s "$d/stdout.ts" ] || { echo "no samples or stdout in $d"; exit 0; }
ttft() { grep -a 'TTFT' "$d/stdout.ts" | awk '{ for (i = 1; i <= NF; i++) if ($i == "TTFT") printf "%.3f %.3f\n", $1 - $(i + 1), $1 }'; }
awk 'FNR == NR { s[++k] = $1; e[k] = $2; next }
     FNR == 1 { t0 = $1 }
     { t = $1; v = $9
       for (i = 1; i <= k; i++) { if (t <= s[i]) lo[i] = v; if (t >= e[i] && !(i in hi)) hi[i] = v }
       if (v > last) last = v }  # the counter disappears with the process: keep its maximum
     END { if (k == 0) { printf "no prefill window | total %d\n", last; exit }
           if (!(1 in lo)) lo[1] = 0
           printf "load +%d", lo[1]; prev = lo[1]
           for (i = 1; i <= k; i++) { if (!(i in lo)) lo[i] = prev; if (!(i in hi)) hi[i] = last
               printf " | pf%d +%d (%.1f-%.1f s)", i, hi[i] - lo[i], s[i] - t0, e[i] - t0; prev = hi[i] }
           printf " | after +%d | total %d\n", last - hi[k], last }' <(ttft) "$d/samples.tsv"
EOS
}
mkdir -p -- "$out"; rm -f -- "$out/evict.done"
vm > "$out/vm0.before"; peer cat /proc/vmstat < /dev/null | awk -v keys="$keys" 'BEGIN { n = split(keys, k, " "); for (i = 1; i <= n; i++) want[k[i]] = 1 } ($1 in want) { print $1, $2 }' > "$out/vm1.before"
# Mapping evidence for rank 0, once, after its first TTFT line (load done, every runtime buffer exists): how many
# shared anonymous host maps (the runtime's user-pointer memory) and how many device maps the process has.
(
    for ((i = 0; i < 170; i++)); do
        sleep 1
        grep -aq 'TTFT' "$out/$arm-r0/stdout.ts" 2>/dev/null || continue
        for p in $(cat "$out/$arm-r0/pid" 2>/dev/null) $(pgrep -P "$(cat "$out/$arm-r0/pid" 2>/dev/null)" 2>/dev/null); do
            [[ -r /proc/$p/maps ]] || continue
            awk -v p="$p" '/\/dev\/zero \(deleted\)/ && $2 ~ /s/ { z++ } /\/dev\/dri\/renderD|\/dev\/kfd/ { g++ }
                END { if (z + g > 0) printf "shmem maps %d, device maps %d (pid %s)\n", z, g, p }' "/proc/$p/maps"
        done > "$out/maps.r0" 2>/dev/null
        break
    done
) &
watcher=$!
bash "$here/../common/ticket.sh" "$out" minimax/fast-gate run --bin "$bin" --model "$model" --arm "$arm||$envs" --sessions "$sessions" --rows 0
rc=$?
wait "$watcher" 2>/dev/null
vm > "$out/vm0.after"; peer cat /proc/vmstat < /dev/null | awk -v keys="$keys" 'BEGIN { n = split(keys, k, " "); for (i = 1; i <= n; i++) want[k[i]] = 1 } ($1 in want) { print $1, $2 }' > "$out/vm1.after"
{
    grep -a "^== $arm" "$out/ticket.log" 2>/dev/null | cut -c1-230 | sed 's/^/verdict /'
    mode=$(grep -ah '^runtime host memory:' "$out/$arm-r0/stderr" "$out/$arm-r0/stdout" 2>/dev/null | head -1)
    echo "mode $arm r0 ${mode:-runtime default} | $(head -1 "$out/maps.r0" 2>/dev/null || echo 'maps not sampled')"
    echo "ev $arm r0 $(windows | bash -s -- "$out/$arm-r0")"
    echo "ev $arm r1 $(windows | peer bash -s -- "$peer_out/$arm-r1")"
    echo "vm $arm r0$(vm_delta "$out/vm0.before" "$out/vm0.after")"
    echo "vm $arm r1$(vm_delta "$out/vm1.before" "$out/vm1.after")"
    grep -ahi 'EFAULT\|Bad address\|pread (O_DIRECT)\|restart failed' "$out/$arm-r0/stderr" "$out/ticket.log" 2>/dev/null | head -3 | cut -c1-200 | sed 's/^/error /'
} > "$out/evict.txt" 2>&1
cat "$out/evict.txt"
: > "$out/evict.done"
exit "$rc"
