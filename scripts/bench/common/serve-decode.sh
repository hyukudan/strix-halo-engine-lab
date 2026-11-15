#!/usr/bin/env bash
# Served decode by prompt class (prose / edit / json) at 1/4/8 concurrent chats, through tools/serve_bench.py.
#
#   serve-decode.sh --tag TAG --port P [--classes prose,edit,json] [--concurrency 1,4,8] [--max-tokens 128]
#                   [--rounds 1] [--ready-path /v1/models | --ready-log REGEX] [--ready-s 240] [--chat]
#                   [--body-extra JSON] [--arm NAME] [--round N (ABAB index of this server start)] [--peer-cmd 'RANK1 SERVER CMD'] [--client-cmd 'CMD']
#                   -- SERVER_CMD...
# --client-cmd replaces the class bursts by any client (run with KIT_PORT, KIT_URL and KIT_RUN_DIR exported), so
# other served kits (vision battery, long-context probes) reuse the same server start / validity / stop.
# --peer-cmd starts rank 1 of a pair server on KIT_PEER first (common/run1.sh there: same sampler and receipt,
# capped by the ticket budget, so it exits with rank 0); --ready-log polls the server stderr instead of HTTP
# (DS41 serve prints "READY", the Qwen server "listening on").
# Starts SERVER_CMD under the shared sampler (OUT/TAG/server: stamped stderr, samples, klog, meta), waits for
# GET http://127.0.0.1:P<ready-path> to answer, runs one warm-up request, then for each round and class one
# serve_bench.py burst set (aggregate tok/s from the server's cycle receipts when the log carries them, per-request
# and end-to-end tok/s, TTFT, token ids), stops the server and writes one receipt per (class, round) with
# metrics agg_C<n> / e2e_C<n>. Class prompts: scripts/bench/prompts/classes/<class>.txt (or --prompt-dir).
# The decode class matters: drafting (MTP / n-gram) acceptance differs by an order of magnitude between them.
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/bench/common/runlib.sh
source "$here/runlib.sh"
tag='' port='' classes=prose,edit,json conc=1,4,8 maxt=128 rounds=1 ready=/v1/models ready_log='' peer_cmd='' client_cmd='' round_base=0 ready_s=240 chat=() extra='{}' arm=''
pdir=$here/../prompts/classes
while (($#)); do
    case $1 in
        --tag) tag=$2 ;; --port) port=$2 ;; --classes) classes=$2 ;; --concurrency) conc=$2 ;; --max-tokens) maxt=$2 ;;
        --rounds) rounds=$2 ;; --ready-path) ready=$2 ;; --ready-log) ready_log=$2 ;; --peer-cmd) peer_cmd=$2 ;; --client-cmd) client_cmd=$2 ;; --round) round_base=$2 ;; --ready-s) ready_s=$2 ;; --body-extra) extra=$2 ;;
        --prompt-dir) pdir=$2 ;; --arm) arm=$2 ;;
        --chat) chat=(--chat); shift; continue ;;
        --) shift; break ;;
        *) kit_die "unknown argument $1" ;;
    esac
    shift 2
done
[[ -n $tag && -n $port && $# -gt 0 ]] || kit_die 'need --tag, --port and -- SERVER_CMD'
out=${KIT_OUT:-bench-out}; d=$out/$tag; sdir=$d/server
rm -rf -- "$d"; mkdir -p -- "$sdir"
kit_clean_env
printf '%q ' "$@" >"$sdir/cmd.txt"; echo >>"$sdir/cmd.txt"
ppid_=''
if [[ -n $peer_cmd ]]; then
    pdir=${KIT_PEER_OUT:-$out}/$tag-r1
    kit_peer "cd $(printf %q "$(kit_peer_root)") && KIT_TAG=$tag-r1 KIT_HOST_TAG=peer bash scripts/bench/common/run1.sh $(printf %q "$pdir") $(($(kit_left) - 5)) -- $peer_cmd" >"$d/peer.log" 2>&1 &
    ppid_=$!; sleep "${KIT_PEER_LEAD_S:-2}"
fi
mem=$(kit_mem); kit_sampler_start "$sdir"
k0=$(date '+%Y-%m-%d %H:%M:%S'); s0=$(date +%s%N)
( echo "$BASHPID" >"$sdir/pid"; exec "$@" ) >"$sdir/stdout" 2> >(kit_py stamp.py "$sdir/stderr") &
spid=$!
url=http://127.0.0.1:$port
up=0
for ((i = 0; i < ready_s * 2; i++)); do
    if [[ -n $ready_log ]]; then grep -aqE "$ready_log" "$sdir/stderr" "$sdir/stdout" 2>/dev/null && { up=1; break; }
    else curl -fsS -m 2 "$url$ready" >/dev/null 2>&1 && { up=1; break; }; fi
    kill -0 "$spid" 2>/dev/null || break
    sleep 0.5
done
echo "  server ready=$up after $((($(date +%s%N) - s0) / 1000000)) ms"
if ((up)) && [[ -n $client_cmd ]]; then
    KIT_PORT=$port KIT_URL=$url KIT_RUN_DIR=$d timeout -k 5 "$(($(kit_left) > 15 ? $(kit_left) - 10 : 5))" bash -c "$client_cmd" >"$d/client.txt" 2>&1
    echo "  client rc=$? ($(wc -l <"$d/client.txt") lines in client.txt)"
elif ((up)); then
    IFS=, read -ra C <<<"$classes"
    for ((r = 0; r < rounds; r++)); do
        for c in "${C[@]}"; do
            [[ -s $pdir/$c.txt ]] || { echo "== $tag-$c SKIP: no $pdir/$c.txt"; continue; }
            kit_fits 20 || { echo "== $tag-$c-$r SKIP (left $(kit_left) s)"; continue; }
            off=$(stat -c %s "$sdir/stderr" 2>/dev/null || echo 0)
            ((r == 0)) && [[ $c == "${C[0]}" ]] && "$KIT_PY" -I "$KIT_ROOT/tools/serve_bench.py" --url "$url" "${chat[@]}" \
                --prompt-file "$pdir/$c.txt" --concurrency 1 --max-tokens 16 --body-extra "$extra" >/dev/null 2>&1
            "$KIT_PY" -I "$KIT_ROOT/tools/serve_bench.py" --url "$url" "${chat[@]}" --prompt-file "$pdir/$c.txt" \
                --concurrency "$conc" --max-tokens "$maxt" --log "$sdir/stderr" --body-extra "$extra" \
                --timeout "$(($(kit_left) > 30 ? $(kit_left) - 10 : 20))" --out "$d/$c-$r.json" >"$d/$c-$r.txt" 2>&1
            echo "  $c round $r: serve_bench rc=$? (log offset $off)"
        done
    done
fi
kill "$spid" 2>/dev/null; wait "$spid" 2>/dev/null
if [[ -n $ppid_ ]]; then
    # Rank 1 normally exits when rank 0 goes away; after 20 s stop it through its own pid file (never other jobs).
    for ((i = 0; i < 40; i++)); do kill -0 "$ppid_" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$ppid_" 2>/dev/null && kit_peer "kill \$(cat $(printf %q "$pdir/pid")) 2>/dev/null"
    wait "$ppid_"; grep -a '^== ' "$d/peer.log" | sed 's/^/  r1| /'
fi
pkill -P "$(cat "$sdir/pid" 2>/dev/null || echo 0)" 2>/dev/null || true
e0=$(date +%s%N); sleep 0.3; kit_sampler_stop
journalctl -k --since "$k0" --no-pager 2>/dev/null | grep -aiE 'evict|amdgpu|kfd|thunderbolt|nhi|oom|reset' | grep -v '^-- ' |
    head -20 >"$sdir/klog.txt" || true
kit_py receipt.py meta "$sdir" --rc 0 --wall-ms $(((e0 - s0) / 1000000)) --mem "$mem" --host-tag "${KIT_HOST_TAG:-local}" \
    --bin-sha "$( [[ -f $1 ]] && kit_sha "$1" || echo -)"
kit_py serve_receipts.py "$d" --tag "$tag" --arm "${arm:-$tag}" --server-up "$up" --round-base "$round_base" --client "$([[ -n $client_cmd ]] && echo 1 || echo 0)"
