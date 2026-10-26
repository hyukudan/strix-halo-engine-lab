#!/usr/bin/env bash
# Profile globals cross sourced module boundaries.
# shellcheck disable=SC2034,SC2154
set -eu
# Sourced modules exchange profile globals and command arrays with bench.sh.
# No peer is implicit. SSH commands are shell-quoted argv, never interpolated unquoted.
remote_quote() {
    local arg
    for arg in "$@"; do
        arg=${arg//\'/\'\\\'\'}
        printf "'%s' " "$arg"
    done
}
peer_exec() {
    timeout --signal=TERM --kill-after=5 "${2:-${BENCH_SSH_TIMEOUT:-60}}s" \
        ssh -o BatchMode=yes -o ConnectTimeout=10 -- "$peer" "bash -c $(remote_quote "$1")"
}
pair_check() {
    [[ -n $peer && $peer != -* ]] || { printf 'pair requires --peer HOST\n' >&2; return 1; }
    local tmp rc=0 cmd local_job peer_job
    tmp=$(mktemp -d)
    py inventory >"$tmp/local.json" &
    local_job=$!
    cmd="cd $(remote_quote "$peer_repo") && BENCH_VERIFY_MODELS=$(remote_quote "$verify_models") BENCH_ENGINE=$(remote_quote "$engine") BENCH_PROFILE=$(remote_quote "$identity") BENCH_BIN_DIR=$(remote_quote "$peer_bin_dir") BENCH_MODEL_DIR=$(remote_quote "$peer_model_dir") BENCH_PYTHON=$(remote_quote "${BENCH_PYTHON:-python3}") $(remote_quote "${BENCH_PYTHON:-python3}" scripts/bench/lib/kit.py inventory)"
    peer_exec "$cmd" "$limit" >"$tmp/remote.json" 2>"$tmp/ssh.err" &
    peer_job=$!
    wait "$local_job" || rc=1
    wait "$peer_job" || rc=1
    if [[ $rc == 0 ]]; then
        py compare "$tmp/local.json" "$tmp/remote.json" || rc=1
        local_inventory=$(cat "$tmp/local.json")
        peer_inventory=$(cat "$tmp/remote.json")
    fi
    if [[ $rc != 0 ]]; then printf 'Pair check failed: reachability, ROCm, USB4, commit or content differs.\n{"pair_ok":false}\n' >&2; fi
    # The fixed temporary directory contains only this check's bounded outputs.
    rm -f -- "$tmp/local.json" "$tmp/remote.json" "$tmp/ssh.err"
    rmdir -- "$tmp"
    return "$rc"
}
pair_remote_command() {
    local arg
    remote_command=()
    for arg in "$@"; do
        arg=${arg/#"$bin_dir"/"$peer_bin_dir"}
        arg=${arg//"$model_dir"/"$peer_model_dir"}
        # Prompt files have been uploaded to the run-owned directory.
        if [[ -n ${out:-} ]]; then arg=${arg//"$out"/"$remote_dir"}; fi
        remote_command+=("$arg")
    done
    # MiniMax rank uses positional MODEL_DIR RANK TRANSPORT OUT_DIR.
    local i
    for ((i=0; i<${#remote_command[@]}-1; i++)); do
        if [[ ${remote_command[i]} == *strix-minimax-rank* ]]; then remote_command[i+2]=1; fi
    done
    for ((i=0; i<${#remote_command[@]}-1; i++)); do
        if [[ ${remote_command[i]} == --rank ]]; then remote_command[i+1]=1; fi
    done
}
pair_start() {
    local cmd
    remote_dir=${planned_remote_dir:-".bench-runs/$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"}
    remote_active=1
    pair_remote_command env "${rank_env[@]}" "$@"
    peer_exec "cd $(remote_quote "$peer_repo") && mkdir -p $(remote_quote "$remote_dir" "$remote_dir/native")" >/dev/null
    # Upload only deterministic token lists; never ship a local configuration or environment dump.
    local f
    for f in "$out"/*.ids "$out"/*.u32 "$out"/*.tsv; do
        [[ -f $f ]] || continue
        # Remote destination and local input are on different nodes.
        # shellcheck disable=SC2094
        peer_exec "cd $(remote_quote "$peer_repo") && cat >$(remote_quote "$remote_dir/$(basename -- "$f")")" <"$f" >/dev/null
    done
    # Set the working directory before backgrounding. PID is the setsid group leader.
    # Variables in this script expand on node B.
    # shellcheck disable=SC2016
    cmd="cd $(remote_quote "$peer_repo") && { setsid bash -c $(remote_quote 'echo $$ > "$1/pid"; shift; exec "$@"' bench-rank "$remote_dir" timeout --signal=TERM --kill-after=5 "$(remaining)s" "${remote_command[@]}") >$(remote_quote "$remote_dir/rank1.log") 2>&1 < /dev/null & }"
    cmd+="; for i in {1..50}; do test ! -s $(remote_quote "$remote_dir/pid") || exit 0; sleep 0.1; done; exit 1"
    peer_exec "$cmd" >/dev/null
}
pair_stop() {
    [[ $remote_active == 1 ]] || return 0
    local script rc=0
    # Variables in this cleanup script expand on node B.
    # shellcheck disable=SC2016
    script='set -eu
cd "$1"
d=$2
test -f "$d/pid" || exit 1
p=$(cat "$d/pid")
case "$p" in ""|*[!0-9]*) exit 1;; esac
test "$p" -gt 1
# Only signal the process group created by this run. Record descendants before termination.
members=$(ps -eo pid=,pgid= | awk -v g="$p" '\''$2 == g {print $1}'\'')
kill -TERM -- "-$p" 2>/dev/null || true
sleep 2
kill -KILL -- "-$p" 2>/dev/null || true
sleep 1
for child in $members; do
    if test -d "/proc/$child/fd"; then
        for fd in /proc/"$child"/fd/*; do
            target=$(readlink "$fd" 2>/dev/null || true)
            test "$target" != /dev/kfd || exit 1
        done
    fi
done
python3 scripts/bench/lib/kit.py sanitize "$d/rank1.log"
cat "$d/rank1.log"
'
    if ! peer_exec "bash -c $(remote_quote "$script" bench-clean "$peer_repo" "$remote_dir")" >"$out/rank1-${remote_dir##*/}.log" 2>/dev/null; then rc=1; fi
    remote_active=0
    py sanitize-all "$out"
    [[ $rc == 0 ]] || printf 'Remote teardown could not be verified; inspect the run PID file on node B.\n' >&2
    return "$rc"
}
pair_run() {
    local log=$1 rc=0
    shift
    pair_start "$@"
    run_local "$log" env "${rank_env[@]}" "$@" || rc=$?
    pair_stop || rc=1
    return "$rc"
}
pair_preview() {
    local remote_dir=REMOTE_RUN_DIR
    local preview_repo=$peer_repo
    [[ $peer_repo != "$BENCH_ROOT" ]] || preview_repo=REPO
    pair_remote_command env "${rank_env[@]}" "$@"
    printf 'node A: '; show_command env "${rank_env[@]}" "$@"
    printf 'node B (SSH, setsid, PID file, same timeout): '
    show_command ssh -o BatchMode=yes -- "$peer" "bash -c $(remote_quote "cd $(remote_quote "$preview_repo") && setsid timeout --signal=TERM --kill-after=5 ${limit}s $(remote_quote "${remote_command[@]}")")"
}
