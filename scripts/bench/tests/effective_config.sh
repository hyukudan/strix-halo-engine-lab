#!/usr/bin/env bash
set -eu
export BENCH_ROOT=$(pwd)
source scripts/bench/lib/common.sh
source scripts/bench/lib/pair.sh
source scripts/bench/lib/batch.sh
test_dir=$(mktemp -d)
mkdir -p "$test_dir/bin" "$test_dir/models-a" "$test_dir/models-b"
cat >"$test_dir/bin/config-shim" <<'SHIM'
#!/usr/bin/env bash
set -eu
rank=0; printed=0
while (($#)); do
    case $1 in
        --rank) rank=$2; shift;;
        --print-effective-config) printed=1;;
    esac
    shift
done
[[ $printed == 1 ]] || exit 94
printf 'app serve\nenv STRIX_DS41_CB4=1\nenv STRIX_DS41_OVERLAY=%s/c-u6.gguf\n' "$STRIX_DS41_PACKS"
printf 'arg --rank\narg %s\n' "$rank"
if [[ ${CONFIG_CASE:-} == mismatch && $rank == 1 ]]; then printf 'env STRIX_DS41_CB4=0\n'; fi
if [[ ${CONFIG_CASE:-} == dropped ]]; then printf 'deepseek41 defaults: production packs off (not found: ours-K4)\n' >&2; fi
SHIM
chmod +x "$test_dir/bin/config-shim"
# This is a local transport shim: no SSH or remote execution takes place.
peer_exec() { bash -c "$1"; }
bin_dir="$test_dir/bin"; peer_bin_dir=$bin_dir
model_dir="$test_dir/models-a"; peer_model_dir="$test_dir/models-b"
peer_repo=$BENCH_ROOT; pair=1; engine=deepseek-pair; export BENCH_ENGINE=$engine; deadline=$((SECONDS+60))
rank_env=("STRIX_DS41_PACKS=$model_dir")
for config_case in match mismatch dropped; do
    out="$test_dir/$config_case"; mkdir -p "$out"
    export CONFIG_CASE=$config_case
    if effective_config decode-greedy --print-effective-config "$bin_dir/config-shim" --rank 0 >"$out/check.txt" 2>&1; then
        [[ $config_case == match ]] || exit 95
        "$BENCH_PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["decode-greedy"]["STRIX_DS41_CB4"]=="1"; assert d["decode-greedy"]["STRIX_DS41_OVERLAY"]=="MODEL_DIR:c-u6.gguf"' "$out/effective-environment.json"
        printf 'PASS effective-config rank0 == rank1 with translated pack paths\n'
    else
        [[ $config_case != match ]] || { cat "$out/check.txt"; exit 96; }
        if [[ $config_case == mismatch ]]; then grep -q 'effective configuration mismatch' "$out/check.txt"
        else grep -q 'deepseek41 defaults: production packs off' "$out/check.txt"; fi
        printf 'PASS effective-config %s fails closed\n' "$config_case"
    fi
done
# Execute both ranks locally through a transport shim, using the actual env handoff.
export CONFIG_CASE=match
out="$test_dir/native"; mkdir -p "$out"
cat >"$bin_dir/strix-deepseek41-dspark-rank" <<'SHIM'
#!/usr/bin/env bash
set -eu
[[ $STRIX_DS41_CB4 == 1 && $STRIX_DS41_OVERLAY == "$STRIX_DS41_PACKS/c-u6.gguf" ]]
printf '%s\n' "$*" >>"$BENCH_NATIVE_RECORD"
printf 'tokens (2, fnv abc): 1 2\nprompt 3 tokens in 1.00 s; generation 1 tokens in 0.025 s = 40.00 tok/s\n'
SHIM
chmod +x "$bin_dir/strix-deepseek41-dspark-rank"
cp "$bin_dir/config-shim" "$bin_dir/strix-deepseek41-rank"
export BENCH_NATIVE_RECORD="$out/native-record"
dry=0; config_only=1
command=("$bin_dir/strix-deepseek41-dspark-rank" model draft --rank 0 --mode greedy)
deepseek_native_decode decode-greedy "$out/manifest.json"
"$BENCH_PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["env_source"]=="strix-deepseek41-rank --print-effective-config"; assert d["rank0"]==d["rank1"]' "$out/effective-decode-greedy.json"
printf '[{"prompt_id":"code-ctx3","mode":"greedy","generation":2,"warmup":false}]' >"$out/manifest.json"
pair_run() {
    local log=$1 remote_dir=unused; shift
    pair_remote_command env "${rank_env[@]}" "$@"
    "${remote_command[@]}" >"$out/peer-native.log"
    env "${rank_env[@]}" "$@" >"$log"
}
config_only=0
deepseek_native_decode decode-greedy "$out/manifest.json"
[[ $(wc -l <"$BENCH_NATIVE_RECORD") == 2 ]]
grep -q -- '--rank 0 --mode greedy' "$BENCH_NATIVE_RECORD"
grep -q -- '--rank 1 --mode greedy' "$BENCH_NATIVE_RECORD"
"$BENCH_PYTHON" -c 'import json,sys; d=json.loads(open(sys.argv[1]).read()); assert d["timing_scope"]=="native" and d["values"]==[40]' "$out/samples.jsonl"
printf 'PASS native dspark-rank shim receives rank-resolver environment on both ranks\n'
# Only this absolute mktemp-owned directory is removed.
[[ $test_dir == /* && ${test_dir##*/} == tmp.* ]] || exit 97
rm -rf -- "$test_dir"
