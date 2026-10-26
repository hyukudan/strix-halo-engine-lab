#!/usr/bin/env bash
# Local executable shims only: no device or remote execution.
set -eu
export BENCH_ROOT=$(pwd) BENCH_ENGINE=qwen BENCH_SELECTED_PROFILE=default
source scripts/bench/lib/common.sh
source scripts/bench/engines/qwen.sh
test_dir=$(mktemp -d)
bin_dir="$test_dir/bin"; model_dir="$test_dir/models"; out="$test_dir/run"
mkdir -p "$bin_dir" "$model_dir" "$out"
export BENCH_MODEL_DIR="$model_dir" BENCH_BIN_DIR="$bin_dir" BENCH_OUT="$out"
cat >"$bin_dir/strix-hip-decode-token" <<'SHIM'
#!/usr/bin/env bash
set -eu
printed=0; explicit=0; golden=''
for arg in "$@"; do
    [[ $arg != --print-effective-config ]] || printed=1
    [[ $arg != --known-prefix-window && $arg != --baseline ]] || explicit=1
done
while (($#)); do
    if [[ $1 == --mtp-golden-dump ]]; then golden=$2; shift; fi
    shift
done
if [[ $explicit == 0 ]]; then
    case ${CONFIG_CASE:-admitted} in
        admitted) printf 'Qwen auto-prefill: automatic T16 context=8192\n';;
        declined) printf 'Qwen auto-prefill declined: test-capacity; using serial prefill\n';;
        missing) :;;
    esac
fi
if [[ $printed == 1 ]]; then
    printf '{"admitted":true,"environment":{"STRIX_PREFILL_PIPE":{"value":"255","source":"automatic"}},"argv":["--context","8192"],"T":16,"context":8192,"decode_profile":"dense-a8","prefill_profile":"fast-prefill-f16wmma-v1"}\n'
else
    [[ -z $golden ]] || printf golden >"$golden"
    printf '{"model_content_identity":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","generated_tokens":[1,2],"prompt_ms":100,"generation_step_elapsed_ms_total":50,"mtp_generation_cycle_elapsed_ms":50}\n'
fi
SHIM
chmod +x "$bin_dir/strix-hip-decode-token"
printf model >"$model_dir/${assets[0]}"
printf '1,2,3\n' >"$out/code-ctx16.ids"
cp "$out/code-ctx16.ids" "$out/prefill-16.ids"
dry=0; profile=default; only=decode,prefill; serve=0; warmup=1; repeats=1; gen_tokens=2
context_values=(16); prompt_names=(code); rank_env=(); deadline=$((SECONDS+120))
qwen_execute
"$BENCH_PYTHON" -c 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); e=json.load(open(p/"effective-environment.json")); assert all(any(kind in label for label in e) for kind in ("plain","spec","calibration","ttft")); assert all(v=={"STRIX_PREFILL_PIPE":"255"} for v in e.values()); s=[json.loads(x) for x in (p/"samples.jsonl").read_text().splitlines()]; assert next(x for x in s if x["mode"]=="ttft")["auto_prefill"]["T"]==16; assert not list(p.glob("*rank1*"))' "$out"
printf 'PASS Qwen launch config prints and admitted warmup/measured prefill shims\n'
# Isolate the runtime-log guard from the earlier inspection guard.
effective_config() { :; }
for config_case in declined missing; do
    export CONFIG_CASE=$config_case
    for rep in 1 2; do
        if (qwen_command prefill-16 1; p=prefill-16; qwen_prefill_production; qwen_emit "$p" ttft 1 "$rep") >"$out/reject.log" 2>&1; then exit 98; fi
        if [[ $config_case == declined ]]; then grep -q 'test-capacity' "$out/reject.log"
        else grep -q 'missing or ambiguous Qwen auto-prefill admission' "$out/reject.log"; fi
    done
    printf 'PASS Qwen %s warmup/measured admission fails closed\n' "$config_case"
done
unset CONFIG_CASE
source scripts/bench/lib/common.sh
source scripts/bench/engines/deepseek-rtx.sh
export BENCH_ENGINE=deepseek-rtx
cat >"$bin_dir/strix-cuda-ds41-replica" <<'SHIM'
#!/usr/bin/env bash
set -eu
[[ " $* " == *' --print-config '* ]] || exit 94
if [[ " $* " == *' --baseline '* ]]; then
    printf '{"load_serial":true,"draft_vocab":"full","draft_keep":0}\n'
else
    printf '{"load_serial":false,"draft_vocab":"auto:vocab","draft_keep":64}\n'
fi
SHIM
chmod +x "$bin_dir/strix-cuda-ds41-replica"
for profile in default baseline; do
    export BENCH_SELECTED_PROFILE=$profile
    rank_env=(); rtx_command decode 96
    effective_config "rtx-$profile" --print-config "${command[@]}"
done
"$BENCH_PYTHON" -c 'import json,sys; e=json.load(open(sys.argv[1])); assert e["rtx-default"]=={"DG_LOAD_SERIAL":"0"}; assert e["rtx-baseline"]=={"DG_LOAD_SERIAL":"1"}' "$out/effective-environment.json"
printf 'PASS RTX default/baseline config-print shims\n'
[[ $test_dir == /* && ${test_dir##*/} == tmp.* ]] || exit 97
rm -rf -- "$test_dir"
