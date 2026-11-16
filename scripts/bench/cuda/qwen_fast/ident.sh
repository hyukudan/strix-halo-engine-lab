#!/usr/bin/env bash
# ident.sh: byte identity of teacher-forced logit dumps between arms (proof that a "bitwise" lever is bitwise).
# Each arm runs qwen_tf_logits once (one process, all prompts) and every dump file is compared with the first arm's.
#
#   ident.sh --tf TF --model M --tables T --rope-table R [--mtp-sidecar G] --out DIR [--rows 64] [--max-positions N] \
#            --prompt NAME=ids.u32 [--prompt ...] --arm 'NAME|ENV=v ENV2=w|--flags' [--arm ...]
#
# Arm: name, space-separated environment assignments (may be empty), qwen_tf_logits flags (e.g. --profile fast).
# Prints one line per (arm, file): IDENTICAL <bytes> <sha256 prefix> or DIFFERENT; exit 0 = all identical, 1 = a
# difference, 2 = an arm failed. No host names or paths are built in.
set -u
tf= model= tables= rope= mtp= out= rows=64 pos=
prompts=() arms=()
while [ $# -gt 0 ]; do
    case $1 in
        --tf) tf=$2; shift 2;;
        --model) model=$2; shift 2;;
        --tables) tables=$2; shift 2;;
        --rope-table) rope=$2; shift 2;;
        --mtp-sidecar) mtp=$2; shift 2;;
        --out) out=$2; shift 2;;
        --rows) rows=$2; shift 2;;
        --max-positions) pos=$2; shift 2;;
        --prompt) prompts+=(--prompt "$2"); shift 2;;
        --arm) arms+=("$2"); shift 2;;
        *) echo "ident.sh: unknown argument $1" >&2; exit 2;;
    esac
done
[ -n "$tf" ] && [ -n "$model" ] && [ -n "$out" ] && [ ${#prompts[@]} -gt 0 ] && [ ${#arms[@]} -ge 2 ] ||
    { echo "ident.sh: need --tf --model --out, a --prompt and two --arm" >&2; exit 2; }
extra=()
[ -n "$tables" ] && extra+=(--tables "$tables")
[ -n "$rope" ] && extra+=(--rope-table "$rope")
[ -n "$mtp" ] && extra+=(--mtp-sidecar "$mtp")
[ -n "$pos" ] && extra+=(--max-positions "$pos")
names=()
for a in "${arms[@]}"; do
    n=${a%%|*}; rest=${a#*|}; envs=${rest%%|*}; flags=${rest#*|}
    names+=("$n"); mkdir -p "$out/$n"
    # shellcheck disable=SC2086
    env $envs "$tf" --model "$model" "${extra[@]}" $flags "${prompts[@]}" --rows "$rows" --runs 1 --out "$out/$n" --prefix q \
        > "$out/$n.log" 2>&1
    rc=$?
    [ $rc -eq 0 ] || { echo "arm $n rc $rc (see $out/$n.log)"; exit 2; }
done
status=0
ref=${names[0]}
for n in "${names[@]:1}"; do
    for f in "$out/$ref"/*.f32; do
        b=$(basename "$f")
        if cmp -s "$f" "$out/$n/$b"; then
            echo "$ref == $n $b IDENTICAL $(stat -c %s "$f") B $(sha256sum "$f" | cut -c1-16)"
        else
            echo "$ref vs $n $b DIFFERENT"; status=1
        fi
    done
done
exit $status
