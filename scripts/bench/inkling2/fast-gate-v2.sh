#!/usr/bin/env bash
# Run "arm" once per GPU ticket on the same machine and binary. Reduce on CPU.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
rows=64
vocabulary=200058
weight_policy=inkling2-mini-attention-head-bf16-q8_0@1

usage() {
    echo "usage: $0 arm BIN MODEL MANIFEST OUT REF|REF2|NULL|CAND|CANDF|NULLF prefill|decode [--case NAME ...] [ENGINE_ARGS...]" >&2
    echo "       $0 qual MANIFEST OUT prefill|decode [GATE_ARGS...]" >&2
    echo "       $0 --expect-identical MANIFEST OUT prefill|decode" >&2
    exit 2
}

fail() { echo "$*" >&2; exit 2; }

read_windows() {
    case "$mode" in prefill|decode) ;; *) usage ;; esac
    manifest=$(realpath -- "$manifest")
    out=$(realpath -m -- "$out")
    local evidence
    evidence=$(python3 - "$manifest" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
manifest_bytes = manifest.read_bytes()
digest = hashlib.sha256(manifest_bytes)
names = {}
for line in manifest_bytes.decode("utf-8").splitlines():
    fields = line.split()
    if not fields or fields[0].startswith("#"):
        continue
    name = fields[0]
    if len(fields) != 3 or not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in names:
        sys.exit(f"invalid manifest line: {line!r}")
    names[name] = None
    digest.update(hashlib.sha256((manifest.parent / fields[2]).read_bytes()).digest())
if len(names) < 16:
    sys.exit("the kit requires at least 16 manifest windows")
classes = manifest.with_name("classes.json")
if not classes.is_file():
    sys.exit("declare a content class and prompt-length band for every window in classes.json")
classes_bytes = classes.read_bytes()
labels = json.loads(classes_bytes)
if not isinstance(labels, dict) or labels.keys() != names.keys() or any(
        not isinstance(label, str) or not label.strip() for label in labels.values()):
    sys.exit("classes.json must label each manifest window exactly once")
digest.update(hashlib.sha256(classes_bytes).digest())
bands = manifest.with_name("bands.json")
if bands.is_file():
    bands_bytes = bands.read_bytes()
    labels = json.loads(bands_bytes)
    if not isinstance(labels, dict) or labels.keys() != names.keys() or any(
            not isinstance(label, str) or not label.strip() for label in labels.values()):
        sys.exit("bands.json must label each manifest window exactly once")
    digest.update(hashlib.sha256(bands_bytes).digest())
print(digest.hexdigest())
print(*names, sep="\n")
PY
    )
    mapfile -t cases <<< "${evidence//$'\r'/}"
    inputs=${cases[0]}
    cases=("${cases[@]:1}")
}

check_outputs() {
    local arm=$1 case_name dump storage=plain.bf16@1
    case "$arm" in CAND|CANDF) storage=gguf.q8_0@1 ;; esac
    shift
    for case_name in "$@"; do
        dump="$out/$case_name-$arm.f32"
        [[ -f $dump && $(stat -c %s -- "$dump") == "$((rows * vocabulary * 4))" ]] ||
            fail "missing or wrong-size dump: $dump"
        [[ -f "$out/$case_name-$arm.ids" && $(wc -l < "$out/$case_name-$arm.ids") == "$rows" ]] ||
            fail "missing or wrong-size ID dump: $case_name $arm"
        [[ -s "$out/log/$case_name.$arm.stderr" ]] || fail "missing route receipt for $case_name $arm"
        grep -Fxq -- "strix-weight-load/v1 $weight_policy $storage" "$out/log/$case_name.$arm.stderr" ||
            fail "missing effective $storage receipt for $case_name $arm"
        case "$arm" in
            NULL|NULLF)
                grep -Eq '^strix-route-file/v1 .*"null":\{"order":"[a-zA-Z0-9_.@-]+","applications":[1-9][0-9]*\}' \
                    "$out/log/$case_name.$arm.stderr" || fail "missing evaluation-order null receipt for $case_name $arm"
                ;;
        esac
        case "$arm" in
            REF|REF2|CANDF|NULLF)
                [[ -s "$out/routes/$case_name.$arm.rte" ]] || fail "missing routes for $case_name $arm"
                ;;
        esac
    done
}

check_ticket() {
    local arm=$1
    [[ -s "$out/$arm.ticket" ]] || fail "missing completed ticket: $arm"
    [[ $(head -n 1 "$out/$arm.ticket") == "$mode $rows $inputs" ]] || fail "different gate inputs: $arm"
    cmp -s -- "$out/REF.ticket" "$out/$arm.ticket" || fail "different binary or checkpoint: $arm"
    shift
    if [[ -d "$out/tickets" ]]; then
        for case_name in "$@"; do
            [[ -f "$out/tickets/$arm-$case_name" ]] || fail "missing completed case: $arm $case_name"
        done
    fi
    check_outputs "$arm" "$@"
}

action=${1:-}
case "$action" in
    arm)
        [[ $# -ge 7 ]] || usage
        binary=$(realpath -- "$2")
        model=$(realpath -- "$3")
        manifest=$4 out=$5 arm=$6 mode=$7
        shift 7
        read_windows
        mkdir -p -- "$out/routes"
        selected=()
        capture_manifest=$manifest
        while [[ ${1:-} == --case ]]; do
            [[ $# -ge 2 ]] || usage
            case_name=$2
            [[ " ${cases[*]} " == *" $case_name "* && "$case_name" =~ ^[a-zA-Z0-9_-]+$ ]] ||
                fail "unknown manifest case: $case_name"
            if (( ${#selected[@]} > 0 )); then
                [[ " ${selected[*]} " != *" $case_name "* ]] || fail "duplicate requested case: $case_name"
            fi
            selected+=("$case_name")
            shift 2
        done
        if (( ${#selected[@]} > 0 )); then
            [[ -d "$out/tickets" || ! -e "$out/REF.ticket" ]] || fail "cannot resume a whole-manifest ticket by case"
            mkdir -p -- "$out/tickets"
            for case_name in "${selected[@]}"; do
                [[ ! -e "$out/tickets/$arm-$case_name" ]] || fail "case already completed: $arm $case_name"
            done
            capture_manifest=$(mktemp -- "$(dirname "$manifest")/.stripe-inkling-windows.XXXXXX")
            trap 'rm -f -- "$capture_manifest"' EXIT
            awk -v selected="${selected[*]}" 'BEGIN {split(selected, cases); for (i in cases) wanted[cases[i]]=1}
                $1 in wanted {print}' "$manifest" > "$capture_manifest"
        else
            [[ ! -d "$out/tickets" ]] || fail "resume this capture with --case"
            [[ ! -e "$out/$arm.ticket" ]] || fail "ticket already completed: $arm"
            selected=("${cases[@]}")
        fi
        unset STRIX_MOE_ROUTES_OUT STRIX_MOE_ROUTES_IN STRIX_MOE_ROUTE_NULL
        tier=exact
        weight_args=(--no-weight-q8)
        for option in "$@"; do
            case "$option" in
                --case|--profile|--qualify-q8|--no-weight-q8|--gate-*) fail "the gate kit controls $option" ;;
            esac
        done
        case "$arm" in
            REF|REF2|CANDF|NULLF) export STRIX_MOE_ROUTES_OUT="$out/routes/{case}.$arm.rte" ;;
            CAND|NULL) export STRIX_MOE_ROUTES_IN="$out/routes/{case}.REF.rte" ;;
            *) usage ;;
        esac
        case "$arm" in CAND|CANDF) tier=fast; weight_args=(--qualify-q8) ;; esac
        case "$arm" in NULL|NULLF) weight_args+=(--gate-order-null) ;; esac
        binary_hash=$(sha256sum < "$binary")
        # Checkpoint identity for this same-machine kit without re-reading the weights.
        model_identity=$(stat -Lc '%d %i %s %y %n' -- "$model")
        engine_args_hash=$(printf '%s\0' "$@" | sha256sum)
        ticket=$(printf '%s\n' "$mode $rows $inputs" "$binary_hash" "$model_identity" "$engine_args_hash")
        if [[ $arm != REF ]]; then
            check_ticket REF "${selected[@]}"
        fi
        if [[ -e "$out/REF.ticket" ]]; then
            [[ $(< "$out/REF.ticket") == "$ticket" ]] || fail "use the reference binary, checkpoint and engine options"
        fi
        timeout "${STRIX_INKLING_GATE_TIMEOUT:-110s}" "$binary" "$model" "$@" --profile "$tier" "${weight_args[@]}" \
            --gate-manifest "$capture_manifest" --gate-out "$out" --gate-arm "$arm" --gate-rows "$rows" --gate-mode "$mode"
        check_outputs "$arm" "${selected[@]}"
        printf '%s\n' "$ticket" > "$out/$arm.ticket"
        if [[ -d "$out/tickets" ]]; then
            for case_name in "${selected[@]}"; do touch -- "$out/tickets/$arm-$case_name"; done
        fi
        ;;
    qual)
        [[ $# -ge 4 ]] || usage
        manifest=$2 out=$3 mode=$4
        shift 4
        read_windows
        for arm in REF NULL CAND CANDF NULLF; do check_ticket "$arm" "${cases[@]}"; done
        for option in "$@"; do
            case "$option" in
                --allow-null-floor) ;;   # Gate v2's explicit diagnostic policy; default bars are absolute.
                *) fail "unsupported qual option: $option" ;;
            esac
        done
        reducer=(python3 "$repo/tools/strix-gate/gate_v2.py" qual "$out" --format head --manifest "$manifest"
            --head-rows "$rows" --ref REF --null NULL --cand CAND --cand-free CANDF --null-free NULLF "$@")
        "${reducer[@]}" --classes "$(dirname "$manifest")/classes.json" --json "$out/gate-$mode.json"
        if [[ -f "$(dirname "$manifest")/bands.json" ]]; then
            "${reducer[@]}" --classes "$(dirname "$manifest")/bands.json" --json "$out/gate-$mode-bands.json"
        fi
        ;;
    --expect-identical)
        [[ $# == 4 ]] || usage
        manifest=$2 out=$3 mode=$4
        read_windows
        check_ticket REF "${cases[@]}"
        check_ticket REF2 "${cases[@]}"
        for case_name in "${cases[@]}"; do
            cmp -- "$out/$case_name-REF.f32" "$out/$case_name-REF2.f32"
            cmp -- "$out/$case_name-REF.ids" "$out/$case_name-REF2.ids"
            cmp -- "$out/routes/$case_name.REF.rte" "$out/routes/$case_name.REF2.rte"
        done
        echo "PASS: REF and REF2 IDs, logits and routes are byte-identical across ${#cases[@]} windows"
        ;;
    *) usage ;;
esac
