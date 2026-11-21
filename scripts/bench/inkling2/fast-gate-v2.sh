#!/usr/bin/env bash
# Run "arm" once per GPU ticket on the same machine and binary. Reduce on CPU.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
rows=64
vocabulary=200058

usage() {
    echo "usage: $0 arm BIN MODEL MANIFEST OUT REF|REF2|NULL|CAND|CANDF|NULLF prefill|decode [ENGINE_ARGS...]" >&2
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
if classes.is_file():
    digest.update(hashlib.sha256(classes.read_bytes()).digest())
print(digest.hexdigest())
print(*names, sep="\n")
PY
    )
    mapfile -t cases <<< "$evidence"
    inputs=${cases[0]}
    cases=("${cases[@]:1}")
}

check_outputs() {
    local arm=$1 case_name dump
    for case_name in "${cases[@]}"; do
        dump="$out/$case_name-$arm.f32"
        [[ -f $dump && $(stat -c %s -- "$dump") == "$((rows * vocabulary * 4))" ]] ||
            fail "missing or wrong-size dump: $dump"
        [[ -s "$out/log/$case_name.$arm.stderr" ]] || fail "missing route receipt for $case_name $arm"
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
    check_outputs "$arm"
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
        [[ ! -e "$out/$arm.ticket" ]] || fail "ticket already completed: $arm"
        unset STRIX_MOE_ROUTES_OUT STRIX_MOE_ROUTES_IN STRIX_MOE_ROUTE_NULL
        tier=exact
        case "$arm" in
            REF|REF2|CANDF|NULLF) export STRIX_MOE_ROUTES_OUT="$out/routes/{case}.$arm.rte" ;;
            CAND|NULL) export STRIX_MOE_ROUTES_IN="$out/routes/{case}.REF.rte" ;;
            *) usage ;;
        esac
        case "$arm" in CAND|CANDF) tier=fast ;; esac
        case "$arm" in NULL|NULLF) export STRIX_MOE_ROUTE_NULL=0:1 ;; esac
        binary_hash=$(sha256sum < "$binary")
        # Checkpoint identity for this same-machine kit without re-reading the weights.
        model_identity=$(stat -Lc '%d %i %s %y %n' -- "$model")
        ticket=$(printf '%s\n' "$mode $rows $inputs" "$binary_hash" "$model_identity")
        if [[ $arm != REF ]]; then
            check_ticket REF
            [[ $(< "$out/REF.ticket") == "$ticket" ]] || fail "use the reference binary and checkpoint"
        fi
        timeout 110s "$binary" "$model" "$@" --profile "$tier" --gate-manifest "$manifest" \
            --gate-out "$out" --gate-arm "$arm" --gate-rows "$rows" --gate-mode "$mode"
        check_outputs "$arm"
        printf '%s\n' "$ticket" > "$out/$arm.ticket"
        ;;
    qual)
        [[ $# -ge 4 ]] || usage
        manifest=$2 out=$3 mode=$4
        shift 4
        read_windows
        for arm in REF NULL CAND CANDF NULLF; do check_ticket "$arm"; done
        classes=()
        if [[ -f "$(dirname "$manifest")/classes.json" ]]; then
            classes=(--classes "$(dirname "$manifest")/classes.json")
        fi
        exec python3 "$repo/tools/strix-gate/gate_v2.py" qual "$out" --format head --manifest "$manifest" \
            --head-rows "$rows" --ref REF --null NULL --cand CAND --cand-free CANDF --null-free NULLF \
            "${classes[@]}" --json "$out/gate-$mode.json" "$@"
        ;;
    --expect-identical)
        [[ $# == 4 ]] || usage
        manifest=$2 out=$3 mode=$4
        read_windows
        check_ticket REF
        check_ticket REF2
        for case_name in "${cases[@]}"; do
            cmp -- "$out/$case_name-REF.f32" "$out/$case_name-REF2.f32"
            cmp -- "$out/routes/$case_name.REF.rte" "$out/routes/$case_name.REF2.rte"
        done
        echo "PASS: REF and REF2 logits and routes are byte-identical across ${#cases[@]} windows"
        ;;
    *) usage ;;
esac
