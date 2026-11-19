#!/usr/bin/env bash
# Hy4-flash G1 op fixtures on device vs the shared host contract oracles (no model load: seeded inputs at the real HY4
# shapes, the host oracle of each shared contract runs in the same process on the same inputs).
#
#   scripts/bench/common/ticket.sh OUT hy4/g1-ops --bin OPS_BIN [--ops engram,hc,mla-dsa,mla-swa,indexer,router,experts]
#       [--seeds 3] [--cap 100]
#
# OPS_BIN is the backend's op gate binary (built by the device binding package, HIP or CUDA). Contract
# (gates-plan.md section 2, one line per case, nothing else starts with "G1 "):
#   "G1 <op> <case> <class> PASS|FAIL <metric>=<value> [<metric>=<value> ...]"
#   class: bitwise | ids-equal | tol:<contract name>   (declared per case in tests/hy4_fixtures/g1-cases.tsv)
#   and a final "G1 done cases=<N>". `OPS_BIN --list` prints the case names; the kit fails a case that is in
#   g1-cases.tsv but absent from the output (MISSING).
# Expected summary: "== G1 <op> <case> PASS ..." per case and "== G1 summary pass=N fail=0 missing=0".
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/bench/common/runlib.sh
source "$here/../common/runlib.sh"
bin='' ops=engram,hc,mla-dsa,mla-swa,indexer,router,experts seeds=3 cap=100
while (($#)); do
    case $1 in
        --bin) bin=$2 ;; --ops) ops=$2 ;; --seeds) seeds=$2 ;; --cap) cap=$2 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) kit_die "unknown argument $1" ;;
    esac
    shift 2
done
[[ -x $bin ]] || kit_die 'need --bin OPS_BIN'
out=${KIT_OUT:-bench-out}
cases=$KIT_ROOT/tests/hy4_fixtures/g1-cases.tsv
kit_fits 20 || kit_die "budget $(kit_left) s < 20 s"
export KIT_TAG=g1 KIT_ARM=g1 KIT_ROUND=0 KIT_TOKENS=0 KIT_EXPECTED=0
kit_clean_env
echo "#### g1 ops=$ops seeds=$seeds bin=$(kit_sha "$bin") t+$(($(date +%s) - KIT_T0))s"
kit_run "$out/g1" "$(( cap < $(kit_left) - 3 ? cap : $(kit_left) - 3 ))" "$bin" --ops "$ops" --seeds "$seeds"
pass=0 fail=0 miss=0
IFS=, read -ra O <<<"$ops"
while IFS=$'\t' read -r op case cls _; do
    [[ $op == op || -z $op ]] && continue
    [[ " ${O[*]} " == *" $op "* ]] || continue
    line=$(grep -a "^G1 $op $case " "$out/g1/stdout" | tail -1)
    if [[ -z $line ]]; then
        echo "== G1 $op $case MISSING"; miss=$((miss + 1))
    elif [[ $line == "G1 $op $case $cls PASS"* ]]; then
        echo "== $line"; pass=$((pass + 1))
    else
        echo "== $line (declared class $cls)"; fail=$((fail + 1))
    fi
done <"$cases"
echo "== G1 summary pass=$pass fail=$fail missing=$miss"
