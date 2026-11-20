#!/usr/bin/env bash
# Hy4-flash G0 (host, no GPU): every shared contract's host test + the CPU oracle on the tiny config + the gate fixtures.
#
#   scripts/bench/hy4/g0-host.sh --build DIR [--wfmt-build DIR] [--pack PART1.gguf] [--tokenizer BIN]
#
# DIR is a CMake build of the engine (MSVC: DIR/Release/<test>.exe; Ninja/Make: DIR/<test>); the test list is data:
# tests/hy4_fixtures/g0-tests.tsv (name, args, status, contract). The binaries run directly (no ctest), with the
# registered arguments; PACK replaces the "PACK" argument (only the GGUF header is read), SRC:<path> is repo-relative. With --tokenizer
# (strix-tokenize) and --pack the gate prompt ids are re-encoded from their text and must be equal.
# Output, one line each:  "G0 <test> PASS|FAIL rc=N|MISSING|PENDING <contract>"  then
#                         "G0 summary pass=N fail=M missing=K pending=P verdict=GREEN|RED"
# GREEN = every committed row PASS and the fixture check PASS; pending rows are reported, not counted.
set -u
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../../.." && pwd)
build='' wfmt='' pack='' tok=''
while (($#)); do
    case $1 in
        --build) build=$2 ;; --wfmt-build) wfmt=$2 ;; --pack) pack=$2 ;; --tokenizer) tok=$2 ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "g0-host: unknown argument $1" >&2; exit 64 ;;
    esac
    shift 2
done
[[ -d $build ]] || { echo 'g0-host: need --build DIR' >&2; exit 64; }
py=${KIT_PYTHON:-python3}
command -v "$py" >/dev/null 2>&1 || py=python
find_bin() {  # find_bin NAME DIR...: first existing binary
    local n=$1 d; shift
    for d in "$@"; do
        [[ -n $d ]] || continue
        for c in "$d/Release/$n.exe" "$d/RelWithDebInfo/$n.exe" "$d/$n.exe" "$d/$n" "$d/Release/$n"; do
            [[ -f $c ]] && { echo "$c"; return 0; }
        done
    done
    return 1
}
pass=0 fail=0 miss=0 pend=0
while IFS=$'\t' read -r name args status contract; do
    [[ $name == name || -z $name ]] && continue
    if [[ $status == pending ]]; then
        echo "G0 $name PENDING $contract"; pend=$((pend + 1)); continue
    fi
    bin=$(find_bin "$name" "$build" "$wfmt") || { echo "G0 $name MISSING $contract"; miss=$((miss + 1)); continue; }
    a=()  # args: "-" none | PACK (the --pack file, optional) | SRC:<repo-relative path>
    [[ $args == PACK && -n $pack ]] && a=("$pack")
    [[ $args == SRC:* ]] && a=("$root/${args#SRC:}")
    log=$(mktemp)
    "$bin" "${a[@]}" >"$log" 2>&1
    rc=$?
    if ((rc == 0)); then
        echo "G0 $name PASS rc=0 $contract"; pass=$((pass + 1))
    else
        echo "G0 $name FAIL rc=$rc $contract"; tail -5 "$log" | sed 's/^/    /'; fail=$((fail + 1))
    fi
    rm -f -- "$log"
done <"$root/tests/hy4_fixtures/g0-tests.tsv"
fx=(check --fixtures "$root/tests/hy4_fixtures")
[[ -n $tok && -n $pack ]] && fx+=(--tokenizer "$tok" --pack "$pack")
"$py" -I "$here/check_fixtures.py" "${fx[@]}" | tail -4
fxrc=${PIPESTATUS[0]}
"$py" -I "$here/check_fixtures.py" self-test | tail -1
strc=${PIPESTATUS[0]}
verdict=GREEN
((fail + miss + fxrc + strc > 0)) && verdict=RED
echo "G0 summary pass=$pass fail=$fail missing=$miss pending=$pend fixtures_rc=$fxrc selftest_rc=$strc verdict=$verdict"
[[ $verdict == GREEN ]]
