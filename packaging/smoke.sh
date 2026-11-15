#!/usr/bin/env bash
# Post-install smoke test of the Strix Linux packages (installed system-wide or extracted into a directory).
#   smoke.sh [--root DIR] [--prefix /usr] [--port 18080] [--model FILE.gguf] [--require-health]
# --root: where the packages were extracted (default /, i.e. installed); binaries are <root><prefix>/bin.
# Checks: every installed strix-* executable loads and answers --help (no missing library, no crash);
# strix-skeleton-serve answers GET /health = 200 without weights; with --model, strix-hip-decode-token
# --print-effective-config resolves the model. Prints one PASS/FAIL/SKIP line per check; exit 1 on any FAIL.
set -u
root=/ prefix=/usr port=18080 model="" require_health=0
while [ $# -gt 0 ]; do
    case "$1" in
        --root) root="$2"; shift 2 ;;
        --prefix) prefix="$2"; shift 2 ;;
        --port) port="$2"; shift 2 ;;
        --model) model="$2"; shift 2 ;;
        --require-health) require_health=1; shift ;;
        -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
        *) echo "smoke.sh: unknown option $1" >&2; exit 2 ;;
    esac
done
base="${root%/}${prefix}"
bindir="$base/bin" datadir="$base/share/strix"
fails=0 passes=0 nohelp=0
pass() { passes=$((passes + 1)); echo "PASS $*"; }
fail() { fails=$((fails + 1)); echo "FAIL $*"; }
skip() { echo "SKIP $*"; }

[ -d "$bindir" ] || { echo "FAIL no bin directory at $bindir"; exit 1; }
for exe in "$bindir"/strix-*; do
    [ -x "$exe" ] || continue
    name=$(basename "$exe")
    if file -b "$exe" 2>/dev/null | grep -q ELF && command -v ldd >/dev/null 2>&1; then
        missing=$(ldd "$exe" 2>/dev/null | grep "not found" | head -3 | tr -s ' \t' ' ')
        if [ -n "$missing" ]; then fail "$name: missing libraries:$missing"; continue; fi
    fi
    out=$(timeout 30 "$exe" --help 2>&1 </dev/null); rc=$?
    if [ $rc -eq 124 ] || [ $rc -ge 128 ]; then
        fail "$name --help: rc=$rc (timeout or signal)"
    elif [ $rc -eq 0 ] || printf '%s' "$out" | grep -qiE "usage|options"; then
        pass "$name --help (rc=$rc)"
    elif printf '%s' "$out" | grep -qiE "error while loading shared libraries|symbol lookup error|device gate"; then
        fail "$name --help: rc=$rc: $(printf '%s' "$out" | head -2 | tr '\n' ' ' | cut -c1-160)"
    elif [ -n "$out" ]; then
        # The program loaded and its own parser answered (it has no --help yet): the package is intact.
        nohelp=$((nohelp + 1))
        pass "$name loads (no --help: $(printf '%s' "$out" | head -1 | cut -c1-100))"
    else
        fail "$name --help: rc=$rc and no output"
    fi
done

probe() {
    if command -v curl >/dev/null 2>&1; then
        curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$port/health"
    elif command -v python3 >/dev/null 2>&1; then
        python3 -c "import sys,urllib.request as u
try: print(u.urlopen('http://127.0.0.1:$port/health', timeout=2).status)
except Exception: print(0)"
    else
        echo 0
    fi
}
skel="$bindir/strix-skeleton-serve" reg="$datadir/examples/skeleton_engine"
if [ -x "$skel" ] && [ -f "$reg/models/registry.toml" ]; then
    log=$(mktemp)
    "$skel" --registry-root "$reg" --registry models/registry.toml --model skeleton --bind 127.0.0.1 \
        --port "$port" >"$log" 2>&1 </dev/null &
    pid=$!
    code=0
    for _ in $(seq 1 50); do
        code=$(probe)
        [ "$code" = "200" ] && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.2
    done
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    if [ "$code" = "200" ]; then pass "strix-skeleton-serve GET /health = 200"
    else fail "strix-skeleton-serve GET /health = $code: $(tail -n 3 "$log" | tr '\n' ' ' | cut -c1-200)"; fi
    rm -f "$log"
elif [ $require_health -eq 1 ]; then
    fail "health: strix-skeleton-serve or its registry not installed (install strix-core)"
else
    skip "health: strix-skeleton-serve not in this package (strix-core has it)"
fi

hip="$bindir/strix-hip-decode-token"
if [ -x "$hip" ] && [ -n "$model" ]; then
    out=$(timeout 120 "$hip" "$model" --print-effective-config 2>&1 </dev/null); rc=$?
    if [ $rc -eq 0 ]; then pass "strix-hip-decode-token --print-effective-config ($(printf '%s' "$out" | wc -l) lines)"
    else fail "strix-hip-decode-token --print-effective-config rc=$rc: $(printf '%s' "$out" | tail -n 2 | tr '\n' ' ' | cut -c1-200)"; fi
elif [ -x "$hip" ]; then
    skip "--print-effective-config: needs --model FILE.gguf"
fi
echo "smoke: $passes passed ($nohelp without --help), $fails failed"
[ $fails -eq 0 ]
