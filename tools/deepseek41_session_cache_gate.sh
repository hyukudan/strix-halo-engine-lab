#!/usr/bin/env bash
# Run from the lead node; execution phases require an exclusive two-node GPU allocation.
# prewarm and compare-off are CPU-only. Artifacts stay outside Git.
set -euo pipefail
: "${NODE1:?peer node for the two-rank gate}"
: "${OUT:?absolute evidence directory outside Git on persistent storage}"
PHASE=${1:?usage: gate PHASE}
# This phase is entirely CPU-only: no model, SSH, or rank launch.
if [[ "$PHASE" = compare-off ]]; then
    [[ -f "$OUT/unnamed.done" && ! -e "$OUT/compare-off.done" ]] || exit 2
    LIMIT=${LIMIT:-690}
    [[ "$LIMIT" =~ ^[0-9]+$ && "$LIMIT" -ge 30 && "$LIMIT" -le 690 ]] || exit 2
    if [[ ${DS41_GATE_INNER:-0} != 1 ]]; then
        export DS41_GATE_INNER=1
        exec timeout --signal=TERM --kill-after=2 "$((LIMIT-2))" bash "$0" "$@"
    fi
    start=$SECONDS
    python3 "$(dirname "$0")/deepseek41_session_cache_compare_off.py" "$OUT"
    printf 'phase=compare-off cpu_only=1 wall_seconds=%s hashing_seconds=0 checkpoint_io_seconds=0\n' "$((SECONDS-start))" > "$OUT/compare-off.receipt"
    mv -- "$OUT/compare-off.receipt" "$OUT/compare-off.done"
    exit 0
fi
: "${BIN:?absolute candidate rank path on both nodes}"
: "${BASELINE_BIN:?absolute rank binary built from exactly 05fedd80 on both nodes}"
: "${MODEL:?absolute model path on both nodes}"
: "${DSPARK:?absolute drafter path for the DSpark continuation gate}"
BASELINE_REV=05fedd8011408fada8688b675d4ad2c699016c02
[[ $(cat "$BASELINE_BIN.revision") == "$BASELINE_REV" ]] || { echo "baseline provenance mismatch"; exit 2; }
case "$PHASE" in
  prewarm) PREVIOUS= ;;
  old) PREVIOUS=prewarm ;;
  off) PREVIOUS=old ;;
  unnamed) PREVIOUS=off ;;
  target-a) PREVIOUS=compare-off ;;
  target-l) PREVIOUS=target-a ;;
  target-b) PREVIOUS=target-l ;;
  negatives) PREVIOUS=target-b ;;
  dspark-a) PREVIOUS=negatives ;;
  dspark-l) PREVIOUS=dspark-a ;;
  dspark-b) PREVIOUS=dspark-l ;;
  *) echo "unknown phase"; exit 2 ;;
esac
if [[ "$PHASE" = prewarm ]]; then
    LIMIT=${LIMIT:-7200}
    [[ "$LIMIT" =~ ^[0-9]+$ && "$LIMIT" -ge 30 && "$LIMIT" -le 86400 ]] || exit 2
else
    LIMIT=${LIMIT:-690}
    [[ "$LIMIT" =~ ^[0-9]+$ && "$LIMIT" -ge 30 && "$LIMIT" -le 690 ]] || exit 2
fi
# One deadline includes setup, SSH, comparisons, and cleanup. Reserve 14s for cleanup and watchdog escalation.
if [[ ${DS41_GATE_INNER:-0} != 1 ]]; then
    export DS41_GATE_INNER=1 DS41_GATE_END=$(( $(date +%s) + LIMIT ))
    export DS41_GATE_DEADLINE=$((DS41_GATE_END - 14))
    if [[ "$PHASE" = prewarm ]]; then
        echo "EXPECTED CPU-only prewarm <= ${LIMIT}s; no GPU allocation or model upload"
    else
        echo "EXPECTED phase $PHASE <= ${LIMIT}s including cold loads, setup, comparisons and verified cleanup; one exclusive GPU allocation; DSpark ${DSPARK_POLICY:-controller}"
    fi
    exec timeout --signal=TERM --kill-after=10 "$((DS41_GATE_END - $(date +%s) - 12))" bash "$0" "$@"
fi
DSPARK_POLICY=${DSPARK_POLICY:-controller}
[[ "$DSPARK_POLICY" == controller || "$DSPARK_POLICY" =~ ^fixed:[0-5]$ ]] || exit 2
remaining() { local n=$((DS41_GATE_DEADLINE - $(date +%s))); ((n > 0)) || exit 124; echo "$n"; }
[[ "$OUT" = /* && "$OUT" != / && "$OUT" != *".."*  ]] || { echo "OUT must be an absolute run directory"; exit 2; }
if [[ -z "$PREVIOUS" ]]; then
    [[ ! -e "$OUT" ]] || { echo "prewarm requires a fresh OUT"; exit 2; }
else
    [[ -f "$OUT/$PREVIOUS.done" && ! -e "$OUT/$PHASE.done" ]] || { echo "missing prerequisite $PREVIOUS or phase already completed"; exit 2; }
    [[ $(cat "$OUT/policy") == "$DSPARK_POLICY" ]] || { echo "policy changed between phases"; exit 2; }
fi
# Receipts alone are insufficient if run artifacts were removed between GPU allocations.
require_arm() {
    local arm=$1 rank
    for rank in 0 1; do
        [[ -s "$OUT/rank$rank/$arm.log" && -s "$OUT/rank$rank/$arm-r$rank.f32" && -f "$OUT/rank$rank/$arm.ended" ]] || { echo "missing prerequisite artifacts: $arm rank$rank"; exit 2; }
    done
}
case "$PHASE" in
  off) require_arm old ;;
  unnamed) require_arm off ;;
  target-a) require_arm unnamed ;;
  target-l) require_arm A ;;
  target-b) require_arm A; require_arm L ;;
  negatives) require_arm B ;;
  dspark-l) require_arm DA ;;
  dspark-b) require_arm DA; require_arm DL ;;
esac
case "$PHASE" in
  target-b|negatives|dspark-b) [[ -s "$OUT/checkpoints/gate.rank0.group" ]] || { echo "missing local group manifest"; exit 2; } ;;
esac
PHASE_START=$SECONDS
PEER_PID=
REMOTE_READY=0
cleanup() {
    local rc=$? command p f receipt
    trap - EXIT TERM INT HUP
    for f in "$OUT"/rank0/*.pid; do
        [[ -f "$f" ]] || continue
        read -r p < "$f"
        kill -KILL -- "-$p" 2>/dev/null || true
        if kill -0 -- "-$p" 2>/dev/null; then rc=1; fi
    done
    printf -v command 'for f in %q/rank1/*.pid; do [ -f "$f" ] || continue; read -r p < "$f"; kill -KILL -- "-$p" 2>/dev/null || true; if kill -0 -- "-$p" 2>/dev/null; then exit 1; fi; done' "$OUT"
    if [[ "$REMOTE_READY" = 1 ]]; then
        timeout -k 1 5 ssh -o BatchMode=yes -o ConnectTimeout=3 "$NODE1" "$command" || rc=1
    fi
    if [[ -n "$PEER_PID" ]]; then kill -KILL "$PEER_PID" 2>/dev/null || true; wait "$PEER_PID" 2>/dev/null || true; fi
    if [[ "$rc" = 0 && "$REMOTE_READY" = 1 ]]; then
        receipt=$(printf 'phase=%s wall_seconds=%s P=%s cleanup_both_nodes=verified\n' "$PHASE" "$((SECONDS-PHASE_START))" "$P"
                  grep '^TIMING ' "$OUT/results.txt") || exit 1
        printf '%s\n' "$receipt" > "$OUT/$PHASE.receipt" || exit 1
        mv -- "$OUT/$PHASE.receipt" "$OUT/$PHASE.done" || rc=1
    fi
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 124' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
SUFFIX=${SUFFIX:-101,102,103,104}
[[ "$OUT" = /* && "$OUT" != / && "$OUT" != *".."* ]] || exit 2
mkdir -p "$OUT/rank0" "$OUT/rank1"
if [[ -z "$PREVIOUS" ]]; then
    printf '%s\n' "$DSPARK_POLICY" > "$OUT/policy"
    printf -v setup 'test ! -e %q && mkdir -p %q && date +%%s' "$OUT" "$OUT/rank1"
else
    printf -v setup 'test -d %q && date +%%s' "$OUT/rank1"
fi
peer_clock=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$NODE1" "$setup")
REMOTE_READY=1
case "$PHASE" in
  target-b|negatives|dspark-b)
    printf -v check 'test -s %q' "$OUT/checkpoints/gate.rank1.group"
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$NODE1" "$check" || { echo "missing peer group manifest"; exit 2; } ;;
esac
clock_delta=$(( $(date +%s) - peer_clock ))
((clock_delta >= -1 && clock_delta <= 1)) || { echo "FAIL node clocks differ by more than 1s"; exit 1; }
ENV_ARGS=()
while IFS= read -r key; do ENV_ARGS+=("$key=${!key}"); done < <(compgen -e | LC_ALL=C sort | grep -E '^(STRIX_|HIP_|ROCR_|HSA_|AMD_|GPU_)' || true)
# Scheduler allocation variables may change between launches; numerical settings do not.
SETTINGS_ENV=()
for entry in "${ENV_ARGS[@]}"; do
    case "${entry%%=*}" in
      GPU_TURN_*|GPU_WINDOW|STRIX_HASH_CACHE|STRIX_SESSION_DIR|STRIX_SESSION_CACHE|STRIX_SESSION_DISK|STRIX_SESSION_DISK_BYTES|STRIX_SESSION_RAM_BYTES|STRIX_SESSION_MAX_PARKED|STRIX_SESSION_POOL_IDENTITY_TEST) ;;
      *) SETTINGS_ENV+=("$entry") ;;
    esac
done
P=${P:-4096}
[[ "$P" =~ ^[0-9]+$ && "$P" -ge 1 && "$P" -le 16000 ]] || exit 2
# Rank session adapter supports sequential R1 only, not the separate tiled fill path.
# Lock prompt length and binary provenance across all allocated turns.
# All ldd-resolved libraries (including system libraries) are required identical.
# This is deliberately stronger than only binding non-system numerical libraries.
library_inventory() {
    python3 - "$@" <<'PY'
import hashlib,pathlib,re,subprocess,sys
files=set()
for binary in sys.argv[1:]:
    run=subprocess.run(['ldd',binary],capture_output=True,text=True)
    if run.returncode or 'not found' in run.stdout:
        raise RuntimeError('cannot resolve runtime libraries: '+binary+' '+run.stdout+run.stderr)
    for line in run.stdout.splitlines():
        m=re.search(r'(?:=>\s+)?(/\S+)\s+\(0x',line)
        if m: files.add(m[1])
for name in sorted(files):
    h=hashlib.sha256()
    with open(name,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    print(h.hexdigest()+'  '+name)
PY
}
# The PID file records the remote process-group leader before work starts.
# A remote-side absolute deadline bounds work even if the SSH client disappears.
tracked_runner='pidfile=$1; ended=$2; deadline=$3; shift 3; limit=$((deadline - $(date +%s))); [ "$limit" -gt 0 ] || exit 124; setsid timeout -k 2 "$limit" "$@" & p=$!; echo "$p" > "$pidfile"; wait "$p"; rc=$?; if kill -0 -- "-$p" 2>/dev/null; then kill -KILL -- "-$p"; exit 125; fi; rm -f "$pidfile"; echo "$rc" > "$ended"; exit "$rc"'
libraries=$(library_inventory "$BIN" "$BASELINE_BIN")
printf '%s\n' "$libraries" > "$OUT/rank0/$PHASE-libraries.sha256"
printf -v library_command '%s; library_inventory %q %q' "$(declare -f library_inventory)" "$BIN" "$BASELINE_BIN"
printf -v library_remote '%q ' bash -c "$tracked_runner" gate "$OUT/rank1/$PHASE-libraries.pid" "$OUT/rank1/$PHASE-libraries.ended" "$DS41_GATE_DEADLINE" bash -c "$library_command"
timeout -k 2 "$(remaining)" ssh -o BatchMode=yes -o ConnectTimeout=5 "$NODE1" "$library_remote" > "$OUT/rank1/$PHASE-libraries.sha256"
peer_libraries=$(cat "$OUT/rank1/$PHASE-libraries.sha256")
[[ "$libraries" == "$peer_libraries" ]] || { echo "peer runtime-library mismatch"; exit 2; }
binary_identity=$(sha256sum "$BIN" "$BASELINE_BIN")
printf -v identity_check 'cat %q; sha256sum %q %q' "$BASELINE_BIN.revision" "$BIN" "$BASELINE_BIN"
printf -v identity_remote '%q ' bash -c "$tracked_runner" gate "$OUT/rank1/$PHASE-identity.pid" "$OUT/rank1/$PHASE-identity.ended" "$DS41_GATE_DEADLINE" bash -c "$identity_check"
peer_identity=$(timeout -k 2 "$(remaining)" ssh -o BatchMode=yes -o ConnectTimeout=5 "$NODE1" "$identity_remote")
[[ "$peer_identity" == "$BASELINE_REV"$'\n'"$binary_identity" ]] || { echo "peer binary/provenance mismatch"; exit 2; }
printf 'TIMING %s runtime_libraries=identical inventory_sha256=%s\n' "$PHASE" "$(printf '%s' "$libraries" | sha256sum | cut -d' ' -f1)" >> "$OUT/results.txt"
settings=$(printf '%s\n' "$BIN" "$BASELINE_BIN" "$BASELINE_REV" "$binary_identity" "$libraries" "$P" "$MODEL" "$DSPARK" "$NODE1" "$SUFFIX" "$DSPARK_POLICY" "${SETTINGS_ENV[@]}")
if [[ -z "$PREVIOUS" ]]; then printf '%s\n' "$settings" > "$OUT/settings";
else [[ $(cat "$OUT/settings") == "$settings" ]] || { echo "settings changed between phases"; exit 2; }; fi
# A separate rank invocation hashes every bound target + draft asset without
# opening transport or constructing DeviceForward. Each invocation has a fresh
# process memo, so the second prewarm pass proves persistent-cache readiness.
[[ -n "${STRIX_HASH_CACHE:-}" && "$STRIX_HASH_CACHE" != 0 ]] || { echo "STRIX_HASH_CACHE must name a persistent cache directory"; exit 2; }
hash_assets() {
    local pass=$1 rank command log
    for rank in 0 1; do
        log="$OUT/rank$rank/$PHASE-hash-$pass.log"
        local -a args=(env "${ENV_ARGS[@]}" "$BIN" "$MODEL" --rank "$rank" --layers 40 --max-positions 16384 --tp-attention
            --session-hash-only --session-dspark --stage-dspark "$DSPARK" --stage-dspark-policy "$DSPARK_POLICY")
        if [[ "$rank" = 0 ]]; then
            timeout -k 2 "$(remaining)" bash -c "$tracked_runner" gate "$OUT/rank0/$PHASE-hash-$pass.pid" "$OUT/rank0/$PHASE-hash-$pass.ended" "$DS41_GATE_DEADLINE" "${args[@]}" > "$log" 2>&1
        else
            printf -v command '%q ' bash -c "$tracked_runner" gate "$OUT/rank1/$PHASE-hash-$pass.pid" "$OUT/rank1/$PHASE-hash-$pass.ended" "$DS41_GATE_DEADLINE" "${args[@]}"
            timeout -k 2 "$(remaining)" ssh -o BatchMode=yes -o ConnectTimeout=5 "$NODE1" "$command" > "$log" 2>&1
        fi
        python3 - "$log" "$PHASE" "$pass" "$rank" "$OUT/results.txt" <<'PY'
import pathlib,re,sys
log,phase,run,rank,out=sys.argv[1:]
m=re.search(r'session hashing rank \d+ seconds ([\d.]+) cache_hits (\d+) cache_misses (\d+)',pathlib.Path(log).read_text())
if not m: raise ValueError('missing hashing telemetry')
seconds,hits,misses=m.groups()
if (phase!='prewarm' or run=='2') and (int(hits)==0 or int(misses)!=0):
    raise ValueError('expected warm persistent hash cache for every bound file')
with open(out,'a') as f:
    f.write(f'TIMING {phase} rank={rank} hash_pass={run} hashing_seconds={seconds} cache_hits={hits} cache_misses={misses} checkpoint_io_seconds=0\n')
PY
    done
}
hash_assets 1
if [[ "$PHASE" = prewarm ]]; then
    hash_assets 2
    echo "PASS CPU-only prewarm on both nodes, second pass all persistent-cache hits"
    exit 0
fi
PROMPT=$(python3 -c "print(','.join(str(100+i%100) for i in range($P)))")
printf '%s\n' "$PROMPT" > "$OUT/prompt.ids"
printf '%s\n' "$SUFFIX" > "$OUT/suffix.ids"
BASE=("$BIN" "$MODEL" --layers 40 --max-positions 16384 --tp-attention --transport usb4auto --session-cache-mb 512 --session-id gate)
pair() {
    local name=$1 mode=$2 expect=${3:-pass} start=$SECONDS rc0 rc1 command
    local -a extra rank0 rank1 base=("${BASE[@]}") pair_env=(env -u STRIX_SESSION_CACHE "${ENV_ARGS[@]}")
    case "$mode" in
      ordinary)
        base=("$BIN" "$MODEL" --layers 40 --max-positions 16384 --tp-attention --transport usb4auto)
        if [[ "$name" = old ]]; then base[0]=$BASELINE_BIN; fi
        if [[ "$name" = off ]]; then base+=(--session-cache off); pair_env+=(STRIX_SESSION_CACHE=0); fi
        # Unnamed uses actual defaults; remove any inherited session switch.
        if [[ "$name" = unnamed || "$name" = old ]]; then pair_env+=(env -u STRIX_SESSION_CACHE); fi
        extra=(--prompt-ids "$PROMPT" --generate 64)
        ;;
      live) pair_env+=(STRIX_SESSION_CACHE=1); extra=(--prompt-ids "$PROMPT" --generate 8 --session-save "$OUT/checkpoints" --session-suffix-ids "$SUFFIX" --session-suffix-tokens 64) ;;
      cold) pair_env+=(STRIX_SESSION_CACHE=1); extra=(--prompt-ids "$PROMPT" --generate 8 --session-suffix-ids "$SUFFIX" --session-suffix-tokens 64) ;;
      load) pair_env+=(STRIX_SESSION_CACHE=1); extra=(--prompt-ids "$SUFFIX" --generate 64 --session-load "$OUT/checkpoints") ;;
      *) return 2 ;;
    esac
    [[ "$expect" != refuse ]] || extra+=(--session-validate-only)
    local budget runner
    budget=$DS41_GATE_DEADLINE
    remaining > /dev/null
    # An ended marker requires wait and a process-group liveness check.
    runner=$tracked_runner
    # Line-buffered stdio: with stdout block-buffered into the shared log, a stderr line could land
    # inside a session result line.
    rank0=(stdbuf -oL -eL "${pair_env[@]}" "${base[@]}" --rank 0 "${extra[@]}" --logits "$OUT/rank0/$name" --state-out "$OUT/rank0/$name")
    rank1=(stdbuf -oL -eL "${pair_env[@]}" "${base[@]}" --rank 1 "${extra[@]}" --logits "$OUT/rank1/$name" --state-out "$OUT/rank1/$name")
    if [[ "$mode" = ordinary ]]; then
        rank0+=(--logits "$OUT/rank0/$name-r0.f32")
        rank1+=(--logits "$OUT/rank1/$name-r1.f32")
    fi
    printf -v command '%q ' bash -c "$runner" gate "$OUT/rank1/$name.pid" "$OUT/rank1/$name.ended" "$budget" "${rank1[@]}"
    printf -v command '%s > %q 2>&1' "$command" "$OUT/rank1/$name.log"
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$NODE1" "$command" > "$OUT/rank1/$name.ssh.log" 2>&1 & PEER_PID=$!
    bash -c "$runner" gate "$OUT/rank0/$name.pid" "$OUT/rank0/$name.ended" "$budget" "${rank0[@]}" > "$OUT/rank0/$name.log" 2>&1 &
    local wrapper=$!
    # Observe whichever rank/SSH wrapper finishes first. Unexpected termination
    # must enter cleanup immediately rather than waiting out the other deadline.
    local completed first_rc=0 wanted=0
    [[ "$expect" != refuse ]] || wanted=2
    wait -n -p completed "$wrapper" "$PEER_PID" || first_rc=$?
    [[ "$first_rc" = "$wanted" ]] || { echo "FAIL $name first rank exit $first_rc (wanted $wanted)"; exit 1; }
    if [[ "$completed" = "$wrapper" ]]; then
        rc0=$first_rc; rc1=0; wait "$PEER_PID" || rc1=$?
    else
        rc1=$first_rc; rc0=0; wait "$wrapper" || rc0=$?
    fi
    PEER_PID=
    printf -v command 'cat %q; cat %q >&2' "$OUT/rank1/$name.log" "$OUT/rank1/$name.ended"
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$NODE1" "$command" > "$OUT/rank1/$name.log" 2> "$OUT/rank1/$name.ended" || { echo "FAIL $name SSH/termination verification"; exit 1; }
    [[ $(cat "$OUT/rank0/$name.ended") == "$rc0" && $(cat "$OUT/rank1/$name.ended") == "$rc1" ]] || { echo "FAIL $name processes did not end normally"; exit 1; }
    printf 'TIMING %s wall_seconds=%s rank0_exit=%s rank1_exit=%s\n' "$name" "$((SECONDS-start))" "$rc0" "$rc1" | tee -a "$OUT/results.txt"
    python3 - "$OUT" "$name" "$mode" <<'PY'
import pathlib,re,sys
root=pathlib.Path(sys.argv[1]); name=sys.argv[2]; mode=sys.argv[3]
for rank in range(2):
    log=(root/f'rank{rank}/{name}.log').read_text()
    loads=re.findall(r'(?:loaded in |session model load rank \d+ seconds )([\d.]+)',log)
    # Validation-only refusal intentionally performs no model load.
    if name not in ('wrong-flag','corrupt') and not loads: raise ValueError('missing model load time')
    draft=re.findall(r'session drafter load rank \d+ seconds ([\d.]+)',log)
    hashing=re.findall(r'session hashing rank \d+ seconds ([\d.]+)',log)
    # Session arms compute the key in the rank itself; that lookup must also be warm.
    counts=re.findall(r'session hashing rank \d+ seconds [\d.]+ cache_hits (\d+) cache_misses (\d+)',log)
    if mode!='ordinary' and (not counts or any(int(h)==0 or int(m)!=0 for h,m in counts)):
        raise ValueError('session arm hashing missing or not fully warm')
    checkpoint=re.findall(r'write_fsync_seconds ([\d.]+)',log)
    checkpoint+=re.findall(r'session restore rank \d+ seconds ([\d.]+)',log)
    checkpoint+=re.findall(r'session checkpoint io rank \d+ seconds ([\d.]+)',log)
    # Ordinary baseline/OFF paths do not compute a session key; the phase's
    # hash-only invocation above records their full bound-asset hashing time.
    with (root/'results.txt').open('a') as f:
        f.write(f'TIMING {name} rank={rank} session_hashing_seconds={sum(map(float,hashing)):.6f} checkpoint_io_seconds={sum(map(float,checkpoint)):.6f}\n')
    if name in ('DA','DL','DB') and len(draft)!=1: raise ValueError('missing drafter load time')
    with (root/'results.txt').open('a') as f:
        f.write(f'TIMING {name} rank={rank} load_seconds={loads[0] if loads else "0-validation-only"} drafter_load_seconds={draft[0] if draft else "0"}\n')
PY
    grep -hE '^session (checkpoint|restore|turn|final)' "$OUT/rank0/$name.log" "$OUT/rank1/$name.log" >> "$OUT/results.txt" || true
    if [[ "$expect" = refuse ]]; then
        if [[ "$rc0" = 2 && "$rc1" = 2 ]] &&
           grep -qE 'session cache: restore.*refused by one or both ranks' "$OUT/rank0/$name.log" &&
           grep -qE 'session cache: restore.*refused by one or both ranks' "$OUT/rank1/$name.log"; then
            echo "PASS $name coordinated refusal" | tee -a "$OUT/results.txt"
        else echo "FAIL $name refusal" | tee -a "$OUT/results.txt"; exit 1; fi
    elif [[ "$rc0" = 0 && "$rc1" = 0 ]]; then
        ssh "$NODE1" "cat $(printf '%q' "$OUT/rank1/$name-r1.f32")" > "$OUT/rank1/$name-r1.f32"
        if [[ "$mode" = ordinary ]]; then
            ssh "$NODE1" "cat $(printf '%q' "$OUT/rank1/$name-r1.state")" > "$OUT/rank1/$name-r1.state"
        fi
        echo "PASS $name execution" | tee -a "$OUT/results.txt"
    else echo "FAIL $name execution" | tee -a "$OUT/results.txt"; exit 1; fi
}
if [[ "$PHASE" = old || "$PHASE" = off || "$PHASE" = unnamed ]]; then
    pair "$PHASE" ordinary
    python3 "$(dirname "$0")/deepseek41_session_cache_compare_off.py" "$OUT" --extract "$PHASE" "$P"
fi
if [[ "$PHASE" = target-a ]]; then pair A live; fi
if [[ "$PHASE" = target-l ]]; then pair L cold; fi
if [[ "$PHASE" = target-b ]]; then
pair B load
if ! python3 - "$OUT" <<'PY' | tee -a "$OUT/results.txt"
import pathlib,re,subprocess,sys
root=pathlib.Path(sys.argv[1]); failures=[]
for rank in range(2):
    rows={}
    for phase in ('A','L','B'):
        log=(root/f'rank{rank}/{phase}.log').read_text()
        ids=re.findall(r'^session token rank \d+ index \d+ id (\d+)$',log,re.M)
        digest=re.findall(r'^session final rank \d+ canonical_sha256 ([0-9a-f]{64})$',log,re.M)
        data=(root/f'rank{rank}/{phase}-r{rank}.f32').read_bytes()
        if len(ids)!=64 or len(digest)!=1 or len(data)!=64*129280*4: failures.append(f'{phase}/rank{rank} incomplete dumps')
        rows[phase]=(ids,digest,data)
    if rows['A']!=rows['B']: failures.append(f'rank{rank} live vs restart bytes differ')
    if rows['A']!=rows['L']: failures.append(f'rank{rank} saved vs never-saved bytes differ')
    for phase in ('L','B'):
        subprocess.run(['cmp',str(root/f'rank{rank}/A-r{rank}.f32'),str(root/f'rank{rank}/{phase}-r{rank}.f32')],check=True)
if failures:
    print('FAIL exactness: '+'; '.join(failures)); sys.exit(1)
print('PASS DSpark OFF: both ranks, 64 ids, every FP32 logit byte, canonical state SHA-256')
PY
then exit 1; fi
fi
if [[ "$PHASE" = negatives ]]; then
# Invert a numerical flag on BOTH ranks, then restore the inherited environment.
if [[ "${STRIX_DS41_IDX_DIRECT:-unset}" = 1 ]]; then ENV_ARGS+=(STRIX_DS41_IDX_DIRECT=0); else ENV_ARGS+=(STRIX_DS41_IDX_DIRECT=1); fi
pair wrong-flag load refuse
unset 'ENV_ARGS[${#ENV_ARGS[@]}-1]'
# Mutate only the rank-1 node's checkpoint. Parse the current manifest to find its generation.
mutate() {
    local command
    printf -v command 'python3 - %q %q' "$OUT/checkpoints" "$1"
    ssh "$NODE1" "$command" <<'PY'
import pathlib,shutil,struct,sys
root=pathlib.Path(sys.argv[1]); mode=sys.argv[2]; backup=root.with_name(root.name+'-backup')
if mode=='backup':
    if backup.exists(): shutil.rmtree(backup)
    shutil.copytree(root,backup); sys.exit()
if mode=='reset':
    if root.exists(): shutil.rmtree(root)
    shutil.copytree(backup,root); sys.exit()
def string(data,at):
    n=struct.unpack_from('<Q',data,at)[0]; at+=8
    if n>len(data)-at: raise ValueError('bounds')
    return data[at:at+n],at+n
manifest=root/'gate.rank1.group'; data=manifest.read_bytes()
_,at=string(data,0); at+=8
_,at=string(data,at); descriptor,_=string(data,at)
_,at=string(descriptor,0); generation,at=string(descriptor,at)
state=root/f'gate.{generation.decode()}.rank1.state'
if mode=='corrupt':
    b=bytearray(state.read_bytes()); b[len(b)//2]^=1; state.write_bytes(b)
else: raise ValueError(mode)
PY
}
mutate backup
mutate corrupt
pair corrupt load refuse
mutate reset
fi
# Remaining negative cases run in the CPU suite.
# DSpark ON: one selected policy per bounded turn (controller by default).
if [[ "$PHASE" = dspark-* ]]; then
BASE+=(--session-dspark --stage-dspark "$DSPARK")
for policy in "$DSPARK_POLICY"; do
    BASE+=(--stage-dspark-policy "$policy")
    if [[ "$PHASE" = dspark-a ]]; then pair DA live; fi
    if [[ "$PHASE" = dspark-l ]]; then pair DL cold; fi
    if [[ "$PHASE" = dspark-b ]]; then
    pair DB load
    if ! python3 - "$OUT" "$policy" <<'PY' | tee -a "$OUT/results.txt"
import pathlib,re,subprocess,sys
root=pathlib.Path(sys.argv[1])
for rank in range(2):
    rows={}
    for phase in ('DA','DL','DB'):
        log=(root/f'rank{rank}/{phase}.log').read_text()
        ids=re.findall(r'^session token rank \d+ index \d+ id (\d+)$',log,re.M)
        digest=re.findall(r'^session final rank \d+ canonical_sha256 ([0-9a-f]{64})$',log,re.M)
        acceptance=re.findall(r'^session acceptance rank \d+ windows (\d+) verified (\d+) accepted (\d+)$',log,re.M)
        data=root/f'rank{rank}/{phase}-r{rank}.f32'
        assert len(ids)==64 and len(digest)==1 and len(acceptance)==1 and int(acceptance[0][0])>0
        assert data.stat().st_size==64*129280*4
        rows[phase]=(ids,digest,acceptance)
    assert rows['DA']==rows['DL']==rows['DB'],f'rank{rank} DSpark continuation differs'
    for phase in ('DL','DB'):
        subprocess.run(['cmp',str(root/f'rank{rank}/DA-r{rank}.f32'),str(root/f'rank{rank}/{phase}-r{rank}.f32')],check=True)
print('PASS DSpark',sys.argv[2],'A == L == B: ids, cmp every FP32 row, final target+drafter SHA, acceptance counts')
PY
    then exit 1; fi
    # Preserve each policy's complete local evidence before the next policy reuses DA/DL/DB.
    python3 - "$OUT" "$policy" <<'PY' | tee -a "$OUT/results.txt"
import pathlib,re,shutil,sys
root=pathlib.Path(sys.argv[1]); policy=sys.argv[2].replace(':','-')
for rank in range(2):
    source=root/f'rank{rank}'; dest=root/f'dspark-{policy}'/f'rank{rank}'
    dest.mkdir(parents=True,exist_ok=True)
    for phase in ('DA','DL','DB'):
        for p in source.glob(phase+'*'):
            if p.is_file(): shutil.copy2(p,dest/p.name)
    logs=[(source/f'{p}.log').read_text() for p in ('DA','DL','DB')]
    capture,write,n=re.search(r'capture_seconds ([\d.]+) write_fsync_seconds ([\d.]+) bytes (\d+)',logs[0]).groups()
    restore=re.search(r'session restore rank \d+ seconds ([\d.]+)',logs[2])[1]
    cold=re.search(r'input_seconds ([\d.]+)',logs[1])[1]
    print(f'PER_RANK dspark-{policy} rank={rank} save_stall_seconds={float(capture)+float(write):.6f} write_fsync_seconds={write} bytes={n} restore_seconds={restore} cold_prefill_P_seconds={cold}')
PY
    fi
    unset 'BASE[${#BASE[@]}-1]' 'BASE[${#BASE[@]}-1]'
done
fi
if [[ "$PHASE" = target-b ]]; then
python3 - "$OUT" <<'PY' | tee -a "$OUT/results.txt"
import pathlib,re,sys
root=pathlib.Path(sys.argv[1])
for label,phases in [('target',('A','L','B'))]:
    for rank in range(2):
        logs=[(root/f'rank{rank}/{p}.log').read_text() for p in phases]
        capture,write,n=re.search(r'capture_seconds ([\d.]+) write_fsync_seconds ([\d.]+) bytes (\d+)',logs[0]).groups()
        restore=re.search(r'session restore rank \d+ seconds ([\d.]+)',logs[2])[1]
        cold=re.search(r'input_seconds ([\d.]+)',logs[1])[1]
        print(f'PER_RANK {label} rank={rank} save_stall_seconds={float(capture)+float(write):.6f} write_fsync_seconds={write} bytes={n} restore_seconds={restore} cold_prefill_P_seconds={cold}')
PY
fi
echo "PASS phase $PHASE; artifacts $OUT"
