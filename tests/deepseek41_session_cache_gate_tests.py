"""Execute the gate's refusal predicate with CPU fixtures; never invokes SSH."""
import ast
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

bash = shutil.which("bash")
if not bash:
    raise SystemExit("bash is required")
source = (pathlib.Path(__file__).resolve().parents[1] / "tools/deepseek41_session_cache_gate.sh").read_text()
predicate = re.search(r'if \[\[ "\$rc0" = 2.*?; then', source, re.S).group()[3:-6]
with tempfile.TemporaryDirectory(prefix="ds41-gate-") as tmp:
    root = pathlib.Path(tmp)
    for rank in range(2):
        (root / f"rank{rank}").mkdir()
    refused = "error: ds41 session cache: restore validation (missing/corrupt/incompatible rank or generation) refused by one or both ranks\n"
    for rc0, rc1, log0, log1, expected in [
        (2, 2, refused, refused, True),
        (2, 137, refused, refused, False),
        (124, 2, refused, refused, False),
        (2, 255, refused, refused, False),
        (143, 2, refused, refused, False),
        (2, 2, refused, "", False),
        (2, 2, "", refused, False),
        (0, 2, refused, refused, False),
        (2, 2, "unrelated failure", refused, False),
    ]:
        (root / "rank0/test.log").write_text(log0)
        (root / "rank1/test.log").write_text(log1)
        # Bash on Windows needs a POSIX path; pwd in the fixture directory supplies it.
        command = 'OUT=$PWD; name=test; rc0=$1; rc1=$2; if ' + predicate + '; then exit 0; else exit 1; fi'
        result = subprocess.run([bash, "-c", command, "gate-test", str(rc0), str(rc1)], cwd=root)
        assert (result.returncode == 0) == expected, (rc0, rc1, log0, log1)
# Parse embedded comparisons without running remote commands.
for block in re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY", source, re.S):
    ast.parse(block)
# Execute only startup validation, stopping before any trap, setup or SSH.
preamble = source[:source.index("PEER_PID=\nREMOTE_READY=0")]
assert ':-690}' in preamble and '-le 690' in preamble
with tempfile.TemporaryDirectory(prefix="ds41-phases-") as tmp:
    root = pathlib.Path(tmp)
    baseline = root / "baseline"
    baseline.with_suffix(".revision").write_text("05fedd8011408fada8688b675d4ad2c699016c02\n")
    env = dict(os.environ, NODE1="mock", BASELINE_BIN=str(baseline), BIN="unused", MODEL="unused", DSPARK="unused", DS41_GATE_INNER="1", OUT="/missing-run", LIMIT="690")
    for phase in ("off", "unnamed", "target-a", "target-l", "target-b", "negatives", "dspark-a", "dspark-l", "dspark-b", "unknown"):
        result = subprocess.run([bash, "-c", preamble, "gate-test", phase], env=env, capture_output=True)
        assert result.returncode == 2, (phase, result.stdout, result.stderr)
    for limit in ("29", "691", "1200", "invalid"):
        result = subprocess.run([bash, "-c", preamble, "gate-test", "old"], env=dict(env, LIMIT=limit), capture_output=True)
        assert result.returncode == 2, (limit, result.stdout, result.stderr)
    # Valid predecessor receipts still require artifacts.
    for previous, phase in (("target-a","target-l"), ("target-l","target-b"), ("target-b","negatives"), ("dspark-a","dspark-l"), ("dspark-l","dspark-b")):
        (root / (previous + ".done")).write_text(previous)
        (root / "policy").write_text("controller")
        command = 'OUT=$PWD; BASELINE_BIN=$PWD/baseline; export OUT BASELINE_BIN; ' + preamble
        result = subprocess.run([bash, "-c", command, "gate-test", phase], cwd=root, env={k:v for k,v in env.items() if k != "OUT"}, capture_output=True)
        assert result.returncode == 2 and b"missing prerequisite artifacts" in result.stdout, (phase, result.stdout, result.stderr)
# Exercise phase dispatch with CPU stubs; strip embedded Python comparisons.
dispatch = source[source.index('if [[ "$PHASE" = target-a ]]; then pair A live; fi'):]
dispatch = re.sub(r"python3 - [^\n]*<<'PY'[^\n]*\n.*?\nPY", "true", dispatch, flags=re.S)
with tempfile.TemporaryDirectory(prefix="ds41-dispatch-") as tmp:
    for phase, expected in (("target-a","A"), ("target-l","L"), ("target-b","B"), ("dspark-a","DA"), ("dspark-l","DL"), ("dspark-b","DB")):
        command = 'set -euo pipefail; PHASE=$1; DSPARK_POLICY=controller; DSPARK=unused; OUT=$PWD; BASE=(); FAILURES=0; pair() { echo "ARM $1"; }; ' + dispatch
        result = subprocess.run([bash, "-c", command, "gate-test", phase], cwd=tmp, capture_output=True, text=True)
        assert result.returncode == 0 and re.findall(r'^ARM (\w+)$', result.stdout, re.M) == [expected], (phase, result.stdout, result.stderr)
print("PASS phase dispatch: save/control/restore run in separate invocations")
print("PASS phase limits, prerequisite receipts/artifacts, embedded comparisons; no SSH executed")
print("PASS gate refusal predicate: both ranks required; timeout/signal/SSH/unrelated failures rejected")

# A failed first refusal must stop before backup/corruption or another launch.
with tempfile.TemporaryDirectory(prefix="ds41-fail-fast-") as tmp:
    command = 'set -euo pipefail; PHASE=negatives; OUT=$PWD; ENV_ARGS=(); pair() { echo "ARM $1"; return 1; }; ssh() { echo MUTATED; }; ' + dispatch
    result = subprocess.run([bash, "-c", command], cwd=tmp, capture_output=True, text=True)
    assert result.returncode != 0 and "ARM wrong-flag" in result.stdout and "MUTATED" not in result.stdout and "ARM corrupt" not in result.stdout

# Execute cleanup itself with CPU process-group/SSH stubs. Receipt publication must
# follow successful cleanup on both nodes, including after a signal/watchdog exit.
cleanup = source[source.index("cleanup() {"):source.index("trap cleanup EXIT")]
with tempfile.TemporaryDirectory(prefix="ds41-cleanup-") as tmp:
    root = pathlib.Path(tmp)
    (root / "rank0").mkdir()
    (root / "rank0/test.pid").write_text("99999999")
    (root / "results.txt").write_text("TIMING A rank=0 load_seconds=1.0\n")
    for label, ssh_rc, initial_rc, live, expected in [
        ("success", 0, 0, False, True), ("ssh-failed", 255, 0, False, False),
        ("watchdog", 0, 124, False, False), ("leftover", 0, 0, True, False),
        ("peer-leftover", 1, 0, False, False),
    ]:
        command = ('OUT=$PWD; PHASE=' + label + '; P=4096; SECONDS=0; PHASE_START=0; PEER_PID=; REMOTE_READY=1; NODE1=mock; '
                   + 'timeout() { "$@"; }; ssh() { return ' + str(ssh_rc) + '; }; '
                   + 'timeout() { shift 3; "$@"; }; '
                   + ('kill() { return 0; }; ' if live else 'kill() { return 1; }; ')
                   + cleanup + '(exit ' + str(initial_rc) + '); cleanup')
        result = subprocess.run([bash, "-c", command], cwd=root, capture_output=True, text=True)
        assert (result.returncode == 0) == expected, (label, result.stdout, result.stderr)
        assert (root / (label + ".done")).exists() == expected, label
print("PASS immediate refusal failure, two-node cleanup, SSH, leftover and watchdog receipt suppression")

# Full driver integration with CPU rank/SSH stand-ins. The peer has a separate
# filesystem tree, exercising artifact retrieval, option/env selection and receipts.
with tempfile.TemporaryDirectory(prefix="ds41-driver-") as tmp:
    root = pathlib.Path(tmp)
    bindir = root / "bin"
    bindir.mkdir()
    ssh = bindir / "ssh"
    ssh.write_text("""#!/usr/bin/env python3
import os,subprocess,sys
command=sys.argv[-1]
if os.environ.get('FAIL_CLEANUP') == '1' and command.startswith('for f in '): sys.exit(255)
command=command.replace(os.environ['GATE_OUT'],os.environ['GATE_OUT']+'-peer')
run=subprocess.run(['bash','-c',command],capture_output=True)
output=run.stdout
if os.environ.get('BAD_LIBRARY') == '1' and 'library_inventory' in command: output=b'changed peer library inventory\\n'
sys.stdout.buffer.write(output); sys.stderr.buffer.write(run.stderr)
sys.exit(run.returncode)
""")
    library = root / "runtime.so"
    library.write_bytes(b"runtime library fixture")
    ldd = bindir / "ldd"
    ldd.write_text('#!/usr/bin/env bash\nprintf "libfixture.so => %s (0x1234)\\n" "$MOCK_LIBRARY"\n')
    ldd.chmod(0o755)
    rank = bindir / "candidate"
    rank.write_text("""#!/usr/bin/env python3
import os,pathlib,sys
args=sys.argv[1:]
def value(k): return args[len(args)-1-args[::-1].index(k)+1]
r=int(value('--rank'))
if '--session-hash-only' in args:
    assert '--stage-dspark' in args and '--session-dspark' in args
    print(f"session hashing rank {r} seconds 0.125000 cache_hits 5 cache_misses {os.environ.get('BAD_CACHE', '0')}")
    sys.exit(0)
prompt=value('--prompt-ids').split(','); n=len(prompt)+int(value('--generate'))
if pathlib.Path(sys.argv[0]).name=='baseline': assert '--session-cache' not in args
elif '--session-cache' in args: assert value('--session-cache')=='off' and os.environ['STRIX_SESSION_CACHE']=='0'
else: assert 'STRIX_SESSION_CACHE' not in os.environ
if os.environ.get('BAD_RANK') == str(r): sys.exit(7)
print(f'rank {r} of 2: loaded in 1.5 s')
for i in range(n): print(f'pos {i} fed 100 argmax 101 top 0.0000 logits_fnv 0000000000000000 (0.0 s)')
print(f'state rank {r} position {n} digest 0123456789abcdef file ignored')
with open(value('--logits'),'wb') as f: f.truncate(n*129280*4)
pathlib.Path(value('--state-out')+f'-r{r}.state').write_bytes(b'complete state bytes')
""")
    baseline = bindir / "baseline"
    baseline.write_text(rank.read_text())
    baseline.with_suffix('.revision').write_text('05fedd8011408fada8688b675d4ad2c699016c02\n')
    for executable in (ssh, rank, baseline): executable.chmod(0o755)
    gate = pathlib.Path(__file__).resolve().parents[1] / 'tools/deepseek41_session_cache_gate.sh'
    compare = gate.with_name('deepseek41_session_cache_compare_off.py')
    out = root / 'evidence'
    env = dict(os.environ, PATH=str(bindir)+os.pathsep+os.environ['PATH'], BIN=str(rank), BASELINE_BIN=str(baseline),
               MOCK_LIBRARY=str(library), MODEL='unused', DSPARK='unused', NODE1='mock', OUT=str(out), GATE_OUT=str(out), P='1', LIMIT='30', STRIX_HASH_CACHE=str(root/'hash-cache'))
    for phase in ('prewarm','old','off','unnamed','compare-off'):
        result=subprocess.run([bash,str(gate),phase],env=env,capture_output=True,text=True)
        assert result.returncode == 0, (phase,result.stdout,result.stderr)
        receipt=(out/(phase+'.done')).read_text()
        assert 'wall_seconds=' in receipt
        if phase != 'compare-off':
            assert 'cleanup_both_nodes=verified' in receipt and 'wall_seconds=' in receipt
            for r in range(2):
                assert f'rank={r} hash_pass=1 hashing_seconds=0.125000' in receipt
                assert 'checkpoint_io_seconds=' in receipt
                if phase != 'prewarm': assert f'rank={r} load_seconds=1.5' in receipt
                else: assert f'rank={r} hash_pass=2 hashing_seconds=0.125000 cache_hits=5 cache_misses=0' in receipt
    inventory0=(out/'rank0/prewarm-libraries.sha256').read_text()
    inventory1=(out/'rank1/prewarm-libraries.sha256').read_text()
    assert inventory0 == inventory1 and str(library) in inventory0
    library.write_bytes(b"changed library fixture")
    result=subprocess.run([bash,str(gate),'target-a'],env=env,capture_output=True,text=True)
    assert result.returncode != 0 and 'settings changed' in result.stdout, (result.stdout,result.stderr)
    library.write_bytes(b"runtime library fixture")
    mismatch=root/'bad-library'
    result=subprocess.run([bash,str(gate),'prewarm'],env=dict(env,OUT=str(mismatch),GATE_OUT=str(mismatch),BAD_LIBRARY='1'),capture_output=True,text=True)
    assert result.returncode != 0 and 'peer runtime-library mismatch' in result.stdout and not (mismatch/'prewarm.done').exists(), (result.stdout,result.stderr)
    bad_cache=root/'bad-cache'
    result=subprocess.run([bash,str(gate),'prewarm'],env=dict(env,OUT=str(bad_cache),GATE_OUT=str(bad_cache),BAD_CACHE='1'),capture_output=True,text=True)
    assert result.returncode != 0 and not (bad_cache/'prewarm.done').exists(), (result.stdout,result.stderr)
    # Every evidence type on either rank is mandatory and compared byte-for-byte.
    for r in range(2):
        for suffix in ('.ids','.digest',f'-r{r}.f32',f'-r{r}.state'):
            artifact=out/f'rank{r}/unnamed{suffix}'
            with artifact.open('r+b') as f:
                before=f.read(1); f.seek(0); f.write(bytes([before[0]^1]))
            result=subprocess.run([sys.executable,str(compare),str(out)],capture_output=True)
            assert result.returncode != 0, (r,suffix)
            with artifact.open('r+b') as f: f.write(before)
    for label, overrides in (('cleanup-failure',{'FAIL_CLEANUP':'1'}),('rank0-failure',{'BAD_RANK':'0'}),('rank1-failure',{'BAD_RANK':'1'})):
        failed=root/label
        ready=subprocess.run([bash,str(gate),'prewarm'],env=dict(env,OUT=str(failed),GATE_OUT=str(failed)),capture_output=True,text=True)
        assert ready.returncode == 0, (ready.stdout,ready.stderr)
        result=subprocess.run([bash,str(gate),'old'],env=dict(env,OUT=str(failed),GATE_OUT=str(failed),**overrides),capture_output=True,text=True)
        assert result.returncode != 0 and not (failed/'old.done').exists(), (label,result.stdout,result.stderr)
print('PASS CPU driver: old/off/unnamed/compare-off, bitwise failures on both ranks, rank/SSH cleanup failure receipts')

# The remote runner itself bounds a detached process group, even without a client.
runner=re.search(r"^tracked_runner='(.*)'$",source,re.M).group(1)
assert 'setsid timeout -k 2' in runner and 'echo "$p" > "$pidfile"' in runner
with tempfile.TemporaryDirectory(prefix='ds41-hash-deadline-') as tmp:
    command='deadline=$(($(date +%s)+1)); bash -c "$1" gate "$PWD/hash.pid" "$PWD/hash.ended" "$deadline" sleep 10'
    result=subprocess.run([bash,'-c',command,'gate-test',runner],cwd=tmp,capture_output=True,text=True,timeout=8)
    assert result.returncode == 124, (result.stdout,result.stderr)
    assert pathlib.Path(tmp,'hash.ended').read_text().strip() == '124'
print('PASS remote hash deadline/process group, runtime-library inventory and phase drift rejection')

# Compare-off must actually time out without publishing success. This fixture
# runs only a sleeping CPU comparator under the production LIMIT watchdog.
with tempfile.TemporaryDirectory(prefix='ds41-compare-watchdog-') as tmp:
    root=pathlib.Path(tmp)
    gate=root/'gate.sh'; gate.write_text(source)
    (root/'deepseek41_session_cache_compare_off.py').write_text('import time; time.sleep(45)')
    (root/'unnamed.done').write_text('fixture')
    env=dict(os.environ,NODE1="mock",OUT=str(root),LIMIT='30')
    env.pop('DS41_GATE_INNER',None)
    result=subprocess.run([bash,str(gate),'compare-off'],env=env,capture_output=True,text=True,timeout=34)
    assert result.returncode == 124 and not (root/'compare-off.done').exists(), (result.returncode,result.stdout,result.stderr)
print('PASS compare-off LIMIT watchdog suppresses receipt on timeout; success records wall time')
