#!/usr/bin/env bash
# Linux only; opt-in SINGLE phases. Invoke each phase in a separate turn; never loop GPU phases.
# Usage: bash GATE PHASE BIN MODEL P_IDS S_IDS OUT [-- COMMON_FLAGS...]
# 1: 32K save A; 2: never-saved L; 3: cross-turn restore B + CPU preflight negatives.
# 4: CPU comparison of 64 IDs, every FP32 logit byte, digests and MTP counts.
# 5/6/7: preserved baseline / candidate OFF / candidate ON unnamed identity (one GPU turn each).
# BASE_BIN is required in every phase. PASS is withheld until phase 7.
# Unaligned target-only case: use unaligned-1..unaligned-4 in separate turns,
# P=32768, FIRST_GENERATE=2, tile=1024, S=2046. Restore begins at 32769,
# consumes pending plus 1022 suffix IDs serially, then executes one complete tile.
# Phase 4 compares saved/live/restored tokens, every logit byte and final digest;
# this case alone does not qualify automatic serving admission.
set -euo pipefail
export SESSION_GATE_HELPERS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# < 6 )); then
  echo "usage: $0 {1|2|3|4|5|6|7} BIN MODEL P_IDS S_IDS OUT [-- COMMON_FLAGS...]" >&2
  exit 2
fi
# Absolute budget includes Python setup, preflights, model load and verified cleanup.
export SESSION_GATE_DEADLINE=$(( $(date +%s) + 440 ))
exec timeout --signal=TERM --kill-after=5s 440s python3 - "$@" <<'PY'
import hashlib, json, os, pathlib, re, shutil, signal, struct, subprocess, sys, time
sys.path.insert(0, os.environ['SESSION_GATE_HELPERS'])
from gate_identity import compare_identity, restore_turn1, http_identity, numerical_digests
started = time.monotonic()
deadline = started + max(0, int(os.environ['SESSION_GATE_DEADLINE']) - time.time() - 10)
phase, binary, model, pfile, sfile, output, *flags = sys.argv[1:]
unaligned = phase.startswith("unaligned-")
if unaligned:
    phase = phase.removeprefix("unaligned-")
    if phase not in ("1", "2", "3", "4"): raise SystemExit("unaligned case phases are 1..4")
if phase not in ('1', '2', '3', '4', '5', '6', '7'): raise SystemExit('PHASE must be 1..7 or unaligned-1..unaligned-4')
if flags[:1] == ['--']: flags = flags[1:]
binary, model = str(pathlib.Path(binary).resolve()), str(pathlib.Path(model).resolve())
out = pathlib.Path(output).resolve()
def left():
    remaining = deadline-time.monotonic()
    if remaining <= 0: raise TimeoutError('phase absolute deadline exhausted')
    return remaining
def deadline_signal(signum, frame): raise TimeoutError('phase deadline signal')
signal.signal(signal.SIGTERM, deadline_signal)
def ids(path):
    values = [int(x) for x in re.split(r'[\s,]+', pathlib.Path(path).read_text().strip()) if x]
    if not values or any(x < 0 or x > 0xffffffff for x in values):
        raise SystemExit('invalid token-ID file: '+str(path))
    return values
p, suffix = ids(pfile), ids(sfile)
for path in (binary, model):
    if not pathlib.Path(path).is_file(): raise SystemExit('missing prerequisite: '+path)
if phase != '4' and not os.access(binary, os.X_OK): raise SystemExit('BIN must be executable')
if len(p) != 32768: raise SystemExit('CLI P must contain exactly 32768 token IDs')
if any((f.startswith('--session-') and f not in ('--session-tiled-suffix','--session-mtp-fixed')) or
       f.startswith(('--http-', '--generate', '--tokens-file')) for f in flags):
    raise SystemExit('gate owns generation, input, HTTP and session flags')
if '--session-tiled-suffix' not in flags: flags += ['--session-tiled-suffix']
# Normalize this option so admission and the runtime always see the same tile width.
normalized=[]
for f in flags:
    normalized.extend(f.split('=',1) if f.startswith('--known-prefix-prefill-rows=') else [f])
flags=normalized
if flags.count('--known-prefix-prefill-rows') > 1: raise SystemExit('duplicate prefill row option')
if '--known-prefix-prefill-rows' not in flags: flags += ['--known-prefix-prefill-rows','1024']
rows = int(flags[flags.index('--known-prefix-prefill-rows')+1])
first = int(os.environ.get('FIRST_GENERATE', '2' if unaligned else '1')); ram = None
if unaligned and (first != 2 or rows != 1024 or len(suffix) != 2046 or '--session-mtp-fixed' in flags):
    raise SystemExit('unaligned target case requires FIRST_GENERATE=2, tile=1024, S=2046, no MTP')
if first < 1 or rows < 1024: raise SystemExit('FIRST_GENERATE must be positive and prefill rows >=1024')
align = (rows-(len(p)+first-1)%rows)%rows
if 1+len(suffix) < align+rows: raise SystemExit('S requires >=%d IDs for tiled continuation'%(align+rows-1))
# Admission uses cold load even if weights are warm. Rates are planning inputs, not qualification.
# Bound envelope size by the runtime's four-copy capture budget (HTTP uses its RAM budget).
if rows & (rows-1): raise SystemExit('phase prefill rows must be a power of two')
if first > 65 or rows > 8192 or len(suffix) > 4095 or align > 1024:
    raise SystemExit('phase workload cap: FIRST_GENERATE<=65, rows<=8192, S<=4095, alignment<=1024')
tail = (1+len(suffix)-align) % rows
serial = align+tail
if serial > 1024: raise SystemExit('phase workload cap: alignment plus serial tail <=1024 rows')
checkpoint_bytes = (ram if ram is not None else (8 << 30)) // 4
existing = out/'checkpoint/session.bin'
if existing.is_file() and existing.stat().st_size > checkpoint_bytes:
    raise SystemExit('checkpoint exceeds estimated four-copy envelope bound')
rates = {'cold_load_s':260, 'warm_load_s':30, 'prefill_s_per_token':0.0007,
         'serial_s_per_token':0.05, 'generation_s_per_token':0.05,
         'checkpoint_write_bytes_per_s':1e9, 'restore_bytes_per_s':1e9}
# Reserve 20 s for hashing/CPU negative checks, 10 s setup/report and 10 s cleanup.
reserve = 40
continuation = serial*0.05 + (1+len(suffix)-serial)*0.0007
initial = len(p)*0.0007 + first*0.05
# Counts include capture/restore copies, checkpoint publication/backup, negative copies/reads.
if ram is None:
    work = {'1':(initial+continuation+64*0.05,1,2),
            '2':(initial+continuation+64*0.05,0,0),
            '3':(continuation+64*0.05,2,10), '4':(0,0,2)}
else:
    work = {'1':(initial+continuation+64*0.05,0,0),
            '2':(initial+continuation+64*0.05,2,5),
            '3':(len(p)*0.0007+continuation+66*0.05,2,10), '4':(0,0,2)}
work.update({n:(len(p)*0.0007+64*0.05,0,0) for n in ('5','6','7')})
compute, writes, restores = work[phase]
io = checkpoint_bytes*(writes/rates['checkpoint_write_bytes_per_s']+restores/rates['restore_bytes_per_s'])
cold = (260 if phase != '4' else 0)+compute+io+reserve
warm = cold-(230 if phase != '4' else 0)
print('Phase '+phase+' estimate: '+json.dumps(dict(rates, first_generate=first, suffix_tokens=len(suffix),
      prefill_rows=rows, alignment_rows=align, serial_rows=serial, checkpoint_bound_bytes=checkpoint_bytes,
      write_copies=writes, restore_read_copies=restores, compute_s=compute, io_s=io,
      hashing_negatives_setup_cleanup_s=reserve, cold_s=cold, warm_s=warm),sort_keys=True),flush=True)
if cold > 400: raise SystemExit('phase workload refused: cold estimate exceeds 400 s')
# Record all numerical inputs; reject mixed binaries, flags, inputs, model files or STRIX settings.
env = dict(os.environ, STRIX_PREFIX_CACHE_LOOKUP='0', STRIX_PREFIX_CACHE_DIGEST='1', STRIX_HASH_LOG='1', STRIX_PROMPT_STATE_DIGEST='1', STRIX_SERIAL_LOGITS_DIGEST='1')
def file_hash(path):
    h=hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            left(); block=f.read(1<<20)
            if not block: break
            h.update(block)
    return h.hexdigest()
baseline = pathlib.Path(os.environ.get('BASE_BIN','')).resolve()
if not baseline.is_file() or not os.access(baseline,os.X_OK):
    raise SystemExit('BASE_BIN must name the preserved pre-change executable')
st = pathlib.Path(model).stat()
config = {'baseline_binary':str(baseline), 'baseline_sha256':file_hash(baseline), 'binary': binary, 'binary_sha256': file_hash(binary), 'model': model,
          'model_stat': [st.st_size, st.st_mtime_ns], 'p': p, 'suffix': suffix,
          'flags': flags, 'first': first, 'ram': ram, 'unaligned': unaligned,
          'numerical_env': {k:v for k,v in env.items()
              if k.startswith(('STRIX_', 'HIP_', 'HSA_', 'AMD_', 'ROCR_', 'GPU_'))
              and not k.startswith('GPU_TURN_') and k not in ('GPU_WINDOW', 'STRIX_HASH_CACHE')}}
if phase == '1':
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): raise SystemExit('phase 1 requires empty OUT')
    (out/'config.json').write_text(json.dumps(config, sort_keys=True)+'\n')
else:
    if not (out/'config.json').is_file(): raise SystemExit('missing prerequisite: phase 1 config.json')
    if json.loads((out/'config.json').read_text()) != config: raise SystemExit('phase configuration differs from phase 1')
for n in range(1, int(phase)):
    receipt_path=out/('phase-%d.json'%n)
    if not receipt_path.is_file(): raise SystemExit('missing prerequisite: '+str(receipt_path))
    receipt=json.loads(receipt_path.read_text())
    if receipt.get('phase_status') != 'COMPLETE': raise SystemExit('prerequisite phase %d did not complete'%n)
    for name in receipt['artifacts']:
        if not (out/name).is_file() or not (out/name).stat().st_size:
            raise SystemExit('missing prerequisite artifact: '+name)
    turn=os.environ.get('GPU_TURN_ID')
    if phase != '4' and turn and turn == receipt.get('gpu_turn_id'):
        raise SystemExit('each GPU phase requires a separate GPU turn')
receipt_path=out/('phase-'+phase+'.json')
if receipt_path.exists(): raise SystemExit('phase already attempted; use a new run directory')
summary={'qualification':'not_assessed', 'phase':phase, 'phase_status':'FAIL',
         'gpu_turn_id':os.environ.get('GPU_TURN_ID'), 'runs':{}, 'artifacts':[],
         'estimate_cold_s':cold, 'estimate_warm_s':warm}
def terminate(proc):
    if proc is None: return
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try: proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=2)
    if proc.poll() is None: raise RuntimeError('child cleanup not verified')
def execute(cmd, stdout, stderr):
    left()
    proc=subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    try:
        proc.wait(timeout=left())
        return proc.returncode
    finally: terminate(proc)
checkpoint = out/'checkpoint'
def run(name, inputs, extra=(), expect=None, exe=binary, generation=64, common=None):
    cmd = [str(exe), model, *map(str, inputs), *(flags if common is None else common),
           '--generate', str(generation), *map(str, extra)]
    t = time.perf_counter()
    left()
    with (out / (name + '.json')).open('wb') as stdout, (out / (name + '.stderr')).open('wb') as stderr:
        returncode = execute(cmd, stdout, stderr)
    elapsed = (time.perf_counter()-t)*1000
    log = (out / (name + '.stderr')).read_text(errors='replace')
    summary['runs'][name] = {'wall_ms_including_model_load': elapsed, 'exit_code': returncode}
    for field in ('initial_prefill_ms', 'save_stall_ms', 'write_fsync_ms', 'restore_ms', 'suffix_ms', 'generation_ms'):
        found = re.findall(r'\b' + field + r'=([0-9.eE+-]+)', log)
        if found: summary['runs'][name][field] = float(found[-1])
    print(name, 'wall_ms=%.3f' % elapsed, flush=True)
    for line in log.splitlines():
        if line.startswith('session '): print(' ', line, flush=True)
    if expect is not None:
        if returncode == 0 or expect not in log:
            raise RuntimeError(name + ': unexpected negative-case result; see stderr')
        print('PASS ' + name + ' refused before model upload', flush=True)
    elif returncode:
        raise RuntimeError(name + ' failed; see ' + str(out / (name + '.stderr')))
    return log

try:
    initial = ['--session-first-generate', first, '--session-then-prompt', pathlib.Path(sfile).resolve()]
    if phase == '1':
        a = run('A', p, ['--session-save', checkpoint, *initial, '--session-logits', out/'A.f32'])
        boundary = re.findall(r'session save_stall_ms=.* consumed=(\d+) pending=(\S+)', a)
        if len(boundary) != 1 or int(boundary[0][0]) != len(p)+first-1 or boundary[0][1] == 'none':
            raise RuntimeError('save boundary differs from requested first-turn frontier (possibly early EOS)')
        summary['artifacts']=['A.json','A.stderr','A.f32','checkpoint/session.bin']
    elif phase == '2':
        run('L', p, ['--session-live-control', *initial, '--session-logits', out/'L.f32'])
        summary['artifacts']=['L.json','L.stderr','L.f32']
    elif phase == '3':
        run('B', suffix, ['--session-load', checkpoint, '--session-logits', out/'B.f32'])
        positive=run('positive-control', suffix, ['--session-load', checkpoint, '--session-validate-only'])
        if 'session preflight PASS' not in positive: raise RuntimeError('positive preflight missing CPU-only receipt')
        # ALT_BIN is always made from this run's executable with an appended byte.
        alt_bin = pathlib.Path(os.environ.get('ALT_BIN', str(out/'alternate-bin'))).resolve()
        if alt_bin.exists(): raise RuntimeError('ALT_BIN must be a new path')
        if out not in alt_bin.parents: raise RuntimeError('ALT_BIN must be inside OUT')
        shutil.copy2(binary, alt_bin)
        with alt_bin.open('ab') as f: f.write(b'\0')
        run('negative-executable', suffix, ['--session-load', checkpoint, '--session-validate-only'],
            expect='session compatibility mismatch', exe=alt_bin)
        # Change a real numerical/layout CLI setting, avoiding duplicate options.
        negative_flags = list(flags)
        # A numerics-affecting option that stays admissible with the qualified flags (the prefill tile plan);
        # an inadmissible one would be refused by option validation before the session check.
        key = '--known-prefix-prefill-rows'
        old = '8192'
        if key in negative_flags:
            i = negative_flags.index(key); old = negative_flags[i+1]; del negative_flags[i:i+2]
        negative_flags += [key, '4096' if old != '4096' else '8192']
        run('negative-numerics', suffix, ['--session-load', checkpoint, '--session-validate-only'],
            expect='session compatibility mismatch', common=negative_flags)
        source = checkpoint/'session.bin'
        with source.open('rb') as f:
            def u64():
                b = f.read(8)
                if len(b)!=8: raise RuntimeError('checkpoint parser truncation')
                return struct.unpack('<Q',b)[0]
            def skip_string(): f.seek(u64(),1)
            skip_string(); u64(); f.seek(64,1); skip_string(); u64(); f.seek(32,1)
            for _ in range(5): u64()
            skip_string(); f.seek(u64()*8,1)
            if not u64(): raise RuntimeError('checkpoint has no sections')
            skip_string(); u64(); u64(); count=u64(); f.seek(32,1); payload=u64()
            if not count or payload!=f.tell(): raise RuntimeError('invalid section geometry')
        corrupt = out/'corrupt'; corrupt.mkdir(); shutil.copyfile(source, corrupt/'session.bin')
        with (corrupt/'session.bin').open('r+b') as f:
            f.seek(payload); v=f.read(1); f.seek(payload); f.write(bytes([v[0]^1]))
        run('negative-corrupt', suffix, ['--session-load', corrupt, '--session-validate-only'],
            expect='session manifest root mismatch')
        truncated = out/'truncated'; truncated.mkdir(); shutil.copyfile(source, truncated/'session.bin')
        with (truncated/'session.bin').open('r+b') as f: f.truncate(source.stat().st_size-1)
        run('negative-truncated', suffix, ['--session-load', truncated, '--session-validate-only'],
            expect='session manifest root mismatch')

        summary['artifacts']=['B.json','B.stderr','B.f32','positive-control.stderr']+[
            'negative-'+name+'.stderr' for name in ('executable','numerics','corrupt','truncated')]
    elif phase == '4':
        logs = [(out/(name+'.stderr')).read_text() for name in ('A','L','B')]
        generated = [json.loads((out/(name+'.json')).read_text())['generated_tokens'] for name in ('A','L','B')]
        if any(len(t) != 64 for t in generated): raise RuntimeError('early EOS: requires exactly 64 output IDs')
        rows = [re.findall(r'session logits_rows=(\d+) vocabulary=(\d+)', log) for log in logs]
        if any(len(r)!=1 for r in rows) or not rows[0]==rows[1]==rows[2] or int(rows[0][0][0])!=64:
            raise RuntimeError('missing/incomplete 64-row logits diagnostic')
        size = (out/'A.f32').stat().st_size
        if size != 64*int(rows[0][0][1])*4 or not size: raise RuntimeError('logits geometry mismatch')
        # Required literal byte comparisons, no tolerance or floating point decoding.
        comparisons = [execute(['cmp', str(out/'A.f32'), str(out/(name+'.f32'))], None, None) == 0
                       for name in ('L','B')]
        byte_equal = all(comparisons)
        digests = [re.findall(r'session final_digest=([0-9a-f]{64})', log) for log in logs]
        digests = [d[-1:] for d in digests]
        if not all('session suffix_route=aligned-tiled' in log for log in logs): raise RuntimeError('tiled suffix route was not exercised')
        if unaligned and not all('session suffix_route=aligned-tiled alignment_steps=1023' in log for log in logs):
            raise RuntimeError('unaligned restore/live suffix did not exercise 1023 serial alignment rows')
        checks = {'64_token_ids_A_L_B': generated[0]==generated[1]==generated[2],
                  'every_fp32_logit_bit_cmp': byte_equal,
                  'final_state_digest_A_L_B': len(digests[0])==1 and digests[0]==digests[1]==digests[2]}
        for name, ok in checks.items(): print(('PASS ' if ok else 'FAIL ') + name, flush=True)
        if '--session-mtp-fixed' in flags:
            receipts = [json.loads((out/(name+'.json')).read_text()) for name in ('A','L','B')]
            checks['mtp_acceptance_counts_A_L_B'] = len({(r['accepted'], r['proposed']) for r in receipts}) == 1
            print(('PASS ' if checks['mtp_acceptance_counts_A_L_B'] else 'FAIL ') + 'mtp_acceptance_counts_A_L_B', flush=True)
            if not all(r['proposed'] > 0 for r in receipts): raise RuntimeError('MTP arm proposed no candidates')
        if not all(checks.values()): raise RuntimeError('A/L/B bitwise comparison failed')

        summary.update(qualification='awaiting_identity', comparisons=checks, logits_bytes=size,
                       checkpoint_bytes=(checkpoint/'session.bin').stat().st_size)
        print('PASS session A == never-saved L == cross-phase restored B', flush=True)
    else:
        name={'5':'identity-baseline','6':'identity-off','7':'identity-on-unnamed'}[phase]
        ordinary=[]
        i=0
        while i<len(flags):
            flag=flags[i]
            if flag.startswith('--session-'):
                if flag in ('--session-logits','--session-dir','--session-ram-bytes','--session-max-parked'): i+=1
            else: ordinary.append(flag)
            i+=1
        extra=[] if phase=='5' else ['--session-cache','off' if phase=='6' else 'on','--session-disk','off']
        log=run(name,p,extra,exe=baseline if phase=='5' else binary,common=ordinary)
        result=json.loads((out/(name+'.json')).read_text())
        tokens=result['generated_tokens']
        if len(tokens)!=64: raise RuntimeError('identity arm early EOS')
        identity={'tokens':tokens,'digests':numerical_digests(result,log)}
        (out/(name+'.identity.json')).write_text(json.dumps(identity,sort_keys=True)+'\n')
        summary['artifacts']=[name+'.json',name+'.identity.json',name+'.stderr']
        if phase=='7':
            compare_identity(out)
            summary['qualification']='PASS'
    left()
    for artifact in summary['artifacts']:
        if not (out/artifact).is_file() or not (out/artifact).stat().st_size:
            raise RuntimeError('missing phase output: '+artifact)
    summary['phase_status']='COMPLETE'
except Exception as error:
    summary.update(qualification='FAIL', error=str(error))
    print('FAIL', error, file=sys.stderr)
finally:
    summary['total_wall_s']=time.monotonic()-started
    receipt_path.write_text(json.dumps(summary,indent=2)+'\n')
    if phase in ('4','7'): (out/'gate-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if summary['phase_status'] != 'COMPLETE': sys.exit(1)
PY
