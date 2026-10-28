#!/usr/bin/env bash
# Linux only; opt-in SINGLE phases. Invoke each phase in a separate turn; never loop GPU phases.
# Usage: bash GATE PHASE BIN MODEL P_IDS S_IDS OUT [-- COMMON_FLAGS...]
# 1: 8K never-captured L turns 1/2; 2: A turn 1, disk checkpoint, turn 2.
# 3: restarted disk restore, A->B->A parking, turn 2, validate-only negatives.
# 4: CPU comparison of IDs, every raw FP32 byte, digests and MTP counts.
# 5/6/7: preserved baseline / candidate OFF / candidate ON unnamed identity (one GPU turn each).
# BASE_BIN is required in every phase. PASS is withheld until phase 7.
set -euo pipefail
export SESSION_GATE_HELPERS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# < 6 )); then
  echo "usage: $0 {1|2|3|4|5|6|7} BIN MODEL P_IDS S_IDS OUT [-- COMMON_FLAGS...]" >&2
  exit 2
fi
# Absolute budget includes Python setup, preflights, model load and verified cleanup.
export SESSION_GATE_DEADLINE=$(( $(date +%s) + 440 ))
exec timeout --signal=TERM --kill-after=5s 440s python3 - "$@" <<'PY'
import hashlib, json, os, pathlib, re, shutil, signal, socket, subprocess, sys, time
sys.path.insert(0, os.environ['SESSION_GATE_HELPERS'])
from gate_identity import compare_identity, restore_turn1, http_identity, numerical_digests
started = time.monotonic()
deadline = started + max(0, int(os.environ['SESSION_GATE_DEADLINE']) - time.time() - 10)
phase, binary, model, pfile, sfile, output, *flags = sys.argv[1:]
if phase not in ('1', '2', '3', '4', '5', '6', '7'): raise SystemExit('PHASE must be 1..7')
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
# Smaller HTTP prompt leaves room for first-turn MTP, parking and preflights in one SINGLE turn.
if len(p) != 8192: raise SystemExit('HTTP P must contain exactly 8192 token IDs (CLI gate remains 32768)')
owned = ('--http-', '--session-', '--generate', '--tokens-file')
if any(any(f.startswith(key) for key in owned) for f in flags): raise SystemExit('gate owns HTTP/session/generation/input flags')
# Normalize this option so admission and the runtime always see the same tile width.
normalized=[]
for f in flags:
    normalized.extend(f.split('=',1) if f.startswith('--known-prefix-prefill-rows=') else [f])
flags=normalized
if flags.count('--known-prefix-prefill-rows') > 1: raise SystemExit('duplicate prefill row option')
if '--known-prefix-prefill-rows' not in flags: flags += ['--known-prefix-prefill-rows','1024']
rows = int(flags[flags.index('--known-prefix-prefill-rows')+1])
first = int(os.environ.get('FIRST_GENERATE','65'))
if first < 1 or rows < 1024: raise SystemExit('FIRST_GENERATE must be positive and prefill rows >=1024')
align = (rows-(len(p)+first-1)%rows)%rows
if align == 0: raise SystemExit('FIRST_GENERATE must give an off-grid frontier (default 65)')
if 1+len(suffix) < align+rows: raise SystemExit('S requires >=%d IDs for tiled continuation'%(align+rows-1))
ram = int(os.environ.get('SESSION_RAM_BYTES', str(12 << 30)))
if ram <= 0: raise SystemExit('SESSION_RAM_BYTES must be positive')
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
          'flags': flags, 'first': first, 'ram': ram,
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
server=None; logs=[]
with socket.socket() as s: s.bind(('127.0.0.1',0)); port=s.getsockname()[1]
base='http://127.0.0.1:%d'%port
common=[*flags,'--session-mtp-fixed','--session-tiled-suffix','--session-live-control','--session-dir',str(out/'sessions'),
        '--session-ram-bytes',str(ram),'--session-max-parked','2','--http-serve','--http-batch','1',
        '--http-bind','127.0.0.1','--http-port',str(port),'--http-sampling','greedy','--generate','64','--session-logits',str(out/'request.f32')]
def curl(name, method, path, body=None, health=False):
    cmd=['curl','--silent','--show-error','--fail','--max-time',str(min(left(),2 if health else left())),
         '--request',method,base+path]
    if body is not None:
        payload=out/(name+'.request.json'); payload.write_text(json.dumps(body))
        cmd+=['--header','Content-Type: application/json','--data-binary','@'+str(payload)]
    proc=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    try:
        stdout,stderr=proc.communicate(timeout=left())
        result=subprocess.CompletedProcess(cmd,proc.returncode,stdout,stderr)
    finally: terminate(proc)
    if result.returncode: raise RuntimeError(name+': '+result.stderr.decode(errors='replace'))
    response=json.loads(result.stdout)
    if not health: (out/(name+'.json')).write_text(json.dumps(response,indent=2)+'\n')
    return response
def start(name, exe=binary, options=None):
    global server
    stdout=(out/(name+'.stdout')).open('wb'); stderr=(out/(name+'.stderr')).open('wb');logs.extend([stdout,stderr])
    server=subprocess.Popen([str(exe),model,*map(str,p),*(common if options is None else options)],env=env,stdout=stdout,stderr=stderr,start_new_session=True)
    while True:
        left()
        if server.poll() is not None: raise RuntimeError(name+' exited during startup; see stderr')
        try: curl('health','GET','/health',health=True); break
        except RuntimeError: time.sleep(min(0.5,left()))
    print(name+' ready after %.2f s'%(time.monotonic()-started),flush=True)
def stop():
    global server
    if server is None:return
    terminate(server)
    server=None
    for log in logs: log.flush()
def turn(name, session, prompt, count):
    t=time.monotonic()
    body={'prompt':prompt,'temperature':0,'max_tokens':count,'return_token_ids':True}
    if session is not None: body['session']=session
    response=curl(name,'POST','/v1/completions',body)
    receipt=response['strix']['session']
    if len(receipt['tokens'])!=count or len(receipt['fp32_logits_sha256'])!=count: raise RuntimeError(name+': early EOS/incomplete logits')
    if response['strix']['token_ids']!=receipt['tokens']: raise RuntimeError(name+': transport token IDs differ')
    raw=out/'request.f32'
    data=raw.read_bytes()
    if not data or len(data) % (count*4): raise RuntimeError(name+': raw FP32 geometry mismatch')
    row_bytes=len(data)//count
    hashes=[hashlib.sha256(data[i*row_bytes:(i+1)*row_bytes]).hexdigest() for i in range(count)]
    if hashes != receipt['fp32_logits_sha256']: raise RuntimeError(name+': raw FP32 rows differ from response hashes')
    shutil.copyfile(raw,out/(name+'.f32'))
    summary['runs'][name]={'wall_s':time.monotonic()-t,'accepted':receipt['accepted'],'proposed':receipt['proposed']}
    return response,receipt
def validate(name, source, expect=None, exe=binary, options=None):
    options=common if options is None else options
    cmd=[str(exe),model,*map(str,suffix),*options,'--session-load',str(source),'--session-validate-only']
    with (out/(name+'.out')).open('wb') as stdout, (out/(name+'.stderr')).open('wb') as stderr:
        rc=execute(cmd,stdout,stderr)
    log=(out/(name+'.stderr')).read_text(errors='replace')
    if expect:
        if rc == 0 or expect not in log: raise RuntimeError(name+': unexpected preflight result')
    elif rc or 'session preflight PASS' not in log: raise RuntimeError(name+': positive CPU preflight failed')
    print('PASS '+name+' CPU-only preflight',flush=True)
try:
    if phase == '1':
        start('live-server')
        _,control_first=turn('control-turn1',None,p,first)
        full=p+control_first['tokens']+suffix
        control,_=turn('control-turn2',None,full,64)
        if control['usage']['prompt_tokens_details']['cached_tokens'] != len(p)+first-1:
            raise RuntimeError('never-captured control did not continue live frontier')
        control_log=(out/'live-server.stderr').read_text(errors='replace')
        if ('session boundary_source=never-captured-live consumed=%d'%(len(p)+first-1) not in control_log or
            'session checkpoint_capture' in control_log or 'session restore_ms=' in control_log):
            raise RuntimeError('control did not use a never-captured boundary')
        if curl('control-list','GET','/v1/sessions')['sessions']: raise RuntimeError('control parked a session')
        summary['artifacts']=['control-turn1.json','control-turn2.json','control-turn1.f32','control-turn2.f32','live-server.stderr']
    elif phase == '2':
        start('save-server')
        _,first_receipt=turn('turn1','A',p,first)
        curl('checkpoint','POST','/v1/sessions/A/checkpoint',{'durability':'disk'})
        # Preserve the durable frontier for preflights even after API delete in phase 3.
        checkpoint=out/'checkpoint'; checkpoint.mkdir()
        shutil.copyfile(out/'sessions/A/session.bin',checkpoint/'session.bin')
        # Save arm A continues after capture, so CPU comparison includes A/L/B turn 2.
        full=p+first_receipt['tokens']+suffix
        live,_=turn('live-turn2','A',full,64)
        if live['usage']['prompt_tokens_details']['cached_tokens'] != len(p)+first-1:
            raise RuntimeError('save arm did not continue saved frontier')
        summary['artifacts']=['turn1.json','turn1.f32','live-turn2.f32','checkpoint.json','checkpoint/session.bin','live-turn2.json','save-server.stderr']
    elif phase == '3':
        if not (out/'sessions/A/session.bin').is_file(): raise RuntimeError('missing prerequisite: durable HTTP A')
        restore_turn1(out)
        start('restarted-server')
        # No model or lane survives the previous turn. Explicit resume loads A from disk.
        listed=curl('list-restarted','GET','/v1/sessions')['sessions']
        if not any(r['id']=='A' and r['state']=='disk' and r['durability']=='disk' for r in listed):
            raise RuntimeError('restart did not begin with disk-only A')
        curl('restore-A','POST','/v1/sessions/A/resume',{})
        curl('park-A','POST','/v1/sessions/A/park',{})
        turn('parking-B','B',p,1)
        listed=curl('list-before','GET','/v1/sessions')
        if not {'A','B'} <= {r['id'] for r in listed['sessions'] if r['state']=='parked'}:
            raise RuntimeError('A/B were not both parked')
        curl('resume-A','POST','/v1/sessions/A/resume',{})
        first_receipt=json.loads((out/'turn1.json').read_text())['strix']['session']
        full=p+first_receipt['tokens']+suffix
        restarted,_=turn('restarted-turn2','A',full,64)
        if restarted['usage']['prompt_tokens_details']['cached_tokens'] != len(p)+first-1:
            raise RuntimeError('restart/parking A->B->A failed to restore')
        stats=curl('stats-live','GET','/v1/cache/stats')
        if stats['ram_bytes']>stats['ram_budget'] or stats['parked']>2: raise RuntimeError('parking budget exceeded')
        miss,_=turn('prefix-mismatch','A',p[:2],1)
        if miss['usage']['prompt_tokens_details']['cached_tokens'] != 0: raise RuntimeError('prefix mismatch was not a miss')
        curl('delete-A','DELETE','/v1/sessions/A')
        if any(r['id']=='A' for r in curl('list-after-delete','GET','/v1/sessions')['sessions']): raise RuntimeError('delete failed')
        stop()  # Validate-only subprocesses never share a live server/GPU allocation.
        checkpoint=out/'checkpoint'
        validate('positive-control',checkpoint)
        alt=out/'alternate-bin'; shutil.copy2(binary,alt)
        with alt.open('ab') as f: f.write(b'\0')
        validate('negative-executable',checkpoint,'session compatibility mismatch',exe=alt)
        negative_flags=list(common); key='--known-prefix-prefill-rows'; old='1024'
        if key in negative_flags:
            i=negative_flags.index(key); old=negative_flags[i+1]; del negative_flags[i:i+2]
        negative_flags += [key,'4096' if old != '4096' else '8192']
        validate('negative-numerics',checkpoint,'session compatibility mismatch',options=negative_flags)
        source=checkpoint/'session.bin'
        for name in ('corrupt','truncated'):
            target=out/name; target.mkdir(); shutil.copyfile(source,target/'session.bin')
            with (target/'session.bin').open('r+b') as f:
                if name == 'truncated': f.truncate(source.stat().st_size-1)
                else:
                    pos=source.stat().st_size//2; f.seek(pos); b=f.read(1); f.seek(pos); f.write(bytes([b[0]^1]))
            validate('negative-'+name,target,'session manifest root mismatch')
        summary['artifacts']=['restarted-turn2.json','restarted-turn2.f32','restarted-server.stderr','stats-live.json',
                              'list-before.json','list-after-delete.json','prefix-mismatch.json',
                              'positive-control.stderr']+['negative-'+n+'.stderr' for n in ('executable','numerics','corrupt','truncated')]
    elif phase == '4':
        control_first,first_receipt=[json.loads((out/(n+'.json')).read_text())['strix']['session']
                                     for n in ('control-turn1','turn1')]
        if control_first != first_receipt: raise RuntimeError('saved/control initial turn differs')
        responses=[json.loads((out/(n+'.json')).read_text()) for n in ('control-turn2','live-turn2','restarted-turn2')]
        receipts=[r['strix']['session'] for r in responses]
        if not receipts[0]==receipts[1]==receipts[2]: raise RuntimeError('L/A/B tokens/raw FP32 row hashes/digest/acceptance differ')
        # Compare complete native FP32 files on CPU; per-row hashes are additional evidence.
        for a,b in (('control-turn1','turn1'), ('control-turn2','live-turn2'),
                    ('control-turn2','restarted-turn2')):
            if execute(['cmp',str(out/(a+'.f32')),str(out/(b+'.f32'))],None,None):
                raise RuntimeError(a+'/'+b+': full raw FP32 cmp failed')
        summary['every_fp32_logit_bit_cmp']=True
        if not all(r['proposed']>0 for r in receipts): raise RuntimeError('MTP proposed no candidates')
        if not responses[0]['choices'][0].get('logprobs') == responses[1]['choices'][0].get('logprobs') == responses[2]['choices'][0].get('logprobs'):
            raise RuntimeError('HTTP logprobs differ')
        for name in ('live-server','save-server','restarted-server'):
            log=(out/(name+'.stderr')).read_text(errors='replace')
            if 'session suffix_route=aligned-tiled alignment_steps=%d'%align not in log:
                raise RuntimeError(name+': missing off-grid tiled continuation receipt')
        summary['qualification']='awaiting_identity'
        print('PASS HTTP never-captured L == saved A == cross-phase restored B; parking, negatives, limits, prefix miss and delete',flush=True)
    else:
        name={'5':'identity-baseline','6':'identity-off','7':'identity-on-unnamed'}[phase]
        # Ordinary batch-1 scheduler and seeded sampling, with all numerical flags held fixed.
        options=[*flags,'--http-serve','--http-batch','1','--http-bind','127.0.0.1',
                 '--http-port',str(port),'--http-sampling','sample','--generate','64']
        if phase == '6': options += ['--session-cache','off','--session-disk','off']
        if phase == '7': options += ['--session-cache','on','--session-disk','off',
            '--session-mtp-fixed','--session-tiled-suffix','--session-ram-bytes',str(ram)]
        start(name, baseline if phase=='5' else binary, options)
        response=curl(name,'POST','/v1/completions',{'prompt':p,'temperature':0.7,'seed':12345,
            'max_tokens':64,'return_token_ids':True,'logprobs':5,'ngram_draft':True,'draft_depth_max':0})
        if len(response['strix']['token_ids'])!=64: raise RuntimeError('identity arm early EOS')
        # Strip only transport IDs and timings; retain generated IDs, logprobs and sampling identity.
        identity=http_identity(response)
        identity['digests']=numerical_digests({},(out/(name+'.stderr')).read_text())
        (out/(name+'.identity.json')).write_text(json.dumps(identity,sort_keys=True)+'\n')
        summary['artifacts']=[name+'.json',name+'.identity.json',name+'.stderr']
        if phase=='7':
            compare_identity(out)
            summary['qualification']='PASS'
    stop()
    left()
    for artifact in summary['artifacts']:
        if not (out/artifact).is_file() or not (out/artifact).stat().st_size:
            raise RuntimeError('missing phase output: '+artifact)
    summary['phase_status']='COMPLETE'
except Exception as error:
    summary.update(qualification='FAIL',error=str(error)); print('FAIL',error,file=sys.stderr)
finally:
    try: stop()
    except Exception as error:
        summary.update(qualification='FAIL',phase_status='FAIL',error='cleanup: '+str(error))
    for log in logs: log.close()
    summary['total_wall_s']=time.monotonic()-started
    receipt_path.write_text(json.dumps(summary,indent=2)+'\n')
    if phase in ('4','7'): (out/'gate-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if summary['phase_status'] != 'COMPLETE': sys.exit(1)
PY
