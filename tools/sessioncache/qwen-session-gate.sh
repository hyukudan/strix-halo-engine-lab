#!/usr/bin/env bash
# Linux only; opt-in qualification. No SSH, deployment or GPU discovery.
# Usage: FIRST_GENERATE=1 bash qwen-session-gate.sh BIN MODEL P_IDS S_IDS OUT -- COMMON_FLAGS...
set -euo pipefail
if (( $# < 5 )); then
  echo "usage: $0 BIN MODEL P_IDS S_IDS OUT [-- COMMON_FLAGS...]" >&2
  exit 2
fi
python3 - "$@" <<'PY'
import json, os, pathlib, re, shutil, struct, subprocess, sys, time
binary, model, pfile, sfile, out, *flags = sys.argv[1:]
if flags[:1] == ['--']: flags = flags[1:]
binary = str(pathlib.Path(binary).resolve())
model = str(pathlib.Path(model).resolve())
out = pathlib.Path(out).resolve()
out.mkdir(parents=True, exist_ok=True)
if any(out.iterdir()): raise SystemExit('OUT must be empty')
def tokens(path):
    values = [int(v) for v in re.split(r'[\s,]+', pathlib.Path(path).read_text().strip()) if v]
    if not values or any(v < 0 or v > 0xffffffff for v in values):
        raise SystemExit('invalid token-ID file: ' + str(path))
    return values
p, suffix = tokens(pfile), tokens(sfile)
if len(p) != 32768: raise SystemExit('P must contain exactly 32768 token IDs')
if any(f.split('=')[0].startswith('--session-') or f.split('=')[0] == '--generate' for f in flags):
    raise SystemExit('gate owns generation and session flags')
first = int(os.environ.get('FIRST_GENERATE', '1'))
if first < 1: raise SystemExit('FIRST_GENERATE must be positive')
env = dict(os.environ, STRIX_PREFIX_CACHE_LOOKUP='0', STRIX_PREFIX_CACHE_DIGEST='1', STRIX_HASH_LOG='1')
checkpoint = out / 'checkpoint'
# Four expensive launches, four CPU preflights. A/L: P + first turn + S + 64; B: restore + S + 64; C: P + 1.
print('Expected total: 8 engine launches = 4 model uploads (~240-280 s) + 4 CPU-only negatives; '
      '3 P prefills + 3 suffix/64-token continuations + checkpoint I/O/hash misses. Budget <=1200 s.', flush=True)
print('Expected wall: 240-280 s + measured compute/I/O; plan 15-20 min, hard deadline 20 min. '
      'No throughput estimate establishes qualification.', flush=True)
started = time.perf_counter()
summary = {'qualification': 'not_assessed', 'first_generate': first, 'prompt_tokens': len(p),
           'suffix_tokens': len(suffix), 'expected_processes': 8, 'model_uploads': 4, 'runs': {}}
def run(name, inputs, extra=(), expect=None, exe=binary, generation=64, common=None):
    cmd = [str(exe), model, *map(str, inputs), *(flags if common is None else common),
           '--generate', str(generation), *map(str, extra)]
    t = time.perf_counter()
    remaining = 1200 - (t-started)
    if remaining <= 0: raise RuntimeError('20-minute gate budget exhausted')
    with (out / (name + '.json')).open('wb') as stdout, (out / (name + '.stderr')).open('wb') as stderr:
        result = subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr, timeout=remaining)
    elapsed = (time.perf_counter()-t)*1000
    log = (out / (name + '.stderr')).read_text(errors='replace')
    summary['runs'][name] = {'wall_ms_including_model_load': elapsed, 'exit_code': result.returncode}
    for field in ('initial_prefill_ms', 'save_stall_ms', 'write_fsync_ms', 'restore_ms', 'suffix_ms', 'generation_ms'):
        found = re.findall(r'\b' + field + r'=([0-9.eE+-]+)', log)
        if found: summary['runs'][name][field] = float(found[-1])
    print(name, 'wall_ms=%.3f' % elapsed, flush=True)
    for line in log.splitlines():
        if line.startswith('session '): print(' ', line, flush=True)
    if expect is not None:
        if result.returncode == 0 or expect not in log:
            raise RuntimeError(name + ': unexpected negative-case result; see stderr')
        print('PASS ' + name + ' refused before model upload', flush=True)
    elif result.returncode:
        raise RuntimeError(name + ' failed; see ' + str(out / (name + '.stderr')))
    return log
try:
    initial = ['--session-first-generate', first, '--session-then-prompt', pathlib.Path(sfile).resolve()]
    a = run('A', p, ['--session-save', checkpoint, *initial, '--session-logits', out/'A.f32'])
    boundary = re.findall(r'session save_stall_ms=.* consumed=(\d+) pending=(\S+)', a)
    if len(boundary) != 1 or int(boundary[0][0]) != len(p)+first-1 or boundary[0][1] == 'none':
        raise RuntimeError('save boundary differs from requested first-turn frontier (possibly early EOS)')
    l = run('L', p, ['--session-live-control', *initial, '--session-logits', out/'L.f32'])
    b = run('B', suffix, ['--session-load', checkpoint, '--session-logits', out/'B.f32'])
    run('C', p, ['--session-live-control'], generation=1)  # cold P only; prefill measured separately
    logs = [a, l, b]
    generated = [json.loads((out/(name+'.json')).read_text())['generated_tokens'] for name in ('A','L','B')]
    if any(len(t) != 64 for t in generated): raise RuntimeError('early EOS: requires exactly 64 output IDs')
    rows = [re.findall(r'session logits_rows=(\d+) vocabulary=(\d+)', log) for log in logs]
    if any(len(r)!=1 for r in rows) or not rows[0]==rows[1]==rows[2] or int(rows[0][0][0])!=64:
        raise RuntimeError('missing/incomplete 64-row logits diagnostic')
    size = (out/'A.f32').stat().st_size
    if size != 64*int(rows[0][0][1])*4 or not size: raise RuntimeError('logits geometry mismatch')
    # Required literal byte comparisons, no tolerance or floating point decoding.
    comparisons = [subprocess.run(['cmp', str(out/'A.f32'), str(out/(name+'.f32'))]).returncode == 0
                   for name in ('L','B')]
    byte_equal = all(comparisons)
    digests = [re.findall(r'session final_digest=([0-9a-f]{64})', log) for log in logs]
    checks = {'64_token_ids_A_L_B': generated[0]==generated[1]==generated[2],
              'every_fp32_logit_bit_cmp': byte_equal,
              'final_state_digest_A_L_B': len(digests[0])==1 and digests[0]==digests[1]==digests[2]}
    for name, ok in checks.items(): print(('PASS ' if ok else 'FAIL ') + name, flush=True)
    if not all(checks.values()): raise RuntimeError('A/L/B bitwise comparison failed')
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
    summary.update(qualification='PASS', comparisons=checks, logits_bytes=size, checkpoint_bytes=source.stat().st_size)
    print('PASS session A == never-saved L == restarted B; C is cold P-only timing', flush=True)
except Exception as e:
    summary.update(qualification='FAIL', error=str(e))
    print('FAIL', e, file=sys.stderr)
finally:
    print('Timing ms (wall includes load; - = not applicable):')
    fields = ('wall_ms_including_model_load', 'initial_prefill_ms', 'save_stall_ms', 'write_fsync_ms', 'restore_ms', 'suffix_ms', 'generation_ms')
    print('arm        wall     P-prefill    save     fsync    restore    S-prefill   generation')
    for name, row in summary['runs'].items():
        print(name, ' '.join('%10.2f'%row[f] if f in row else '%10s'%'-' for f in fields))
    summary['total_wall_ms'] = (time.perf_counter()-started)*1000
    (out/'gate-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
if summary['qualification'] != 'PASS': sys.exit(1)
PY
