"""CPU-only resolved configuration contract; no models or CUDA devices required.

Optional STRIX_DS41_CONFIG_PRIOR / STRIX_DS41_CONFIG_KEEP supply private measured
inputs. Otherwise equivalent synthetic parser fixtures are made in a temp directory.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

exe = sys.argv[1]
baseline_exe = os.environ.get("STRIX_DS41_CONFIG_BASELINE_EXE")
baseline_comparisons = 0
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "-1"
for key in ('STRIX_DS41_RTX_BASELINE', 'DG_LOAD_SERIAL', 'STRIX_SESSION_CACHE', 'STRIX_SESSION_DISK', 'STRIX_DS41_CE_PACING', 'STRIX_DS41_WB_STREAM'):
    env.pop(key, None)


def call(args=(), extra=None, base=None, ok=True):
    e = dict(env, **(extra or {}))
    p = subprocess.run([exe, *(base or ['MODEL', '--import-state', 'S', '--ref', 'R', '--draft', 'D']),
                        '--print-config', *args], env=e, capture_output=True, text=True)
    if not ok:
        assert p.returncode != 0, p.stdout
        return p
    assert p.returncode == 0, (p.returncode, p.stderr)
    # The public baseline has no session inspection fields or session switches.
    if baseline_exe and not any(a.startswith('--session-') or a == '--serve-session-mib' for a in args):
        global baseline_comparisons
        reference = subprocess.run([baseline_exe, *(base or ['MODEL', '--import-state', 'S', '--ref', 'R', '--draft', 'D']),
                                    '--print-config', *args], env=e, capture_output=True, text=True)
        ordinary_output = re.sub(r',"session_cache":.*}\n?$', '}\n', p.stdout)
        assert (p.returncode, ordinary_output, p.stderr) == (reference.returncode, reference.stdout, reference.stderr), args
        baseline_comparisons += 1
    lines = p.stdout.splitlines()
    assert len(lines) == 1, p.stdout
    return json.loads(lines[0])


def diff(a, b):
    return {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}


with tempfile.TemporaryDirectory(prefix='ds41-config-', dir=Path.cwd()) as tmp:
    root = Path(tmp)
    vocab = root / 'vocab'
    vocab.mkdir()
    seed = root / 'seed'
    header = (Path(__file__).resolve().parents[1] / 'include/strix/deepseek41_model.hpp').read_text()
    sha = re.search(r'PinnedFileKind::target_q2,.*?"([0-9a-f]{64})"', header, re.S)[1]
    seed.write_text(f'ds41-residency-v1 sha256={sha} layers=40 experts=384 n=3\n0 5 15359\n')
    env.update(STRIX_DS41_DRAFT_VOCAB=str(vocab), STRIX_DS41_RESIDENCY=str(seed))
    default = call()
    # Make portable equivalent inputs. Private receipt inputs can replace them at test time.
    stats = root / 'keep.jsonl'
    counts = []
    for ids in default['draft_keep_ids']:
        row = [0] * 128
        for rank, expert in enumerate(ids):
            row[expert] = 128 - rank
        counts += row
    stats.write_text(json.dumps({'event': 'route_stats', 'counts': counts}, separators=(',', ':')) + '\n')
    prior = root / 'prior'
    seen, hits = default['controller_prior'].split('/')
    seen = [int(x) for x in seen.split(',') if x]
    hits = [int(x) for x in hits.split(',') if x]
    prior.write_text('ds41-dspark-survival-v1 bins 8\n' + ''.join(
        f'd{d+1} ' + ' '.join(map(str, seen[8*d:8*d+8] + hits[8*d:8*d+8])) + '\n' for d in range(5)))
    prior_arg = os.environ.get('STRIX_DS41_CONFIG_PRIOR', str(prior) + ':0')
    keep_arg = os.environ.get('STRIX_DS41_CONFIG_KEEP', str(stats))
    production = call(['--arena-gib', '74.8', '--spares', '96', '--host-spares', '48', '--predict', '8',
                       '--pressure-spares', '8', '--max-rows', '6', '--window', 'controller', '--controller-scope', 'chat',
                       '--draft-vocab', 'auto:' + str(vocab), '--draft-keep', '64', '--draft-keep-from', keep_arg,
                       '--controller-prior', prior_arg, '--controller-costs', '14.8,18.7,26.6,28.9,33.2,37.3,1.6',
                       '--controller-explore', '0', '--wb-defer'])
    assert default == production, {k: (default[k], production[k]) for k in diff(default, production)}
    assert default['census'] is False and default['evict'] in ('auto', 'lru')
    assert default['ce_pacing'] is True
    assert diff(default, call(['--no-ce-pacing'])) == {'ce_pacing'}
    assert call(extra={'STRIX_DS41_CE_PACING': '0'})['ce_pacing'] is False
    assert call(['--ce-pacing'], extra={'STRIX_DS41_CE_PACING': '0'})['ce_pacing'] is True
    assert call(['--no-ce-pacing'], extra={'STRIX_DS41_CE_PACING': '1'})['ce_pacing'] is False
    assert call(extra={'STRIX_DS41_RTX_BASELINE': '1'})['ce_pacing'] is False
    assert call(['--ce-pacing', '--baseline'])['ce_pacing'] is False
    assert call(['--baseline', '--ce-pacing'])['ce_pacing'] is True
    assert default['wb_stream'] is True
    assert diff(default, call(['--no-wb-stream'])) == {'wb_stream'}
    assert call(extra={'STRIX_DS41_WB_STREAM': '0'})['wb_stream'] is False
    assert call(extra={'STRIX_DS41_WB_STREAM': '1'})['wb_stream'] is True
    assert call(['--wb-stream'], extra={'STRIX_DS41_WB_STREAM': '0'})['wb_stream'] is True
    assert call(['--no-wb-stream'], extra={'STRIX_DS41_WB_STREAM': '1'})['wb_stream'] is False
    assert call(extra={'STRIX_DS41_RTX_BASELINE': '1'})['wb_stream'] is False
    assert call(['--wb-stream', '--baseline'])['wb_stream'] is False
    assert call(['--baseline', '--wb-stream'])['wb_stream'] is True
    assert diff(call(['--no-ce-pacing']), call(['--no-ce-pacing', '--no-wb-stream'])) == {'wb_stream'}
    assert diff(default, call(['--census'])) == {'census'}
    for policy, hook in [('scored', 'ts_dg_set_policy'), ('auto', 'ts_dg_end_rows')]:
        probe = subprocess.run([exe, 'MODEL', '--print-config', '--evict', policy], env=env, capture_output=True, text=True)
        if probe.returncode == 0:
            assert call(['--evict', policy])['evict'] == policy
            assert call(['--evict', policy, '--baseline'])['evict'] == 'lru'
        else:
            assert hook in probe.stderr, probe.stderr
            assert default['evict'] == 'lru'
            call(['--evict', policy], ok=False)
    assert diff(default, call(['--evict', 'lru'])) == ({'evict'} if default['evict'] != 'lru' else set())
    call(['--evict', 'invalid'], ok=False)
    old = call(['--baseline'])
    expected = dict(dsrtxk=False, fuse=False, records_inline=True, router_form=0, qakv_s1=False,
                    q8_r1=False, q8_r1_pad=False, draft_keep=0, draft_keep_ids=[], controller_prior='none',
                    controller_costs='none', wb_defer=False, controller_scope='run', host_spares=96,
                    arena_gib=71.5, draft_vocab='full', warmup_graphs='none', residency_seed='none',
                    residency_save='none', engram_cache_mb=0, load_serial=True, attn_warps=0, max_rows=1, pressure_spares=4, lean_state=False, arena_auto=False, evict='lru', ce_pacing=False, wb_stream=False)
    for key, value in expected.items():
        assert old[key] == value, (key, old[key], value)
    for key in ('decode_fuse_hc', 'decode_fuse_shared', 'decode_router_v2', 'decode_ob_one',
                'decode_shared_fork', 'decode_attn_fork', 'decode_ob_expand', 'decode_moe_expand', 'moe_v1b3', 'hc_cluster'):
        expected[key] = False
    assert old == dict(default, **expected), diff(old, dict(default, **expected))
    assert call(extra={'STRIX_DS41_RTX_BASELINE': '1'}) == old
    for flags, keys in [(['--dsrtxk-off'], {'dsrtxk', 'router_form', 'qakv_s1', 'attn_warps', 'decode_router_v2', 'decode_ob_one', 'decode_shared_fork', 'decode_attn_fork', 'decode_ob_expand', 'decode_moe_expand', 'moe_v1b3', 'hc_cluster'}),
                        (['--fuse-off'], {'fuse', 'decode_fuse_hc', 'decode_fuse_shared'}), (['--records-inline'], {'records_inline'}),
                        (['--router-pf0'], {'router_form'}), (['--qakv-auto'], {'qakv_s1'}),
                        (['--q8-r1-off'], {'q8_r1'}), (['--q8-r1-pad'], {'q8_r1_pad'}),
                        (['--draft-keep', '0'], {'draft_keep', 'draft_keep_ids', 'arena_gib'}),
                        (['--controller-prior', 'none'], {'controller_prior'}),
                        (['--controller-costs', 'none'], {'controller_costs'}),
                        (['--no-wb-defer'], {'wb_defer'}), (['--engram-cache-mb', '0'], {'engram_cache_mb'}),
                        (['--draft-vocab', 'full'], {'draft_vocab'}),
                        (['--residency-seed', 'none'], {'residency_seed'}),
                        (['--controller-scope', 'run'], {'controller_scope'}),
                        (['--host-spares', '96'], {'host_spares'}),
                        (['--no-lean-state'], {'lean_state'}), (['--arena-gib', 'auto'], {'arena_auto'})]:
        assert diff(default, call(flags)) == keys, flags
    assert diff(default, call(extra={'DG_LOAD_SERIAL': '1'})) == {'load_serial'}
    assert call(['--baseline'], extra={'DG_LOAD_SERIAL': '0'})['load_serial'] is False
    # Ordering: baseline resets earlier options; later flags override their own items.
    assert call(['--host-spares', '7', '--baseline']) == old
    assert call(['--baseline', '--host-spares', '7'])['host_spares'] == 7
    assert call(['--arena-gib', 'auto', '--baseline'])['arena_auto'] is False
    assert call(['--baseline', '--lean-state'])['lean_state'] is True
    assert call(['--host-spares', '7'], extra={'STRIX_DS41_RTX_BASELINE': '1'})['host_spares'] == 7
    for flags, key, value in [(['--dsrtxk'], 'dsrtxk', True), (['--fuse'], 'fuse', True),
                              (['--q8-r1'], 'q8_r1', True), (['--router-pf'], 'router_form', 2),
                              (['--qakv-s1'], 'qakv_s1', True), (['--records-deferred'], 'records_inline', False),
                              (['--wb-defer'], 'wb_defer', True), (['--draft-keep', '64'], 'draft_keep', 64)]:
        assert call(['--baseline', *flags])[key] == value, flags
    assert diff(default, call(['--arena-gib', '12'])) == {'arena_gib'}
    assert call(['--draft-keep', '32', '--draft-keep-from', str(stats)])['arena_gib'] == 76.5
    assert call(['--draft-keep', '0'])['arena_gib'] == 71.5
    assert call(base=['MODEL', '--import-state', 'S', '--ref', 'R'])['arena_gib'] == 80
    assert call(['--arena-gib', '12', '--draft-keep', '0'])['arena_gib'] == 12
    worker = ['--serve', str(root / 'worker'), '--max-positions', '4096']
    persistent = call(worker)
    assert persistent['session_cache'] and persistent['session_disk']
    assert persistent['session_route'] == 'rtx-tp1-decode-rows'
    assert persistent['session_explicit_opt_in'] is False
    disabled_policy = call([*worker, '--session-cache', 'off'])
    assert {k: v for k, v in persistent.items() if not k.startswith('session_')} == {
        k: v for k, v in disabled_policy.items() if not k.startswith('session_')}
    assert persistent['session_off_reason'] == ''
    assert call([*worker, '--session-cache', 'on'])['session_cache']
    assert call([*worker, '--session-cache', 'on'])['session_explicit_opt_in'] is True
    assert call([*worker, '--serve-session-mib', '512'])['session_cache']
    assert persistent['session_max_parked'] == 8
    assert 0 < persistent['session_ram_bytes'] <= 8 * 1024**3
    assert call([*worker, '--session-cache', 'off'])['session_cache'] is False
    assert call(worker, extra={'STRIX_SESSION_CACHE': '0'})['session_cache'] is False
    assert call([*worker, '--baseline'])['session_cache'] is False
    assert call([*worker, '--session-disk', 'off'])['session_disk'] is False
    assert call(worker, extra={'STRIX_SESSION_DISK': '0'})['session_disk'] is False
    custom = call([*worker, '--session-ram-bytes', '1048576', '--session-max-parked', '2',
                   '--session-disk-bytes', '2097152', '--session-dir', str(root / 'sessions')])
    assert (custom['session_ram_bytes'], custom['session_max_parked'], custom['session_disk_bytes']) == (1048576, 2, 2097152)
    assert custom['session_dir'] == str(root / 'sessions')
    # Operational session controls may change only session policy fields.
    numeric = lambda cfg: {k: v for k, v in cfg.items() if not k.startswith('session_')}
    for flags in (['--session-cache', 'off'], ['--session-cache', 'on'],
                  ['--session-disk', 'off'], ['--session-ram-bytes', '1'],
                  ['--session-max-parked', '0'], ['--session-disk-bytes', '1']):
        assert numeric(call([*worker, *flags])) == numeric(persistent), flags
    assert persistent['warmup_graphs'] == '0,8'
    assert persistent['residency_save'] == str(seed)
    for flags, key in [(['--warmup-graphs', 'none'], 'warmup_graphs'),
                       (['--residency-save', 'none'], 'residency_save')]:
        assert diff(persistent, call([*worker, *flags])) == {key}
    valid_seed = seed.read_text()
    model_seed = root / 'dummy.gguf.residency'
    model_seed.write_text(valid_seed)
    implicit = call(extra={'STRIX_DS41_RESIDENCY': ''}, base=[str(root / 'dummy.gguf'), '--draft', 'D'])
    assert implicit['residency_seed'] == 'auto:' + str(model_seed)
    assert call([*worker, '--baseline'])['residency_save'] == 'none'
    assert call([*worker, '--baseline'])['warmup_graphs'] == 'none'
    for invalid in (valid_seed.replace('layers=40', 'layers=39'), valid_seed.replace('0 5 15359', '15360'),
                    valid_seed.replace('0 5 15359', 'garbage'), valid_seed.replace('0 5 15359', '')):
        seed.write_text(invalid)
        assert call()['residency_seed'] == 'none'
        call(['--residency-seed', str(seed)], ok=False)
    seed.write_text('invalid\n')
    assert call()['residency_seed'] == 'none'
    call(['--residency-seed', str(seed)], ok=False)
    seed.unlink()
    assert call()['residency_seed'] == 'none'
    assert call(extra={'STRIX_DS41_DRAFT_VOCAB': str(root / 'missing')})['draft_vocab'] == 'full'
    assert call(['--draft-vocab', 'full'])['draft_vocab'] == 'full'
    sibling = root / 'ds41-draft-vocab'
    sibling.mkdir()
    assert call(extra={'STRIX_DS41_DRAFT_VOCAB': str(root / 'missing')}, base=['MODEL', '--draft', str(root / 'D.gguf')])['draft_vocab'] == 'auto:' + str(sibling)
    assert call(extra={'STRIX_DS41_DRAFT_VOCAB': ''}, base=['MODEL', '--draft', str(root / 'D.gguf')])['draft_vocab'] == 'auto:' + str(sibling)
    for label, config in [('i', default), ('ii', production), ('iii', old)]:
        print(label + ': ' + json.dumps(config, sort_keys=True))
    print('PASS: default == production (all keys); baseline; per-feature switches; ordering; arena; serve; auto fallbacks')

if baseline_exe:
    print(f"PASS: {baseline_comparisons} ordinary config outputs/stderr match the public baseline")
