import hashlib
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile

cli = sys.argv[1]
def u32(n): return struct.pack('<I', n)
def u64(n): return struct.pack('<Q', n)
def text(s):
    b = s.encode()
    return u64(len(b)) + b
with tempfile.TemporaryDirectory(prefix='strix-assets-cli-') as tmp:
    root = pathlib.Path(tmp)
    fixture = b'GGUF' + u32(3) + u64(1) + u64(2)
    fixture += text('general.architecture') + u32(8) + text('qwen4exp')
    fixture += text('general.alignment') + u32(4) + u32(32)
    fixture += text('blk.0.weight') + u32(2) + u64(16) + u64(32) + u32(1) + u64(0)
    fixture += b'\0' * (-len(fixture) % 32) + b'\0' * 1024
    gguf = root / 'tiny.gguf'
    gguf.write_bytes(fixture)
    output = subprocess.check_output([cli, str(gguf)])
    golden = pathlib.Path(__file__).with_name('asset_manifest_legacy.golden.json').read_bytes()
    assert output == golden, 'legacy CLI bytes changed'
    spec = dict(format='strix-assets', revision=1, engine='qwen', model_key='tiny',
                profile_pair='test', required={'target': ['tiny']}, provenance=[], licenses=[],
                files=[dict(role='target', key='tiny', member='tiny.gguf',
                            source_revision='test', audit='gguf')])
    source = root / 'spec.json'
    manifest = root / 'assets.json'
    source.write_text(json.dumps(spec))
    def run(*args):
        return subprocess.run([cli, *args], capture_output=True)
    assert run('bundle-create', '--spec', str(source), '--root', tmp, '--output', str(manifest)).returncode == 0
    # A forged metadata-cache hit must not replace full byte verification.
    cache = root / 'private-cache'
    cache.mkdir()
    st = gguf.stat()
    stamp = f'v1 {st.st_size} {st.st_mtime_ns // 10**9}.{st.st_mtime_ns % 10**9:09d} {st.st_ctime_ns // 10**9}.{st.st_ctime_ns % 10**9:09d}'
    (cache / f'{st.st_dev:x}-{st.st_ino:x}').write_text(stamp + '\n' + '1' * 64 + '\n')
    os.environ['STRIX_HASH_CACHE'] = str(cache)
    result = run('bundle-verify', '--manifest', str(manifest), '--root', tmp)
    assert result.returncode == 0
    summary = json.loads(result.stdout)
    assert summary['verification'] == 'full'
    canonical = json.dumps(json.loads(manifest.read_bytes()), sort_keys=True, separators=(',', ':')).encode()
    assert manifest.read_bytes() == canonical + b'\n'
    assert summary['manifest_digest'] == hashlib.sha256(canonical).hexdigest()
    row = json.loads(canonical)['files'][0]
    role_bytes = json.dumps([[row['key'], row['bytes'], row['sha256']]], separators=(',', ':')).encode()
    assert summary['roles']['target'] == dict(count=1, digest=hashlib.sha256(role_bytes).hexdigest())
    # The same hash-valid GGUF fails an MTP structural contract.
    spec['files'][0]['audit'] = 'mtp'
    source.write_text(json.dumps(spec))
    assert run('bundle-create', '--spec', str(source), '--root', tmp, '--output', str(root / 'mtp.json')).returncode == 2
    spec['files'][0]['audit'] = 'gguf'
    source.write_text(json.dumps(spec))
    assert tmp.encode() not in result.stdout + result.stderr
    gguf.write_bytes(b'bad')
    failed = run('bundle-verify', '--manifest', str(manifest), '--root', tmp)
    assert failed.returncode == 2 and tmp.encode() not in failed.stdout + failed.stderr
    report = json.loads(failed.stdout)
    assert report['verification'] == 'failed' and report['reason'] == 'asset size mismatch' and report['role'] == 'target' and report['index'] == 0 and report['key_digest'] == hashlib.sha256(b'tiny').hexdigest()[:16]
    assert run('bundle-verify', '--manifest', str(source), '--root', tmp).returncode == 1
    source.write_text('{bad')
    bad = run('bundle-create', '--spec', str(source), '--root', tmp, '--output', str(manifest))
    assert bad.returncode == 1 and tmp.encode() not in bad.stderr
print('legacy golden and bundle CLI passed')

# Contract regression cases use atomically allocated directories and exact failures.
import copy
import re
with tempfile.TemporaryDirectory(prefix='strix-assets-contract-') as tmp:
    root = pathlib.Path(tmp)
    (root / 'a').write_bytes(b'abc')
    (root / 'b').write_bytes(b'def')
    spec = dict(format='strix-assets', revision=1, engine='qwen', model_key='tiny',
                profile_pair='test', required={'target': ['target'], 'profile': ['profile']},
                provenance=[], licenses=[], files=[
                    dict(role='target', key='target', member='a', source_revision='test'),
                    dict(role='profile', key='profile', member='b', source_revision='test')])
    source, output = root / 'input.json', root / 'output.json'
    def invoke(document, create=False, roles=(), code=0, expected=None):
        if isinstance(document, bytes):
            source.write_bytes(document)
        else:
            source.write_text(json.dumps(document), encoding='utf-8')
        output.unlink(missing_ok=True)
        args = ['bundle-create', '--spec'] if create else ['bundle-verify', '--manifest']
        args += [str(source), '--root', tmp]
        args += ['--output', str(output)] if create else [x for r in roles for x in ('--role', r)]
        r = subprocess.run([cli, *args], capture_output=True)
        assert r.returncode == code, (args, r.returncode, r.stdout, r.stderr)
        assert tmp.encode() not in r.stdout + r.stderr
        if code:
            assert not output.exists()
        if expected is not None:
            assert json.loads(r.stdout) == expected, (r.stdout, expected)
        if code == 1:
            assert not r.stdout and r.stderr == b'error: invalid bundle input or I/O failure\n'
        return r
    invoke(spec, create=True)
    valid = json.loads(output.read_bytes())
    for roles in ((), ('target', 'profile')):
        receipt = json.loads(invoke(valid, roles=roles).stdout)
        assert receipt['verification'] == 'full' and receipt['hash_mode'] == 'full'
        assert receipt['checked_roles'] == ['profile', 'target'] and receipt['unchecked_roles'] == []
    for branch in ('missing', 'corrupt', 'pin', 'audit'):
        case = copy.deepcopy(valid)
        if branch == 'missing': (root / 'a').unlink()
        elif branch == 'corrupt': (root / 'a').write_bytes(b'xyz')
        elif branch == 'pin': case['files'][1]['expected_sha256'] = '0' * 64
        else: case['files'][1]['audit'] = 'gguf'
        receipt = json.loads(invoke(case, roles=('profile',)).stdout)
        assert receipt['verification'] == 'partial' and receipt['hash_mode'] == 'full'
        assert receipt['checked_roles'] == ['profile'] and receipt['unchecked_roles'] == ['target']
        reason = {'missing':'asset missing or not regular', 'corrupt':'asset hash mismatch',
                  'pin':'asset hash mismatch', 'audit':'asset structural mismatch'}[branch]
        invoke(case, code=2, expected=dict(index=1, key='target', role='target',
               reason=reason, verification='failed'))
        (root / 'a').write_bytes(b'abc')

    for raw in (b'\xff', b'\xc0\xaf', b'\xe0\x80\xaf', b'\xed\xa0\x80', b'\xf0\x80\x80\xaf',
                b'\xf4\x90\x80\x80', b'\xf5\x80\x80\x80', b'\xe2\x82', b'\x80',
                b'\\ud800', b'\\udfff', b'\\ud800x', b'\\ud800\\u0041'):
        for document, create in ((spec, True), (valid, False)):
            case = copy.deepcopy(document)
            case['provenance'] = [dict(type='https', value='https://github.com/PLACEHOLDER')]
            blob = json.dumps(case).encode().replace(b'PLACEHOLDER', raw)
            invoke(blob, create=create, code=1)
    case = copy.deepcopy(spec)
    case['provenance'] = [dict(type='https', value='https://github.com/public/caf\u00e9-\U0001f600')]
    invoke(case, create=True)
    data = output.read_bytes()
    canonical = json.dumps(json.loads(data), sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    assert data == canonical + b'\n'
    assert json.loads(invoke(json.loads(data)).stdout)['manifest_digest'] == hashlib.sha256(canonical).hexdigest()

    seeds = ['C:/Users/PRIVACY_ACCOUNT/private.spec', '/private/SECRET_123', 'PRIVACY_HOST.internal',
             'SECRET_123@host', 'x"\nINJECT', r'\\PRIVACY_HOST\share', 'SECRET_123']
    for field in ('provenance', 'licenses'):
        for seed in seeds:
            for ref in (seed, dict(type='public-id', value=seed), dict(type='https', value=seed)):
                case = copy.deepcopy(spec)
                case[field] = [ref]
                invoke(case, create=True, code=1)
                manifest_case = copy.deepcopy(valid)
                manifest_case[field] = [ref]
                invoke(manifest_case, code=1)
            case[field] = [dict(type='member', value=seed)]
            invoke(case, create=True, code=1)
            manifest_case = copy.deepcopy(valid)
            manifest_case[field] = case[field]
            invoke(manifest_case, code=1)
    case = copy.deepcopy(spec)
    case['provenance'] = [dict(type='member', value='notices/source.md')]
    case['licenses'] = [dict(type='public-id', value='Qwen-Community-License')]
    invoke(case, create=True)
    assert json.loads(output.read_bytes())['licenses'] == case['licenses']
    # Conventional root notice names stay expressible; other bare names do not.
    for name, code in (('LICENSE.txt', 0), ('LICENSE.md', 0), ('NOTICE.txt', 0), ('COPYING', 0),
                       ('LICENSE-MIT', 0), ('README', 0), ('notes.txt', 1), ('LICENSE_x', 1), ('LICENSEX', 1)):
        case = copy.deepcopy(spec)
        case['licenses'] = [dict(type='member', value=name)]
        invoke(case, create=True, code=code)
    case = copy.deepcopy(spec)
    case['engine'] = 'PRIVACY_HOST'
    invoke(case, create=True, code=1)

    for url in ('http://github.com/public/source', 'https://PRIVACY_HOST.internal/source',
                'https://SECRET_123@github.com/source', 'https://github.com:443/source',
                'https://github.com/source?SECRET_123', 'https://github.com/source#SECRET_123',
                'https://github.com/%2e%2e/source', 'https://github.com/../source'):
        case = copy.deepcopy(spec)
        case['provenance'] = [dict(type='https', value=url)]
        invoke(case, create=True, code=1)
    # Input/output I/O and invalid option branches never echo private argument text.
    for args in (
        ['bundle-create', '--spec', str(root / 'PRIVACY_HOST.internal-SECRET_123'), '--root', tmp, '--output', str(output)],
        ['bundle-create', '--spec', str(source), '--root', tmp, '--output', str(root / 'SECRET_123' / 'out')],
        ['bundle-verify', '--manifest', str(source), '--root', tmp, '--PRIVATE_HOST.internal', 'SECRET_123'],
        ['bundle-verify', '--manifest', str(source), '--root', tmp, '--role', 'SECRET_123'],
    ):
        source.write_text(json.dumps(spec if args[0] == 'bundle-create' else valid), encoding='utf-8')
        result = subprocess.run([cli, *args], capture_output=True)
        assert result.returncode == 1 and not result.stdout
        assert result.stderr == b'error: invalid bundle input or I/O failure\n'

    for member in ('a.', 'dir./a', 'dir /a', 'a ', 'CON', 'prn.txt', 'AUX', 'nul.gguf',
                   'COM1', 'com9.txt', 'LPT1.bin', 'lpt9', '../a', 'C:/a', '/a'):
        case = copy.deepcopy(spec)
        case['files'][0]['member'] = member
        invoke(case, create=True, code=1)
    case = copy.deepcopy(spec)
    case['files'][1]['member'] = 'A'
    invoke(case, create=True, code=1)
    for name, link in (('alias', lambda p: p.symlink_to(root / 'a')),
                       ('hardlink', lambda p: os.link(root / 'a', p))):
        link(root / name)
        case = copy.deepcopy(spec)
        case['files'][1]['member'] = name
        invoke(case, create=True, code=1)
        case = copy.deepcopy(valid)
        case['files'][0]['member'] = name
        invoke(case, code=1)

    # All stable exit-2 branches redact arbitrary keys and preserve only public names.
    for seed in ('PRIVACY_HOST.internal', 'SECRET_123', 'target'):
        case = copy.deepcopy(spec)
        case['required']['target'] = [seed]
        case['files'][0]['key'] = seed
        invoke(case, create=True)
        seeded = json.loads(output.read_bytes())
        assert json.loads(output.read_bytes())['files'][1]['key'] == seed
        for branch in ('missing', 'size', 'hash', 'pin', 'content', 'audit'):
            case = copy.deepcopy(seeded)
            entry = case['files'][1]
            if branch == 'missing': (root / 'a').unlink()
            elif branch == 'size': entry['bytes'] += 1
            elif branch == 'hash': entry['sha256'] = '0' * 64
            elif branch == 'pin': entry['expected_sha256'] = '0' * 64
            elif branch == 'content':
                entry['content_key'] = 'model'
                entry['content_identity'] = '0' * 64
            else: entry['audit'] = 'gguf'
            reason = {'missing':'asset missing or not regular', 'size':'asset size mismatch',
                      'hash':'asset hash mismatch', 'pin':'asset hash mismatch',
                      'content':'asset content identity mismatch', 'audit':'asset structural mismatch'}[branch]
            expected = dict(index=1, role='target', reason=reason, verification='failed')
            expected.update(key=seed) if seed == 'target' else expected.update(key_digest=hashlib.sha256(seed.encode()).hexdigest()[:16])
            for create in (False, True):
                result = invoke(case, create=create, code=2, expected=expected)
                if seed != 'target': assert seed.encode() not in result.stdout + result.stderr
            (root / 'a').write_bytes(b'abc')
        case = copy.deepcopy(seeded)
        case['files'].pop()
        invoke(case, code=2, expected=dict(reason='asset role closure mismatch', verification='failed'))
    for seed in seeds:
        if seed in ('PRIVACY_HOST.internal', 'SECRET_123'): continue
        case = copy.deepcopy(spec)
        case['required']['target'] = [seed]
        case['files'][0]['key'] = seed
        invoke(case, create=True, code=1)

repo = pathlib.Path(__file__).resolve().parents[1]
release = json.loads((repo / 'release/qwen-assets.spec.json').read_bytes())
shell = (repo / 'scripts/bench/engines/qwen.sh').read_text()
def shell_array(name):
    return re.search(r'^' + name + r'=\(([^)]+)\)', shell, re.M)[1].split()
for i, role in enumerate(('target', 'draft')):
    entry = next(f for f in release['files'] if f['role'] == role)
    assert entry['member'] == shell_array('assets')[i]
    assert entry['expected_sha256'] == shell_array('asset_sha256s')[i]
    if shell_array('asset_bytes')[i] != 'null':
        assert entry['bytes'] == int(shell_array('asset_bytes')[i])
# Every member reference of the shipped spec names a file inside the documented package layout.
members = {f['member'] for f in release['files']}
for ref in release['provenance'] + release['licenses']:
    if ref['type'] == 'member':
        assert ref['value'] in members or '/' not in ref['value'], ref
    if ref['type'] == 'https':
        assert ref['value'].startswith('https://')
assert release['engine'] == 'qwen'
print('strict schema, privacy, coverage, portability and pin drift passed')
