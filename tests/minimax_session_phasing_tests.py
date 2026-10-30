#!/usr/bin/env python3
"""Phase admission checks execute without SSH, models or GPUs."""
import re
import os
import pathlib
import shutil
import subprocess
import tempfile

script = pathlib.Path(__file__).resolve().parents[1] / 'tools/minimax/session_cache_gate.sh'
bash = shutil.which('bash')
assert bash, 'bash required'
subprocess.run([bash, '-n', str(script)], check=True)
with tempfile.TemporaryDirectory(prefix='mm-phases-') as directory:
    root = pathlib.Path(directory)
    posix = subprocess.check_output([bash, '-c', 'pwd'], cwd=root, text=True).strip()
    env = dict(os.environ, OUT=posix, BIN='/absent', MODEL='/absent', DSPARK='/absent', CPU_TEST='/absent', NODE1='peer', STRIX_HASH_CACHE='/hash-cache')
    def refuses(phase, message):
        result = subprocess.run([bash, str(script), phase], env=env, capture_output=True, text=True, timeout=10)
        assert result.returncode == 2, result
        assert message in result.stdout, result
        assert not list(root.glob('turn-*')), 'phase started before admission checks'
    refuses('restore-on', 'missing prepare receipt')
    (root / 'prepare.done').write_text('prepare\n')
    refuses('restore-on', 'missing prewarm receipt')
    (root / 'prewarm.done').write_text('prewarm\n')
    refuses('restore-on', 'prewarm cache directory changed')
    (root / 'prewarm.cache-dir').write_text('/hash-cache\n')
    refuses('restore-on', 'save phase prerequisites missing')
    (root / 'save-on.done').write_text('save-on\n')
    refuses('restore-on', 'save phase prerequisites missing')
    (root / 'restore-on.done').write_text('restore-on\n')
    refuses('restore-on', 'phase already attempted')
# Execute the actual prewarm launch wrapper with a harmless system binary.
# It must pass each staged binary and ldd-resolved library to hash-only mode.
source = script.read_text()
runner = re.search(r"    hash_runner='(.*?)'\n    localcmd=", source, re.S).group(1)
echo = shutil.which('echo')
result = subprocess.run([bash, '-c', runner, 'gate', echo, '/model', '/draft', '0', echo],
                        capture_output=True, text=True, check=True)
assert '--session-hash-only' in result.stdout and '--session-dspark' in result.stdout
assert '--session-hash-file ' + echo in result.stdout
assert re.search(r'--session-hash-file /.*libc', result.stdout), result.stdout
assert source.index('launch_pair prewarm') < source.index('prewarm.receipt')
assert 'SESSION_GATE_END - 30' in source and '+ 690' in source
print('PASS phase syntax, prerequisite refusal before setup, no restore without save evidence, immutable phase receipts')

# Both scheduler arms live inside each existing serve phase, with distinct witnesses.
assert 'default_pair_arm "$@" off' in source
assert 'default_pair_arm "$1-adaptive" "$2" "$3" "$4" default' in source
assert 'unset_env+=(-u STRIX_MM_SERVE_SCHED' in source
assert 'for suffix in "" "-adaptive"' in source
assert 'tag.removesuffix("-adaptive")' in source

# Execute both launch arrays with inherited and forwarded scheduler overrides.
functions = source[source.index('default_pair() {'):source.index('compare_default() {')]
with tempfile.TemporaryDirectory(prefix='mm-scheduler-') as directory:
    root = pathlib.Path(directory)
    binary = root / 'engine'
    binary.write_text('#!/bin/bash\nprintf "%s %s %s\\n" "${STRIX_MM_SERVE_SCHED-unset}" "${STRIX_MM_SERVE_NEST-unset}" "${MM_DEFAULT_PROBE}"\n')
    binary.chmod(0o755)
    harness = functions + r'''
set -eu
OUT=$1; MODEL=model; DSPARK=draft; NODE1=peer
ENV_ARGS=(STRIX_MM_SERVE_SCHED=off STRIX_MM_SERVE_NEST=0)
export STRIX_MM_SERVE_SCHED=off STRIX_MM_SERVE_NEST=0
check_session_receipts() { :; }
copy() { :; }
launch_pair() { "${localcmd[@]}"; "${remote[@]}"; }
default_pair off-serve-off serve "$2" off
'''
    result = subprocess.run([bash, '-c', harness, 'gate', str(root), str(binary)],
                            capture_output=True, text=True, check=True, timeout=10)
    records = result.stdout.splitlines()
    assert len(records) == 4, records
    assert all(line.startswith('off 0 ') and line.endswith('/off-serve-off') for line in records[:2]), records
    assert all(line.startswith('unset unset ') and line.endswith('/off-serve-off-adaptive') for line in records[2:]), records
print('PASS serve phases execute scheduler-off and unmodified default scheduler arms')
