"""Host checks for stored INT4, including required literal-byte hygiene.
Run: python tests/stored_i4_checks.py
"""
from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def changed_sources():
    opts = dict(cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
    tracked = subprocess.check_output(['git', '-c', 'core.autocrlf=false', 'diff', '--name-only'], **opts)
    added = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], **opts)
    return sorted({ROOT / p for p in (tracked + '\n' + added).splitlines()
                   if Path(p).suffix in {'.hpp', '.hip', '.cpp', '.inc', '.h'}})

def literals(path):
    raw = path.read_bytes()
    if b'\0' in raw:
        raise AssertionError(f'{path.relative_to(ROOT)}: raw NUL')
    s = raw.decode('utf-8')
    i = 0
    while i < len(s):
        if s.startswith('//', i):
            end = s.find('\n', i)
            i = len(s) if end < 0 else end + 1
        elif s.startswith('/*', i):
            end = s.find('*/', i + 2)
            if end < 0: raise AssertionError(f'{path.name}: unterminated comment')
            i = end + 2
        elif s.startswith('R"', i):
            paren = s.index('(', i + 2)
            end = s.find(')' + s[i + 2:paren] + '"', paren + 1)
            if end < 0: raise AssertionError(f'{path.name}: unterminated raw literal')
            i = end + len(s[i + 2:paren]) + 2
        elif s[i] in {'"', "'"}:
            if s[i] == "'" and i > 0 and i + 1 < len(s) and s[i - 1].isalnum() and s[i + 1].isalnum():
                i += 1
                continue
            quote, start = s[i], i
            i += 1
            while i < len(s) and s[i] != quote:
                if s[i] == '\\': i += 2
                elif s[i] in '\r\n':
                    raise AssertionError(f'{path.relative_to(ROOT)}:{s.count(chr(10), 0, start) + 1}: raw newline inside literal')
                else: i += 1
            if i >= len(s): raise AssertionError(f'{path.name}: unterminated literal')
            i += 1
        else:
            i += 1

paths = changed_sources()
for path in paths: literals(path)
production = [p for p in paths if 'tests' not in p.relative_to(ROOT).parts]
for path in production:
    text = path.read_text(encoding='utf-8')
    if re.search(r'STRIX_(?:GLM|MMX)_FAST_I4_(?:JIT|PREFETCH)|\b(?:I4Jit|JitSlot|jit_acquire|jit_release|convert_async|TrellisIU4)\b', text):
        raise AssertionError(f'{path.relative_to(ROOT)}: retired conversion path')
core = (ROOT / 'src/core/wmma_tiled.hip').read_text()
assert len(re.findall(r'hipError_t convert_exl3_iu4\(', core)) == 1
print(f'stored-i4 source bytes: PASS ({len(paths)} changed source files; no raw NUL/newline literals or retired converter paths)')

with tempfile.TemporaryDirectory(prefix='strix-s4-') as temp:
    temp = Path(temp)
    source = ROOT / 'tests/stored_i4_host_tests.cpp'
    log = temp / 'build.log'
    if os.name == 'nt':
        bases = [Path(os.environ.get('ProgramFiles', 'C:/Program Files')),
                 Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)'))]
        envs = [p for base in bases for p in base.glob('Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat')]
        if not envs: raise RuntimeError('MSVC vcvars64.bat required for host test')
        exe = temp / 'stored-i4-host-tests.exe'
        command = (f'call "{envs[-1]}" > "{temp / "env.log"}" 2>&1 && '
                   f'cl /nologo /EHsc /std:c++20 /W4 /WX /I"{ROOT / "src"}" "{source}" '
                   f'/Fe:"{exe}" /Fo:"{temp / "test.obj"}" > "{log}" 2>&1')
        batch = temp / 'build.cmd'
        batch.write_text('@echo off\n' + command + '\nexit /b %errorlevel%\n')
        build = subprocess.run(['cmd.exe', '/d', '/c', str(batch)], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    else:
        compiler = os.environ.get('CXX') or shutil.which('c++')
        if not compiler: raise RuntimeError('C++20 host compiler required')
        exe = temp / 'stored-i4-host-tests'
        with log.open('w') as out:
            build = subprocess.run([compiler, '-std=c++20', '-Wall', '-Wextra', '-Werror', '-I', str(ROOT / 'src'), str(source), '-o', str(exe)], stdout=out, stderr=subprocess.STDOUT)
    if build.returncode:
        print('\n'.join(log.read_text(errors='replace').splitlines()[-20:]) if log.exists() else build.stdout)
        raise SystemExit(build.returncode)
    subprocess.run([str(exe)], check=True, cwd=ROOT)
