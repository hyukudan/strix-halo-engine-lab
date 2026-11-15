#!/usr/bin/env python3
"""Compare the SASS of two CUDA build trees, function by function.

The byte gate of a CUDA refactor that must not change arithmetic (moves, helper extraction, core lifting): every
CUDA object (*.cu.obj / *.cu.o) and executable present in both trees under the same name must have the same
instruction stream per function. Symbol names are normalised (anonymous-namespace ids carry the source path, and a
helper that moved into a shared namespace changes its mangled name), everything else is compared verbatim.

  sass_compare.py --base BUILD_A --new BUILD_B [--base ... --new ...] [--rename old.cu.obj=new.cu.obj]
                  [--cuobjdump PATH] [--out report.txt] [--no-exe]

Exit status 0 when nothing differs, 1 otherwise. One line per difference in the report.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

OBJ = ('.cu.obj', '.cu.o')
SYM = re.compile(r'_Z[0-9A-Za-z_$]+|_GLOBAL__N__[0-9A-Za-z_]+|\$__internal_[0-9A-Za-z_$]+|__nv_static_[0-9A-Za-z_]+')
SKIP = ('.', '---', '=====', 'Fatbin', 'code for', 'arch =', 'host =', 'compile_size', 'identifier')


def collect(root, with_exe):
    found = {}
    for d, _, names in os.walk(root):
        for n in names:
            path = os.path.join(d, n)
            binary = n.endswith(('.exe', '.dll', '.so')) or ('.' not in n and os.access(path, os.X_OK))
            if n.endswith(OBJ) or (with_exe and binary):
                found.setdefault(n, []).append(path)
    return found


def sass(cuobjdump, path):
    r = subprocess.run([cuobjdump, '-sass', path], capture_output=True, text=True, errors='replace')
    if r.returncode != 0 or 'Function :' not in r.stdout:
        return None
    funcs, body, name = {}, [], None
    for line in r.stdout.splitlines():
        if 'Function :' in line:
            if name is not None:
                funcs.setdefault(SYM.sub('S', name), []).append('\n'.join(body))
            name, body = line.split('Function :', 1)[1].strip(), []
        elif name is not None:
            s = line.strip()
            if s and not s.startswith(SKIP):
                body.append(SYM.sub('S', s))
    if name is not None:
        funcs.setdefault(SYM.sub('S', name), []).append('\n'.join(body))
    return {k: sorted(v) for k, v in funcs.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', action='append', required=True)
    ap.add_argument('--new', action='append', required=True)
    ap.add_argument('--rename', action='append', default=[], help='old_name=new_name for renamed sources')
    ap.add_argument('--cuobjdump', default=shutil.which('cuobjdump') or 'cuobjdump')
    ap.add_argument('--out')
    ap.add_argument('--no-exe', action='store_true', help='objects only (faster)')
    a = ap.parse_args()
    if len(a.base) != len(a.new):
        ap.error('--base and --new come in pairs')
    rename = dict(x.split('=', 1) for x in a.rename)
    lines, same, diff = [], 0, 0
    for b, n in zip(a.base, a.new):
        fb, fn = collect(b, not a.no_exe), collect(n, not a.no_exe)
        for name, paths in sorted(fb.items()):
            tgt = rename.get(name, name)
            if tgt not in fn or len(paths) != 1 or len(fn[tgt]) != 1:
                continue
            sb, sn = sass(a.cuobjdump, paths[0]), sass(a.cuobjdump, fn[tgt][0])
            if sb is None and sn is None:
                continue
            if sb == sn:
                same += 1
                continue
            diff += 1
            if sb is None or sn is None:
                lines.append('NOSASS %s base=%s new=%s' % (name, sb is not None, sn is not None))
                continue
            kb, kn = set(sb), set(sn)
            bad = sorted(k for k in kb & kn if sb[k] != sn[k])
            lines.append('DIFF %s: functions %d/%d, only-base %d, only-new %d, body differs %d %s'
                         % (name, len(kb), len(kn), len(kb - kn), len(kn - kb), len(bad), bad[:2]))
        renamed = {rename.get(x, x) for x in fb}
        lines += ['NEW-ONLY %s' % x for x in sorted(set(fn) - renamed) if x.endswith(OBJ)]
        lines += ['BASE-ONLY %s' % x for x in sorted(fb) if x.endswith(OBJ) and rename.get(x, x) not in fn]
    summary = 'SASS compare: %d equal, %d different' % (same, diff)
    report = summary + '\n' + '\n'.join(lines) + '\n'
    if a.out:
        with open(a.out, 'w', newline='\n') as f:
            f.write(report)
    print(report if len(report) < 4000 else summary)
    return 1 if diff else 0


if __name__ == '__main__':
    sys.exit(main())
