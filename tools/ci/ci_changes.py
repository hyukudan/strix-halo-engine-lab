"""Decide which CI jobs a change set can affect.

Prints `code=true|false` and `hip=true|false` (one per line; append to $GITHUB_OUTPUT in CI).

- code: the host build and its tests. False only when every changed path is documentation that no
  tracked non-Markdown file names (a test that reads a document names its path, see
  `referenced_paths`). Lint (always run) covers the checks that read whole documentation trees.
- hip: the offline gfx1151 HIP build. False when every changed path is outside its inputs
  (documentation, Python, tools/scripts, CUDA-only sources, labs it does not build, other workflows).

Any doubt runs the job: `--all`, an empty or failed diff, or an unknown path answer true.
Usage: ci_changes.py --all | ci_changes.py --base REV --head REV
"""
import argparse
import subprocess
import sys
from pathlib import PurePosixPath

DOC_SUFFIXES = {'.md', '.png', '.jpg', '.jpeg', '.gif', '.svg'}
DOC_PREFIXES = ('LICENSES/',)
DOC_NAMES = {'LICENSE'}

# Paths the offline HIP build never compiles or configures (see the hip-offline-build job).
HIP_LABS = ('lab/basic_kernels/', 'lab/packed_projection/', 'lab/uma_residency/')
NON_HIP_PREFIXES = ('docs/', 'tools/', 'scripts/', 'recipes/', 'release/', 'packaging/', 'examples/', 'data/',
                    'artifacts/', 'LICENSES/', 'src/cuda/', 'tests/cuda/', 'tests/cuda_core/',
                    '.github/')
NON_HIP_SUFFIXES = DOC_SUFFIXES | {'.py', '.cu', '.cuh'}
HIP_WORKFLOW = '.github/workflows/ci.yml'


def is_doc(path):
    p = PurePosixPath(path)
    return p.suffix.lower() in DOC_SUFFIXES or path.startswith(DOC_PREFIXES) or path in DOC_NAMES


def needs_hip(path):
    if path == HIP_WORKFLOW:
        return True
    if path.startswith('lab/'):
        return path.startswith(HIP_LABS)
    if PurePosixPath(path).suffix.lower() in NON_HIP_SUFFIXES or path in DOC_NAMES:
        return False
    return not path.startswith(NON_HIP_PREFIXES)


def referenced_paths(paths, grep=None):
    """Documentation paths (with a directory part) that some tracked non-Markdown file names."""
    grep = grep or _git_grep_names
    return {p for p in paths if '/' in p and grep(p)}


def _git_grep_names(path):
    done = subprocess.run(['git', 'grep', '-q', '-F', '-e', path, '--', '.', ':(exclude)*.md'],
                          capture_output=True)
    return done.returncode != 1  # 0 = found; errors count as found (run the build)


def classify(paths, grep=None):
    if not paths:
        return {'code': True, 'hip': True}
    docs = [p for p in paths if is_doc(p)]
    docs_only = len(docs) == len(paths) and not referenced_paths(docs, grep)
    return {'code': not docs_only, 'hip': any(needs_hip(p) for p in paths)}


def changed_paths(base, head):
    out = subprocess.run(['git', 'diff', '--name-only', '--no-renames', base, head],
                         check=True, capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--base')
    ap.add_argument('--head')
    args = ap.parse_args(argv)
    paths = []
    if not args.all:
        try:
            paths = changed_paths(args.base, args.head)
        except (subprocess.CalledProcessError, TypeError) as exc:
            print(f'diff failed ({exc}); running everything', file=sys.stderr)
    result = classify(paths)
    print(f'{len(paths)} changed path(s)', file=sys.stderr)
    for key in ('code', 'hip'):
        print(f'{key}={"true" if result[key] else "false"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
