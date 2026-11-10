"""Fail if any tracked source or text file contains a raw NUL byte.

Raw NUL bytes inside string or character literals compile on some toolchains and
break strict builds on others; they also make diffs unreadable. Use the escape
'\0' instead.
"""
import pathlib
import subprocess
import sys

SUFFIXES = {'.c', '.cc', '.cpp', '.h', '.hpp', '.hip', '.cu', '.cuh', '.inc', '.inl',
            '.cmake', '.txt', '.md', '.py', '.sh', '.toml', '.json', '.yml', '.yaml'}


def main():
    files = subprocess.run(['git', 'ls-files', '-z'], check=True, capture_output=True).stdout.split(b'\0')
    bad = []
    for raw in files:
        if not raw:
            continue
        path = pathlib.Path(raw.decode())
        if path.suffix.lower() not in SUFFIXES and path.name != 'CMakeLists.txt':
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        offset = data.find(b'\0')
        if offset >= 0:
            line = data.count(b'\n', 0, offset) + 1
            bad.append(f'{path}:{line}: raw NUL byte (use the escape \0)')
    for item in bad:
        print(item)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
