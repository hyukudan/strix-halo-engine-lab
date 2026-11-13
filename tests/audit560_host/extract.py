"""Compile verbatim adapter sections with deterministic host stand-ins, not HIP.

The standalone CMake project also generates HEAD controls; each audit regression
must fail on that source. No kernel execution or numerical qualification is implied.
"""
import argparse
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--original', action='store_true')
args = parser.parse_args()
relative = 'apps/hip_mtp_decode_bridge.hpp'
source = (subprocess.check_output(['git', 'show', f'HEAD:{relative}'], cwd=args.root).decode()
          if args.original else (args.root / relative).read_text())
# Mask strings/comments so braces inside them cannot end an extracted block.
mask = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
              lambda m: ''.join('\n' if c == '\n' else ' ' for c in m[0]), source, flags=re.S)

def block_at(start):
    first = mask.index('{', start)
    depth = 1
    end = first + 1
    while depth:
        depth += (mask[end] == '{') - (mask[end] == '}')
        end += 1
    return source[start:end]

def emit(name, text):
    (args.out / f'{name}.inc').write_text(text + '\n')

args.out.mkdir(parents=True, exist_ok=True)
a = source.index('        auto results =', source.index('const bool joint=strix::envcfg'))
b = source.index('        runtime_done(results);', a)
emit('cohort', source[a:b])
a = source.index('    bool fork_on_=')
b = source.index('    serve::PrefixDirectory', a)
emit('fork', source[a:b])
a = source.index('            const std::uint32_t units[2]')
b = source.index('        } else {\n            for (auto& s : slots_)', a)
emit('plan', source[a:b])
emit('execute', block_at(source.index('        if (prompt && (planner_on')))
emit('account', block_at(source.index('        const auto account_step=')) + ';\naccount_step();')
# Include completion loops verbatim: progress must be captured before callbacks reset prefill.
a = source.index('        for (std::size_t k = 0; k < members.size(); ++k)', source.index('    bool run_prompt_group('))
emit('group', block_at(a))
a = source.index('        for (std::size_t k = 0; k < members.size(); ++k)', source.index('    bool run_prompt_windows('))
emit('windows', block_at(a))
