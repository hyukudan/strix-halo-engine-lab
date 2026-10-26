#!/usr/bin/env python3
"""Gate prompts for the MiniMax RTX PRO prefill: json / code / prose chat requests whose
templated token count is close to a target (1K, 8K), written as little-endian u32 token files.

usage: rtxpf_prompts.py MODEL_DIR SOURCE_ROOT OUT_DIR [--targets 1024,8192]
  MODEL_DIR    tokenizer.json + chat_template.jinja of MiniMax M3.1 Flash
  SOURCE_ROOT  engine tree (code from apps/ + src/minimax/, English prose from README.md and docs/*.md)
"""
import argparse
import glob
import os
import random
import struct

import jinja2
from tokenizers import Tokenizer


def render(template, user):
    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)
    env.globals['raise_exception'] = lambda m: (_ for _ in ()).throw(RuntimeError(m))
    return env.from_string(template).render(messages=[{'role': 'user', 'content': user}], add_generation_prompt=True)


def json_body(rng, rows):
    cities = ['Madrid', 'Lisbon', 'Paris', 'Berlin', 'Rome', 'Vienna', 'Prague', 'Oslo', 'Dublin', 'Athens']
    lines = ['id,name,city,age,score,active']
    for i in range(rows):
        lines.append(f'{i + 1},user_{rng.randrange(10**6):06d},{rng.choice(cities)},{rng.randrange(18, 90)},'
                     f'{rng.randrange(0, 10000) / 100:.2f},{rng.choice(["true", "false"])}')
    return '\n'.join(lines)


def build(kind, tok, template, target, src, rng):
    def make(n):
        if kind == 'json':
            body = json_body(random.Random(7), n)
            return ('Convert the following CSV table into a JSON array of objects with typed fields (numbers as '
                    'numbers, booleans as booleans). Output only the JSON.\n\n' + body)
        if kind == 'code':
            return ('Here is part of a C++/HIP inference engine. Review it, then write a new function that validates '
                    'every input argument of the main entry point and explain your changes.\n\n```cpp\n'
                    + src['code'][:n] + '\n```')
        return ('Summarize the following technical document in five paragraphs for an engineering manager, then list '
                'the three biggest risks.\n\n' + src['prose'][:n])
    lo, hi = 1, (200000 if kind != 'json' else 4000)
    best = None
    while lo <= hi:  # largest body whose templated length is <= target
        mid = (lo + hi) // 2
        ids = tok.encode(render(template, make(mid)), add_special_tokens=False).ids
        if len(ids) <= target:
            best = ids
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model_dir')
    ap.add_argument('source_root')
    ap.add_argument('out_dir')
    ap.add_argument('--targets', default='1024,8192')
    a = ap.parse_args()
    tok = Tokenizer.from_file(os.path.join(a.model_dir, 'tokenizer.json'))
    template = open(os.path.join(a.model_dir, 'chat_template.jinja'), encoding='utf-8').read()
    code = ''.join(open(p, encoding='utf-8', errors='replace').read() for p in
                   sorted(glob.glob(os.path.join(a.source_root, 'apps', 'minimax_*.cpp')) +
                          glob.glob(os.path.join(a.source_root, 'src', 'minimax', '*', '*.hip'))))
    prose = ''.join(open(p, encoding='utf-8', errors='replace').read() for p in
                    [os.path.join(a.source_root, 'README.md')] + sorted(glob.glob(os.path.join(a.source_root, 'docs', '*.md'))))
    src = {'code': code, 'prose': prose}
    os.makedirs(a.out_dir, exist_ok=True)
    for target in (int(t) for t in a.targets.split(',')):
        for kind in ('json', 'code', 'prose'):
            ids = build(kind, tok, template, target, src, random.Random(1))
            name = f'{kind}{target // 1024}k'
            with open(os.path.join(a.out_dir, name + '.u32'), 'wb') as f:
                f.write(struct.pack(f'<{len(ids)}I', *ids))
            print(f'{name}: {len(ids)} tokens, first {ids[:4]}')


if __name__ == '__main__':
    main()
