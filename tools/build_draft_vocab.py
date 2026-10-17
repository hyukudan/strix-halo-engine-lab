#!/usr/bin/env python3
"""Build a static MTP draft vocabulary (little-endian uint32 token ids).

The draft head only steers acceptance (the target verifies every token), so the
set is a coverage heuristic, not a numerical contract. Sources, in priority:
  1. control/special tokens of the GGUF tokenizer (EOS, chat markers);
  2. corpus frequencies: --corpus-ids (uint32 LE id dumps) and/or --corpus-text
     encoded with a tokenizer.json (--tokenizer-json, needs `tokenizers`);
  3. Spanish-friendly coverage: tokens whose text carries Spanish letters or
     punctuation (accents, enye, inverted marks), lowest ids first;
  4. lowest ids (byte-level BPE ids follow merge rank, i.e. merge frequency).
Only the GGUF metadata header is read; tensor data is never touched.

usage: build_draft_vocab.py MODEL.gguf OUT.u32 [--size N] [--spanish N]
           [--corpus-ids FILE ...] [--corpus-text FILE ... --tokenizer-json PATH]
"""
import argparse
import collections
import struct
import sys

SCALARS = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}
WANTED = ('tokenizer.ggml.tokens', 'tokenizer.ggml.token_type', 'tokenizer.ggml.eos_token_id')


def read_metadata(path):
    out = {}
    with open(path, 'rb') as f:
        def take(fmt):
            n = struct.calcsize(fmt)
            return struct.unpack(fmt, f.read(n))[0]

        def string():
            return f.read(take('<Q')).decode('utf-8', 'replace')

        def value(kind, keep):
            if kind in SCALARS:
                return take(SCALARS[kind])
            if kind == 8:
                return string() if keep else f.seek(take('<Q'), 1)
            if kind == 9:
                item, count = take('<I'), take('<Q')
                if item in SCALARS and not keep:
                    f.seek(struct.calcsize(SCALARS[item]) * count, 1)
                    return None
                return [value(item, keep) for _ in range(count)]
            raise ValueError(f'unsupported GGUF metadata type {kind}')

        if f.read(4) != b'GGUF':
            raise ValueError('not a GGUF file')
        take('<I')
        take('<Q')
        for _ in range(take('<Q')):
            key = string()
            keep = key in WANTED
            v = value(take('<I'), keep)
            if keep:
                out[key] = v
    if 'tokenizer.ggml.tokens' not in out:
        raise ValueError('GGUF has no tokenizer.ggml.tokens')
    return out


# GPT-2 byte-level BPE maps bytes to printable code points; invert it for text checks.
def byte_decoder():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('\xa1'), ord('\xac') + 1)) + list(range(ord('\xae'), ord('\xff') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_text(token, decoder):
    try:
        return bytes(decoder[c] for c in token).decode('utf-8')
    except (KeyError, UnicodeDecodeError):
        return token


SPANISH = set('áéíóúüñÁÉÍÓÚÜÑ¿¡')
# Letters outside the Spanish alphabet mark other Latin-script languages.
FOREIGN = set('àèìòùâêîôûãõçäëïöåæøœßčďěňřšťžůłśźżőűăâșțÀÈÌÒÙÂÊÎÔÛÃÕÇÄËÏÖ')
WORDS = '''de la que el en y a los se del las un por con no una su para es al lo como más o pero sus le ha me si sin
sobre este ya entre cuando todo esta ser son dos también fue había era muy años hasta desde está mi porque qué sólo
han yo hay vez puede todos así nos ni parte tiene él uno donde bien tiempo mismo ese ahora cada e vida otro después
te otros aunque esa eso hace otra gobierno tan durante siempre día tanto ella tres sí dijo sido gran país según menos
año antes estado contra sino forma caso nada hacer general estaba poco estos presidente mayor ante unos les algo hacia
casa ellos ayer hecho primera mucho mientras además quien momento millones esto españa hombre están pues hoy lugar
madrid nacional trabajo otras mejor nuevo decir algunos entonces todas días debe política cómo casi toda tal luego
pasado primer medio va estas sea tenía nunca poder aún veces embargo partido personas grupo cuenta pueden tienen misma
nueva cual fueron mujer frente josé tras cosas fin ciudad he social manera tener sistema será historia muchos juan tipo
cuatro dentro nuestro punto dice ello cualquier noche aquí mundo respuesta pregunta ejemplo información problema
gracias favor necesito quiero puedes usted ustedes nosotros vosotros también través través número'''.split()


def spanish(text):
    chars = set(text)
    return bool(chars & SPANISH) and not chars & FOREIGN and all(c.isascii() or c in SPANISH for c in chars)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gguf')
    ap.add_argument('out')
    ap.add_argument('--size', type=int, default=32768, help='total ids (default 32768)')
    ap.add_argument('--spanish', type=int, default=4096, help='max Spanish-coverage ids (default 4096)')
    ap.add_argument('--corpus-ids', nargs='*', default=[])
    ap.add_argument('--corpus-text', nargs='*', default=[])
    ap.add_argument('--tokenizer-json')
    a = ap.parse_args()

    meta = read_metadata(a.gguf)
    tokens = meta['tokenizer.ggml.tokens']
    types = meta.get('tokenizer.ggml.token_type') or [1] * len(tokens)
    vocab = len(tokens)
    chosen, seen = [], set()

    def add(ids, limit=None):
        added = 0
        for i in ids:
            if len(chosen) >= a.size or (limit is not None and added >= limit):
                return
            if 0 <= i < vocab and i not in seen:
                seen.add(i)
                chosen.append(i)
                added += 1

    eos = meta.get('tokenizer.ggml.eos_token_id')
    add(([eos] if eos is not None else []) + [i for i, t in enumerate(types) if t in (3, 4)])
    counts = collections.Counter()
    for path in a.corpus_ids:
        data = open(path, 'rb').read()
        counts.update(struct.unpack(f'<{len(data) // 4}I', data[:len(data) // 4 * 4]))
    if a.corpus_text:
        if not a.tokenizer_json:
            sys.exit('--corpus-text requires --tokenizer-json')
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(a.tokenizer_json)
        for path in a.corpus_text:
            counts.update(tok.encode(open(path, encoding='utf-8').read()).ids)
    add(i for i, _ in counts.most_common())
    decoder = byte_decoder()
    index = {token_text(t, decoder): i for i, t in enumerate(tokens) if types[i] == 1}
    add((index[v] for w in WORDS for v in (w, ' ' + w, w.capitalize(), ' ' + w.capitalize()) if v in index), a.spanish)
    add((i for i, t in enumerate(tokens) if types[i] == 1 and spanish(token_text(t, decoder))), a.spanish)
    add(i for i in range(vocab) if types[i] in (1, 6))
    with open(a.out, 'wb') as f:
        f.write(struct.pack(f'<{len(chosen)}I', *chosen))
    print(f'{a.out}: {len(chosen)} ids of {vocab} (corpus distinct {len(counts)})')


if __name__ == '__main__':
    main()
