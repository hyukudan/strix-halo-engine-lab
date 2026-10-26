"""G4 prompt token files (design notes §2): raw text, no chat template ('[gMASK]<sop>' + text), exactly T + 1
tokens each (inputs 0..T-1; the last 64 positions are scored, token T is only a target). Writes g4-<name>-<T>.u32 and
g4-manifest.txt (source sha256, start marker, token-file sha256, a decode check).
usage: make_g4_prompts.py TOKENIZER_JSON OUT_DIR 'NAME|SOURCE|T[|START_MARKER]' ...
"""
import hashlib, struct, sys
from pathlib import Path


def main():
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    prefix = tk.encode('[gMASK]<sop>', add_special_tokens=False).ids
    lines = []
    for spec in sys.argv[3:]:
        name, src, t, *marker = spec.split('|', 3)
        t = int(t)
        raw = Path(src).read_bytes()
        text = raw.decode('utf-8').replace('\r\n', '\n')
        if marker:
            i = text.find(marker[0])
            if i < 0:
                raise SystemExit(f'{name}: start marker not found')
            text = text[i:]
        ids = prefix + tk.encode(text, add_special_tokens=False).ids
        if len(ids) < t + 1:
            raise SystemExit(f'{name}: only {len(ids)} tokens < {t + 1}')
        ids = ids[:t + 1]
        data = struct.pack(f'<{len(ids)}I', *ids)
        f = out / f'g4-{name}-{t}.u32'
        f.write_bytes(data)
        lines.append(f'{f.name} tokens {len(ids)} (T {t}) sha256 {hashlib.sha256(data).hexdigest()} | source '
                     f'{Path(src).name} sha256 {hashlib.sha256(raw).hexdigest()} start {marker[0] if marker else "<file start>"!r} '
                     f'| head {tk.decode(ids[:16])!r} | scored targets {tk.decode(ids[t - 63:t + 1])[:80]!r}...')
        print(lines[-1])
    (out / 'g4-manifest.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
