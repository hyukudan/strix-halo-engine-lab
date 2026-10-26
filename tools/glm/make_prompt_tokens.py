"""Prompt token files (.u32 little-endian) for the GLM engine benches and gates, from the model's own tokenizer.json.

tok-<N>.u32 = '[gMASK]<sop>' (the chat template's prefix) + the raw lab bench text, truncated to exactly N tokens, so
ms/token at context N compares with the other engines' bench-prompt runs. Also writes chat-<N>.u32 (the full chat
template rendering of the same text as one user turn, generation prompt appended, not truncated) for sane-text decode.
usage: make_prompt_tokens.py TOKENIZER_JSON CHAT_TEMPLATE OUT_DIR TEXT:N [TEXT:N ...]
"""
import struct, sys
from pathlib import Path


def main():
    tok_json, template, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    from tokenizers import Tokenizer
    import jinja2
    tk = Tokenizer.from_file(tok_json)
    prefix = tk.encode('[gMASK]<sop>', add_special_tokens=False).ids
    tmpl = jinja2.Environment(trim_blocks=False, lstrip_blocks=False, extensions=['jinja2.ext.loopcontrols']).from_string(Path(template).read_text(encoding='utf-8'))
    out.mkdir(parents=True, exist_ok=True)
    for spec in sys.argv[4:]:
        path, n = spec.rsplit(':', 1)
        n = int(n)
        text = Path(path).read_text(encoding='utf-8', errors='replace')
        ids = prefix + tk.encode(text, add_special_tokens=False).ids
        if len(ids) < n:
            print(f'{path}: only {len(ids)} tokens < {n}; writing {len(ids)}')
        ids = ids[:n]
        (out / f'tok-{n}.u32').write_bytes(struct.pack(f'<{len(ids)}I', *ids))
        chat = tk.encode(tmpl.render(messages=[{'role': 'user', 'content': text}], add_generation_prompt=True),
                         add_special_tokens=False).ids
        (out / f'chat-{n}.u32').write_bytes(struct.pack(f'<{len(chat)}I', *chat))
        print(f'tok-{n}.u32 {len(ids)} tokens (prefix {prefix}), chat-{n}.u32 {len(chat)} tokens, '
              f'decode check: {tk.decode(ids[:24])!r}')


if __name__ == '__main__':
    main()
