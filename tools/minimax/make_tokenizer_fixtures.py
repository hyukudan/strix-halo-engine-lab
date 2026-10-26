#!/usr/bin/env python3
"""Generate the MiniMax tokenizer / chat-template fixtures checked by tests/minimax_tokenizer_tests.cpp.

Reference implementations: HuggingFace `tokenizers` (Tokenizer.from_file(tokenizer.json)) for
encode/decode/NFC and `transformers` apply_chat_template(tokenize=False) with the model's
chat_template.jinja for the chat renderings.

usage: make_tokenizer_fixtures.py --model-dir path/to/fixture \
           [--oracle-tokens path/to/tokens.u32] \
           [--random 4000] [--seed 1234] OUT.json

Output (JSON, UTF-8):
  encode: [{text, ids, ids_plain, decoded, decoded_skip}]   ids_plain = special tokens not parsed
  decode: [{ids, text, text_skip}]                           arbitrary id sequences (broken UTF-8, added, out of range)
  nfc:    [{text, nfc}]                                      HF normalizers.NFC
  chat:   [{name, messages_json, tools_json, options, text, ids}]
  oracle: {source, ids, text, reencoded_ids, reproduces}
"""
import argparse
import json
import os
import random
import struct
import sys

from tokenizers import Tokenizer
from tokenizers.normalizers import NFC

FIXED_TEXTS = [
    "", " ", "  ", "\n", "\n\n", "\r\n", " \n", "\t", "a", "Hello world", "Hello, world!", " Hello", "  Hello  ",
    "The quick brown fox jumps over the lazy dog.",
    "I'm here, you're there, they've gone, we'll see, he'd know, it's fine, don't.",
    "I'M HERE, YOU'RE THERE, THEY'VE GONE, WE'LL SEE, HE'D KNOW, IT'S FINE, DON'T.",
    "I'm You'Re They'vE We'Ll He'D It'S Don'T", "o'clock rock'n'roll l'été 'tis 'S 'sup", "'s's's 's 't 're 've 'm 'll 'd 'x",
    "Don'ſ stop", "camelCaseIdentifier snake_case_identifier SCREAMING_CASE PascalCase iPhone eBay McDonald's",
    "ALLCAPS WORD", "HTTPServerError XMLHttpRequest", "ÀÉÎÕÜ àéîõü ÑñÇç",
    # Spanish, NFC and NFD forms
    "¿Dónde está la estación? ¡Mañana será otro día! El niño comió piñas y cigüeñas.",
    "¿Do\u0301nde esta\u0301 la estacio\u0301n? ¡Man\u0303ana sera\u0301 otro di\u0301a! El nin\u0303o comio\u0301 pin\u0303as y cigu\u0308en\u0303as.",
    "Canción, corazón, acción, pingüino, ÁRBOL, Ñandú",
    "Cancio\u0301n, corazo\u0301n, accio\u0301n, pingu\u0308ino, A\u0301RBOL, N\u0303andu\u0301",
    "e\u0301\u0301 a\u0328\u0301 o\u031b\u0323 \u1e9b\u0323 A\u030a \u212b \u2126 \u0390",
    # CJK, Hangul, other scripts
    "日本語のテキストです。東京は大きい都市です。", "中文文本测试，包括标点符号！还有数字123和English混合。",
    "한국어 텍스트입니다. 서울은 큰 도시입니다.", "\u1100\u1161\u11a8 \u1112\u1161\u11ab\u1100\u1173\u11af",
    "Русский текст: Привет, мир! ПРИВЕТ", "Ελληνικά: Καλημέρα κόσμε", "עברית: שלום עולם", "العربية: مرحبا بالعالم",
    "हिन्दी: नमस्ते दुनिया", "ไทย: สวัสดีชาวโลก", "Tiếng Việt: Xin chào thế giới", "ქართული ენა", "Ὀδυσσεύς",
    # emoji
    "Emoji: 😀😃😄 👍🏽 👨‍👩‍👧‍👦 🇪🇸🇯🇵 ❤️ ✨🚀", "🙂🙂🙂", "a😀b", " 😀", "x\ufe0f",
    # code / JSON
    "def f(x):\n    return x ** 2  # square\n\nprint(f(3))\n",
    "for (int i = 0; i < n; ++i) {\n\tsum += a[i];\n}\n",
    '{"name": "MiniMax", "version": 3.1, "tags": ["a", "b"], "nested": {"k": null, "t": true}}',
    "<html><body><div class=\"x\">Hi</div></body></html>", "SELECT * FROM users WHERE id = 42;",
    "path/to/file.txt ./a/b ../c //comment /* block */ http://example.com/a/b?c=d&e=f#g",
    "a!/b ?// .\n/ ;\r\n/x !!//\n\n// )/\r", "...\n\n\n---\n***\n", "&&|| ->=> ::<>",
    # numbers
    "1 12 123 1234 12345 123456 1234567", "3.14159 2,718 1e10 -42 +7 0x1F 1_000_000", "١٢٣٤٥ ١٢٣ ۴۵۶ ४५६ ໑໒໓", "²³¹ ½ ⅓ Ⅻ ⑳",
    "year2024 v1.2.3 abc123def 12ab34",
    # whitespace runs
    "a  b   c    d", "a \n b \n\n c", "end with spaces   ", "   start with spaces", "tabs\t\t\there", "mixed \t \n \r\n\t x",
    "\n\n\n\n", "    \n    \n", "a\u00a0b\u3000c\u2028d\u2029e\u202ff", "x\x1cy\x1dz\x1e\x1f", "\u2000\u2001\u2002 x", "\u000b\u000c x",
    # special tokens inline
    "]~b]user\nHello[e~[\n", "text<think>reasoning</think>answer", "<mm:think>t</mm:think>done", "]<]minimax[>[<tool_call>x",
    "]<]start of speech[>[ hi ]<]end of speech[>[", "<fim_prefix>a<fim_suffix>b<fim_middle>", "]~b]]~b][e~[[e~[",
    "]~!b[]~b]system\n", "<|content_altered_placeholder|>", "]!p~[]!d~[", "x]~b", "<think", "</think>>", "<<think>>",
    "]<]image[>[]<]video[>[]<]frame[>[", "a]<]minimax[>[b]<]minimax[>[c", "<tool_call></tool_call>",
    # misc unicode
    "\u0897\u1c89\u10d40 Unicode 16", "\u07fd\u0334 \u08d3\u0345", "ſtraße STRASSE ß ẞ", "ﬁ ﬂ ﬃ", "Ǆ ǅ ǆ ǈ",
    "\u0300start mark", " \u0301", "\u200b\u200c\u200d zero width", "\ue000 private", "\U0001d400\U0001d41a math",
]


def oracle_section(tok, path, count):
    with open(path, 'rb') as f:
        data = f.read(4 * count)
    ids = list(struct.unpack('<%dI' % (len(data) // 4), data))
    text = tok.decode(ids, skip_special_tokens=False)
    re_ids = tok.encode(text, add_special_tokens=False).ids
    return {'source': os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path), 'ids': ids, 'text': text,
            'reencoded_ids': re_ids, 'reproduces': re_ids == ids}


POOLS = [
    'abcdefghijklmnopqrstuvwxyz', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', '0123456789', ' ' * 6 + '\n\t\r',
    "'''sStTrReEvVmMlLdD", '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~', '//\n\n', 'áéíóúñüÁÉÍÓÚÑÜçàèìòù',
    '\u0300\u0301\u0302\u0303\u0308\u0323\u0327\u0328\u031b\u0334\u0345\u05b0\u093c\u3099\u302a',
    '日本語中文字漢字テキストカタカナひらがな', '한국어텍스트', '\u1100\u1161\u11a8\u1112\u1173\u11af', 'привет ПРИВЕТ',
    'αβγ ΑΒΓ', 'مرحبا', 'नमस्ते', 'สวัสดี', '😀👍🏽👨‍👩‍👧🇪🇸❤️\ufe0f\u200d', '١٢٣४५६²½Ⅻ', '\u00a0\u3000\u2028\u2029\u202f\u2000\u0085\x1c\x1f',
    'ſǅǈǋᾈᾙ', '\u0897\u1c89\u1c8a\ua7cb\ua7dc\U00010d40\U00010d4a\U000116d0', '\u07fd\u08d3\u0d3b\u0d3c\u0eba\u0c3c',
    '\ue000\U000f0000\ufffd\u0378\U0001d400', '\u0b47\u0b3e\u0dd9\u0dcf\u1025\u102e\u0915\u0f73\u0f75\u0f81',
]
SPECIAL_SNIPPETS = [']~b]', '[e~[', '<think>', '</think>', '<mm:think>', '</mm:think>', ']<]minimax[>[', '<tool_call>', ']<]image[>[',
                    ']!p~[', '<fim_prefix>', ']~!b[', '<|content_altered_placeholder|>', ']<]start of video[>[']


def random_texts(rng, n):
    out = []
    for _ in range(n):
        pools = rng.sample(POOLS, rng.randint(1, 4))
        length = rng.randint(1, 60)
        s = []
        for _ in range(length):
            if rng.random() < 0.03:
                s.append(rng.choice(SPECIAL_SNIPPETS))
            else:
                p = rng.choice(pools)
                s.append(rng.choice(p))
        out.append(''.join(s))
    return out


CHAT_CASES = [
    ('basic', '[{"role":"user","content":"Hello!"}]', None, {}),
    ('no_gen_prompt', '[{"role":"user","content":"Hello!"}]', None, {'add_generation_prompt': False}),
    ('system_user', '[{"role":"system","content":"You are a pirate."},{"role":"user","content":"¿Qué hora es?"}]', None, {}),
    ('thinking_enabled', '[{"role":"user","content":"Solve 2+2"}]', None, {'thinking_mode': 'enabled'}),
    ('thinking_disabled', '[{"role":"user","content":"Hi"}]', None, {'thinking_mode': 'disabled'}),
    ('thinking_adaptive', '[{"role":"user","content":"Hi"}]', None, {'thinking_mode': 'adaptive'}),
    ('thinking_other', '[{"role":"user","content":"Hi"}]', None, {'thinking_mode': 'weird'}),
    ('multi_turn', '[{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello! How can I help?"},'
                   '{"role":"user","content":"Tell me a joke\\nplease"}]', None, {}),
    ('assistant_reasoning_field', '[{"role":"user","content":"Q"},{"role":"assistant","content":"A","reasoning_content":"thinking..."},{"role":"user","content":"Q2"}]', None, {}),
    ('assistant_empty_reasoning_field', '[{"role":"user","content":"Q"},{"role":"assistant","content":"A</mm:think>B","reasoning_content":""}]', None, {'add_generation_prompt': False}),
    ('assistant_think_in_content', '[{"role":"user","content":"Q"},{"role":"assistant","content":"<mm:think>\\nstep 1\\nstep 2\\n</mm:think>\\n\\nThe answer."},{"role":"user","content":"Q2"}]', None, {}),
    ('assistant_think_no_begin', '[{"role":"user","content":"Q"},{"role":"assistant","content":"just reasoning\\n</mm:think>answer\\n"}]', None, {}),
    ('assistant_two_ends', '[{"role":"user","content":"Q"},{"role":"assistant","content":"a<mm:think>b<mm:think>c</mm:think>d</mm:think>\\ne\\n"}]', None, {}),
    ('assistant_null_content', '[{"role":"user","content":"Q"},{"role":"assistant","content":null}]', None, {}),
    ('root_system', '[{"role":"root","content":"High priority rules.","current_date":"2026-10-24","current_location":"Madrid"},{"role":"system","content":"Dev rules."},{"role":"user","content":"Hi"}]', None, {}),
    ('root_empty', '[{"role":"root","content":""},{"role":"user","content":"Hi"}]', None, {}),
    ('root_only_date', '[{"role":"root","content":"R","current_date":"2026-01-01"},{"role":"user","content":"Hi"}]', None, {}),
    ('developer_role', '[{"role":"developer","content":"Dev msg"},{"role":"user","content":"Hi"},{"role":"system","content":"mid system"},{"role":"developer","content":""}]', None, {}),
    ('model_identity', '[{"role":"user","content":"Hi"}]', None, {'model_identity': 'You are Strix.'}),
    ('model_id_effort', '[{"role":"user","content":"Hi"}]', None, {'template_model_id': 'mm-3.1-flash', 'reasoning_effort': 'high', 'effort': 'low'}),
    ('effort_only', '[{"role":"user","content":"Hi"}]', None, {'effort': 'medium'}),
    ('content_parts', '[{"role":"user","content":[{"type":"text","text":"Look: "},{"type":"image","image":"x.png"},{"type":"text","text":" and "},{"type":"video"},{"type":"frame"},"plain",{"type":"audio"}]}]', None, {}),
    ('unknown_role', '[{"role":"user","content":"a"},{"role":"function","content":"ignored"},{"role":"user","content":"b"}]', None, {}),
    ('tools_and_calls',
     '[{"role":"system","content":"Use tools."},{"role":"user","content":"Weather in Madrid and Tokyo?"},'
     '{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"get_weather","arguments":{"city":"Madrid","units":"celsius","days":3,"ratio":0.5,"big":1e5,"neg":-2.50,"tiny":1.5e-7,"huge":12345678901234567890,"flag":true,"off":false,"none":null,"list":[1,"two",{"k":"v","z":null},[true,null]],"obj":{"a":{"b":"c"}},"uni":"東京 ñ"}}},'
     '{"type":"function","function":{"name":"get_weather","arguments":{"city":"Tokyo"}}}]},'
     '{"role":"tool","tool_call_id":"c1","content":"{\\"temp\\": 21}"},{"role":"tool","content":"{\\"temp\\": 15}"},'
     '{"role":"assistant","content":"Madrid 21, Tokyo 15."},{"role":"user","content":"Thanks"}]',
     '[{"type":"function","function":{"name":"get_weather","description":"Get the weather — «quoted» \\"x\\" \\\\ \\n tab\\t ñ 東京","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer","minimum":1,"maximum":1.5e1},"units":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]}}},'
     '{"type":"function","function":{"name":"noop","parameters":{}}}]', {}),
    ('tool_call_flat', '[{"role":"user","content":"x"},{"role":"assistant","content":"Calling.","tool_calls":[{"name":"f","arguments":{"a":1}}]},{"role":"tool","content":[{"type":"text","text":"res1"},{"output":"out2"},{"type":"image"},{"type":"video"},{"type":"frame"},"str3",{"output":42}]},{"role":"user","content":"y"}]', None, {}),
    ('tools_no_calls', '[{"role":"user","content":"hi"}]', '[{"type":"function","function":{"name":"search","description":"Search","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}]', {'thinking_mode': 'enabled'}),
    ('empty_tools', '[{"role":"user","content":"hi"}]', '[]', {}),
    ('special_in_content', '[{"role":"user","content":"Say ]~b] and [e~[ and <think>x</think>"}]', None, {}),
    ('multi_tool_turns', '[{"role":"user","content":"a"},{"role":"assistant","content":null,"tool_calls":[{"function":{"name":"t1","arguments":{"x":[]}}}]},{"role":"tool","content":"r1"},'
                         '{"role":"assistant","reasoning_content":"think more","content":"","tool_calls":[{"function":{"name":"t2","arguments":{"y":{}}}}]},{"role":"tool","content":"r2"},{"role":"assistant","content":"done"}]', None, {'add_generation_prompt': False}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--oracle-tokens')
    ap.add_argument('--oracle-count', type=int, default=4096)
    ap.add_argument('--random', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=1234)
    a = ap.parse_args()
    tok = Tokenizer.from_file(os.path.join(a.model_dir, 'tokenizer.json'))
    rng = random.Random(a.seed)

    texts = list(FIXED_TEXTS)
    texts += [t + ' ' + t.upper() for t in FIXED_TEXTS[:40] if t]
    texts += [str(n) for n in [0, 7, 42, 999, 1000, 12345, 123456, 1234567, 99999999]]
    texts += [''.join(rng.choice('0123456789') for _ in range(k)) for k in range(1, 7) for _ in range(3)]
    texts += random_texts(rng, a.random)

    encode = []
    for t in texts:
        enc = tok.encode(t, add_special_tokens=False).ids
        tok.encode_special_tokens = True
        plain = tok.encode(t, add_special_tokens=False).ids
        tok.encode_special_tokens = False
        encode.append({'text': t, 'ids': enc, 'ids_plain': plain, 'decoded': tok.decode(enc, skip_special_tokens=False),
                       'decoded_skip': tok.decode(enc, skip_special_tokens=True)})

    vocab_size = tok.get_vocab_size(with_added_tokens=True)
    decode = []
    for _ in range(600):
        ids = []
        for _ in range(rng.randint(1, 12)):
            r = rng.random()
            if r < 0.6:
                ids.append(rng.randrange(0, 200000))
            elif r < 0.75:
                ids.append(rng.randrange(200000, vocab_size))
            elif r < 0.95:
                ids.append(rng.randrange(0, 256 + 1000))  # byte tokens: broken UTF-8
            else:
                ids.append(rng.choice([vocab_size, vocab_size + 5, 300000]))
        decode.append({'ids': ids, 'text': tok.decode(ids, skip_special_tokens=False), 'text_skip': tok.decode(ids, skip_special_tokens=True)})

    hf_nfc = NFC()
    nfc = []
    for _ in range(3000):
        s = ''.join(rng.choice(rng.choice(POOLS[7:])) for _ in range(rng.randint(1, 12)))
        nfc.append({'text': s, 'nfc': hf_nfc.normalize_str(s)})

    from transformers import PreTrainedTokenizerFast
    hf = PreTrainedTokenizerFast(tokenizer_file=os.path.join(a.model_dir, 'tokenizer.json'))
    template = open(os.path.join(a.model_dir, 'chat_template.jinja'), encoding='utf-8').read()
    chat = []
    for name, messages_json, tools_json, options in CHAT_CASES:
        kwargs = dict(options)
        add_gen = kwargs.pop('add_generation_prompt', True)
        text = hf.apply_chat_template(json.loads(messages_json), tools=json.loads(tools_json) if tools_json else None,
                                      chat_template=template, tokenize=False, add_generation_prompt=add_gen, **kwargs)
        chat.append({'name': name, 'messages_json': messages_json, 'tools_json': tools_json, 'options': options, 'text': text,
                     'ids': tok.encode(text, add_special_tokens=False).ids})

    out = {'tokenizers_version': __import__('tokenizers').__version__,
           'transformers_version': __import__('transformers').__version__,
           'vocab_size': vocab_size, 'bos': tok.token_to_id(']~b]'), 'eos': tok.token_to_id('[e~['),
           'encode': encode, 'decode': decode, 'nfc': nfc, 'chat': chat}
    if a.oracle_tokens:
        out['oracle'] = oracle_section(tok, a.oracle_tokens, a.oracle_count)
    with open(a.out, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(out, f, ensure_ascii=False, separators=(',', ':'))
        f.write('\n')
    print('%d encode, %d decode, %d nfc, %d chat cases%s -> %s' % (
        len(encode), len(decode), len(nfc), len(chat),
        ', oracle reproduces=%s' % out['oracle']['reproduces'] if 'oracle' in out else '', a.out))


if __name__ == '__main__':
    sys.exit(main())
