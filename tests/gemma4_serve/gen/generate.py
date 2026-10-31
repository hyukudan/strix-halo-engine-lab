"""Generate small native-test fixtures with the official local HF tokenizer assets.

Usage: python generate.py MODEL_DIR FIXTURE_DIR
Only generation needs transformers/tokenizers; native tests do not need Python.
"""
import hashlib
import json
from pathlib import Path
import random
import sys

from transformers import AutoTokenizer

model, dest = map(Path, sys.argv[1:])
dest.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
rng = random.Random(40521)
languages = [
    "The quick brown fox walks through the city today",
    "El niño pidió café y encontró una canción en el jardín",
    "Le cœur connaît déjà cette forêt près de la rivière",
    "Grüße aus München für die große Straße und schöne Häuser",
    "Perché questa città è più bella sotto il cielo azzurro",
    "A informação está disponível amanhã para os cidadãos",
    "机器学习是人工智能的一个分支 今天城市里下雨了",
    "東京で雨の日に楽しめる観光スポット こんにちは世界",
    "한국어 텍스트입니다 오늘은 좋은 날씨입니다",
    "Привет мир сегодня мы изучаем новые языки и числа",
    "مرحبا بالعالم هذه تجربة جديدة في المدينة اليوم",
    "नमस्ते दुनिया आज हम नई भाषाएँ सीख रहे हैं",
    "สวัสดีครับ วันนี้อากาศดีมาก ภาษาไทยน่าสนใจ",
    "def f(x): return x ** 2 # comment {\"value\": true} std::vector<int>",
    "🙂 🚀 👍🏽 👨‍👩‍👧 🏳️‍🌈 🧑🏾‍💻 🇪🇸 ❤️ 1️⃣",
]
sentences = []
for i in range(1000):
    words = languages[i % len(languages)].split()
    chosen = [rng.choice(words) for _ in range(rng.randint(3, 20))]
    sep = rng.choice([" ", "  ", "\n", "\t", "\u00a0", "\u3000", " / "])
    s = rng.choice(["", " ", "\n\t"]) + sep.join(chosen) + rng.choice([".", "!?", "\n", "  "])
    if i % 7 == 0:
        s += str(rng.randrange(10**12))
    if i % 11 == 0:
        s += " a\u0301 ﬁ ﬃ ＡＢＣ ▁ <bos><eos><|channel>thought\n<channel|>"
    ids = tok.encode(s, add_special_tokens=False)
    sentences.append(dict(text=s, ids=ids, decoded=tok.decode(ids)))

specials = list(tok.backend_tokenizer.get_added_tokens_decoder().values())
edges = [str(t) for t in specials] + ["<unused0>", "<0x41>", "▁▁hello ▁", "a" * 4096,
    " " * 100 + "\n" * 80 + "\t" * 80, "\r\n\r\n\u2028\u2029", "\ufeff\u200b\u200d",
    "".join(chr(i) for i in range(128)), "\U0010ffff\U00010000\ue000\ufffd", "é e\u0301 ① ﬁ Ａ"]
for s in edges:
    ids = tok.encode(s, add_special_tokens=False)
    sentences.append(dict(text=s, ids=ids, decoded=tok.decode(ids)))

def chat(name, messages, **kwargs):
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    return dict(name=name, messages=messages, kwargs=kwargs, rendered=rendered,
                ids=tok.encode(rendered, add_special_tokens=False))

tools = [{"type": "function", "function": {"name": "inspect", "description": "Inspect a structured value",
    "parameters": {"type": "object", "properties": {
        "z": {"type": "array", "items": {"type": ["string", "null"], "description": "Values"}},
        "Alpha": {"type": "object", "description": "Nested", "nullable": True, "properties": {
            "flag": {"type": "boolean"}, "count": {"type": "integer"}}, "required": ["flag"]},
        "b": {"type": "string", "enum": ["x", "y"]}}, "required": ["Alpha"]},
    "response": {"type": "object", "description": "Result"}}}]
call = {"id": "t0", "type": "function", "function": {"name": "inspect", "arguments": {
    "z": ["a\\b", 'a"b', None, 1.5], "Alpha": {"count": 2, "flag": True}}}}
user = {"role": "user", "content": "Inspect the data."}
assistant = {"role": "assistant", "content": "  Working.  ", "reasoning_content": "Plan first.", "tool_calls": [call]}
cases = [
    chat("developer_parts", [{"role": "developer", "content": [{"type": "text", "text": "  One  "},
        {"type": "text", "text": "Two"}]}, user]),
    chat("nested_schema", [user], tools=tools),
    chat("call_pending", [user, assistant], enable_thinking=True),
    chat("tool_id_and_continuation", [user, assistant, {"role": "tool", "tool_call_id": "t0", "name": "wrong",
        "content": [{"type": "text", "text": "A "}, {"type": "text", "text": "B"}]},
        {"role": "assistant", "content": "Done."}], tools=tools, enable_thinking=True),
    chat("strip_earlier_thoughts", [user, {"role": "assistant", "content": "<|channel>thought\nsecret<channel|>  OK "},
        {"role": "user", "content": "Next"}], enable_thinking=False),
    chat("reasoning_guard", [user, assistant, {"role": "tool", "tool_call_id": "t0", "content": "ok"}, user],
        preserve_thinking=False),
    chat("preserve_call_reasoning", [user, assistant, {"role": "tool", "tool_call_id": "t0", "content": "ok"}, user],
        preserve_thinking=True),
    chat("legacy_response", [user, {"role": "assistant", "tool_calls": [call], "tool_responses": [
        {"name": "inspect", "response": {"success": True, "value": [1, None]}}]}], enable_thinking=True),
    chat("spoken_system", [{"role": "system", "content": "  Be kind. "}, user], conversational_mode=True, enable_thinking=True),
    chat("empty_assistant", [user, {"role": "assistant", "content": None}]),
    chat("consecutive_assistants", [user, {"role": "assistant", "content": "A"}, {"role": "assistant", "content": "B"}]),
]
for i in range(32):
    keys = ["z", "A", "a", "b", "Ä", "ä", "İ", "i", "ΟΣ", "Ος", "Οσ", "Σ", "ΜΣ\u0345Α", "Я", "я", "𐐀", "𐐨", "中文"]
    rng.shuffle(keys)
    numbers = [1.0, -0.0, 1e-4, 1e-5, 1e15, 1e16, 1e20, 1.7976931348623157e308,
               5e-324, 2.2250738585072014e-308, 1.2345678901234567, 100000000000000000000001]
    data = {k: rng.choice(numbers) for k in keys[:rng.randint(3, len(keys))]}
    c = {"type": "function", "function": {"name": "inspect", "arguments": data}}
    cases.append(chat(f"unicode_sort_numbers_{i}", [user, {"role": "assistant", "tool_calls": [c]}]))

# Include arbitrary-byte fallback groups to check decoder error replacement, not just encodable text.
byte_ids = [tok.convert_tokens_to_ids(f"<0x{i:02X}>") for i in range(256)]
decode = []
for raw in [list(range(256)), [0x41, 0xC3, 0xA9], [0x41, 0xFF, 0x42], [0xE2, 0x82], [0xF0, 0x9F, 0x99, 0x82]]:
    ids = [byte_ids[b] for b in raw]
    decode.append(dict(ids=ids, decoded=tok.decode(ids)))
for i in range(128):
    sample_ids = [rng.randrange(tok.vocab_size) for _ in range(32)]
    decode.append(dict(ids=sample_ids, decoded=tok.decode(sample_ids)))
no_special = []
for s in ["<bos><eos>", "hi <|channel>thought\n世界<channel|>", "<|tool_call>call:f{}<tool_call|>"]:
    # encode_special_tokens=True asks the backend to treat even added special tokens as ordinary text.
    tok.backend_tokenizer.encode_special_tokens = True
    try:
        sample_ids = tok.backend_tokenizer.encode(s, add_special_tokens=False).ids
    finally:
        tok.backend_tokenizer.encode_special_tokens = False
    no_special.append(dict(text=s, ids=sample_ids))
provenance = {"generator": "transformers 5.21 / official tokenizer assets", "seed": 40521,
    "sha256": {f: hashlib.sha256((model / f).read_bytes()).hexdigest()
               for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]}}
(dest / "extended.json").write_text(json.dumps(dict(provenance=provenance, encode=sentences, chat=cases, decode=decode, encode_no_special=no_special),
    ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(f"Generated {len(sentences)} encode, {len(cases)} chat, {len(decode)} decode cases")
