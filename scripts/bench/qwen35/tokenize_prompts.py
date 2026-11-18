#!/usr/bin/env python3
"""Chat-template + tokenize a prompt set into <name>.ids (comma-separated ids) for strix-cuda-qwen35-spec --prompt-file.
Thinking is off unless the prompt file sets "thinking": true (chart/think.json).
Usage: python tokenize_prompts.py MODEL_DIR [prompts_dir]   (MODEL_DIR holds tokenizer.json + chat_template.jinja)
Sets: prompts/ (copy/json/prose classes), chart/ (short inputs: English prose, Japanese prose, code, thinking on).
Renders the model's own chat_template.jinja (jinja2) and encodes with the tokenizers library (no transformers needed)."""
import json, sys, pathlib

import jinja2
import jinja2.ext
from tokenizers import Tokenizer

model = pathlib.Path(sys.argv[1])
d = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else pathlib.Path(__file__).parent / "prompts")
tok = Tokenizer.from_file(str(model / "tokenizer.json"))


def raise_exception(msg):
    raise ValueError(msg)


class GenerationBlock(jinja2.ext.Extension):
    """{% generation %}...{% endgeneration %} (assistant-mask marker of the HF template dialect): render the body as is."""
    tags = {"generation"}

    def parse(self, parser):
        next(parser.stream)
        return parser.parse_statements(("name:endgeneration",), drop_needle=True)


env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols", GenerationBlock])
env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False, **kw)
template = env.from_string((model / "chat_template.jinja").read_text(encoding="utf-8"))
for name in json.loads((d / "manifest.json").read_text(encoding="utf-8")):
    spec = json.loads((d / name).read_text(encoding="utf-8"))
    text = template.render(messages=spec["messages"], add_generation_prompt=True,
                           enable_thinking=bool(spec.get("thinking", False)), raise_exception=raise_exception)
    ids = tok.encode(text, add_special_tokens=False).ids
    (d / name.replace(".json", ".ids")).write_text(",".join(map(str, ids)), encoding="utf-8")
    print(name, len(ids), "tokens")
