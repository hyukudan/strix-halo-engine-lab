#!/usr/bin/env python3
"""Render chat-template test cases with Jinja2 for tests/data/serve/chat_template_cases.json.

The template text is read from the model file's metadata (tokenizer.chat_template); the cases
file stores only the inputs and the rendered prompts. The environment follows the usual chat
template conventions: sandboxed, trim_blocks/lstrip_blocks, loop controls, a tojson filter that
is json.dumps(ensure_ascii=False) with key order preserved, and raise_exception.

usage: gen_chat_template_cases.py TEMPLATE.jinja OUT.json
"""
import json
import sys

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather in a city (°C).",
           "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. Málaga"},
                                                          "days": {"type": "integer"}},
                          "required": ["city"]}}}
SEARCH = {"type": "function", "function": {"name": "search", "description": "Search the web.",
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "filters": {"type": "object"}},
                         "required": ["query"]}}}

CASES = [
    {"name": "user-default", "messages": [{"role": "user", "content": "Hello!"}]},
    {"name": "system-no-thinking", "messages": [{"role": "system", "content": "  You are terse.  "},
                                                {"role": "user", "content": "Hi\n"}], "enable_thinking": False},
    {"name": "merged-systems-low", "messages": [{"role": "system", "content": "Rule one."}, {"role": "developer", "content": "Rule two."},
                                                {"role": "user", "content": "Go"}], "reasoning_effort": "low"},
    {"name": "effort-medium", "messages": [{"role": "user", "content": "Q"}], "reasoning_effort": "medium"},
    {"name": "effort-high", "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}], "reasoning_effort": "high"},
    {"name": "parts", "messages": [{"role": "user", "content": [{"type": "text", "text": "first part"}, {"type": "text", "text": " second"}]}]},
    {"name": "history", "messages": [{"role": "user", "content": "One?"},
                                     {"role": "assistant", "content": "Uno.", "reasoning_content": "  translate  "},
                                     {"role": "user", "content": "Two?"}]},
    {"name": "history-no-preserve", "preserve_thinking": False,
     "messages": [{"role": "user", "content": "One?"}, {"role": "assistant", "content": "Uno.", "reasoning_content": "translate"},
                  {"role": "user", "content": "Two?"}, {"role": "assistant", "content": "Dos.", "reasoning_content": "again"}],
     "add_generation_prompt": False},
    {"name": "tools-call", "tools": [WEATHER, SEARCH],
     "messages": [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "Weather in Málaga for 3 days?"},
                  {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
                      "name": "get_weather", "arguments": {"city": "Málaga", "days": 3}}}]},
                  {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 21}"},
                  {"role": "user", "content": "And search for it."}]},
    {"name": "tools-two-calls-two-results", "tools": [WEATHER, SEARCH],
     "messages": [{"role": "user", "content": "Do both"},
                  {"role": "assistant", "content": "Calling tools now.", "tool_calls": [
                      {"function": {"name": "search", "arguments": {"query": "a b", "filters": {"lang": "es", "n": [1, 2]}}}},
                      {"function": {"name": "get_weather", "arguments": {"city": "Oslo"}}}]},
                  {"role": "tool", "content": "r1"}, {"role": "tool", "content": "r2"}], "enable_thinking": False},
    {"name": "tool-after-system", "messages": [{"role": "system", "content": "S"}, {"role": "tool", "content": "orphan"},
                                               {"role": "user", "content": "U"}]},
    {"name": "tools-only-no-system", "tools": [SEARCH], "messages": [{"role": "user", "content": "Find x"}], "enable_thinking": False},
    {"name": "null-content", "messages": [{"role": "user", "content": None}, {"role": "assistant", "content": None},
                                          {"role": "user", "content": "again"}]},
    {"name": "unicode-whitespace", "messages": [{"role": "user", "content": "　 leading ideographic space "}]},
]


def main(template_path, out_path):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters['tojson'] = tojson
    env.globals['raise_exception'] = raise_exception
    template = env.from_string(open(template_path, encoding='utf-8').read())
    out = []
    for case in CASES:
        kwargs = {k: v for k, v in case.items() if k not in ('name', 'messages', 'tools')}
        kwargs.setdefault('add_generation_prompt', True)
        rendered = template.render(messages=case['messages'], tools=case.get('tools'), **kwargs)
        entry = dict(case)
        entry['expected'] = rendered
        out.append(entry)
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write('\n')
    print(len(out), 'cases')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
