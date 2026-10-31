# Gemma 4 text serving adapter

This host module implements the tokenizer and text conversation protocol from
[NVIDIA's Nemotron 3.5 Bolt E4B BF16 assets](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16).
The native renderer follows the published Nemotron Gemma 4 Jinja template,
including its assistant-content ordering and reasoning-off generation suffix.
It is a translation of those model semantics, not a newly invented template.
The tokenizer, weights and full template files are loaded or inspected externally;
they are not bundled here. The official model card identifies OpenMDW-1.1 for
the model materials; retain its [license](notices/OPENMDW-LICENSE.txt) and applicable
notices when distributing adopted model material.

`Tokenizer::load("assets/tokenizer.json")` validates the supported asset contract:
262144 vocabulary entries, ranked Unicode BPE, space-to-U+2581 normalization,
the resulting no-op space split, byte fallback, added special tokens and the
identity postprocessor. Unsupported asset settings fail loading. Raw encoding
does not add BOS/EOS. Chat rendering owns BOS once and the turn boundaries.
U+2581 normalizes to a space on decode, so normalized round trips are deliberate.
Consecutive invalid byte-token groups decode as one replacement per byte, as
specified by the asset decoder. `token_bytes` returns raw pieces, possibly inside
a UTF-8 scalar. The shared UTF-8 stream decoder can assemble valid scalar runs;
it does not reproduce the asset's whole-group invalid-byte replacement rule.

The ranked merge queue is exact by default. Set `STRIX_G4_TOKENIZER_HEAP=0` before
loading to select the literal pair-scan reference; both routes have the same ID
fixtures. Other environment values fail explicitly. This module changes no
model arithmetic or numerical identity.

`ChatInput` supports an initial system/developer message, text content and text
parts, user/model history, thinking on/off, preserved tool reasoning,
conversational mode, function declarations, nested schema objects/arrays/enums,
assistant calls, consecutive tool results and embedded native tool responses.
String arguments are rejected: deserialize them to JSON objects first. Thinking
defaults to false, as in the actual template asset. The model card's prose default
does not override the asset. Unspecified generation prompts are enabled.
Schema types are uppercase in declarations. Argument and declaration ordering
matches Jinja's case-insensitive, stable `dictsort`; Python-compatible number
formatting preserves the template's integer/float distinction.

Unknown template kwargs (including a budget passed as a template kwarg), unsupported
roles, late system/developer turns, orphan tool results, non-text parts, non-object
arguments, unsupported response declaration types, ambiguous quote delimiters and
keys that cannot be represented unquoted fail explicitly. Runtime tool-selection
policies, constrained generation and multimodal execution are outside this module.

`OutputParser` owns one request's incremental state. Feed it decoded output bytes;
it publishes complete UTF-8 content/reasoning deltas and complete tool calls.
Tool arguments remain JSON objects, with literal backslashes and quotes inside
the model's `<|"|>` delimiters. Calls have deterministic request-local IDs
`call_0`, `call_1`, etc. Truncated calls are flagged and withheld. A truncated
channel header is flagged separately. Malformed closed frames, duplicate keys,
unknown channels, nesting above 64 and frames above the configurable byte limit
fail. Stop boundaries are `<turn|>`, `<eos>` and `<|tool_response>`; trailing
bytes after a stop do not become assistant content. Reasoning whitespace is
preserved verbatim.

Thinking budgets are a runtime two-step contract from the model card, not a
Jinja token-counting feature. Execute a thinking-enabled first job with an output
limit equal to the budget. On natural stop, return it normally. On length cutoff,
call `continue_after_budget` with the actual used tokens and partial parsed output.
It renders the partial assistant with preserved reasoning, closes that channel,
continues the same assistant turn, and returns `max_tokens - used_tokens` for the
second job. Reuse the tools and sampling policy; retain all first-job output IDs
for accounting. Do not feed calls or a partial control header into this helper;
completed tool calls belong in the tool-execution loop.
It requires a positive budget and answer room, and is an explicit token-changing
request policy. No executing engine budget capability is advertised in phase 1.

`Adapter::prepare` gives rendered text, raw prompt IDs, stop IDs `[1,106]` and
whether generation starts inside an already-open thought channel. This flag is
derived from the prompt suffix, independently of thinking being enabled.
`PreparedPrompt::job` builds the shared `GenerationJob`; a later engine implements
`GenerationEngine::generate` and uses the existing `GenerationQueue`. The shared
API already has renderer/output-parser hooks, but its concrete `BpeTokenizer`
parameter and reasoning assumptions still need a family-neutral interface before
HTTP integration. No shared serving files or hooks are changed here.

On Windows the package CMake file compiles the shared portable JSON, tokenizer
utilities and trimming implementation unchanged into a small support library,
because the full shared HTTP target uses POSIX headers. Linux builds link
`strix_serve`. This is host qualification, not an HTTP server or model qualification.

The generated lowercase/context table freezes Unicode 15.1 semantics queried from
CPython's Unicode database. Its generator contains no copied Python implementation
code. Unicode data attribution and permission are in the accompanying
[Unicode notice](notices/UNICODE-LICENSE.txt). Fixture provenance records SHA-256
of the official assets; fixtures are reference outputs, not comparative records.

Build with `STRIX_FAMILY_GEMMA4=ON`, HIP off, and an optional
`STRIX_GEMMA4_TOKENIZER_JSON=assets/tokenizer.json`. Build target
`strix-gemma4-serve-tests`, then run `ctest -R '^gemma4-serve-' --output-on-failure`.
Missing tokenizer assets yield `NO-REFERENCE` and CTest skip status, never an
exactness pass. The host renderer/parser gate runs without model assets or Python.
Generation scripts live under `tests/gemma4_serve/gen/`; the native tests need only
the committed small fixtures and, for the tokenizer gates, the runtime JSON asset.
