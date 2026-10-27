# Serving over HTTP

`strix-hip-decode-token --http-serve` loads the model once, qualifies the MTP predictor once and
then serves OpenAI-compatible and Anthropic-compatible HTTP requests until it receives SIGINT or
SIGTERM. Every request runs the same native path as a command-line request: tiled prompt prefill,
MTP speculative decode with target verification, greedy selection. Output tokens are those of the
command-line route for the same prompt and flags, up to the first stop token (the command line stops
only at `--eos-token`, by default `<|endoftext|>`; the server also stops at `<|im_end|>`).

The HTTP front end (`strix_serve`: JSON, tokenizer, chat template, HTTP server, request queue and the
API routes) is host-only code with no GPU dependency; its contracts are tested without a GPU.

## Starting the server

Serving runs on the tiled MTP route, so it takes the flags of a tiled MTP request (see
[prefix-state-cache.md](prefix-state-cache.md#mtp) and [mtp-native-cli.md](../speculative/mtp-native-cli.md)) plus:

| Flag | Default | Meaning |
|---|---|---|
| `--http-serve` | off | serve requests after the calibration replay instead of running the command-line request |
| `--http-bind ADDR` | `127.0.0.1` | listen address (numeric) |
| `--http-port N` | `8080` | listen port (`0`: any free port, printed on stderr) |
| `--http-model-id NAME` | `qwen3.8-flash-next` | model name reported by `/v1/models` and responses |
| `--http-api-key-file FILE` | none | require `Authorization: Bearer KEY` or `x-api-key: KEY` on the API routes |
| `--http-batch N` | `0` | `1`..`8`: batched decoding with N request slots ([below](#batched-decoding)); `0`: one request at a time |
| `--http-sampling greedy\|reject\|sample` | `greedy` | requests with `temperature > 0`: run greedy and say so in the response, answer 400, or sample (sampling-v1, batched route only) |
| `--http-default-temperature X` | `0` (built-in) | Qwen HTTP default in [0, 2]; positive values require `--http-sampling sample` |
| `--http-default-top-k N` | `0` (built-in) | Qwen HTTP default non-negative integer, at most 4294967295 |
| `--http-default-top-p X` | `1` (built-in) | Qwen HTTP default in (0, 1] |
| `--http-default-min-p X` | `0` (built-in) | Qwen HTTP default in [0, 1) |
| `--http-default-max-tokens N` | `4096` | generation budget when a request gives none |
| `--http-max-queue N` | `64` | waiting requests before new ones get 503 |
| `--http-max-connections N` | `64` | open connections before new ones get 503 |
| `--http-responses on\|off` | `on` | the stateless `POST /v1/responses` route ([below](#responses-stateless)); `off`: 404 |
| `--http-capabilities on\|off` | `on` | `GET /v1/capabilities` and its `capabilities_id` in `/health`, `/v1/models`, `/api/status`; `off`: 404, no id |
| `--http-stream-receipts on\|off` | `on` | the applied-policy `strix` object at the start and end of every stream ([below](#applied-policy-and-keepalive)); `off` together with `--http-sse-keepalive 0 --http-sse-progress-interval 0`: the previous build's bytes |
| `--http-constraints on\|off` | `on` | constrained JSON output ([below](#constrained-json-output)); `off`: such requests get 400 and capabilities say unavailable |
| `--http-sse-progress-interval SECONDS` | `1` | coalesced completed prefill progress as SSE comments; `0`: none; request opt-out: `"strix":{"progress":false}` |
| `--http-sse-keepalive SECONDS` | `10` | SSE comment `: keepalive` after this long without a write while a stream waits (queue, prefill, decode); `0`: none |

The token IDs on the command line are the **calibration prompt**: its first
`--mtp-calibration-rows R` pairs are replayed against the golden (`--mtp-golden`) to qualify the
predictor, exactly as for a command-line request. Requests do not need to share anything with it.
`--generate` is ignored while serving. The prefix state cache flags (`--prefix-cache-host-bytes`,
`--prefix-cache-every`, `--prefix-cache-dir`) apply across requests.

The prompt may run a separate prefill profile (`--prefill-numerical-profile`, see
[prefill-numerical-profile.md](../prefill/prefill-numerical-profile.md)), for example the fast prefill profile
`fast-prefill-f16wmma-v1` with the dense-A8 decode profile and `--known-prefix-prefill-rows 8192` (T
above 4096 needs `--known-prefix-host-metadata-budget-bytes` above the plan's metadata, about 286 MB
at T = 8192). Its complete tiled blocks run the prefill profile; the calibration rows, the serial
suffix and decoding run the decode profile, and decoding is exact from the state the blocks produced.
The calibration golden is that of the decode profile. The prefix state cache keys its checkpoints by
the pair, both executions and T.

The process prints `strix serve: listening on ADDR:PORT` on stderr once it accepts requests, and a
one-line JSON receipt (`"mode":"native-mtp-http-serve"`, the number of served requests) on stdout
when it stops.

## Execution model

- One engine thread (the decoder's thread) runs requests one at a time in arrival order. HTTP
  connections run on their own threads; they exchange tokens with the engine through a ticket that
  the engine fills after every verify cycle. Streaming therefore delivers tokens per cycle (one to
  eight tokens).
- Each request starts from a fresh target state and session, a fresh predictor and no pending
  verify-window rows, so its output does not depend on earlier requests. With the prefix state cache
  the longest cached block boundary of its prompt is restored first; restored tokens are reported as
  `usage.prompt_tokens_details.cached_tokens` (`cache_read_input_tokens` on the Anthropic route).
- Generation stops at a stop token (`<|im_end|>`, `<|endoftext|>` and `--eos-token` if given), at the
  request budget, at the context capacity, at a stop string, or when the client disconnects (the
  engine stops after the current cycle).
- A failure inside the engine leaves the target state unknown: the failing request gets an error,
  `/health` answers 503 and the process stops with a non-zero exit status.
- Decoding is greedy unless the batched route samples (below). A request with `temperature > 0` is decoded greedy
  and the response carries `strix.note`; with `--http-sampling reject` it gets a 400 instead.

Each sampling filter resolves independently: a non-null request value, then the configured server default,
then the built-in value (`temperature: 0`, `top_k: 0`, `top_p: 1`, `min_p: 0`). The four default-filter flags
are available on the Qwen command; the DeepSeek command remains greedy with built-in defaults.
Temperature, top-p and min-p must be finite numbers in their advertised ranges, even during greedy decoding.
Top-k must be a non-negative integer in uint32 range. Seed must be an integer in [0, 2^63); strings,
fractions and negative seeds are refused with a 400 naming the field before submission.
At temperature 0, filters do not change greedy argmax; non-neutral filters are accepted with the note
`filters ignored at temperature 0 (greedy)`. Positive temperature on a greedy-only engine retains the
configured downgrade or rejection policy.

A sampled request without a seed receives an API-generated 63-bit seed. The engine uses that seed and
the policy receipt echoes it. Reusing it reproduces the run with the same model, numerical identity,
sampler and position-keyed uniforms. Greedy receipts omit the seed.

## Batched decoding

`--http-batch N` keeps N request slots. Each slot owns what a single request owns: target state and session, launch
plan, PLE backend and prefetch state, tiled prefill backends (the prefill arenas are shared; one prompt runs at a
time), prefix state cache view (over the shared store), predictor, target producer and its 2/4/8 window owners on one
window stream shared by all slots. The calibration replay runs once.

- Between decode cycles the engine thread takes queued requests into free slots. A joining request starts exactly
  like a request of the one-at-a-time route: fresh state and session, fresh predictor, its prompt through the tiled
  prefill, its first token from the prompt's last row.
- Its prompt advances one unit at a time (a T-row block, a tail block or 16 serial rows) between the decode cycles of
  the other requests, so that a long prompt does not stop them: one prompt unit, then one decode cycle.
  `STRIX_BATCH_PROMPT_SHARE=X` gives the decode cycles at least X times the time of the prompt units (more decode
  progress while prompts wait, later full cohorts); `STRIX_BATCH_INTERLEAVE=0` runs each prompt whole at admission.
- A cycle decodes every request past its prompt: with two or more it is a cohort cycle (cohort windows, D-039: each
  request drafts up to 8 / requests rows, at least 2, and requests are packed into 4- and 8-row windows whose
  weight-reading steps run once for all rows), with one the ordinary cycle. With five to eight requests,
  `STRIX_COHORT_ROWS=16 STRIX_COHORT_WIDE=all` runs the cycle as one group of up to 16 rows whose GDN/QSA
  projections, LM head and expert union run once for all rows (D-052,
  [multi-request cohort](../scheduling/multi-request-cohort.md#global-row-budget-and-wide-steps-opt-in)).
  The experimental `STRIX_MTP_BATCH_JOINT=1` replaces that fixed per-request cap
  with a joint cost-based choice and supports shared dense steps through 32
  rows; see [joint serving windows](../scheduling/joint-serving-windows.md). It is off by default.
- A request leaves its slot when it ends (stop token, budget, context, stop string, disconnect), without stopping the
  others; its slot is free for the next queued request.
- Greedy requests return exactly the tokens they return alone (and on the one-at-a-time route): every request keeps
  its own state and arithmetic, and the shared steps compute each row as the request's own window would.
- `--http-sampling sample`: a request with `temperature > 0` samples with its own policy (`temperature`, `top_k`,
  `top_p`, `min_p`) and `seed` (random when absent), with speculative decisions on its drafts (D-040; on the GPU as
  in [sampled-decoding.md](../decode/sampled-decoding.md)). Its tokens follow the target distribution exactly; they are not
  the same tokens as alone, because the drafts per cycle depend on how many requests share the cycle.
- Statistics per request are those of the one-at-a-time route; `prefill_ms` runs from admission to the first
  token (it includes the decode cycles interleaved with the prompt), `queue_ms` from submission to admission.
  The receipt at shutdown is `"mode":"native-mtp-batch"` with slot, step and cohort window counts.

## Endpoints

| Method and path | Purpose |
|---|---|
| `POST /v1/chat/completions` | chat; `stream` with server-sent events |
| `POST /v1/completions` | raw prompt (text or token IDs); `stream` supported |
| `POST /v1/messages` | Anthropic-compatible messages; `stream` with typed events |
| `POST /v1/responses` | OpenAI Responses format, stateless; `stream` with typed events |
| `GET /v1/models` | the served model (with `capabilities_id`) |
| `GET /v1/capabilities` | what this server executes: endpoints, sampling policy and modes, stream features, unavailable features |
| `POST /tokenize`, `POST /detokenize` | `{"text"}` ↔ `{"tokens"}` with the model tokenizer |
| `GET /health`, `GET /v1/health` | `{"status":"ok"}`, or 503 `{"status":"failed"}` after an engine failure |
| `GET /metrics` | Prometheus text: request counters, token counters, queue depth, busy time |
| `GET /api/status` | JSON snapshot (`"schema":"strix-serve-status-v1"`): state, queue, totals, recent requests |

`/health`, `/metrics` and `/api/status` do not require the API key.

### Chat requests

- `messages` with roles `system`, `developer`, `user`, `assistant` and `tool`; string content or text
  parts. Images and video are rejected.
- `tools` (function tools) are rendered into the system turn the way the model was trained; assistant
  tool calls in the history are rendered from `tool_calls` (string or object `arguments`).
- Thinking: on by default (the template default). `chat_template_kwargs.enable_thinking: false`
  (or top-level `enable_thinking: false`) disables it; `reasoning_effort` (`low`, `medium`, `high`,
  `xhigh`) and `preserve_thinking` are passed to the template.
- The reply is split into `reasoning_content` (the thinking block), `content` and `tool_calls`.
  Parameter values of a tool call are typed by the tool's JSON schema (strings stay strings, other
  types are parsed as JSON); `finish_reason` is `tool_calls` when the reply contains calls.
- `tool_choice: "none"` renders no tools.
- Other fields: `max_tokens` / `max_completion_tokens`, `stop` (string or list), `stream`,
  `stream_options.include_usage`, `n` (only 1), `temperature`, `top_p`, `top_k`, `seed`.
- Generation fields the server does not execute yet are refused with a 400 naming the field, on every route:
  non-zero `presence_penalty` / `frequency_penalty`, `repetition_penalty` other than 1, a non-empty `logit_bias`,
  `logprobs` / `top_logprobs`, `response_format` other than text / `json_object` / `json_schema` ([constrained output](#constrained-json-output)), `parallel_tool_calls: false`, `required` or named
  `tool_choice` (Anthropic: `any` / `tool`, `disable_parallel_tool_use: true`), `strict` tools, legacy `functions` /
  `function_call`; on completions also `n` / `best_of` other than 1, a `suffix` and a numeric `logprobs`; on
  messages `thinking.type` other than `enabled` / `disabled` and non-null `output_config` members. Neutral values are accepted.
  `thinking.budget_tokens` is accepted (the format requires it) and stated in `strix.note` as not enforced.
  `GET /v1/capabilities` lists these features as unavailable.

Streaming sends `reasoning_content` and `content` deltas as they are decoded (holding back only
bytes that may belong to a stop string, a `</think>` or `<tool_call>` marker, an incomplete UTF-8
sequence or trailing whitespace), tool calls when they are complete, a final chunk with
`finish_reason`, optionally a usage chunk, and `data: [DONE]`.

### Anthropic-compatible messages

`system` (string or text blocks), `messages` with text, `tool_use` and `tool_result` blocks, `tools`
with `input_schema`, `thinking` (`{"type":"enabled"}` or `{"type":"disabled"}`), `stop_sequences`,
`max_tokens`. The reply has `thinking`, `text` and `tool_use` blocks and `stop_reason`
`end_turn`, `max_tokens`, `stop_sequence` or `tool_use`. Streaming emits `message_start`,
`content_block_start`, `content_block_delta` (`thinking_delta`, `text_delta`, `input_json_delta`),
`content_block_stop`, `message_delta` and `message_stop`.

### Responses (stateless)

`POST /v1/responses` is translated onto the chat route: `instructions` becomes the leading system message, `input`
(a string or items) becomes the messages, function tools and `tool_choice` `auto`/`none` map to their chat forms,
`reasoning.effort` to `reasoning_effort`, `max_output_tokens` to `max_tokens`. Message items take `input_text` /
`output_text` parts; `function_call` items (their `arguments` string kept as given) join the preceding assistant turn,
`function_call_output` items become tool messages matched by `call_id`, and plaintext `reasoning` items are attached
to the following assistant message. The prompt token IDs equal those of the equivalent chat request (tested), so a
Responses request generates the same tokens as its chat form.

The route keeps no state: `previous_response_id`, `store: true`, `conversation` and `background` are refused with
400, as are features the server does not execute yet: `text.format` other than text / `json_object` / `json_schema`, `text.verbosity` other than
`medium`, `strict: true` tools, `required` or named `tool_choice`, non-function tools, image or file input,
encrypted-only reasoning, reasoning summaries, `include`, `truncation` other than `disabled`, `max_tool_calls`,
unknown or duplicate `call_id`s; the 400 names the field. A `null` member counts as absent. The reply is a response
object with `reasoning`, `message` and `function_call` output items, `status` `completed` or `incomplete`
(`max_output_tokens`, or `unfinished_tool_call` when a stop string or the end of generation cut a tool call; such a
call has `status: incomplete`), usage and `strix`. As on the chat route, text from a `<tool_call>` marker on is
treated as tool framing, also when the model writes the marker literally. Streaming sends typed events (`event:` line + JSON with the same `type`
and an increasing `sequence_number`): `response.created`, `response.in_progress`, `response.output_item.added`,
`response.content_part.added`, `response.reasoning_text.delta`/`.done`, `response.output_text.delta`/`.done`,
`response.function_call_arguments.delta`/`.done`, `response.content_part.done`, `response.output_item.done`, and
exactly one of `response.completed`, `response.incomplete`, `response.failed`. There is no `[DONE]`.

### Constrained JSON output

`response_format` (chat) or `text.format` (Responses) set to `{"type":"json_object"}` or to a JSON schema
(`{"type":"json_schema","json_schema":{"name","schema","strict"}}` in chat, `{"type":"json_schema","name","schema","strict"}`
in Responses) makes the answer a JSON value that the token selection itself keeps valid: at every answer position the
server picks the highest-logit token among those whose bytes keep the output a valid prefix (lowest id on ties), and
end of sequence only once the value is complete. Thinking, when on, runs unconstrained until the end-of-thinking token;
the constraint applies to the answer after it. The answer bytes are returned exactly (no tool parsing, no trimming).

The schema language is `strix-json-schema-v1`: `type` (also type arrays), `properties`, `required`, boolean
`additionalProperties`, `items`, `minItems` / `maxItems`, `minLength` / `maxLength` (Unicode scalars), `enum`, `const`,
`anyOf`, local acyclic `$defs` / `$ref`, and the annotations `title`, `description`, `$comment`. Any other keyword
(`minimum`, `pattern`, `format`, `oneOf`, ...) is refused with a 400 whose `param` is the keyword's full path (for
example `text.format.schema.properties.age.minimum`) and `code` `unsupported_schema_keyword`; `strict: false` does not
turn the refusal into silent ignoring. Output follows a documented generation language (standard JSON with at most two
whitespace characters between tokens, no duplicate keys, integers without fraction or exponent, `enum` / `const` in
their compact spelling). Compilation is bounded (schema size, nodes, depth, properties, enum values, work, memory);
limits are listed in `GET /v1/capabilities` under `constraints`.

Constraints run greedy only, on a serial host route (`execution: serial-host`): the request's target logits row is read
back and selected on the host, and the request decodes without speculative drafts; other requests in the batch keep
their normal speculative decoding. Constraints are refused with stop strings, with effective sampling
(`temperature > 0` on a sampling server) and with a non-empty tool offering. Every constrained response carries
`strix.constraint` (also with stream receipts off): `identity` (`json-object` or `strix-json-schema-v1`), `selection`,
`execution`, `phase`, `status` (`complete`, `incomplete`, `failed`), `accepting` and `reason` (`max_output_tokens`,
`context`, `reasoning_eos`, `cancelled`, `constraint_failure`, `execution_failure`). A reply cut by the budget is
reported incomplete, never repaired; Responses then has `status: incomplete`.

### Applied policy and keepalive

Streaming replies may emit compact comments such as
`: progress {"phase":"prefill","prefilled_tokens":128,"cached_tokens":64,"total_tokens":256}`
followed by a blank line. Counts describe completed host-visible prefill units: computed tokens and restored
cached tokens separately; their sum never exceeds the prompt total. The phase becomes `decode` at the first
published token. Snapshots are coalesced, so short phases and intermediate counts may be skipped. No progress
is invented for engines that do not report it. `/v1/capabilities` advertises `stream.progress.interval_seconds`
and `engine_reports`; the applied stream receipt reports the request-effective interval. The interval defaults
to one second; `--http-sse-progress-interval 0` or `"strix":{"progress":false}` disables comments.
Progress comments are shared by chat, Messages, Responses and completions streams, never non-stream replies
or data events. They count as writes for keepalive. Reports use existing host counters at completed blocks,
serial rows and windows, including cache restore; they introduce no device synchronization or scheduling changes.
Progress-on/off GPU token/state equality remains a native qualification gate; CPU fake-engine tests cover wire equality.

With `--http-stream-receipts on` (the default), every response's `strix` object states the policy that ran: `sampling_requested` and `sampling_effective`
(`greedy` or `sample`, with `note` when a sampled request ran greedy), `filters` (each of `temperature`,
`top_k`, `top_p`, `min_p` has a resolved `value` and `source`: `request`, `server` or `builtin`),
`seed` for sampled runs, `max_tokens_requested` and
`max_tokens_effective`, and the stream settings. A stream carries it in its first event (chat: the role chunk;
messages: `message_start.message`; responses: `response.created`; completions: the first chunk) and its last,
whether or not `stream_options.include_usage` is set. `GET /v1/capabilities` (`"object":"strix.capabilities"`,
`"version":1`) lists what the server executes and marks every feature it does not as `false`; its
`capabilities_id` is a hash of the document. `sampling.defaults` reports each effective server default
with its `value` and `source` (`server` or `builtin`), and `sampling.seed_echo: true` advertises seed receipts.

While a stream waits for tokens (queued, during a long prefill, between decode steps) and nothing was written for
`--http-sse-keepalive` seconds, the server writes the SSE comment `: keepalive`, which clients ignore and proxies
count as activity. A request turns it off with `"strix": {"keepalive": false}`. Neither the receipts nor the
keepalive change scheduling or generated tokens (GPU-gated: token IDs equal with them on and off).

### Engine fields

Non-streamed responses (and the final usage chunk of a stream) carry a `strix` object: queue,
prefill and decode milliseconds, decode tokens per second, MTP cycles, proposed and accepted drafts.
`"return_token_ids": true` in the request adds `strix.token_ids`, the generated token IDs including
the stop token, for exactness checks against the command line.

## Status for the monitor panel

`/api/status` reports per request the fields of the live monitor's request record (`status`
`done|cancelled|error`, `stop_reason`, `prompt_tokens`, `reused_tokens`, `output_tokens`,
`prefill_tok_s`, `decode_tok_s`, `duration_ms`, `mtp_cycles`, `mtp_proposed`, `mtp_accepted`), so
the monitor page can show served requests without another format. The serving loop publishes the
same facts it already holds on the host; it adds no device synchronization.

## MiniMax serve (`strix-minimax-serve`)

The MiniMax TP2 server (rank 0 serves HTTP; rank 1 replays the control blocks) has its own API module,
`src/minimax/serve/minimax_api.*`, with the same routes plus:

- `GET /api/status[?detail=1]`: `"schema": "strix-serve-status-v1"`, `"engine": "strix-minimax-serve"`, state,
  queue, lifetime totals (from the engine queue: requests, prompt, cached, imported, computed and generated
  tokens, busy seconds), slots, the process `instance` (`id`: random 16 hex chosen once at start; `start_time` and
  `started_at`: the same unix start time; always present; the key for counter
  baselines), `capabilities` (prefix cache and state import are `false` in this build: their counters are
  real zeros, and a consumer shows them as not supported; `draft` is true when a DSpark drafter is loaded), load facts and the recent-request ring (50 records,
  newest first). `detail=1` adds the latency histograms as JSON (`le` = finite bounds, `counts` = cumulative
  counts with the last one for +Inf).
- Per record: `id` (the accepted `X-Request-Id`, else the completion id), `request_id`, `completion_id`,
  route, arrival time, `status` (`done|cancelled|error|refused`), `stop_reason`, `stream`, prompt accounting
  (`prompt_tokens`, `cached_tokens`, `imported_tokens`, `computed_tokens` = prompt - cached - imported;
  `reused_tokens` is the legacy name of `cached_tokens`), `output_tokens` (stop token included), `queue_ms`,
  `prefill_ms`, `decode_ms`, `duration_ms`, `first_token_ms` (submit to the first published token, measured by
  the scheduler; null when no token was published), `max_gap_ms` (the scheduler's longest gap between two
  emitted tokens; null with fewer than two), draft counts when the engine drafted, rates. The decode rate is
  (output - 1) / decode seconds, because this engine's decode time starts at the first token.
- Reply `timings` (non-stream reply and the stream's final chunk): the scheduler's per-request measurements,
  `queue_ms`, `prefill_ms`, `decode_ms`, `decode_tok_s`, `completion_tokens`, `ttft_ms`, `max_gap_ms`,
  `stall_ms` (decode time stalled by prefill units), `prefill_units`, `cycles`, `drafted`, `accepted`, `ids_fnv`,
  `prompt_fnv` (FNV-1a, 16 hex) and `sampling`, for every request, armed or not.
- `GET /metrics`: metrics text with `# HELP`/`# TYPE` for every family. The `mm_*` value lines are
  unchanged (the queue's seven and the scheduler's step, prefill, draft, idle and decode-stall seconds, steps,
  step rows, prefill units, draft cycles, drafted and accepted tokens; `POST /admin/sched` with
  `reset_counters` zeroes the scheduler's, a counter reset); the `strix_*` counters and gauges use the shared names (`strix_requests_completed_total`,
  `strix_prompt_tokens_total`, `strix_generated_tokens_total`, `strix_queue_waiting`, ...), plus
  `strix_prompt_tokens_imported_total`, `strix_prompt_tokens_computed_total`, `strix_engine_info{engine,model,
  instance}`, `strix_engine_start_time_seconds`, `strix_instrumentation_armed` and, once armed, the
  histograms `strix_ttft_seconds` and `strix_decode_tokens_per_second`.
- Armed instrumentation: the first `GET /metrics` or `GET /api/status?detail=1` arms the process (one-way).
  Requests submitted before that cost one relaxed atomic load each and are not recorded in the ring or the
  histograms; lifetime totals are always available. Armed, a request adds one ring record and constant-time
  histogram updates at its end on its HTTP thread, and the scheduler copies one already-measured time into its
  statistics at completion; nothing on the device path changes.
- Request ids: an inbound `X-Request-Id` of 8-64 characters `[A-Za-z0-9._-]` (for example `<16 hex>.<attempt>`)
  is kept whole and echoed in the `X-Request-Id` of every response to that request; a generation request that
  reaches the queue without one echoes its completion id (requests rejected before that echo only an accepted
  inbound id); any other value is ignored.
- `"return_token_ids": true` adds `strix.token_ids` (every generated id, the stop token included, so its length
  is `usage.completion_tokens`) to the reply, and in a stream to the final chunk (the one with `finish_reason`)
  and to the usage chunk. `stream_options.include_usage` adds a chunk with empty `choices` and the `usage`
  after the final chunk, before `data: [DONE]`. These boolean options must be booleans (400 otherwise).
- `--admin`: `GET` / `POST /admin/sched` (scheduler policy and counters at runtime), served by the same API module
  through the app's hook.
- Tests: `strix-minimax-observe-tests` (ids, ring, status JSON, a metrics text validator, histograms) and
  `strix-minimax-api-tests` (the HTTP contracts against a scripted engine thread: echo rules, token ids,
  stream event order, engine errors, failed stream start, client disconnect, queue refusal, arming during a
  request, the scheduler's timings, extras, `mm_*` lines and `/admin/sched` hooks). `tools/minimax/serve_observe_check.py` checks a live server and writes the token-identity and
  overhead receipt across unarmed and armed sets.

## Tokenizer and chat template

The tokenizer is read from the model GGUF (byte-level BPE, 248320 tokens, its merges and its
pre-tokenizer split rule) and implemented natively; special tokens in a prompt string are parsed as
single tokens. The chat template is a native implementation of the template embedded in the GGUF
(system/developer merge, reasoning-effort instructions, tool block, tool-call and tool-response
rendering, thinking on/off generation prompt). `tools/serve/gen_chat_template_cases.py` renders the
embedded template on fixed cases; `strix-serve-frontend-tests` checks the native renderer against
them byte for byte, and the tokenizer against expected splits. `strix-tokenize MODEL.gguf
encode|plain|decode FILE` exposes the tokenizer on the command line.

## Tests

- `strix-serve-frontend-tests`: JSON, tokenizer pre-split and UTF-8 streaming, chat template cases,
  assistant output parsing (with a model GGUF as second argument, also an encode/decode round trip).
- `strix-serve-api-tests`: the HTTP contracts against a scripted engine: chat (plain, streamed, tool
  calls, stop strings, thinking off, length), completions (text and token IDs), Anthropic messages
  (plain, streamed, tool use), errors, authentication, two concurrent clients, cancellation on
  client disconnect, status and metrics.
- On hardware: the same prompt through the command line and through the server gives the same
  token IDs, also when two clients send requests at the same time and when the prefix state cache
  restores part of the prompt.

## Limits

- Without `--http-batch`: one request at a time, greedy only; concurrent requests wait in the queue.
- `--http-batch`: at most eight slots; requests beyond them wait in the queue. Prompts run one at a time (block-
  interleaved with the decode cycles, not batched with each other). With more than four decoding requests a cycle
  forms two or more 8-row cohort windows (two rows per request), so the aggregate rate stays near that of four.
- Every slot holds a full context of state memory and its own window owners; memory grows with N.
- Each request resets the adaptive min-p, depth or survival controller to its configured prior, retaining
  the slot's device owners. Previous requests' acceptance estimates and scheduler-dependent cycle costs
  do not set the new request's draft depth. `STRIX_MTP_REQUEST_RESET=0` retains the earlier policy history
  for controlled comparisons. Target verification continues to decide every token and committed state.
- The window controller switches (`STRIX_MTP_CONTROLLER=survival`, `STRIX_MTP_ACTIVE_ROWS=1`,
  `STRIX_MTP_DRAFT_VOCAB=2`, the window folds) apply to served requests as on the command line. The draft
  vocabulary is chosen per request from that request's prompt.
- `n` > 1, logprobs, images and audio are not supported.
- Plain HTTP/1.1 (keep-alive, chunked responses); put a TLS proxy in front for remote access.

## Independent-session file route (`--serve-prefix-u32le`)

A separate, older route serves a fixed list of token files in one process without HTTP.


Source implementation only. No native equivalence, concurrent execution or
throughput qualification is established by these interfaces.

The decoder accepts repeated `--serve-prefix-u32le FILE` arguments with
`--generate N`. Each file is a nonempty literal sequence of little-endian uint32
token IDs, bounded by the selected context and vocabulary. Ordinary positional
prefix tokens are excluded on this route. All files are parsed before any
request is submitted. The decoder owns one shared mapping and model residency;
weights are not cloned for a request.

Required reservation controls are `--serve-memory-budget-bytes N`,
`--serve-request-budget-bytes N` and `--serve-headroom-bytes N` (zero headroom
is an explicit choice). `--serve-concurrency 1|2|4` defaults to one. At most64
requests are retained. These are caller-declared virtual backing/request budgets,
not a physical-memory fit guarantee. The request floor checks state, scratch,
selector payload and token storage; allocator/driver/stream/metadata overhead
requires the declared additional budget/headroom. Completed handles keep their
reservation and owners charged until released.

Each accepted factory allocates its own state, scratch, PLE backend/prefetch,
stream and sticky D1 completion record, with a fresh allocation epoch. All
operator resources and arenas belong to that request. Its production backend
validates finite output and D1 completion before committing state. Prepared mode
is passed explicitly to that backend; overlap, islands, graph/integer, paired
prompt routes, prefix roundtrip and single-session RAW/reference/final-state or
catalog-inventory output are rejected rather than silently disabled.

Concurrency two/four means independent C1 sessions on scheduler workers and
owned streams, not cross-request tensor batching. One uses the inline path with
no batch-fill wait. The existing token cursor keeps EOS, inclusive output and
pending-last-token semantics; prefill chunks, FIFO/decode priority with aging,
queue timeout and cancellation act only at completed backend boundaries. The
public scheduler exposes cancellation/timeouts; this file-list CLI drains its
fixed submitted family and reports every terminal result.

Setup uses existing guarded allocations and global HIP synchronization, so
submitting another request can wait for GPU work already active. This can limit
observed concurrency. Two streams or workers do not prove physical overlap or a
speed gain. Host service intervals may overlap; they must not be summed into GPU
busy time. The receipt distinguishes queue/service/wall/TTFT and fixed log2
inter-token histograms; histogram quantiles would be bounds, not exact percentiles.

Optional telemetry uses one counter owner per request and binds it on the worker
thread for that request's serialized calls. No non-atomic counter is shared by
concurrent workers. Per-request reports cover factory/worker calls before owner
teardown; the enclosing main-thread report covers remaining setup/teardown.
The normal decoder route creates no scheduler or extra request storage.

### Explicit shared prefix cache

The optional shared pool requires all four positive limits:
`--serve-prefix-cache-entries N`, `--serve-prefix-cache-snapshot-bytes N`,
`--serve-prefix-cache-capture-peak-bytes N` and
`--serve-prefix-cache-max-tokens N`. The token bound cannot exceed the selected
context. Omitting these options preserves the existing no-cache route; partial,
duplicate and out-of-serving controls are rejected.

The pool's checked reservation includes retained snapshot payload, the declared
capture peak and bounded entry/token bookkeeping. It is charged to the scheduler
shared reservation before the pool or request factories are constructed, in
addition to the shared model and caller headroom. Snapshots retain independent
allocator owners rather than retaining a completed request's scratch/state/stream
outside its request reservation. Cache identity and complete-state restoration
remain the production backend's responsibility; an exact-prefix hit carries an
unconsumed terminal token, while a proper-prefix hit continues its known suffix.
The optional telemetry wrapper forwards both cache hooks unchanged.

Receipts report actual pool lookups, hits, restored logical inputs, capture
attempts, inserts, skips, failures and currently retained entries/bytes. These
are operation counters, not measured time savings or physical-memory traffic.
This integration remains source-only until final validation.

### Independent bounded PLE row caches

`--ple-cache-rows N --ple-cache-budget-bytes N` requires both positive explicit
limits and resolved planned PLE I/O. It reserves payload and private row tags
once per backend. Serving adds the pure requested amount to each request's
minimum reservation before factories allocate it. Normal setup records the same
reservation before construction. Each enabled backend retains an independent
source object that itself retains the immutable mapped model; there is no cycle
back to its request owner. Disabled caches preserve the existing borrowed source.
Counters report actual hits/misses/inserts/evictions/invalidations and retained
rows/bytes per backend, not measured I/O time savings. This cache is distinct
from the shared full-state serving prefix pool.
