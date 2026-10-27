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
| `--http-batch N` | AUTO | omitted: as many request slots as memory admits ([below](#batched-decoding)); `N`: at most N; `0`: one request at a time |
| `--http-sampling greedy\|reject\|sample` | `greedy` | requests with `temperature > 0`: run greedy and say so in the response, answer 400, or sample (sampling-v1, batched route only) |
| `--http-default-temperature X` | `0` (built-in) | Qwen HTTP default in [0, 2]; positive values require `--http-sampling sample` |
| `--http-default-top-k N` | `0` (built-in) | Qwen HTTP default non-negative integer, at most 4294967295 |
| `--http-default-top-p X` | `1` (built-in) | Qwen HTTP default in (0, 1] |
| `--http-default-min-p X` | `0` (built-in) | Qwen HTTP default in [0, 1) |
| `--http-default-max-tokens N` | `4096` | generation budget when a request gives none |
| `--http-max-queue N` | `65536` | waiting requests before new ones get 429; explicit small limits retain rejection |
| `--http-max-connections N` | `4096` | open connections before new ones get 503; independent of execution slot occupancy |
| `--http-responses on\|off` | `on` | the stateless `POST /v1/responses` route ([below](#responses-stateless)); `off`: 404 |
| `--http-capabilities on\|off` | `on` | `GET /v1/capabilities` and its `capabilities_id` in `/health`, `/v1/models`, `/api/status`; `off`: 404, no id |
| `--http-stream-receipts on\|off` | `on` | the applied-policy `strix` object at the start and end of every stream ([below](#applied-policy-and-keepalive)); `off` together with `--http-sse-keepalive 0 --http-sse-progress-interval 0`: the previous build's bytes |
| `--http-anthropic-thinking-budget enforce\|note` | `note` | Messages `thinking.budget_tokens`: `note` accepts it and states in `strix.note` that it is not enforced (the request keeps speculative decoding); `enforce` applies `thinking-budget-v1` (resolved host selection route) |
| `--http-thinking-budget on\|off` | `on` | makes the opt-in `thinking-budget-v1` policy available ([below](#thinking-budget)); `off`: explicit policy requests get 400 |
| `--http-selection-mtp on\|off` | `on` | policy selection uses admitted serial-equivalent MTP windows; `off` forces `serial-host` for JSON, budgets, scores and transforms |
| `--http-selection-fast on\|off` | `on` | argmax-first policy selection: rows whose unmasked argmax the constraint admits skip the mask walk (and, on `mtp-host`, the logits readback); `off` walks the full mask for every row |
| `--http-logprobs on\|off` | `on` | Qwen host logprobs; `off` refuses score requests |
| `--http-logit-transforms on\|off` | `on` | Qwen opt-in logit transforms; `off` refuses non-neutral transforms |
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

Before any device work the process prints one `strix-build-info` and one `strix-startup-event`
JSON line on stderr: configured build identity and a read-only memory preflight (GTT/VRAM, host
memory, 2 MiB-equivalent free blocks, compaction counters, memlock, KFD clients), with warnings derived
only from known plan terms ([startup-preflight.md](../../operations/startup-preflight.md); default ON,
`STRIX_PREFLIGHT=0` / `STRIX_RELEASE_EVENTS=0` to switch off).
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

Without `--http-batch` the server is batched with as many request slots as memory admits (AUTO). The batched
route needs prefix windows, the draft chain and window rows `{2,4,8}`; when they are missing, or a single-only
feature is set (state import, copy windows), AUTO serves one request at a time and says why on startup.
`--http-batch N` caps the slots at N (an explicit N with incompatible options is an error) and `--http-batch 0`
selects the one-at-a-time route. AUTO never changes the numerical profile or adds MTP assets.

- Sizing: before allocation the engine builds one extra slot, measures what it took (MemAvailable and GTT
  deltas, at least the static estimate), adds what every request allocates while it runs (window owners from
  their compiled plans, MTP runtime and target capture budgets, prompt taps, draft workspaces) and keeps a
  headroom (`--known-prefix-host-headroom-bytes`, 4 GiB unless set). Shared prefill arenas, sidecar and cache
  budgets count once. Startup prints `http slots N (auto: limited by ...)` or `(cap C: ...)`; there is no fixed
  slot ceiling.
- Startup builds `min(N, STRIX_HTTP_EAGER_SLOTS)` slots (default 4; `0` builds all N). The rest are built one at
  a time between engine steps as soon as every built slot is busy (`STRIX_HTTP_SPARE_SLOTS=0`: only once work is
  queued), so a request arriving during a long prompt finds a free slot, through the same code as the eager ones,
  after re-checking memory (growth stops for the server, with a notice, when a slot no longer fits).
  A build takes 1.4-2.9 s (measured) and delays that step: it is the longest decode gap of a server that grows;
  `STRIX_HTTP_EAGER_SLOTS` trades startup time for fewer builds under load.
- Requests beyond the built slots wait in the queue and are never rejected for busy slots: responses carry
  `X-Strix-Http-Slots` (capacity) and `X-Strix-Queue-Position` (waiting jobs ahead at submission), the final
  `strix` receipt `queue_ms`. The queue and connection limits default to 65536 / 4096.
- Measured (one node, dense serving numerics, T8192, 2026-10-27): AUTO resolved 25 slots (limited by GTT),
  started with 4 and grew to 12 under 16 clients; every response equalled the `--http-batch 0` response in
  full token IDs and every committed target-state component.

Each admitted slot owns what a single request owns: target state and session, launch
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
- Long prompts do not stop the other requests inside a block either. Native T-row blocks (and prompt groups)
  run on a prefill worker thread that can pause at a completed layer: then the engine thread runs one decode
  round of the decoding requests (cohort cycle as usual) and the block continues with the same T, kernels and
  rows. A round starts when `STRIX_BATCH_SERVICE_MS` (default 250, `0` = off) of completed GPU prefill time, or of
  wall time since the last decode round, has passed; `STRIX_BATCH_SERVICE_LOOKAHEAD` (default 2) bounds how many layers the host
  enqueues ahead of the GPU. The prefill stream is drained before the round and the decode streams before the
  block resumes; nothing else (admission, prompt finishing, cache publication) runs inside a pause.
- Short prompts first: between prompt units a prompt with at most `STRIX_BATCH_SHORT_FIRST_ROWS` (default 2048,
  `0` = oldest first) rows left runs before longer ones.
- Nested admission: at a pause, also on an otherwise idle server, a queued or waiting request with at most
  `STRIX_BATCH_NEST_ROWS` (default 2048, `0` = off) uncached rows is prefilled completely on the engine thread
  through its usual 1024/256/64/16 tail units, which use their own arenas (never the paused block's), with
  decode rounds between its units; it then decodes with the others and the long block resumes. One nested
  prompt per pause; T8192 with the default tails and the dense-A8 prefill only (other routes decline, counted
  in `nest_skip_reasons`).
- Cancellation (`STRIX_BATCH_CANCEL_POINTS`, default 1) is observed at those layer boundaries: the request's
  partial block is discarded (nothing is published to the prefix cache), the shared prefill storage is reset
  and the slot starts fresh. Groups and window cohorts finish their unit first.
- `STRIX_BATCH_SERVICE_MS=0 STRIX_BATCH_CANCEL_POINTS=0 STRIX_BATCH_NEST_ROWS=0` runs the previous synchronous
  path (no worker). Receipts (`batch_service`, per request and in the shutdown batch line): `service_rounds`,
  `service_ms_suspended`, `max_bundle_gpu_ms`, `nested_admits`, `nested_rows`, `nested_ms`, `nest_skip_reasons`,
  `cancel_points_cancelled`.
- Exactness: every request's generated IDs and committed target state are bitwise those of the request alone,
  with or without pauses, nesting and cancellation of other requests. Only the MTP predictor's own rows can
  differ with concurrency (draft batching), as before; they never decide a token.
- Measured (3 decoding streams, a 32K prompt, two 1192-token arrivals, default command without `--http-batch`,
  2026-10-27): decode gap p90 during the 32K prefill 14.3 s -> 0.54 s, short-request first token 42-52 s -> 2.3-3.8 s
  (both nested), 32K prompt first token 53.1 -> 63.5 s (the time given to the other requests), identical token IDs;
  the longest gap left (5.6 s) is the spare slot build when the 32K prompt took the last built slot. With 6 slots
  built at startup the longest gap was 2.8 s (12.3 s without these features).
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
| `GET /health`, `GET /v1/health` | Qwen [service health record](../../operations/service-health.md), with build/assets, resolved settings and backend evidence; `STRIX_SERVE_HEALTH_RECORD=0` restores the legacy body (including `capabilities_id` when enabled). Failed/draining: 503 |
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
  `response_format` other than text / `json_object` / `json_schema` ([constrained output](#constrained-json-output)), `parallel_tool_calls: false`, `required` or named
  `tool_choice` (Anthropic: `any` / `tool`, `disable_parallel_tool_use: true`), `strict` tools, legacy `functions` /
  `function_call`; on completions also `n` / `best_of` other than 1, a `suffix`; on
  messages `thinking.type` other than `enabled` / `disabled` and non-null `output_config` members. Neutral values are accepted.
  Enabled `thinking.budget_tokens` is accepted and stated in `strix.note` as not enforced, unless the server runs
  with `--http-anthropic-thinking-budget enforce`, which maps it to `thinking-budget-v1` ([thinking budget](#thinking-budget)).
  `GET /v1/capabilities` lists these features as unavailable. Logprobs and transforms are available only on executing Qwen routes ([below](#logprobs-and-logit-transforms)); other engines refuse them.

Streaming sends `reasoning_content` and `content` deltas as they are decoded (holding back only
bytes that may belong to a stop string, a `</think>` or `<tool_call>` marker, an incomplete UTF-8
sequence or trailing whitespace), tool calls when they are complete, a final chunk with
`finish_reason`, optionally a usage chunk, and `data: [DONE]`.

### Thinking budget

`thinking-budget-v1` is an explicit, token-changing selection policy. Chat and Responses requests opt in with:

```json
"strix": {
  "thinking_policy": "thinking-budget-v1",
  "max_thinking_tokens": 2048,
  "answer_room_tokens": 1024
}
```

Thinking must be enabled. With effective output budget `B`, answer reservation `R`, and closure length `C`,
reasoning may commit at most `min(max_thinking_tokens, B - R - C)` tokens before forced closure starts.
`max_thinking_tokens` must be a positive integer. Optional `answer_room_tokens` is a nonnegative integer;
omitted, the answer gets what remains (`R = B - max_thinking_tokens - C`), and a budget that leaves no answer
room (`max_thinking_tokens + C >= B`) returns 400. Explicit reservations with `R + C > B` return 400 before submission. A zero remaining reasoning allowance closes immediately.
Unknown policies and unavailable execution return 400. Without these fields, generation is unchanged.

The executing Qwen route uses the shared host selector on its resolved selection route. It resolves
`</think>` from the loaded tokenizer, then encodes the template's following two newlines. Forced IDs become
the actual target frontier, pass through target execution, and consume output/context capacity. They carry
`forced_control` provenance in selection; no model-selected probability is claimed. This policy requires
effective greedy selection and rejects nonempty stop strings and active tools. It composes with JSON output:
the answer constraint starts only after the entire forced closure. Natural closure before the limit needs no
injection. EOS during reasoning remains incomplete with reason `reasoning_eos`.

Every policy response includes `strix.thinking` with `policy`, requested `max_thinking_tokens`, resolved
`answer_room_tokens`, `closure` (`natural` or `forced`), committed `reasoning_tokens`, and `forced_ids`.
`status` and an optional `reason` distinguish incomplete output. Stream terminal metadata includes these fields
even with `--http-stream-receipts off`. Reasoning and answer split on the committed end-thinking token ID.
`GET /v1/capabilities` advertises `thinking_budget` and `thinking_budget_policy` only for an executing route.

For `/v1/messages`, `thinking: {"type":"enabled","budget_tokens":2048}` selects the same policy and default
answer reservation when the server runs with `--http-anthropic-thinking-budget enforce`; the response states the
executed policy. By default (`note`) the budget is accepted, the request keeps speculative decoding and `strix.note`
states that the budget is not enforced. Enforcing it uses the resolved host selection route. `--http-thinking-budget off` disables explicit policy requests. Other engines retain that note
until they integrate the executing selection adapter.

CPU contract tests cover the production selection adapter and HTTP translation. Native GPU token/state,
first-token/cache routes, mixed-slot isolation, and complete-workload qualification remain separate gates;
compilation and CPU results do not establish those gates.

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

### Logprobs and logit transforms

Qwen chat, completions and Responses execute these greedy-only policies on the same
resolved host selection route as constrained JSON and thinking budgets. Other slots retain their
selected routes. Effective temperature greater than zero is refused with 400; the
capabilities document advertises availability and execution. DeepSeek and MiniMax
continue to refuse these requests. The two feature switches above default to on;
disabling a feature refuses requests requiring it.

Chat accepts `logprobs:true` and integer `top_logprobs` in [0,20]. Completions accepts
integer `logprobs` in [0,5], including zero to request the chosen-token score only.
Responses accepts `top_logprobs` in [0,20] and
`include:["message.output_text.logprobs"]`. A positive `top_logprobs` also requests
scores. Scores include the first selected token from the terminal prompt row and do
not change selection. They are temperature-one log-softmax over the eligible final
selection row, after transforms and any JSON mask. Top entries sort by descending
logprob, then ascending token ID. `strix.scoring` reports
`basis:"target-row-log-softmax"`, `transform` (`none` or `logit-transform-v1`) and
`constrained`. These scores describe that row, not the unmasked model distribution.

Chat emits `choices[].logprobs.content` records with `token`, `logprob`, `bytes` and
`top_logprobs`. Token text uses lossy UTF-8; bytes preserve the model token bytes.
Completions emits `tokens`, `token_logprobs`, `top_logprobs` and `text_offset`.
Responses emits `output_text.logprobs` and `response.output_text.delta.logprobs`.
Streams carry the same scores as non-stream replies. Thinking-budget forced control
records have `logprob:null`, an empty top list and `forced_control:true`; no model
likelihood is invented for them.

Non-neutral transforms are explicit token-changing request opt-ins. The order is:

1. For IDs seen in the prompt or committed reply, apply `repetition_penalty` rho to the raw logit:
   positive logits become z/rho; zero and negative logits become z*rho.
2. Subtract `presence_penalty` once for each ID seen in the committed reply.
3. Subtract `frequency_penalty` times that ID's committed reply count.
4. Add each `logit_bias` value, then apply any JSON mask and greedy selection.

Presence and frequency accept [-2,2], repetition accepts [0.1,10], and bias accepts
at most 1024 canonical decimal vocabulary token-ID string keys with values in [-100,100]. Reply counts
include reasoning and forced controls and advance only when the selected token
commits. Transform arithmetic is applied to the target float row. Responses label
sampling `greedy-logit-transform-v1`, set `unconstrained_greedy_equal:false` and report `strix.transform` with `presence`,
`frequency`, `repetition` and `bias_entries`. This identity does not claim exact
untransformed greedy output. Neutral values (0, 0, 1, empty bias) without a scoring
request retain the ordinary route. JSON constraints compose as transform, mask,
greedy; scoring uses the resulting eligible row.

CPU tests cover these contracts. Native GPU qualification still needs complete IDs,
first-token route coverage, logits/state digests, mixed-slot isolation and workload
cost measurements for the executing model/profile; a compile check is not that gate.

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

Constraints run greedy only with host target selection on the resolved route described below. Constraints are refused with stop strings, with effective sampling
(`temperature > 0` on a sampling server) and with a non-empty tool offering. Every constrained response carries
`strix.constraint` (also with stream receipts off): `identity` (`json-object` or `strix-json-schema-v1`), `selection`,
`execution`, `phase`, `status` (`complete`, `incomplete`, `failed`), `accepting` and `reason` (`max_output_tokens`,
`context`, `reasoning_eos`, `cancelled`, `constraint_failure`, `execution_failure`). A reply cut by the budget is
reported incomplete, never repaired; Responses then has `status: incomplete`. `host_profile` reports the host
selection cost of the request (wall time, informational, differs between runs): `known_rows` selected
argmax-first (below), `rows` / `select_ms` selected on the full row (transforms, argmax, scores),
`mask_ms` building constraint masks and, on `mtp-host`, `window_rows` / `window_row_ms` for whole window
rows (logits readback and selection). Rows of rejected window suffixes are included.

### Policy selection routes

`--http-selection-mtp on` (default) admits `mtp-host` for closed R2/R4/R8
serial-derived owners whose per-row arithmetic preserves the complete serial target
FP32 logits and state. Both prefix commit and checkpoint/restore owners use this
admission. Tiled owners, GSQ owners, the legacy all-Q8 staged GDN pair, unknown
recipes and window-only inject/normalization or combine folds are excluded here.
These conservative exclusions fall back before drafting. The existing prefix gate
compares full-logit digests and committed state, including rejection depths and
owner changes; matching argmax IDs alone is insufficient.

Argmax-first selection (`--http-selection-fast on`, default) applies on every route. Masked greedy
picks the lowest-ID maximum over admitted IDs, so when a row's unmasked lowest-ID maximum is admitted
it is the masked result: the row needs only a membership check of that one token against the
constraint cursor (EOS only at a complete value, never the end-of-thinking ID), not the full mask
walk. On `mtp-host` the maximum comes from the window's GPU reduction (same finite first-max rule, both
zeros equal), so such rows also skip the logits readback and finite scan; on `serial-host` and serial
replays the host computes it from the row it read. Forced closure and reasoning-phase rows use it
directly. Rows with a nonfinite logit or a refused maximum, and requests with scores or transforms, are
selected on the full row; token IDs are identical either way. One difference to `--http-selection-fast
off` (every row walks the full mask): a full walk can stop a request on a mask-walk evaluation limit
reached by another token, which a membership check does not walk; all routes of one setting behave the same.

Every policy response reports `strix.selection_execution` (`mtp-host` or
`serial-host`) and `strix.selection_fallback` (null when admitted). Fallbacks are
`selection-mtp-off`, `no-window-owner` and `window-arithmetic-not-serial`, in that
order. The switch covers scoring-only and transform-only requests too. Feature
disable switches still refuse requests. Capabilities list routes available on the
loaded owners; the request receipt reports the resolved route.

The first token retains its terminal-prompt transaction. Each later MTP cycle uses
ordinary unconstrained proposals and selects only reachable target rows with the
same host selector, masks, transforms, forced controls and CPU scoring as serial.
Acceptance compares each draft with that policy-selected token. Prefix owners
commit the decided rows only; checkpoint owners restore and replay serially unless
the whole chunk matches without consuming EOS. Context/history/budget limits can
also cause transaction-aware serial verification within an `mtp-host` cycle.
Reconciliation completes before the retained state and every score publish, then
IDs emit. Rejected suffix rows publish nothing. Policy requests have no context,
frame, peer or wide-copy proposals and do not train ordinary shared calibration.

Exactness is relative to the same declared policy and numerical identity: complete
IDs, forced-control records and every chosen/top score must equal serial, with
bit-identical reachable full-logit rows and committed model state. CPU tests and
compilation do not establish GPU or complete-workload qualification. No MTP policy
speed claim is made.

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

- `--http-batch 0`: one request at a time, greedy only; concurrent requests wait in the queue.
- Batched: prompt blocks of several requests may form a segmented group; one native prompt unit owns the shared
  prefill arena at a time. With more than four decoding requests a cycle forms several 8-row cohort windows (two
  rows per request); joint planning (`STRIX_MTP_BATCH_JOINT=1`) runs in groups of at most eight requests.
- Every built slot holds a full context of state memory and its own window owners; memory grows with the built
  slots. A nested prompt pauses the long prompt for its own prefill; boundary cancellation is not on the
  one-at-a-time route.
- Each request resets the adaptive min-p, depth or survival controller to its configured prior, retaining
  the slot's device owners. Previous requests' acceptance estimates and scheduler-dependent cycle costs
  do not set the new request's draft depth. `STRIX_MTP_REQUEST_RESET=0` retains the earlier policy history
  for controlled comparisons. Target verification continues to decide every token and committed state.
- The window controller switches (`STRIX_MTP_CONTROLLER=survival`, `STRIX_MTP_ACTIVE_ROWS=1`,
  `STRIX_MTP_DRAFT_VOCAB=2`, the window folds) apply to served requests as on the command line. The draft
  vocabulary is chosen per request from that request's prompt.
- `n` > 1, images and audio are not supported. Logprobs use the resolved host selection route described above.
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

## Scheduling qualification

CPU: `strix-batch-service-handoff-tests`, `strix-batch-nested-recipe-tests`, `strix-http-slot-capacity-tests`,
`strix-mtp-window-controller-tests`, `strix-serve-api-tests`. GPU (pinned model and MTP arguments after `--`,
output outside the checkout; `STRIX_PROMPT_STATE_DIGEST=1 STRIX_PREFIX_CACHE_DIGEST=1` and
`STRIX_PREFIX_CACHE_DIGEST_COMPONENTS=1` add `prompt_state_digest`, `final_state_digest`, `final_state_position`
and per-component `final_state_components` to the responses):

```bash
# alone vs service off / cancel only / service at every layer; then a short request nested into an idle long one
python3 tests/batch_service_gpu_gate.py --short SHORT.ids --long LONG.ids --out "$OUT/service" --service-ms 1 -- CMD...
python3 tests/batch_service_gpu_gate.py --scenario nest --reference "$OUT/service/alone.json" \
  --short SHORT.ids --long LONG.ids --out "$OUT/nest" -- CMD...
# --http-batch 0 references vs one AUTO server at 1/2/4/8/16 clients (lazy growth witnessed)
python3 tests/http_slots_gpu_gate.py --prompt A.ids --prompt B.ids --out "$OUT/slots" --modes auto -- CMD...
```

The gates compare full token IDs, the final frontier and every committed target component (the prompt digest
only within one route: the batched route takes it before its last prompt window is folded in) and require
witnesses (`service_rounds`, `nested_admits`, `cancel_points_cancelled`, `lazy_slot_builds`).
