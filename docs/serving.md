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
[prefix-state-cache.md](prefix-state-cache.md#mtp) and [mtp-native-cli.md](mtp-native-cli.md)) plus:

| Flag | Default | Meaning |
|---|---|---|
| `--http-serve` | off | serve requests after the calibration replay instead of running the command-line request |
| `--http-bind ADDR` | `127.0.0.1` | listen address (numeric) |
| `--http-port N` | `8080` | listen port (`0`: any free port, printed on stderr) |
| `--http-model-id NAME` | `qwen3.8-flash-next` | model name reported by `/v1/models` and responses |
| `--http-api-key-file FILE` | none | require `Authorization: Bearer KEY` or `x-api-key: KEY` on the API routes |
| `--http-batch N` | `0` | `1`..`8`: batched decoding with N request slots ([below](#batched-decoding)); `0`: one request at a time |
| `--http-sampling greedy\|reject\|sample` | `greedy` | requests with `temperature > 0`: run greedy and say so in the response, answer 400, or sample (sampling-v1, batched route only) |
| `--http-default-max-tokens N` | `4096` | generation budget when a request gives none |
| `--http-max-queue N` | `64` | waiting requests before new ones get 503 |
| `--http-max-connections N` | `64` | open connections before new ones get 503 |

The token IDs on the command line are the **calibration prompt**: its first
`--mtp-calibration-rows R` pairs are replayed against the golden (`--mtp-golden`) to qualify the
predictor, exactly as for a command-line request. Requests do not need to share anything with it.
`--generate` is ignored while serving. The prefix state cache flags (`--prefix-cache-host-bytes`,
`--prefix-cache-every`, `--prefix-cache-dir`) apply across requests.

The prompt may run a separate prefill profile (`--prefill-numerical-profile`, see
[prefill-numerical-profile.md](prefill-numerical-profile.md)), for example the fast prefill profile
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
  [multi-request cohort](multi-request-cohort.md#global-row-budget-and-wide-steps-opt-in)).
- A request leaves its slot when it ends (stop token, budget, context, stop string, disconnect), without stopping the
  others; its slot is free for the next queued request.
- Greedy requests return exactly the tokens they return alone (and on the one-at-a-time route): every request keeps
  its own state and arithmetic, and the shared steps compute each row as the request's own window would.
- `--http-sampling sample`: a request with `temperature > 0` samples with its own policy (`temperature`, `top_k`,
  `top_p`, `min_p`) and `seed` (random when absent), with speculative decisions on its drafts (D-040; on the GPU as
  in [sampled-decoding.md](sampled-decoding.md)). Its tokens follow the target distribution exactly; they are not
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
| `GET /v1/models` | the served model |
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
