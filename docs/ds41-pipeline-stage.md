# deepseek41 pipeline stages (`strix/ds41-pipeline-v1`) and `--describe-json`

Two processes run one deepseek41 decode: the **FRONT** runs the embedding and target layers `[0, K)`, the **TAIL**
runs layers `[K, n)` and the head and answers each step with the greedy tokens. The TAIL is a role of
`strix-deepseek41-rank` (one node or the two-node EP+TP group, rank 0 listens); the FRONT is
`strix-deepseek41-pipeline-front` (backend `hip`: `DeviceForward` over `[0, K)`; other backends implement
`ds41::pipe::FrontBackend`, `include/strix/ds41_pipeline_front.hpp`). Arithmetic: the TAIL runs the same plan steps on
the same inputs as a whole forward, so tokens, logits and every op hash of layers `>= K` are bitwise those of the whole
run (gate below). Nothing changes without the new options: `DeviceForwardOptions::first_layer = 0` and `head = true` are
today's forward.

## Command lines

TAIL (each rank of the group; rank 1 opens no port, it follows rank 0 over the communicator):

```
strix-deepseek41-rank MODEL.gguf --role pipeline-stage --first-layer K [--layers 40] --rank R --transport usb4auto|single \
    --stage-listen HOST:PORT --stage-token-file PATH [--max-positions N] [--stage-once] [--stage-timeout S] \
    [--op-hashes] [--stream-experts]
```

FRONT:

```
strix-deepseek41-pipeline-front MODEL.gguf --backend hip --layers K --stage-connect HOST:PORT --stage-token-file PATH \
    --prompt-ids a,b,c [--generate G] [--max-positions N] [--fill N --fill-ids FILE] [--op-hashes] [--stream-experts] \
    [--export-at N --state-out FILE] [--handoff HOST:PORT] [--tail-layers N] [--close]
```

- `--stage-token-file`: 32 bytes as 64 hex characters (the session secret); the TAIL refuses a HELLO with another token.
- Both print `READY {json}` when they accept work (`role`, `layers`, `listen`/`stage`, `model_identity`); the TAIL also
  prints one `pos P fed T argmax N top X logits_fnv F` line per row and, with `--op-hashes`, the op hashes of its layers.
- `--stage-once`: the TAIL exits when the first session ends (tests); without it, it accepts the next session.
- TAIL wire path: the socket has `TCP_NODELAY`; while waiting for a request rank 0 spins on a non-blocking `recv`
  (busy poll, `SO_BUSY_POLL` requested best effort; `--stage-no-busy-poll` blocks in `recv` instead); a STEP payload is
  read straight into pinned host memory owned by the forward (`DeviceForward::receive_buffer`) and decoded with the
  stream as a view of it (`decode_step(..., view_stream = true)`), so the stream upload reads pinned memory with no
  intermediate copy; rank 0 forwards the same pinned payload to rank 1 over the communicator (rank 1 receives it into
  its own pinned buffer; rank 0 does not download the echo); the RESULT goes out as soon as the step returns (argmax,
  and the gathered logits' FNV unless `--no-gather`), before any printing. One message each way per step: no ack.
- `--stage-timeline FILE`: one CSV row per step (`idle_us` waiting, `recv_us` first byte .. whole payload, `decode_us`,
  `peer_us` rank-1 forward, `compute_us` the step, `send_us`, `tail_us` first byte .. RESULT sent, `device_step_us`);
  each session ends with a `tail timeline` line (medians over the one-row steps after the second).
- Recorded FRONT (tests and TAIL timing): `strix-deepseek41-rank ... --role pipeline-record --layers K --record FILE
  --prompt-ids a,b,c` runs `[0, K)` without the head on the group (world 1 or 2, the same flags as a whole run) and
  writes every step's STEP payload (after a HELLO) to FILE; `strix-deepseek41-pipeline-replay --record FILE
  --stage-connect HOST:PORT --stage-token-file PATH [--tail-layers N] [--timeline CSV] [--close]` sends them to a TAIL
  one at a time and prints each RESULT (`pos P fed T argmax A logits_fnv F rtt_us X`).
- Whole runs: `--export-state FILE` (state after the run) and `--import-state FILE` (continue from a state: its next
  token is fed first).
- `--backend host`: HostForward over `[0, K)` on the CPU (one row per step, no state export): the reference the device
  forward equals op by op, so a host FRONT + a device TAIL equals the whole run too.
- HTTP instead of `--prompt-ids`: `--http HOST:PORT [--context N] [--model-id ID] [--http-api-key-file F]
  [--default-max-tokens N]` serves the `strix_serve` routes (one request at a time, greedy, the prompt in windows of 8
  rows), `GET /health` (`{"status","stage":"connected|disconnected","layers":[0,K]}`), `POST /admin/drain` and
  `POST /admin/export-state` (the merged state after the last request, `application/octet-stream` with `X-State-Format`,
  `X-State-Sha256`, `X-State-Position`, `X-State-Next-Token`).

## The cut: what crosses at K

Per step of `R` rows at positions `p .. p + R - 1` (`ds41::pipe::plan_boundary`, from the model's attention
organisation; `step_bytes` gives the bytes):

| section | when | bytes per row |
|---|---|---|
| token ids | always | 4 |
| stream after layer K-1 | always | hc x hidden x 4 = 81 920 |
| `mix_ffn@K-1` (pre, post, comb of the FFN site; `pre` collapses layer K's attention input) | always | 96 |
| `kv@s`: compressed KV + index-key entries `[p/ratio, (p+R)/ratio)` of KV source `s < K` read by a layer `>= K` | K in 3..7 (s 2), 9..13 (8), 15..19 (14), 21..39 (20) | (288 + 68) / ratio: 178 or 356 |
| `topk@s`: per row the count and the ids of index source `s < K` whose selection a layer `>= K` attends with | K in 3..7, 9..13, 15..19, 21..23 (s 20), 25..27 (24), 29..31, 33..35, 37..39 | <= 4 + 512 x 4 |
| `cand@20`: the candidate blocks of the step's rows (layer 20 is the candidate source, 24/28/32/36 its users) | K in 21..36 | ceil(position / 8) |

At the default K = 23 one row at 8K context carries 81 920 + 96 + 356 + <= 2 052 + 1 024 bytes (+ 120 of framing).
Engram rows never cross: the TAIL reads its own tables from the token ids. The TAIL keeps mirrors of the KV sources'
caches it reads, filled by the `kv@s` sections, so a step carries only the entries it completed.

## Messages

32-byte little-endian header: magic `DS4P`, version 1, type, request id, payload bytes, flags, header check (FNV-32
of the 28 bytes before it). The FRONT sends requests; the TAIL answers each with one message or `ERROR` (text; the
session ends). TCP with `TCP_NODELAY`; every blocking wait is bounded.

| request | answer | payload |
|---|---|---|
| `HELLO` | `HELLO_ACK` (status 0, or 1 token / 2 cut / 3 geometry / 4 model identity / 5 max_positions) | token, K, layers, hidden, hc, vocab, max_positions, model identity (GGUF layout identity), backend, numeric identity |
| `STEP` | `RESULT` | flags (1: full logits, R = 1), position, rows, tokens, stream, sections (kind, layer, first, count, bytes) / position, tokens, logits FNV per row, logits |
| `COMMIT` | `ACK` | rows kept of the last step (verify windows) |
| `RESET` | `ACK` | new sequence |
| `EXPORT_STATE` | `STATE` | manifest JSON + `serialize_attention_state` of the TAIL's layers `[K, n)` |
| `IMPORT_STATE` | `ACK` | manifest + state with at least the TAIL's layers and the caches of the KV sources it mirrors |
| `PING` | `PONG` | JSON status |
| `CLOSE` | `ACK` | end of session |

## State

`AttentionState` (`deepseek41_attention_state.hpp`): rings, compressed KV + index keys, ratio-2 carries, the last top-k,
the token history (Engram n-grams are recomputed from it), the position and the next token; greedy sampling only (the
manifest says so). States are **split by layer range**: a FRONT exports `[0, K)`, a TAIL `[K, n)` (absent layers have an
empty ring), `merge_attention_states(front, tail)` gives the whole state, which a whole forward (the two-node group with
every layer) imports and continues bitwise; `slice_attention_state` cuts one. `serialize_attention_state` is the file's
bytes in memory (the `STATE` message). The FRONT's `--export-at N --state-out FILE` writes the merged state;
`--handoff HOST:PORT` moves the TAIL's state to a second TAIL over TCP mid-run (a TAIL's export carries its mirrors of
the KV sources before K as absent layers, so another TAIL imports it alone).

## `--describe-json`

`strix-deepseek41-rank MODEL.gguf --describe-json` and `strix-deepseek41-serve MODEL.gguf --describe-json` print one
line of `tensormeld/engine-describe-v1` from the model file's header (metadata and tensor table, no weight read; ~50 KB,
~0.05 s): engine (name, source revision, executable SHA-256, backend `hip`), the capabilities `ds41.attention`,
`ds41.core`, `ds41.engram`, `ds41.lookup`, `ds41.moe-shared`, `ds41.moe-routed`, the group mode `ep2-tp` (world 2,
`usb4`, `tcp`), the roles, the state block, argv templates per role and the `tensormeld/model-parts-v1` descriptor
generated from the file (`deepseek41_describe.hpp`): real tensor bytes per part, cone rows of the exact suffix replay
plan, the TP replicated fraction of the attention, per-part state and the pipeline cuts' side bytes. The model argument
is required (the descriptor needs it). The serve's previous launch description (memory per rank) is
`--describe-capabilities-json`.

## Gates

- CPU: `strix-ds41-pipeline-wire-tests` (cut of the released model per K, every message round trip, rejections, token
  file, loopback TCP), `strix-deepseek41-describe-tests`, `strix-deepseek41-attention-state-tests` (serialize == file,
  split/merge at K = 1/20/21/23).
- Device (one node, world 1, experts streamed): FRONT `[0, K)` + TAIL `[K, 40)` over TCP loopback == the whole run
  (tokens, logits FNV, op hashes of the TAIL's layers and of the FRONT's) for K in {1, 5, 20, 21, 23}; windows of 8 rows
  on a trimmed model; state handoff between TAIL processes and the merged state imported by a whole forward == the
  whole run; the rank without the new options == the previous binary (op hashes).

## Speculative decoding across the cut (DSpark on the TAIL)

`include/strix/ds41_pipeline_dspark.hpp` extends strix/ds41-pipeline-v1 so that the drafter runs where its inputs
are (the inputs of the target's last three layers live on the TAIL) and the FRONT only runs each verify window's first
layers. One request / response per window, no extra blocking acknowledgement:

- `STEP` with `kStepFill`: context rows (a prompt fed as decode rows from position 0); the TAIL commits every row and
  appends its hidden to the drafter.
- `SPEC_BEGIN {epoch, position, next token, max_new, eos}` -> `SPEC_ACCEPT` window 0: the plan of window 1
  (`[next, d_1..d_k]`, k chosen by the window controller from measured costs and calibrated survival; k = 0 when a
  single row is cheaper).
- `STEP` with `kStepVerify`, request id = `epoch << 32 | window`: the plan's rows after the FRONT's layers. The TAIL
  verifies, keeps `a + 1` rows (cut at an EOS or max_new), commits locally, appends the kept rows to the drafter,
  drafts and answers `SPEC_ACCEPT {argmax of every row, kept, published tokens, next token, done, plan of w + 1}`. The
  FRONT applies it locally (`CudaPrefill::commit_window(kept, next)`) and runs the next plan: the next VERIFY is the
  commit watermark of the previous decision.
- Idempotent decisions: the TAIL keeps the last decision and answers a VERIFY of the same (epoch, window) with it again
  without running anything (read and dropped before the rank-1 forward); another epoch or window is answered with
  status `stale`. The FRONT takes exactly the decision it waits for and drops the rest; a decision that does not come
  within its timeout is asked again with the same VERIFY bytes.
- `SPEC_COMMIT {epoch, window}` at the end of a sequence (acknowledged with the TAIL's position).
- Fallback: a TAIL without a drafter answers `SPEC_BEGIN` with status `off`; the FRONT then decodes with one-row STEPs
  (pipeline R1 of the same numeric identity).

TAIL flags: `--stage-dspark DRAFT.gguf [--stage-dspark-vocab SPEC] [--stage-dspark-policy controller|fixed:N]
[--stage-dspark-costs COSTS.json] [--stage-dspark-placement split|rank0] [--stage-fault-drop-accept N]` (split: both
ranks draft with the head split by vocabulary; rank 1 learns each decision in one control exchange before the draft).
Cost table: `{"V": {"1": ms, "2": ms, ..}, "D": ms, "A": ms}` (the split's measured window costs; the controller's prior,
then every measured cycle). CUDA FRONT: `--dspark-prompts FILE [--dspark-max-new N] [--dspark-eos ids]
[--dspark-row-fnv] [--dspark-timeout-ms T] [--stage-ports ...]` and test faults `--fault-drop-verify N`,
`--fault-dup-verify N`, `--fault-stale-verify N`. CPU test: `strix-ds41-pipeline-dspark-tests` (sessions over a lossy
channel == greedy).
