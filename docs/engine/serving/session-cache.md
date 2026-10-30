# Complete session cache

The r1b target-only CLI gate passed on a native node: P=32,768 inputs, one pending output,
S=128 inputs and 64 greedy outputs. Saved live A, never-saved L and fresh-process B matched
all token IDs, every raw FP32 vocabulary row (cmp), and the final canonical digest.
Measured P prefill: **22.6 s**; save stall: **2.51 s**; write/fsync: **0.91 s**; file: **975 MiB**;
restore: **1.04 s**; serial suffix: **6.5 s**; 64-output generation: **3.18 s**.
These results qualify the r1b route. New r2 tiled/MTP/HTTP paths remain drafts pending a
Linux/HIP build and native gates. CPU tests and declaration-only checks do not qualify
native completeness or numerics. Existing prefix formats/defaults and the r1 gate are unchanged.

## Serving defaults

Serving admission uses the `session_route_qualifications` table in
`apps/qwen_effective_config.hpp`. **No HTTP or MTP serving route is GPU-qualified today**;
all decline automatic admission with exactly one line:
`session cache: off (route not yet GPU-qualified: <route>)`.
The table contains serving routes only; its set of qualified routes is currently empty.
CLI save/load are explicit commands, independent of `--session-cache on`. Default ON
applies only to qualified serving routes; it does not enable a CLI cache. An explicit
CLI cache switch declines with `session cache: off (cli-save-load-explicit)`.
After a serving route passes its compiler, native execution, fidelity and complete-workload
checks, its table entry is the single admission switch. An explicit `--session-cache on`
still requires that qualification.

Qualification tools explicitly supply both `--session-mtp-fixed` and
`--session-tiled-suffix` to exercise the draft protocol. This diagnostic access is separate
from production admission; `--session-live-control` may then measure a never-captured live
boundary. It does not qualify a serving route or change the table.

Only requests naming a `session` enter the fixed session protocol and its draft/sampling
restrictions. Unnamed requests keep ordinary batching, request draft overrides and seeded
sampling, including `--http-batch 1`. Named requests wait for active batch jobs to finish;
management operations and batch execution share the lane mutex. Named requests with
constrained output are refused with a 400 naming `session`. Batches above one still
decline session management. CLI save/load remain explicit commands.

`--session-cache off` or `STRIX_SESSION_CACHE=0` disables serving sessions;
`--baseline` / `STRIX_BASELINE=1` also disables them. `--session-disk off` or
`STRIX_SESSION_DISK=0` disables disk only. Explicit CLI switches override their session
environment switches; baseline takes priority. A declined serving cache emits one line
`session cache: off (<reason>)`. `--print-effective-config` includes a `session_cache`
object with enabled/disk state, RAM/disk byte budgets, maximum parked count, directory,
cache decline reason and disk decline reason. Its `policy` is
`qualified-serving-routes-only` and `cli_operations` is `explicit-save-load`.

RAM defaults to min(8 GiB, 10% of startup MemAvailable), with at most eight parked
sessions. Admission conservatively reserves the complete target and predictor state,
checkpoint working copies and resident state at the configured context. Unknown or
insufficient memory declines the cache. Device-resident state counts against the guard.

Disk defaults to `$XDG_CACHE_HOME/strix-engine/sessions`, falling back to
`$HOME/.cache/strix-engine/sessions`, and min(64 GiB, 25% of startup filesystem free space).
The directory must be creatable/writable and have room for two checkpoints. Disk failure
leaves RAM sessions enabled. Completed named turns are published opportunistically;
explicit `checkpoint` with disk durability reports publication failures. Disk eviction
removes the least recently used published checkpoint files to enforce the byte cap;
compatible checkpoints can resume after restart. A persistent `.session.lock` file in the
session directory uses an interprocess exclusive `flock` for accounting, eviction, reads,
deletes and publication (Windows uses a file lock). Replacement admission counts the old
published envelope and the new temporary envelope together. An oversized replacement is
refused while preserving its destination; interrupted temporary files are cleaned under
the same lock. Existing checkpoint formats are unchanged.

## Flags

The existing [prefix cache](prefix-state-cache.md) owns the allocator, registry and LRU.

| Flag | Contract |
| --- | --- |
| --session-save DIR | Capture a committed turn and atomically publish DIR/session.bin. |
| --session-load DIR | Verify/restore a complete frontier; supplied IDs are a suffix. |
| --session-then-prompt FILE | Continue the same live state after the first turn (save or live-control required). |
| --session-live-control | Identical turn boundary/continuation without capture/publication. |
| --session-first-generate N | Override only the first turn output budget. Default: --generate. |
| --session-logits FILE | Raw native FP32 vocabulary rows for CLI continuation or latest fixed-MTP HTTP request; no header/conversion. |
| --session-tiled-suffix | New aligned known-prefix continuation; off by default. |
| --session-mtp-fixed | New fixed-depth MTP session protocol with R1 target verification. |
| --session-capture-host-bytes N | Four-copy capture/restore/preflight working-set bound; default 8 GiB. |
| --session-validate-only | CPU envelope/executable/supplied-config preflight; normal restore checks model/device/plan too. |
| --session-cache on\|off | Serving cache switch; default on when admitted. |
| --session-disk on\|off | Disk tier switch; default on when admitted. |
| --session-disk-bytes N | Disk checkpoint byte cap. |
| --session-dir DIR | Override HTTP checkpoint directory; files are DIR/ID/session.bin. |
| --session-ram-bytes N | HTTP envelope/LRU and temporary capture/read working-set admission budget. |
| --session-max-parked N | HTTP RAM LRU session count limit. |

Use the same binary, full assets, numerical/environment settings, context and compiled plan
in every arm. Supplied CLI spelling/order is conservatively bound except input/output,
management paths/budgets, HTTP port and output budgets. STRIX_*, HIP_*, HSA_*, AMD_*, ROCR_*
and GPU_* values are bound except operational hash-cache/GPU-turn variables. Observed device
architecture/name, HIP runtime/driver and rank=0/world=1 are bound. Input-length tail-owner
pruning is disabled for explicit session numerical protocols; operational cache/disk
switches, directories and budgets do not own numerical routes or affect tail pruning.
Session mode bypasses legacy prefix lookup/capture. Session continuations disable L1 PLE
reclaim and its admission credit because their positions differ from the position-zero plan.
Capture and restore intersect the host working-set budget with fresh MemAvailable after
model, prefill and live state backing are retained; unavailable observations refuse admission.
Restore checks the four-copy envelope bound before allocating the pinned read buffer.

Sessions require eager greedy floating output, --known-prefix-window, --known-prefix-full-rows,
--known-prefix-prefill-rows >= 1024, positive output budget and sufficient context.
Sampling/RNG, RAW/teacher/final-state export, graph/integer, GDN islands, imported/tainted states
and other speculative producers are refused. r2 session management uses one execution
lane on --http-serve. Batch sizes above one decline session management and retain
ordinary batching. Unnamed requests always use the ordinary request runner.

## Exact suffix rule

History contains consumed inputs; the final emitted selection is a separate pending token.
The next turn consumes pending once before its suffix. Let p be the consumed position,
T the qualified full block width, n the number of pending-plus-suffix inputs, and
alignment = (T - p % T) % T.

Without --session-tiled-suffix, or when n < alignment + T, all continuation inputs run
serially, preserving r1 behavior. Otherwise run alignment serial rows, enter the same
known-prefix cursor used after a cached boundary, execute full T blocks and configured
tail blocks, then their serial remainder. No padding, rounding or replay of consumed
inputs occurs. The completed terminal head is logged once before generation. A/L/B use
the same rule and tail owners. The new target suffix cursor disables deferred/balanced
scheduling across its observation boundary. MTP uses its existing tiled prompt cursor's
block/tail scheduling. Each producer route must match its own live/restored arms.

## MTP committed-turn protocol

--session-mtp-fixed requires the native draft chain and **full draft vocabulary**, with
no prompt-derived subset/context vocabulary or Q4 compact head, sampling, n-gram/context/tool/copy extension,
cohort serving or adaptive window verification. Fixed draft depth (including a configured
depth cap) and serial target verification read no elapsed time or learned acceptance
controller. Named turns do not use target window verification; HTTP may retain window owners for
ordinary unnamed requests.

The existing registry adds predictor KV/raw-index rows [0,p-1) and the actual final target
wide hidden row h[p-1]. At every committed turn, both live and restored protocols finish
reconciliation, retain durable rows, copy the final ticket's hidden bytes, discard transient
tickets/proposals, and create an idle predictor at source p-1. No device row is recomputed.
The fixed controller's canonical turn-entry state is its bound configuration, with no
adaptive history across turns. A and never-saved L normalize exactly like B. This is an
explicit session numerical mode: fixed-window MTP sessions use a different speculation schedule from ordinary adaptive
MTP serving. Qualification compares exact output against the live session (A == L == B);
identity with session-less adaptive MTP serving is not proven.

CLI fixed-session calibration uses the pinned golden payload. HTTP retains ordinary
startup calibration for unnamed-request identity. Restoring a different suffix still
runs the full native replay/activation gates at startup.
The envelope includes predictor/accepted frontiers, emitted count, consumed IDs and pending
selection. The next turn uses the held row and first new known input to prime the next
predictor row, like a cached-prefix continuation. It never forges a historical execution
ticket. Completed serial and adopted tiled terminal heads have a raw FP32 qualification observer.

## HTTP API

Use the supported tiled greedy MTP serving flags; the resolver supplies session enablement
and default budgets. Override budgets and directory as needed. The request field
"session": "conversation_1" names a completions/chat/messages session. IDs are 1..128 ASCII
letters, digits, underscore or hyphen. Without session enablement, servers ignore this field
(including its type/value), preserving the default-off API behavior.
Request-level draft overrides are refused before queue admission.

Every successfully committed named request is captured into host RAM. Cancellation
invalidates that name's RAM frontier and prevents stale disk fallback within the process. A later prompt resumes only if
its IDs begin with every consumed ID followed by the exact pending ID. Textual similarity
is insufficient. A mismatch, incompatible/corrupt opportunistic disk lookup, or evicted
RAM-only state is a miss and follows fresh prefill. Usage cached-token count reports the
restored consumed position. Chat-template/tokenizer changes can cause prefix misses;
raw-ID /v1/completions requests provide an unambiguous gate.

| Endpoint | Request / result |
| --- | --- |
| POST /v1/sessions/ID/checkpoint | {"durability":"ram"} verifies retention; "disk" atomically publishes the current envelope. |
| POST /v1/sessions/ID/park | Idempotent retention of the already parked committed turn. |
| POST /v1/sessions/ID/resume | Verify and restore target/predictor/hidden state into the execution lane; retain RAM backup. |
| GET /v1/sessions | `sessions` rows with `id`, `position`, `consumed`, `pending` (token ID or null), `bytes`, `state` (`parked` or `disk`), `durability` (`ram` or `disk`). |
| GET /v1/cache/stats | `ram_bytes`, `ram_budget`, `parked`, `disk_bytes`, `hits`, `misses`, `refusals`; additional diagnostics under `engine`. |
| DELETE /v1/sessions/ID | Delete RAM entry and published disk file; no recursive deletion. |

Management and generation share one mutex and wait for a committed turn. Completed turns may automatically replace the durable checkpoint when the disk tier is
enabled. Qualification preserves turn 1 separately and restores it before B starts;
turn 2 cannot silently replace the frontier being tested. After restart durable state is
restored lazily. Within a running process, eviction of a newer RAM turn
never silently resurrects an older disk frontier; lookup misses instead. Explicit restore
errors use `{"error":{"code","message"}}`: unknown IDs are 404 `session_not_found`,
incompatible envelopes 409 `session_incompatible`, corrupt envelopes 422 `session_corrupt`,
and RAM/disk budget refusals 507 `session_budget`. DELETE removes both versions and reports
404 for unknown IDs.

Tombstones for superseded durable names use fixed slots (129 bytes each), capped by
--session-max-parked and reserved inside --session-ram-bytes. RAM-only names retain no
tombstones. If all slots fill, disk fallback is disabled for the rest of the process;
RAM lookup still works, and restart can read the explicit durable checkpoints again.
Stats expose tombstone reservation/use and this fail-closed state.

For qualification, --session-live-control also lets consecutive unnamed HTTP requests
continue the occupied live lane when the exact consumed-plus-pending prefix matches.
These requests never capture/encode/decode/restore an envelope. A management operation,
a named request, cancellation or a prefix mismatch breaks that live chain. The flag is
opt-in; ordinary unnamed requests start fresh.

The response strix.session exposes generated IDs, SHA-256 of **each full raw FP32 output
row**, final canonical state digest and accepted/proposed counts. Standard logprobs are
currently null. These are exact-byte hashes, with no decimal conversion/tolerance.
Both gates additionally compare every byte of the complete FP32 files with cmp.
For fixed-MTP HTTP qualification, --session-logits FILE exports the most recent request's
complete native FP32 rows, truncating FILE on each serialized generation request and
closing it before returning the response. The gate preserves each file before the next request.

## Envelope, validation and memory

strix-session-v1 remains alongside the legacy prefix format. Compatibility binds FULL-file
SHA-256 model/draft/overlay/replay assets and executable, schema, effective numerics, compiled
component identity, dtype/layout and chunk plan. Hash caching uses canonical path, size,
inode, mtime and ctime; identities are rechecked before publication. Sampled model hashes
never authorize restore. MTP mode has a distinct compatibility record.

Every rank-0 replicated section has canonical name/order/offset/length and SHA-256; a final
manifest root commits the envelope, history and all sections. Restore validates schema,
bounds, compatibility, hashes, vocabulary, registry geometry and lineage before replacing
live state. Copy failure poisons the lane. Linux publication uses exclusive temp file,
complete write, fsync, rename and directory fsync. No repository history is rewritten.

Target state includes occupied KV/index rows, recurrent matrices, convolution histories,
host n-gram/position state, generations and arithmetic lineage. The final digest includes
history, pending and all frontiers. Unused future rows/padding are omitted. MTP adds its
occupied predictor rows and held hidden row.

Parking uses the existing pinned allocator/LRU. Returned handles remain charged after
eviction/deletion until released. There is **one preallocated execution lane**, not one
device allocation per parked session. Its baseline backing remains in the process/model
memory plan. Host envelopes are additional physical LPDDR copies on Strix and are counted
once, independently of that baseline; no separate fictitious VRAM allowance is added.
Before capture, LRU eviction leaves the conservative four-copy bound plus retained envelopes
within --session-ram-bytes. RAM lookup reserves parser/lineage space; disk loading reserves a four-copy read/decode bound. Admission can evict
sessions; exceeding either working budget fails explicitly. This accounts cache/temporary
allocations, not complete process RSS or total-machine memory.

Whole envelopes are synchronous without segment deduplication, streaming or O_DIRECT.
Capture can stall inference. Byte/count limits evict least recently used envelopes. The
serving disk tier enforces its byte quota; explicit DELETE removes published checkpoints.
CLI save directories remain under caller management.

## Qualification

Gate scripts require Linux Bash, Python 3, GNU timeout, cmp and curl (HTTP gate).
CPU only:

~~~bash
cmake -S . -B build-session-cpu -DSTRIX_ENABLE_HIP_RUNTIME=OFF -DBUILD_TESTING=ON
cmake --build build-session-cpu --target strix-session-state-cache-tests strix-prefix-state-cache-tests strix-http-session-store-tests -j
ctest --test-dir build-session-cpu -R 'strix-(session-state|prefix-state)-cache-tests|strix-http-session-store-tests' --output-on-failure
~~~

Original r1 gate (unchanged, S=128 supported):

~~~bash
bash tools/sessioncache/qwen-session-gate.sh BIN MODEL P_IDS S_IDS NEW_OUT -- QUALIFIED_TARGET_FLAGS...
~~~

New tiled CLI and HTTP gates require seven separate invocations with the same arguments,
inputs, numerical environment and output directory. Run phases 1–3 in separate SINGLE GPU
turns; never loop them in one allocation. Phase 4 is CPU-only and launches no model.
Each phase requires COMPLETE receipts and nonempty artifacts from all earlier phases;
phase 1 requires a fresh empty output directory outside Git. A failed attempt requires a
new run directory. `STRIX_HASH_CACHE` is operational and may change.

| Phase | CLI target or fixed MTP | HTTP fixed MTP |
| --- | --- | --- |
| 1 | Save A, continue, export logits | Never-captured unnamed L turns 1/2 |
| 2 | Never-saved L, continue | Named A turn 1, disk checkpoint, turn 2 |
| 3 | Fresh-process restore B, CPU preflight negatives | Disk restore, A->B->A parking, B turn 2, prefix miss/delete, CPU negatives |
| 4 | Compare A/L/B; qualification pending identities | Compare L/A/B; qualification pending identities |
| 5 | Ordinary baseline binary, no session options | Baseline binary, batch 1, seeded sampling, unnamed request |
| 6 | Candidate cache/disk OFF | Candidate cache/disk OFF, same request |
| 7 | Candidate explicit CLI cache switch (declined), no session command; compare identities | Explicit draft cache ON, ordinary unnamed batch 1; compare all identities |

For each row above invoke the matching command with PHASE replaced by 1 through 7:

~~~bash
bash tools/sessioncache/qwen-session-cli-mtp-gate.sh PHASE BIN MODEL P32K_IDS S_IDS NEW_OUT -- QUALIFIED_TARGET_FLAGS...
bash tools/sessioncache/qwen-session-cli-mtp-gate.sh PHASE BIN MODEL P32K_IDS S_IDS NEW_OUT -- QUALIFIED_MTP_FLAGS... --session-mtp-fixed
bash tools/sessioncache/qwen-session-http-gate.sh PHASE BIN MODEL P8K_IDS S_IDS NEW_OUT -- QUALIFIED_MTP_FLAGS...
~~~

For the target-only unaligned restore case, invoke `unaligned-1`, `unaligned-2`,
`unaligned-3`, then `unaligned-4` in separate turns with the CLI gate above and a new OUT.
It defaults to FIRST_GENERATE=2 and requires T=1024, S=2046 IDs and no MTP. The saved
frontier is 32769 with a pending token; all three continuations must report 1023
serial alignment rows before a full tile. Phase 4 compares IDs, every logit byte and
state digests. This script arm is a draft; no native result is claimed. The ordinary
CLI ON/OFF identity arms check the explicit-switch decline and do not exercise an
automatically enabled CLI cache.

CLI P contains exactly 32,768 IDs; HTTP P contains exactly 8,192 IDs. With the default
T=1024, CLI FIRST_GENERATE=1 needs S=1023 IDs; HTTP FIRST_GENERATE=65 needs S=1983 IDs
(alignment=960). Use these minimum suffix lengths for the default budget estimates.
Provide enough context for P, first outputs, S and 64 continuation outputs.
HTTP requires an off-grid frontier. SESSION_RAM_BYTES defaults to 12 GiB.

Before any engine launch, every phase prints its input-derived cold/warm estimate and
refuses estimates above 400 s. Admission caps FIRST_GENERATE at 65, S at 4095 IDs,
power-of-two T at 8192, alignment at 1024 rows and alignment plus serial tail at 1024 rows; the
suffix must still cover alignment plus one full tile. Larger tiles or off-grid plans
can therefore be refused even below the individual caps. All limits apply per invocation.
The estimate uses cold load 260 s / warm load 30 s, tiled prefill 0.7 ms/token,
serial alignment/tail 50 ms/token, generation 50 ms/token, and checkpoint write/restore
1 GB/s (decimal). Envelope bounds use one quarter of the four-copy capture budget:
2 GiB CLI, 3 GiB HTTP. Phase-specific copy/read counts include checkpoint backup,
parking/capture/restore and CPU negative copies/reads. A 40 s reserve covers hashing/CPU
checks (20 s), setup/report (10 s), and cleanup (10 s). With default T, FIRST_GENERATE,
RAM and minimum S, CLI phases 1/2/3/4 estimate 333.35/326.90/329.69/44.29 s;
HTTP phases estimate 360.90/383.45/396.41/46.44 s. Phase 4 has no model load.
Hash-cache misses or slower native execution may exceed these planning assumptions;
estimates are admission checks, not measured qualification. Each invocation has a
440 s whole-phase timeout plus at most 5 s kill grace; the internal deadline reserves
10 s for cleanup/report. Identity phases each load one model, prefill P and generate 64
outputs; their cold estimate is 260 s + P*0.7 ms + 64*50 ms + 40 s. Phase 7 grants PASS
only after baseline == OFF == ON-unnamed for IDs and available numerical digests;
HTTP additionally compares choices/logprobs, usage and sampling identity. Missing digest
evidence fails closed. Phase 4 alone never establishes qualification.

Both phase-4 gates compare exactly 64 output IDs, every raw FP32 byte via complete-file
cmp, canonical digest and fixed-MTP acceptance counts (when enabled). HTTP also checks
each exported row against its response hash, initial turn identity, logprobs if exposed,
parking budgets and management behavior. CLI and HTTP phase 3 perform four CPU-only
negative preflights after their GPU work; B always follows A in a later phase.
These r2 paths remain drafts until compiler, native execution, model fidelity and
complete-workload gates pass. No r2 native gate was run here.

Both `--session-tiled-suffix` and `--session-mtp-fixed` are **bit-changing** numerical
modes: each changes the arithmetic schedule relative to its ordinary execution path.
Neither establishes equality to ordinary serial suffix execution or adaptive MTP.
They are independent explicit opt-ins and are suppressed by `--baseline`,
`STRIX_BASELINE=1`, and `STRIX_QPF2K_DEFAULTS=0`. Effective configuration and
`GET /v1/capabilities` disclose each requested/effective mode, suppression reason,
and draft qualification status. Turning off serving cache admission alone does not
select a numerical mode. Native full-workload equality qualification remains pending.
