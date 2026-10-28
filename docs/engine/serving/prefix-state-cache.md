# Prefix state cache

Agent requests resend long, mostly identical prompts: a system prompt, tool schemas and the
conversation so far. `--prefix-cache-host-bytes N` keeps the complete model state at block
boundaries of the tiled prefill, and a later request whose prompt starts with a cached prefix
restores that state and prefills only the rest. The exactness contract is bitwise equality with prefilling the whole prompt under the same
configuration. The new certified MTP path still awaits native qualification.

It runs on the tiled known-prefix prefill
(`--known-prefix-window --known-prefix-full-rows --known-prefix-prefill-rows T`), with serial
greedy decode or with MTP decode (see [MTP](#mtp)). Every other route rejects the options.

## Options

| Option | Meaning |
|---|---|
| `--prefix-cache-host-bytes N` | Explicit byte budget of the host tier (pinned host memory). |
| `--prefix-cache-dir DIR` | Optional disk tier: checkpoints are written behind the request and found by later processes. |
| `--prefix-cache-disk-bytes N` | Optional disk budget (oldest checkpoints are deleted first; 0 = unbounded). |
| `--prefix-cache-every N` | Capture interval in tokens, a multiple of T (default: 4096 rounded up to a multiple of T). The last eligible block boundary is selected; incomplete or uncertified boundaries are skipped. Certified async boundaries use stable snapshots. |
| `--prefix-cache-staging-bytes N` | Device snapshot cap; 0 defaults to one T-block of registered rows plus fixed state. Environment: `STRIX_PREFIX_CACHE_STAGING_BYTES`. |
| `--prefix-cache-then-prompt FILE` | Repeatable. After the main request, run these prompts (token IDs separated by commas or whitespace) in the same process, each from a fresh session, through the same cache. Serial-decode route only. |
| `--known-prefix-tail-rows T2[,T3...]\|none` | Independent of the cache: after the last complete T-row block, run the remainder as complete blocks of the largest tail size that fits (descending sizes, 16..1024, multiples of 8, less than T), and only the rest as serial rows. Default (D-056): the sizes of 1024, 256, 64, 16 below T, whenever the tiled prefill has full rows and an output head; `none` keeps the remainder serial. |

Diagnostics: `STRIX_PREFIX_CACHE_LOOKUP=0` captures without ever restoring (every request is cold);
`STRIX_PREFIX_CACHE_DIGEST=1` adds a SHA-256 of the whole prompt state to the MTP receipt.

The receipt gains `prefix_state_cache` (serial route: identity, geometry, main request, follow-ups,
store counters) or `mtp_tiled_prompt.prefix_cache` (MTP route).

## What a checkpoint holds

A checkpoint at block boundary P = k·T is the session state after the first P prompt tokens:

- per-position rows of every attention layer: KV rows and raw indexer keys, positions [0, P);
- the state overwritten in place: GDN recurrent matrices, GDN and PLE convolution histories, the
  n-gram history and the position history;
- the session counters, state generations and the arithmetic lineage (every block interval with its
  producer identities);
- on the MTP route, the predictor's cache rows of the pairs (h_i, t_{i+1}) for i < P-1 and the target's
  wide hidden row h_{P-1}.

Per-position rows never change once written, so they are stored once per block (a segment) and the
checkpoints of one prompt share their common segments. The rest (about 111 MiB for this model) is
stored per checkpoint. Lookup returns the longest cached boundary P < prompt length (at least one
prompt token always runs, to produce the first output's logits); restore copies the rows and the
fixed state into the fresh session and republishes the recorded counters and lineage.

## Keys and identity

`identity = SHA-256` of: model content identity, executable digest (disk tier only), state layout
identity (includes the context), runtime compatibility identity, decode execution and kernel-plan
identities, prefill (sibling) execution and kernel-plan identities, tiled prefill plan identities,
prefill and decode numerical profile identities and their pair identity, T, the tail sizes and the context size.
On the MTP route it also covers the sidecar content, the MTP plan and native profile identities and
the draft-kernel switches (the predictor's cache rows are built with the draft kernels unless
`--mtp-draft-exact`).

`digest_k = SHA-256(digest_{k-1} || tokens of block k)`, `digest_0 = identity`. A digest names one
identity and one exact token prefix; the stored tokens are compared as well. Why each field matters:

- the tiled prefill reproduces serial rows bit for bit within one numerical profile (D-034), but with
  a split pair (D-033) complete blocks run the prefill profile and the unpadded remainder the decode
  profile, so which rows run which arithmetic depends on T (and the tail sizes). Checkpoints are only taken at
  block boundaries and the suffix of a restored request keeps the same block alignment as the full
  prompt, so the row-to-profile assignment is the same as without the cache;
- the decode identities do not change the state at a block boundary, but are part of the key so that
  no two configurations whose outputs could differ ever share an entry;
- context size and layout fix the state geometry; the executable digest keeps a disk tier from
  serving a state that another build produced.

## Storage

Host tier: pinned host memory, fair retention by default or explicit LRU within the byte budget.
A segment stays alive until its last checkpoint or reader releases it. Disk hits promote into
the host tier unless the request suppresses retention.

Disk tier: `seg-<digest>.tmps` (one block's rows) and `ckpt-<digest>.tmpc` (fixed state, metadata
and the list of segment digests), written to a temporary name and renamed. Headers carry identity,
digests and a 64-bit payload checksum; segment tokens are re-hashed on load and must chain to the
requested digest. A file that fails any check is skipped (counted) and the next shorter prefix is
tried. Writes run on a background thread; the process waits for them before exiting.

## MTP

With `--mtp` and the tiled prefill flags the prompt itself runs through the tiled prefill:

1. Calibration: the first R prompt pairs (`--mtp-calibration-rows R`, default 8) run serially through
   the MTP target producer and are replayed against a golden of exactly those pairs (dumped with
   `--mtp-golden-dump` under the same flags). This qualifies and activates the predictor as before.
2. The target state and the predictor are reset; the longest cached prefix is restored.
3. Complete T-row blocks (then complete tail blocks) run tiled. A step observer copies every
   row's wide hidden state (the MTP target tap) out of the block arena, and the predictor's cache
   rows are built from those copies. Eligible certified boundaries use asynchronous snapshots
   without changing the priming schedule (see [Certified MTP subset](#certified-mtp-subset)).
4. A block (T rows or a tail size) that ends exactly at the last prompt row also computes that
   row's head (terminal-row output policy). The first draft cycle starts from that row: its wide
   hidden state and FP32 logits mint the pending target ticket (`HipMtpTargetHidden::adopt_prefill_row`,
   same first-max selection as a serial row), so no prompt row runs serially.
5. Otherwise the remaining rows, at least the last prompt row, run serially through the MTP target
   producer, whose last ticket starts the first draft cycle. Decode is unchanged (adaptive windows).

`STRIX_MTP_TILED_START=0` restores the previous handoff (the last prompt row always serial; a prompt
of exactly k·T tokens then runs its last T rows through the tail blocks and serial rows). The receipt's
`mtp_tiled_prompt.tiled_start` says which handoff ran. A split numerical-profile pair is rejected on
this route.

## Measured

See docs/decisions/decision-register.md (D-035).

## Prefix retention policy

The [prefix state cache](prefix-state-cache.md) retains exact, complete tiled
boundaries. Explicit serial and MTP cache configuration leaves the cold tiled plan unchanged:
store absence, occupancy, quota and retention do not select merge, balance or predictor
priming. A host-only enumerator supplies the same whole-prompt units and continuation
cursor with storage enabled or disabled. Automatic host budgets still wait for
native route qualification. No resident pool is implemented
by this policy slice, and CPU policy tests do not establish GPU output qualification.

### Certified serial subset

The serial tiled route uses the incumbent `KnownPromptChunkCursor` for execution and
for a CPU-only walk of the whole prompt. Cache presence never disables deferred,
merged or balanced scheduling. Each executed unit records absolute bounds, owner
width, T-owner status, balanced/merged class, deferred completion and head intent.
A checkpoint requires a contiguous prefix of ordinary, completed, solo full-T units
from position zero, matching the preflighted trace and ready generation/lineage
metadata. Balanced, merged, deferred, tail and serial prefixes decline capture.

The versioned full-T certificate records the serial route, resolved plan and profile
identities, each producer unit's class and expected state generations. Its namespace
also binds the canonical effective-execution signature, separately from MTP and batch
reuse. Lookup checks the certificate and complete checkpoint before any state copy.
The consumer cursor must accept the boundary in its original whole-prompt plan;
restore advances that cursor without backend calls and executes its original suffix.
It never constructs a new request from the remaining tokens.

Serial capture retains the synchronous, fenced snapshot path. Asynchronous serial
capture declines as `serial-async-capture-unsupported`; no unordered snapshot is
submitted. Certified serial checkpoints are RAM-only, including explicit legacy
configuration. Old uncertified disk/RAM entries and segment grids smaller than T
cannot authorize reuse. Other legacy disk APIs and formats are unchanged.

Request receipts and published cache statistics expose named `certificate_declines`.
Reasons distinguish unsupported plans/boundaries/units, mismatched producer or state
metadata, missing certificates, unsupported whole-prompt continuation, disabled
lookup and absence of a certified checkpoint. Effective configuration lists the
protocol and refusal names. CPU schedule, continuation and copy-submission tests do
not establish native token/logit/state equality; automatic qualification stays gated.

### Certified MTP subset

The new MTP path uses a distinct full-T plan-family identity covering the existing
model/profile/layout identities plus resolved schedule, output policies, tiled-start
and priming flags. In-memory producer certificates record each full-T unit through P.
Segments share only within this family. Balanced units (including full-sized balanced
units), merged units and tails are conservatively excluded from certified prefixes.

Lookup leases candidates in decreasing P order. It checks tokens, producer proof,
metadata/lineage, section geometry and chain before any device write or accepted-hit
accounting. A rejected candidate increments `plan_misses`, then lookup tries a shorter
candidate. Restore resumes the cursor computed from the whole prompt, preserving its
cold suffix instead of planning again from the remaining length. Partial-copy failure
keeps the existing poison/quarantine behavior.

Capture requires an actual incumbent boundary, 0<P<n, P divisible by the
segment grid, certified ready target metadata and successful target completion
validation. It never creates a numerical unit or forces synchronous priming.
Batch/group/SMALL, named-session suffixes and imported/exported state remain outside
the certified subset. Deferred target validations still pending at the boundary skip.

A process-wide slot admits at most one asynchronous snapshot descriptor. Admission
reserves physical host credits and allocates every pinned destination, device staging
range and event before submitting copies. No credit, an occupied slot, allocation
failure before submission, or an oversized candidate skips without waiting. The
plan enumerator must identify the next mutator as a block of the same T owner,
and its actual stream handle must equal the snapshot producer stream. Tail,
window, serial, group, handoff or unknown next owners skip the boundary; no producer
wait is added to save it. Only under this same-stream condition do target snapshot
reads precede the next block's writes by stream order. An incumbent partial block
of that same T owner is also safe by stream order. Host history and metadata
are copied at P. The existing MTP worker submission copies predictor rows and held
hidden after priming and the incumbent held copy, then records its ready event.
The next predictor/held writer must use that same MTP stream. The next target block
uses the spare tap buffer; the incumbent priming drain joins the worker and completes
the MTP stream before tap reuse or a synchronous held write. Unknown predictor
ordering also refuses capture. `capture_skips_cross_stream` counts these refusals
before snapshot allocation/submission (no-store requests count only no-store skips).
A side D2H stream waits on both ready events and reads only staging.

The normal priming drain validates the P-1 frontier and commits the descriptor.
Service points query its D2H completion event; only committed, successfully completed
descriptors publish. Cancellation or failed validation discards them after completion.
Descriptor ownership survives prompt destruction and slot reset. Shutdown explicitly
drains outstanding D2H. HIP submission/completion/drain errors are fatal and retain
all descriptor backing in quarantine. Capture adds no device fence or host completion
wait on the request path; the incumbent producer/priming waits remain in place.
Snapshot copies, allocation and bookkeeping still have costs.

`--prefix-cache-staging-bytes N` (`STRIX_PREFIX_CACHE_STAGING_BYTES`, CLI wins)
bounds device staging. Zero, the default, uses one T-block of all registered rows
plus the aligned fixed registry. Actual staging charges only missing row slices and
fixed device bytes. Larger catch-up candidates are refused. Pinned destinations and
private row leases remain inside the existing host allocation ledger. Stats expose
`staging_budget_bytes`, `staging_bytes`, `staging_peak_bytes`, `staging_cap_skips`,
`capture_busy_skips`, and `row_prefetches`; pool `move_staging_bytes` mirrors live
staging. Per-request capture counters report reaped work; async `capture_ms` is
descriptor elapsed time, including overlap, rather than a request stall measurement.
This bounded snapshot workspace does not implement a resident KV pool.

When checkpoint cadence exceeds T, intervening ordinary full-T boundaries may stage
rows only. After normal commit plus D2H success, these segments remain private to the
request's certified row chain; they do not publish a checkpoint or enter disk storage.
The next selected checkpoint reuses these charged row leases. This preserves both
the checkpoint cadence and the one-block staging default. A delayed/busy descriptor
can prevent that preparation; missing slices exceeding the cap still skip.

Certified checkpoints remain RAM-only and cannot use the v1 disk writer or reader.
Existing MTP legacy disk APIs/formats and baseline capture/priming behavior remain unchanged.
No old uncertified RAM or disk checkpoint is accepted by certified lookup.

With schedule 12, pipeline 255, default cadence (4096 rounded up to T), tiled-start
and terminal heads, the host shim checks these opportunities. Counts assume timely
D2H completion, available credits and no eviction; they are not promised native
retention counts or GPU qualification evidence.

| Prompt tokens | T | Safe selected checkpoints | Private row preparations | Cross-stream/owner skips |
|---:|---:|---|---:|---:|
| 1280 | 2048 | 0 | 0 | 0 |
| 6000 | 2048 | 1 (4096, next partial T on same stream) | 1 | 0 |
| 8300 | 2048 | 1 (4096) | 2 | 1 (8192, next tail) |
| 10300 | 2048 | 2 (4096, 8192) | 2 | 1 (10240, next tail) |
| 10400 | 2048 | 2 (4096, 8192) | 2 | 1 (10240, next tail) |
| 8192 | 2048 | 2 (4096, 6144) | 1 | 0 |
| 32768 | 2048 | 8 (4096 multiples through 28672, then 30720) | 7 | 0 |
| 1280 / 6000 / 8192 | 8192 | 0 | 0 | 0 |
| 8300 | 8192 | 0 | 0 | 1 (8192) |
| 10300 / 10400 | 8192 | 1 (8192, next partial T on same stream) | 0 | 0 |
| 32768 | 8192 | 3 (8192, 16384, 24576) | 0 | 0 |

These are independent retaining cold requests, with the ladder 1024/256/64/16.
A 10300-token no-store request prepares/publishes zero rows/checkpoints; it may
still restore a compatible checkpoint retained by an earlier request.

With `STRIX_PREFIX_CACHE_DIGEST=1`, each prompt receipt emits
`terminal_logits_digest`: SHA-256 of the raw FP32 vocabulary row at its last
prompt position, or JSON null when unavailable (comparison gates must reject null).
It uses the actual tiled final logits, last accepted prompt-window ticket, or
serial ticket without changing route. Window hashes are retained before producer
scratch reuse. The existing `final_logits_digest` and `serial_logits_digest` fields
and their switches retain their meanings. Beside `state_digest`, `terminal_metadata`
reports committed tokens, next position, the canonical LE generations/lineage
digest (including validity and taint), and the predictor frontier (next source
position, P-1 at prompt completion). These diagnostics add readbacks only when
enabled and do not qualify automatic caching or replace authoritative full-row
and state comparisons.

For the production BF16 geometry, fixed device state is 116,400,128 bytes
(aligned fixed registry 116,400,640 bytes, including host histories/padding).
Selected T=2048 captures with the preceding rows prepared stage **177,741,824 B**;
the first row-only preparation stages **61,339,392 B**, later ones **61,341,696 B**.
For T=8192, the first selected capture stages **361,764,608 B**, later captures
**361,766,912 B**. Default caps are **177,742,336 B** and **361,767,424 B** respectively.
These are registry/formula calculations, not measured device peaks; other storage
geometries use their actual registered sizes. P=n remains excluded.

### Controls and fitted HTTP budget

`--prefix-cache on|off` controls retention (default on). `STRIX_PREFIX_CACHE=0|1`
(or `off|on`) supplies the environment default; CLI wins. Explicit
`--prefix-cache-host-bytes N` retains its existing meaning. Outside HTTP serving,
there is no automatic host budget.

All automatic serving routes (`http-target`, `http-mtp-fixed`, `http-mtp-batch`)
are currently unqualified in `prefix_auto_route_qualifications`. An otherwise
accepted automatic route has no budget, source `off`, and emits one
`prefix cache: off (automatic budget not yet GPU-qualified: <route>)` line.
Explicit host budgets remain opt-in experiments. The gated fit code and test-only
qualified route retain the `min(4 GiB, MemAvailable / 16)` estimate, rounded down
to 4096 bytes, and a complete-checkpoint size check. Before enabling a production
entry, re-observe memory after weight residency and lane plans; include retained
prefix bytes and capture peaks in the shared UMA fit and reserve host headroom.
The early observation alone does not establish capacity. Session admission runs
first: an automatic prefix budget never disables sessions. Disk storage remains
explicitly configured, with the existing version-1 files unchanged.

`--prefix-cache-retention fair|lru` and `STRIX_PREFIX_CACHE_RETENTION=fair|lru`
select the host victim policy; fair is the default. `--baseline` or
`STRIX_BASELINE=1` disables the automatic budget and selects lru; explicit prefix
options (host bytes, directory, cadence and follow-up prompts) remain unchanged.
`--prefix-cache off` or `STRIX_PREFIX_CACHE=0|off` fully disables the cache, including
explicit prefix options. Automatic declines leave explicit options unchanged.
Outside baseline, effective configuration includes `prefix_cache.enabled`, `source`
(`auto`, `explicit`, `off`), `budget`, `retention` and `reason`.

### Fairness and lifetime

A caller-supplied conversation key takes precedence; unnamed checkpoints use their
first complete block digest. A checkpoint can acquire up to 16 conversation
memberships through validated hits. The fair directory is bounded to 256
checkpoints; a full directory evicts an
unleased entry by the fair order, or skips admission when all entries are leased.
LRU has no directory entry cap and preserves the legacy victim order. These keys
are retention groups, not
security or tenant boundaries.

Within a conversation, a branch uses the first block after the deepest digest shared
with another retained checkpoint. If that checkpoint ends at the shared ancestor,
its terminal digest identifies the provisional branch. Associations are recomputed
after removals. Fixed/private bytes are charged to their memberships; physical
segments shared by conversations are charged fractionally across those memberships.
The physical byte ledger counts a payload once.

For budget B, conversation cap is `min(B, max(one full checkpoint, floor(B/3)))`.
The branch cap is `min(conversation cap, max(one branch checkpoint,
floor(conversation cap/live branches)))`. A lone conversation may borrow otherwise
unused capacity. Eviction occurs only under byte-budget or fair directory pressure;
conversation/branch caps order victims and never force eviction with spare capacity.
Under pressure, competing conversations reclaim borrowing first. Over-cap branches,
then over-cap conversations, then other entries supply victims. Victim score is
`marginal bytes * (1 + age epochs) / (1 + successful reuse count)`; larger scores
leave first. Ties use older access epoch, then digest. Epochs are deterministic,
without per-token clock reads. Marginal bytes are recomputed after every removal.

Removing a directory entry never invalidates a shared handle or frees another
checkpoint's segments. Physical host gauges include registered payloads kept alive
by restore or writer handles. `counters()` reads scalar counters and incremental
gauges without walking payloads; the older diagnostic `stats()` reconciliation
continues to expose its existing directory/weak-segment accounting.

### Request no-store and counters

`"state_cache":{"retain":false}` permits validated cache hits but skips eligible
prefix captures before digest work, fences or allocation. It also suppresses disk
promotion and named frontier publication. A named continuation restores copies of
its envelope into the execution lane; the published parent is not overwritten.
Cancellation does not invalidate that parent for a no-store request. The cache
object remains present, so the request policy does not change scheduling inputs.

Absent `state_cache` uses server policy. Present `state_cache` must be an object
with boolean `retain`; invalid shapes and types receive 400 `invalid_request_error`.

Lifetime counters add lookup, hit and restored token totals; completed capture
attempts, capture failures, no-store and budget skips; and evictions for space,
conversation cap,
branch cap, explicit deletion and invalid entries. A suppressed eligible operation
counts once, not once per component. Hits count validated target prefix tokens;
predictor rows do not add token credit. No-store hits can update recency/reuse.
Reason counters have their own scope and do not sum to legacy `evicted`:
explicit deletion increments `evictions_explicit` but does not increment `evicted`.
The [serving snapshot](serving.md) exposes these counters without payload scans on
HTTP readers. Resident/pool gauges remain zero.

Opportunistic capture checks the store's incremental-byte estimate before pinned
allocation. A move-only reservation holds capacity against retained bytes plus
outstanding reservations until insertion or failure. Allocation converts reserved
credits into payload receipts without double charging. Disk reads reserve the
candidate payload before allocating; capacity refusal skips to a shorter candidate
or misses (`budget_skips`). Returned disk payloads remain charged until their last
handle drops, including no-store reads and refused promotion. A checkpoint that
cannot fit after eviction is skipped (`budget_skips`).
Only `PrefixCacheAllocationFailure` (including `std::bad_alloc` from the pinned
allocator) can skip capture, and only after `hipStreamSynchronize` returns
`hipSuccess`. After a successful drain, partial destinations are discarded and
`capture_failures` increments. If completion or a failure drain
returns any error, all possibly referenced checkpoint, fixed-buffer and pending
segment owners move into a process-lifetime quarantine before propagation. These
owners are never freed, even during shutdown; static count/byte totals are logged
on one line per quarantine transfer. The HTTP engine fail-stops, bounding this
retention to the failed operation. Restore source leases follow the same contract.
Every HIP error from the fence, copy
submission, completion or drain propagates, including `hipErrorOutOfMemory`.
Baseline explicit-cache capture preserves the original fatal allocation error
`prefix state cache pinned host allocation failed`, LRU and legacy counters, with
no new prefix notice and no additive `prefix_cache` JSON object, whether selected
by `--baseline` or `STRIX_BASELINE=1`. Other effective-config receipts retain the
additive object and non-baseline HTTP retains its single prefix status notice.
Explicit named-session checkpoint operations retain their existing failure contracts and LRU parking.

Policy selection requests (constrained JSON, thinking budgets, logprobs and logit
transforms) bypass prefix restore and capture, including the legacy cache and RTX
import paths. Their response carries `strix.state_cache_refusal` with
`policy-selection-cache-unqualified` when a store or import registry is present.
The full prompt runs from position zero through the existing policy selector and
admitted MTP verification route; cache certificates do not yet qualify that
composition. Named sessions continue to refuse policy requests before submission.

The canonical effective-execution signature is version 3. It binds resolved rotary mode,
factor, original and effective maximum context, and the complete rotary identity (including
mathematical descriptor and coefficient-table SHA-256). Prefix namespaces and producer
certificates use this signature; session envelope compatibility uses the same versioned
rotary frame. Cross-rotary prefix lookups miss and session loads refuse with
`session compatibility mismatch`. YaRN with session save/restore or prefix state caching
is currently refused by name; this binding does not qualify those combinations.
