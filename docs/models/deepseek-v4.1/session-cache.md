# DeepSeek TP2 conversation checkpoints

Conversation checkpoint/park/resume is a production feature. Automatic admission
requires a GPU-qualified route; the admission table is currently empty. Explicit
rank commands run the two-node qualification gate. CPU tests alone do not qualify
a route for automatic admission.
The existing prefix-cache namespace and attention-state v1 bytes remain unchanged.

Session startup captures device, inode, size, nanosecond mtime and ctime from
model loader descriptors, including separate GGUF metadata/data opens, overlays,
safetensors indexes and shards for base/mixed packs and native FP8, and fixed
DSpark vocabulary files. Compatibility hashes use these captured identities.
The persistent hash cache uses device/inode keys and size/mtime/ctime stamps;
cold hashing checks its descriptor before and after reading. Admission rechecks
all original loader paths after compatibility construction, including cache hits.
Any session setup failure disables automatic sessions on both ranks and reports
`session cache: off (REASON)`. A model change reports
`session cache: off (model files changed during startup)`. Explicit serving
requests refuse only after both ranks agree the setup result.
Fixed DSpark file vocabularies are frozen during session drafter construction.
Ordinary loading and numerical execution retain their existing behavior.


The production defaults resolver admits a TP2 session route only when it appears
in `gpu_qualified_session_routes` in `src/deepseek41_pair_defaults.cpp`. This is
the sole qualification admission table and is currently empty: no route has
passed its native GPU gate. Target-only HTTP and batch HTTP DSpark therefore
retain ordinary serving and decline sessions with one line:
`session cache: off (route not yet GPU-qualified: ROUTE)` (currently
`tp2-http-target` or `tp2-http-batch-dspark`). `--session-cache on` does not bypass
qualification. Resource checks still apply after a route is qualified. Explicit
rank save/load and DSpark qualification phase flags remain available for running
the gate. CPU tests do not establish native exactness or qualification.
Defaults never change producer, scheduling or speculation policy to accommodate
sessions. Only requests with a JSON `session` field enter conversation capture/resume.

Serving controls (apply identically on both ranks):

- `--session-cache off` or `STRIX_SESSION_CACHE=0`: all serving sessions off.
- `STRIX_DS41_BASELINE=1` also disables sessions. Generic baseline settings do
  not select the DeepSeek baseline or disable its sessions.
- `--session-disk off` or `STRIX_SESSION_DISK=0`: disk off, RAM retained.
- `--session-ram-bytes N`: per-rank RAM budget; default min(8 GiB, 10% of
  startup MemAvailable). A conservative context-size and available-memory guard
  disables sessions rather than failing the process.
- `--session-max-parked N`: RAM entry limit, default 8; eviction uses LRU.
- `--session-dir DIR`: default `$XDG_CACHE_HOME/strix-engine/sessions`,
  otherwise `~/.cache/strix-engine/sessions`.
- `--session-disk-bytes N`: per-rank disk quota; default min(64 GiB, 25% of
  startup free space). Disk requires a writable directory and room for two
  conservative checkpoints; failure disables only disk.
- `--print-effective-config`: reports state, byte budgets, directory and off reason
  before device work. Explicit flags override environment settings. Session
  switches accept `true/false`, `on/off` and `1/0`; other nonzero values enable
  the switch. Invalid budgets decline admission on eligible routes and are
  ignored when sessions are off, under baseline or on a route awaiting qualification.

The startup hello agrees the minimum enabled RAM budget, parked count and disk
quota across both ranks. Auto-selected resource budgets are excluded from the
numerical compatibility record, so changing free memory at restart does not
change a checkpoint's identity. A peer with sessions disabled disables them on both
nodes; a peer without disk disables only disk. Resource limits occupy unused
hello fields, retaining the original OFF wire bytes.

Budget environment overrides are `STRIX_SESSION_RAM_BYTES`,
`STRIX_SESSION_MAX_PARKED`, `STRIX_SESSION_DIR` and `STRIX_SESSION_DISK_BYTES`.
The legacy `--session-cache-mb` and `--session-cache-dir` remain aliases.
Rank `--session-save/--session-load` are explicit commands and never injected by
the serving defaults; the master cache/baseline off switches suppress those
commands too. Directories may be shared by multiple engine processes. Persistent
`.session-rankR.lock` files serialize disk transactions with POSIX `flock`
(Windows uses `LockFileEx`). Pairs acquire rank 0 before rank 1, agree each lock
acquisition, and hold both locks through quota reservation, victim deletion,
rank-file and manifest publication, and old-generation cleanup. Explicit deletion
uses the same ordering; restore holds locks through validation and peer agreement.
Lock files must not be removed while any engine uses the directory. Quota
reservation includes the old committed generation plus new payload and manifest
temporary publication space, so replacement peaks cannot bypass the cap.
Each node retains its own rank file and a complete group manifest; no shared
filesystem is required. Keep cache directories outside Git.

The implementation extends `PrefixCache::Sessions` and uses the existing
`AttentionState` serialization and `BatchForward`/`DeviceForward` exports/imports.
Prefix lookup still uses its original sampled identity. Conversation checkpoints
use a separate `strix/ds41-session-v1` namespace and full-content identity.

## Supported boundary and exactness

A checkpoint represents a completed, greedy, target-only turn. It contains the
physical FP8 window rings, compressed KV/index bytes, ratio-2 carries and validity,
token history for Engram, consumed and emitted token counts, accepted frontier,
and the selected but unconsumed pending token. Sampling is explicitly `greedy`;
the drafter frontier is zero. Producer lineage identifies a committed target turn.
Emitted counts are token IDs handed to the serving output adapter, including its
terminal stop ID; text decoding and filtering finish before the next turn.

The pending token has already been emitted and must be consumed exactly once.
For HTTP continuation, the complete rendered prompt must start with the consumed
history followed by that pending token. A changed history is a normal cache miss
and runs a full prefill; its successful turn replaces the named RAM checkpoint.
Clients must preserve tokenization of the prior transcript. A template change that
alters existing IDs causes a cache miss.

Last-row top-k and candidate scratch are not required for the next operation:
the target selects them again. The rank CLI's attention section may retain
diagnostic top-k; the batch export already omits it. Canonical state SHA-256
clears last-row top-k, inactive physical ring slots, and invalid carry payloads.
All live ring bytes, compressed rows, valid carries, tokens, position and pending
token remain bound. These exemptions do not permit changing any data the next
step reads.

DSpark continuation is opt-in in the whole-rank CLI with `--session-dspark`
and `--stage-dspark FILE`. HTTP `--dspark` sessions remain refused: the batch
runtime also owns per-slot staging, controller and scheduler state that this rank
producer does not export. The rank implementation does not qualify the HTTP path.
Split pipeline stages, non-greedy sampling and mid-turn snapshots are unsupported.

Identical complete state bytes and deterministic execution support the conditional
exactness argument. Native imports, kernels and complete workload fidelity still
require the gate below; CPU format tests do not establish GPU equality. Prefix
sharing across prefill and generated producers retains its existing qualification
rules.

## Compatibility

Trellis base/mix and native FP8 directories use the checkpoint loader's shared
resolution: the index and every referenced shard, or every top-level
`*.safetensors` without an index. Hidden shard names, external references and
symlink target contents are bound; files the loader cannot consume are excluded.
Operational environment exclusions are
scheduler allocation variables and `STRIX_HASH_CACHE`,
`STRIX_SESSION_DIR`, `STRIX_SESSION_CACHE`, `STRIX_SESSION_DISK`,
`STRIX_SESSION_DISK_BYTES`, `STRIX_SESSION_RAM_BYTES`, `STRIX_SESSION_MAX_PARKED`
and `STRIX_SESSION_POOL_IDENTITY_TEST`. Unknown session variables remain in the key;
there is no blanket exclusion of `STRIX_SESSION_*`.

The key is SHA-256 of a length-framed, lexically sorted canonical record containing:

- Full-file SHA-256 of the model, draft when applicable, and selected asset files;
  directories bind sorted relative filenames and every complete file hash.
- Full executable SHA-256 and the session/attention schema versions.
- Effective CLI geometry and producer/chunk plan, layout/dtype, greedy sampling,
  TP world/rank, and target-only or DSpark mode. DSpark additionally binds the full
  draft and cost-table file contents, vocabulary specification, policy, split-head
  placement, frozen cost-prior rule and R1 producer plan.
- Exported `STRIX_`, `HIP_`, `HSA_`, `AMD_`, `ROCR_` and `GPU_` settings,
  with explicit operational exclusions listed below. Selectors and
  approximate-prefill switches therefore cannot alias the baseline.
- Explicit immutable numerical environment assets: `STRIX_DS41_OVERLAY`,
  `STRIX_DS41_TRELLIS`, `STRIX_DS41_NATIVE_FP8`, `STRIX_DS41_PREPARED_LOAD`, `STRIX_DS41_EP_MAP`, and read-only
  `STRIX_DS41_FP16_CACHE`. Trellis mix manifests bind their contents and every
  referenced pack. Prepared-load directories are adopted numerical inputs, not
  operational output caches.
- Mutable operational directories (`STRIX_HASH_CACHE`, prefix/output/cache/log
  paths) are never content-hashed. Their exported scalar settings remain bound except for the operational exclusions above;
  adding files to them cannot change the key. Other path-looking settings are
  scalars unless explicitly classified as numerical assets.

Full hashes are memoized by canonical path, size, high-resolution mtime and native
file identity (device/inode plus ctime on Linux; volume/file index on Windows).
A changed identity rehashes the full file. Hashing checks identity again before
publishing the memoized result. Model/asset files must remain immutable while an
engine uses them. First-use full hashing can take substantial startup time;
enable `STRIX_USE_OPENSSL_SHA256` for the accelerated full-file backend.

The key is deliberately conservative: explicit versus unset settings, path changes
and extra environment settings may cause refusals. Both ranks need identical
numerical environments, apart from the rank CLI option. The HTTP batch producer
and rank CLI producer have different identities; their checkpoints are not
interchangeable. An explicit incompatible restore reports an error.

## HTTP API

The HTTP front end is `strix-deepseek41-serve`, using the shared serving session request contract. A chat/completions request
names its conversation with the JSON field `"session":"example"`. IDs contain
1-128 ASCII letters, digits, hyphens or underscores. The field is ignored when
the session budget is zero, including invalid values. With the off switch, hello and API response bytes remain unchanged.
With a qualified route enabled, requests without a session field use the same serving path and
response bytes as OFF. Session routes are registered only when enabled;
otherwise they use the ordinary unknown-route response. The old
`/admin/session-cache` route and session header are no longer session inputs.

A successful named greedy turn is captured into RAM before completion and slot
release. Failed or canceled turns preserve the last committed checkpoint. Only
one active turn may use a name. A later full-transcript request automatically
resumes compatible RAM or disk state when its token history extends the consumed
history and pending token. Missing, invalid, evicted or changed-history lookups
are normal misses. A newer RAM turn prevents fallback to an older disk frontier
after eviction within the same process. Restart uses the last disk generation.

All routes use the existing API-key authentication:

| Method | Route | Body |
|---|---|---|
| POST | /v1/sessions/{id}/checkpoint | {"durability":"ram"} or {"durability":"disk"} |
| POST | /v1/sessions/{id}/park | empty |
| POST | /v1/sessions/{id}/resume | empty |
| GET | /v1/sessions | empty |
| GET | /v1/cache/stats | empty |
| DELETE | /v1/sessions/{id} | empty |

Disk checkpoint success waits for both durable rank files and both group manifest
publications. RAM checkpoint validates retained state. Park retains a host copy
and releases a staged slot; resume validates RAM or published disk state and
admits it to host RAM. The next named request imports it into unused slots on
both ranks, with import agreement before scheduler publication. Parked sessions
therefore retain no extra device copy outside the RAM budget. There is no separate HTTP restore route.

Inventory returns `{"sessions":[...]}`; each row has `id`, `position`, `consumed`,
`pending` (token ID or null), `bytes`, `state` (`active|parked|disk`) and
`durability` (`ram|disk`). TP2 generation IDs are under `engine`. Inventory validates
local metadata; it does not prove peer health. Cache stats return `ram_bytes`,
`ram_budget`, `parked`, `disk_bytes`, `hits`, `misses` and `refusals`. Disk bytes
include local immutable state files and manifests, including older generations.
Mutation replies contain `id` and `parked:true`, or `deleted:true`; TP2 restore
details are under `engine`.

Errors have `{"error":{"code":"...","message":"..."}}`: incompatible checkpoints
use HTTP 409 / `session_incompatible`, corrupt state 422 / `session_corrupt`,
unknown IDs 404 / `session_not_found`, and budget refusal 507 / `session_budget`.
An HTTP operation timeout does not cancel an already queued operation; query
inventory before retrying. Capture failure can occur after tokens stream, so
clients must check final completion status.

## Storage and group publication

Integers in the new envelope are little-endian u64; strings and payloads are
u64-length-framed bytes. Attention v1 retains its existing x86-native encoding.
Each envelope contains magic/schema, session/generation, world/rank, the full
canonical compatibility record, two sections, and a SHA-256 root:

1. `frontier-history`: replicated frontier, lineage, greedy mode, and token IDs.
2. `attention-v1`: rank-local (marked sharded) physical attention export,
   including rank-local diagnostic selections when present.

Every section records name, rank, replication class, section-relative offset
(zero in this version), size and SHA-256. Token IDs are also present in the
unchanged attention v1 payload. The manifest's shared digest binds the replicated
frontier/history and the canonical logical attention state; physical diagnostic
bytes can remain rank-local without permitting divergent logical state.

Rank files are `ID.GENERATION.rankR.state`; manifest copies are
`ID.rankR.group`. A group manifest contains world=2, both ordered rank descriptors,
generation, per-rank compatibility and payload hashes/sizes, the shared frontier,
section descriptors and a SHA-256 manifest root.

Each rank writes a unique temporary file, flushes, fsyncs, renames and syncs the
directory. Existing generation files cannot be replaced with different content.
Only after both ranks acknowledge durable rank files are manifests written.
Checkpoint success also waits for both manifest publications. A crash between
publications can leave one newer manifest; restore compares both rank
descriptors/manifests and refuses that incomplete generation. Orphan rank files
are never considered recoverable checkpoints.

Restore validates bounds, compatibility, schema, both local section hashes and
the manifest before touching device state. Each node validates its local file;
peer exchange establishes that both nodes validated the same complete group.
Missing/corrupt/truncated files, a missing manifest, or different generations
cause a group refusal. Both device imports then target unused slots and pass
another agreement barrier before scheduler publication. Existing active slots
are not restore destinations.

Disk quota reservation evicts complete sessions in rank-0 LRU order on both
ranks. Successful publication removes superseded local generations. Updating an
ID reserves room for both old and new generations (including temporary files left
by an interrupted writer); quota refusal preserves the
previous durable checkpoint. Crash temporaries and unreferenced orphan payloads
can still require operator cleanup. `delete` removes an ID's manifests and local
generation files on both participating ranks.

## RAM and size limits

The LRU charges retained serialized-buffer **capacity**, including both token
histories and envelope bytes. Admission allocates an unpublished node, validates
both ranks, then transfers ownership into the cache. Oversize admission preserves
existing entries. Active execution slots, map bookkeeping, transient export/
validation copies and device staging allocations are additional memory; the RAM
budget is not a process-wide peak-memory cap. Parking on shared-memory hardware
still consumes system memory.

Full target state is approximately 2.9 MB of fixed rings plus 902 bytes per consumed
token in this envelope (890 compressed bytes, four attention-history bytes and
eight frontier-history bytes). Carries, diagnostic selections and framing add
small amounts. At 8,192 tokens this is about 9.8 MiB per rank; at 122,880 tokens,
about 108.5 MiB per rank, twice that for the pair. These are layout estimates;
CLI output reports actual envelope bytes.

Control messages are bounded at 16 MiB; envelope strings at 1 MiB; each rank's
envelope must fit its configured RAM budget even for disk restore. The current
layout supports at most 40 target layers. Forks, incremental chunk sharing for conversation generations, disk compression
and asynchronous session writes are not provided. Existing prefix
cache blocks remain independent and keep their existing sharing behavior.

## CLI and qualification

The existing `strix-deepseek41-rank` adds:

```text
--session-save DIR | --session-load DIR
--session-cache-mb N       (default 512 when save/load is requested)
--session-id ID            (default gate)
--session-suffix-ids A,B,C  (live continuation after save, or cold control)
--session-suffix-tokens N  (default 64)
--session-validate-only    (requires load; CPU validation + peer agreement, no model upload)
```

A nonzero session budget alone enables the cold-control path. This path uses
greedy R1 input steps; `--generate N` emits exactly N tokens and leaves the last
one pending. The live suffix path consumes that pending token once, then the
suffix. Fresh-process load performs the identical operation with the suffix in
`--prompt-ids`. Both ranks must receive identical options except rank and
rank-local output paths. Other state-import, window-benchmark and ABA modes are
refused in combination with the session CLI.

`--logits BASE` writes only the final turn's N raw FP32 rows to
`BASE-rR.f32`; `--state-out BASE` writes `BASE-rR.state`.
Each rank prints token IDs, canonical state SHA-256, capture time,
write/fsync time, restore time/bytes, input-only time and total turn time. Model/asset hashing and weight
loading occur before these timings. The CLI splits restore into read, local verification/allocation, peer agreement,
and device-copy/import agreement time. HTTP responses report total operation time.
Use process wall time separately to assess restart including model loading.

CPU build and tests (no HIP):

```bash
cmake -S . -B build-session-cpu -DSTRIX_ENABLE_HIP_RUNTIME=OFF -DBUILD_TESTING=ON
cmake --build build-session-cpu --target strix-deepseek41-session-cache-tests strix-deepseek41-attention-state-tests strix-deepseek41-prefix-cache-tests
ctest --test-dir build-session-cpu -R 'strix-deepseek41-(session-cache|attention-state|prefix-cache)-tests' --output-on-failure
```

Run `prewarm` from the lead node using CPU only, then run each execution phase in a
**separate allocated PAIR turn**. Retain the same absolute OUT and numerical
environment across invocations:

```bash
export BIN="$(realpath build/strix-deepseek41-rank)"
export BASELINE_BIN="$(realpath baseline-build/strix-deepseek41-rank)"
# Build the baseline from public revision 05fedd80 and record its full SHA
# in the adjacent BASELINE_BIN.revision sidecar.
export MODEL="$(realpath assets/model.gguf)" DSPARK="$(realpath assets/draft.gguf)"
export P=4096
export STRIX_HASH_CACHE="$SESSION_HASH_CACHE"
export NODE1="$SESSION_PEER" OUT="$SESSION_EVIDENCE_DIR"
# SESSION_HASH_CACHE and SESSION_EVIDENCE_DIR must be absolute, persistent
# locations outside Git; the evidence directory must initially be absent.
LIMIT=7200 bash tools/deepseek41_session_cache_gate.sh prewarm
export LIMIT=690
bash tools/deepseek41_session_cache_gate.sh old
# In later allocated turns, one invocation per turn, in this order:
# off, unnamed, compare-off, target-a, target-l, target-b, negatives,
# dspark-a, dspark-l, dspark-b
# compare-off runs only CPU cmp; it never launches ranks or SSH.
```

Every rank phase prints its enforced phase ceiling: **690 seconds including cold
model loads, setup, transfers, comparisons and verified cleanup**. LIMIT may be
lowered to 30..690 seconds. An outer watchdog and autonomous rank deadlines reserve
cleanup time. Completion receipts are written only after cleanup verifies both
process groups are gone. Missing predecessor receipts, logs, logits or required
group manifests refuse phase startup. Numerical settings and selected policy must
remain unchanged across phases. Nodes must have clocks within one second.
The first phase requires a fresh OUT; later phases reuse its files.

The `old` arm uses an unmodified binary built from exactly `05fedd80`;
`off` uses the candidate with `--session-cache off` and `STRIX_SESSION_CACHE=0`;
`unnamed` uses candidate defaults without a named session. All three use identical
prompt and generation inputs on both ranks. They persist IDs, all FP32 logits and
complete state files and position/digests. CPU-only `compare-off` requires bitwise equality using `cmp` for all
IDs, logits, complete state bytes and position/digest on each rank. The revision sidecar records baseline provenance;
the gate checks identical binary hashes and the sidecar on both nodes, then locks
binary hashes and numerical settings across phases. Build provenance must be retained
outside Git; a sidecar alone does not establish how a binary was compiled.

P defaults to 4096 prompt IDs and is fixed across phases. The session adapter uses
sequential R1 input steps; its separate tiled fill path is not supported here.
Receipts record per-phase wall time and each arm's per-rank model load time after
verified cleanup. A 690-second ceiling is enforced, not established by CPU tests.

Target A saves and continues; L runs the same turns without saving; B restores in
a later exclusive GPU allocation than A. DSpark repeats A/L/B in three further phases for the
selected DSPARK_POLICY (controller by default, or fixed:0..5). Testing another
policy requires a new run directory and phase sequence. Each B phase compares all
64 token IDs, every FP32 logit byte with cmp, and final canonical state digests on
both ranks. DSpark also compares acceptance/window/verified counters. Phases
communicate through files in OUT.

One compatibility negative changes a numerical flag; one integrity negative corrupts
only rank 1. Both use --session-validate-only before DeviceForward or drafter
construction. Each must exit 2, print the coordinated restore refusal on BOTH ranks,
and publish process-ended evidence after wait and process-group liveness checks.
Timeout, signal, SSH failure or missing end evidence fails the gate. Missing,
truncated, mixed-generation, bounds and other compatibility negatives remain in
the CPU suite. The script records per-rank save stall, fsync, bytes, restore and
cold prefill timings. All evidence stays outside Git.

**Qualification scope:** the rank adapter can qualify DSpark continuation only from its sequential
R1 producer after native compilation and this gate pass. It avoids the R6-generated
state/prefill mismatch and does not test or qualify that producer. R6, HTTP and batch
DSpark session continuation remain refused. CPU evidence does not establish native
exactness or throughput.

The fresh-process CPU regression can also be run explicitly:

```bash
python3 tests/deepseek41_session_cache_restart_test.py build-session-cpu/strix-deepseek41-session-cache-tests
```

The portable strix-serve-session-disabled-tests harness checks ignored valid and
invalid session headers and JSON fields while disabled, enabled JSON validation,
common route/body/error translation, hello identity and route gating. For
baseline differential verification, build that same harness once with the current
src/serve/api.cpp and once with the file obtained from git show 05fedd80:src/serve/api.cpp
in a local build directory outside Git. Run tests/serve_session_baseline_compare.py CURRENT_EXE
BASELINE_EXE. It compares unmodified response bytes and retries only if the two
service construction times straddle a second boundary. There is no platform-dependent transcript-size assertion.
`serve_session_defaults_compare.py CURRENT_EXE` also compares the default request without
a session against explicit OFF, retaining all raw response bytes. Public session admission callbacks and request-local error receipts are retained. CPU cache tests also check metadata, peer error categories
and full-history/pending-token matching.



## DSpark rank continuation

Add `--session-dspark --stage-dspark assets/draft.gguf` to the session CLI.
`--stage-dspark-policy fixed:0..5|controller` chooses the policy (default
controller); `--stage-dspark-vocab SPEC` chooses the initial vocabulary;
`--stage-dspark-costs FILE` supplies an optional full-content-bound cost prior.
Placement must be split: both TP2 ranks draft, sharing the vocabulary head.
The path requires all 40 layers, greedy generation and whole-rank R1 steps.
`--session-dspark` requires a session budget or save/load operation.

The new envelope `strix/ds41-session-v2` adds a checksummed replicated
`dspark-v2` section to the existing frontier/history and rank attention sections.
The existing v1 target-only format and default execution path are retained.
The group descriptor lists the third section and binds its canonical digest;
both ranks must agree on all replicated continuation state before capture or
restore can publish. Old readers reject v2 envelopes.

The DSpark state inventory is:

| Owner | Captured state / reconstruction |
|---|---|
| DeviceDrafter | All three physical 128 x 528 FP8 rings, including inactive slots; next and start (skip_to lower bound) |
| Vocabulary | Core count and ordered extra IDs; membership bitmap, device IDs and gathered head rows rebuilt exactly from immutable weights |
| WindowController | Every verify cost and sample count, draft/extra costs, all seen/hit cells, decisions, sweep and explore flag |
| SpecTailDrafter | Split-head exchange request sequence; staged count asserted zero at export/import |
| Rank loop | Consumed/emitted/accepted frontiers, target pending ID, complete target token history, drafter frontier, cumulative windows/verified/accepted counters |

Every committed target row is staged from the captured inputs of layers 37-39.
A turn flushes staging even when no checkpoint is requested, and asserts drafter
next == target next. The local Draft object and its tokens exist only within a
verification iteration. Return from a turn discards its unused predictions;
no draft window survives the boundary. The serialized pending-draft and staged-row
counts are zero, and nonzero values are refused. The pending target ID remains
unconsumed and is consumed exactly once before the next suffix.

Device scratch (main projection, RoPE tables, block tokens/entries, stage streams,
mixes, query/attention/MoE intermediates, routing tables, logits, Markov candidates
and confidences) is overwritten before its next read by append/draft. Immutable
weights, identity coefficients and route constants are recreated by construction.
No last-step hidden capture is read after restore: the next input step replaces it
before staging. Export synchronizes the drafter stream; import validates CPU bounds
before copying, rebuilds vocabulary, restores rings and cursors, then synchronizes.

This producer verifies drafts with R1 target steps; it does not use R6 target
windows or promise a speedup. Controller survival adapts, but cost priors remain
frozen, so wall-clock timing cannot change A/L/B scheduling. This is a separate
compatibility identity from measured-cost pipeline/batch decoding. In particular,
generated verify-window state must never be relabelled as prefill state: different
row producers can leave different attention bytes. Restores import the captured
state directly and never replay it through prefill or the prefix cache.

Automatic admission for this producer requires native compilation and the
complete A/L/B GPU gate. No route is currently listed as qualified; explicit rank
commands remain available to run that gate. CPU framing/validation tests do not
establish GPU exactness or performance.

HTTP DSpark session disablement is intentional and remains fail-closed. In the batch implementation,
`BatchSpeculator::Slots::staging` ([slot][128][15360] FP32), `staged` and
`context` retain hidden rows and vocabulary IDs not yet appended to each drafter.
`pending_commits_`, `positions_`, `owners_`, `kept_` and `staged_step_`
retain rank-1 staging/commit progress. `BatchWindowController` also has pooled and
per-slot survival tables and plan counts; `BatchSpeculator` retains cost means,
sample counts, `groups_` LRU regression samples, overhead/probe history and a
wall-clock `last_finish_end_`. The HTTP session envelope has no adapter for these
buffers or a portable clock-gap boundary contract. Restoring only its target
attention and the rank CLI's single-context section would lose this state, so HTTP
DSpark restore remains refused rather than applying this schema to batch slots.

Inventory merges retained checkpoints with live scheduler sessions by ID. First
active turns are present before any capture; running metadata supersedes an older
checkpoint's consumed position, pending token and emitted count. During prompt
prefill the pending token is null until one is emitted. Restored slots awaiting a
request are `parked`, and running requests are `active`. Live rows use durability
`none`, null host-checkpoint bytes and null engine generation because an older
retained checkpoint does not describe their live frontier. Completed retained
rows report their own RAM/disk bytes, durability and generation.

All unsupported compatibility classes (envelope/group/draft schema, TP geometry,
strict model/build/config identity, and DSpark mode) carry `session_incompatible`
through both `inspect()` and two-rank `resume()`, including explicit HTTP resume's
inspection preflight. The API maps this category to HTTP 409; malformed payloads
and checksum failures remain `session_corrupt` / 422.

The phase sequence is `prewarm -> old -> off -> unnamed -> compare-off -> target-a
-> target-l -> target-b -> negatives -> dspark-a -> dspark-l -> dspark-b`.
`prewarm` requires an explicit persistent `STRIX_HASH_CACHE` directory on both
nodes. It runs without a GPU allocation (default CPU limit 7200 seconds, configurable
up to 86400 with `LIMIT`). It runs `--session-hash-only` twice per node, binding target, drafter and
all selected numerical assets without opening transport or uploading a model.
The second process must report hits for every file and zero misses. Each later
execution phase repeats the hash-only measurement and records per-node hashing,
model/drafter load and checkpoint I/O seconds in its receipt. `compare-off` is a
CPU comparison with zero hashing and checkpoint I/O. These measurements must be
collected on the actual nodes before claiming the 690-second budget is met.

Session compatibility validates every consumed safetensors shard header and
refuses tensor names repeated across shards, including indexed checkpoints.
Ordinary loading keeps its existing resolution and overwrite behavior. Trellis
inventory uses the loader's effective environment specification and shared mix
parser; an explicit mix default can leave the base directory unused, and inline
path suffixes remain literal. Manifest comments retain the loader's semantics.

DSpark `auto:` vocabulary selection is refused for explicit sessions with a clear
error. Use `files:`, a fixed core size, or `full`. The ordinary automatic
vocabulary loader is unchanged; unrelated directory files and links are never
traversed by session initialization.

The gate hashes all runtime libraries resolved by `ldd` for both candidate and
baseline binaries, including system libraries. Inventories must match between
nodes and remain identical across phases. Unresolved dependencies refuse the
gate. Remote library, executable and asset hashing run in recorded process groups
with remote deadlines; cleanup verifies those groups. `compare-off` runs under
its `LIMIT` watchdog and publishes wall time only after comparison succeeds.
