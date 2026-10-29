# RTX persistent conversation sessions

Sessions extend the existing attention export/import and persistent spool worker.
Automatic named-session admission with RAM parking is **on by default** for the qualified
`rtx-tp1-decode-rows` route in `apps/cuda_ds41_session_cache.hpp`.
Unknown routes remain disabled unless explicitly admitted. A name-only request
adds `--session NAME` to ordinary full-prompt fields, with no other
`--session-*` field. Committed history must be an exact token prefix of that
full prompt: a hit imports the parked state and feeds only the remainder. A
missing, evicted or non-prefix RAM-resident name executes the full ordinary
request from its launch seed and parks the result under NAME. Implicit
continuation and history matching never read disk or consult checkpoint paths.
Ordinary unnamed requests retain their existing reset and numerical behavior.

## Launch and memory

A normal `--serve SPOOL --max-positions N` launch enables automatic sessions
without a session opt-in flag. All 40 target layers are required. Host RAM
defaults to min(8 GiB, 10% of total host RAM), with at most 8 parked sessions
in addition to the active session. Admission additionally checks available
host memory after loading. Unsupported geometry or insufficient memory disables
automatic sessions with one `session cache: off (<reason>)` line; ordinary serving continues.
Model, drafter and numerical dependencies must remain immutable while loaded.

`--session-cache off` / `STRIX_SESSION_CACHE=0` disables the cache. The existing
`--baseline` / `STRIX_DS41_RTX_BASELINE=1` also disables it. Flags retain the RTX
parser's ordering: a later explicit switch overrides an earlier baseline.
`--session-ram-bytes N` and `--session-max-parked N` override RAM limits.
The legacy `--serve-session-mib N` sets the total budget and enables sessions;
zero disables them. One-shot invocations do not acquire automatic sessions.
`--print-config` (alias `--print-effective-config`) includes session policy,
budgets, directory and the policy off reason without loading CUDA. Exact context
admission happens on the first session request, after loading. The worker's
`ready.json` reports `session_config`, including enablement,
`session_explicit_opt_in: false` and `session_route: rtx-tp1-decode-rows`.

The disk tier defaults to `$XDG_CACHE_HOME/strix-engine/sessions`, otherwise
`~/.cache/strix-engine/sessions` (`USERPROFILE` supplies the Windows home fallback).
Its budget is min(64 GiB, 25% of free filesystem space at first managed-tier operation).
Inspection resolves the RAM default and directory for display and reports
`session_disk_bytes: 0` for the deferred automatic disk budget. Worker startup
leaves these defaults unresolved; its compact policy report contains no resolved
budget or directory.
Disk opening, directory creation, locking, probing, scanning and quota eviction
are lazy; a worker serving only unnamed requests performs no session filesystem
operations. The startup criterion is: no session filesystem operations, no session
allocation above 4 KiB, and no change to engine memory budgets or ordinary output.
Trivial Config construction (`getenv`, small strings) is acceptable. Host RAM
queries and the default directory construction are deferred until session use;
inspection explicitly resolves them. Creation, writability, two-checkpoint
free-space and budget failures leave the tier unavailable. `--session-disk off` / `STRIX_SESSION_DISK=0`
disables disk; `--session-dir DIR` and `--session-disk-bytes N` override its directory and cap.
Managed checkpoints use complete `ID.gN` publication directories and a small
`ID.current` pointer. The first managed-tier operation lazily takes an exclusive, non-blocking OS lock
(Windows `LockFileEx` on `.owner.lock`), held for the worker's lifetime. One worker
owns each managed tier. If another live worker owns it, all managed-tier operations
of the competing worker return `session_tier_busy`/409 without changing RAM,
checkpoints or device state. A new worker can acquire the tier after the owner exits.
The owner's `.session.lock` serializes individual filesystem transactions.
Each save flushes the new files and generation before atomically replacing the
pointer (Windows replace-existing/write-through; POSIX rename and directory fsync).
Only then is the previous generation removed. Both generations and interrupted
writes count against quota until cleanup; insufficient replacement capacity
refuses without removing the current generation. Opening removes unreferenced
generations and pointer temporaries. Only the pointer-selected generation is
restored; a missing or incomplete selected generation refuses.
Explicit external checkpoint paths retain immutable publication and owner-managed
retention. Read-only external copies without a lock file are inspected without
creating one. Finished turns automatically park in RAM. RAM capacity eviction
drops the session and may record a bounded diagnostic tombstone; it never writes
a checkpoint. Checkpoints are explicit save points: advancing RAM leaves an older
checkpoint intact. Only an explicit restore/resume loads that save point, which
may legitimately precede an evicted RAM frontier. Tombstones have no correctness role.
A suffix-only turn on a missing name returns `session_not_found`/404. A name-only
full-prompt request recomputes on eviction. Durable sessions require an explicit checkpoint with
`--session-durability disk` (managed tier) or an explicit `--session-path`.
Checkpoint failures return an error. The registry/name map, LRU, disk object,
session request containers, hashing and pinned capture allocation are constructed
lazily, on the first session request. The session seed references the existing
imported launch state without an additional deep copy.

Let B be the maximum attention capture size plus drafter rings and 2 MiB metadata.
The worker reserves 8*B + 16 MiB for two pinned capture slots, seed, transient
snapshots, validation and serialization. The remainder bounds LRU payload bytes;
at least another B must fit. Stats report both limits. This is additional to
engine weights/cache/scratch; small container overhead and the engine's existing
import staging remain separate.

Pinned slots reuse attn::ExportCapture; parked payloads occupy ordinary RAM.
Copies and saves are serialized at committed boundaries. There are no background
checkpoint DMA submissions, so demand restore cannot queue behind background
saves. Disk durability stalls the spool. Parking releases the logical execution
slot; allocated device-cache capacity remains available for the next session.

DSpark sessions require launch `--draft-vocab full`; other vocabulary routes
refuse up front with HTTP-style status 422 and `session_unsupported`.
Target-only sessions work with any launch vocabulary.
The preflight uses launch runtime values, and compatibility uses the launch
environment snapshot even after ordinary request overrides. Each first session
operation logs `session_compat` with model, drafter, executable, hardware,
arguments and environment component digests.

## Spool protocol

Use existing one-argument-per-line queue/NAME.tmp → NAME.req publication.
Explicit session operations contain only session fields. A name-only automatic
request uses ordinary `--seqs-file`, `--chats`, `--chat-prompt`, `--chat-steps` and
one phase (`chats0` or fixed `schats0f1` through `schats0f5`), plus optional
qualification/output observers. The selected sequence contains the full prompt
after the immutable imported seed; history matching includes that seed. Multiple
chats/phases, numerical overrides and environment changes are refused for reuse. Launch runtime values are restored
for execution. Requests without a session name retain the exact existing per-request reset,
scheduling and output path. With the cache disabled, a `--session ID` field on
an ordinary phase request is ignored. Explicit session operations remain available
subject to geometry and host-memory limits. An explicit `--session-path` save/load
is independent of automatic admission and of `--session-disk off`; its retention
and storage capacity are owner-managed.
A history hit selects the greedy/fixed-window session continuation producer.
A history miss and unnamed requests execute the ordinary producer. Restore versus
live continuation still requires the complete numerical gate; default admission
alone does not establish exactness.

| Field | Meaning |
|---|---|
| --session ID | 1–128 ASCII letters, digits, underscore or hyphen |
| --session-op OP | create, turn (default), checkpoint, park, resume, list, stats, delete; restore remains an alias for loading |
| --session-tokens FILE | Explicit whitespace-separated suffix IDs (legacy transport) |
| --session-history FILE | Complete token history; mutually exclusive with suffix tokens |
| --session-steps N | Generated greedy rows after suffix, default 0 |
| --session-dspark on/off | Default off; repeat the saved mode on each turn |
| --session-fixed K | With DSpark, fixed 1–5 drafts, bounded by loaded max-rows |
| --session-durability ram/disk | Checkpoint durability; explicit disk selects managed restore/resume, default ram |
| --session-path DIR | Immutable checkpoint destination or resume source |
| --session-dump BASE | Qualification outputs for create/turn |
| --session-control-prefix FILE | Qualification-only create: consume P before suffix, with no intermediate capture/import |

Create starts explicitly from the launch's imported attention state and history.
The drafter starts with empty rings at that seed frontier, as in the existing
imported-state path. Explicit create refuses existing RAM names. A `turn` without
`--session-history` never creates: it continues only a session resident in this
worker's RAM, otherwise returns `session_not_found`/404. It never reads the
managed tier or any checkpoint path. This also holds after restart, eviction,
deletion and tombstone overflow, regardless of disk availability. `resume`
without a path or explicit `--session-durability disk` follows the same RAM-only
rule. Durable state is loaded only by `restore`/`resume` with an explicit
`--session-path`, or by name with explicit `--session-durability disk`.
The result is exactly the requested checkpoint, even when an evicted RAM state
was newer. Explicit managed restore or
checkpoint operations report `session_tier_unavailable`/503 (I/O/probe/open) or
507 (space/budget); a missing checkpoint in an available tier is 404.

Each create assigns a random 128-bit lineage. Every committed turn increments
its generation; both fields persist in the manifest and pointer. Within the
exclusive owner, an equal or older generation or a different lineage is refused
with `session_conflict`/409 on managed save. RAM turns leave the last explicit
save point intact; checkpoint again to make the newer frontier durable. External
immutable checkpoints retain owner-managed retention. Restore grants no replacement permission.
With `--session-history FILE`, an exact prefix resumes at the saved frontier;
a mismatch or absent RAM name starts from the launch seed and consumes the complete
supplied history as a miss. History must extend the immutable launch seed.
Explicit suffix-token requests append to saved state without history matching.
Responses report `history_hit` and `recomputed`; automatic misses report false/true.
Automatic snapshot turn/emission metadata describes the current full-prompt request;
explicit suffix continuations accumulate those counters.

Example create request:

~~~text
--session
A
--session-op
create
--session-tokens
fixtures/prompt-tail.ids
--session-dspark
on
--session-fixed
3
~~~

Example durable checkpoint:

~~~text
--session
A
--session-op
checkpoint
--session-durability
disk
--session-path
artifacts/session-A-v1
~~~

Continue with --session A, --session-tokens SUFFIX.ids, --session-steps 64,
--session-dspark on, --session-fixed 3, --session-dump artifacts/live.
After restart submit --session A, --session-op resume,
--session-path artifacts/session-A-v1, then the same continuation.

Every successful create/turn leaves a complete RAM snapshot. The qualification
control-prefix runs two logical turns in one create request, preserving the
prefix/suffix chunk boundary and turn count without capturing at that boundary. Consecutive turns
on the active session continue live device state only when the RAM entry's unique
generation still matches the device identity. Every eviction, deletion or
replacement of that entry invalidates the identity, including a disk reload with
the same name. Switching imports target and
drafter state. Park releases the active slot; resume imports RAM state without
advancing. Restore verifies and admits disk state into RAM before any device
import. LRU replacement can evict sessions; durable files remain. List returns
`sessions` records with id, position, consumed, pending, bytes, state and durability
for RAM sessions. Disk-only records expose id, generation, lineage, state and
durability from the checkpoint pointer without loading snapshot payloads. Finished turns have
parked RAM snapshots; the current device copy may remain reusable until another
request takes the slot. Stats includes ram_bytes, ram_budget, parked, disk_bytes,
hits, misses and refusals. Delete removes RAM and managed disk state; explicit
external checkpoint paths remain owner-managed. Disk checkpoint/restore may omit
`--session-path` with explicit `--session-durability disk` to use the managed directory.

Preflight stages a private CPU snapshot, including managed disk loads and history
fallback. All mode, history, token, shape, capacity and dump-open checks precede
RAM admission, LRU refresh, active-identity changes and disk timestamp updates.
A refused preflight leaves cached sessions, LRU order, active identity and the
disk tier unchanged. CPU regressions cover single-entry eviction/reload,
delete/replacement generations, history fallback and refused disk admission.
Preflight errors return rc 2 without device writes. Disk publication failures
return rc 1 without poisoning the worker. Exceptions after device work starts
return rc 1 and terminate it. Only session responses add session_result to
completion JSON: operation, compatibility, frontier, bytes, attention SHA-256,
read/verify, restore-copy, capture, write/fsync and execution timings. Ordinary
request wall time also includes preflight.

## Format and exact boundary

The directory version is strix-rtx-session-v2:

- attention.bin: unchanged attention serialization: target rings, compressed
  KV/index keys, carries, selections and token history.
- drafter.bin: rtx-dspark-v1: three physical 128×528-byte FP8 rings, start/next
  cursors and ordered vocabulary. Empty for target-only sessions.
- frontier.bin: greedy lineage; consumed/emitted counts; selected, unconsumed
  pending token; accepted/drafter frontiers; turn count and fixed-window
  parameter. This mode has no RNG or mutable timing controller.
- manifest.bin: v2 length-prefixed schema/compatibility/lineage, committed generation, three ordered descriptors
  (name, rank 0, replicated, offset 0, bytes, SHA-256), then SHA-256 of the
  manifest prefix. Integers are little-endian u64; strings/blobs have u64 byte
  lengths. Sections occupy separate files at offset zero.

Compatibility hashes a length-delimited canonical record: full target/drafter
content hashes, executable hash, schema/layout/producer, TP geometry, launch
arguments except explicit transport/output/fixture fields (including controller-save and session transport/budget options),
and sorted STRIX_*, CUDA_* and NVIDIA_* environment values except STRIX_HASH_CACHE, STRIX_SESSION_CACHE, STRIX_SESSION_DISK
and CUDA_VISIBLE_DEVICES. CUDA device properties supply the resolved target and
drafter UUID and architecture; CUDA APIs supply driver/runtime versions. Device
selector spelling and draft-device ordinals are excluded in favor of those resolved
identities. The executable binds compiled defaults.
Existing file-valued numerical arguments (including FILE:CAP controller priors)
also contribute full hashes. Unknown options are included conservatively;
reordered/equivalent arguments can refuse reuse. Full-file hashes use the
repository persistent cache in src/model_content_hash.cpp: size, OS file identity,
mtime and ctime on Linux (unchanged v1 records). Windows v3 uses volume + file ID + size + LastWriteTime + creation time; metadata-only ChangeTime is excluded. A same-size rewrite with deliberately restored write time is outside this cache identity contract.
STRIX_HASH_CACHE selects a private cache directory; 0 disables it. Each session
identity lookup logs hit/miss and elapsed milliseconds. No sampled identity is used.

Payloads are written and flushed; the manifest is written and flushed last.
A sibling .tmp directory is atomically renamed (Windows write-through; POSIX
directory fsync). Existing external destinations are refused. Managed names use the pointer commit
described above. Interrupted generations are never selected automatically. Bounds,
compatibility, all hashes, target geometry/history and drafter/frontier
consistency are checked on CPU before device writes.

Only committed greedy/fixed-window decode-row producers are admitted. Accepted
rows are committed before drafter append/capture. The existing exporter
canonicalizes unused target ring slots. Omitted seed top-k is recomputed by the
next operation. Each generated row emits and consumes the pending token; its greedy successor
becomes the next pending token. Explicit user suffix rows consume supplied IDs
without emitting them. Token dumps and completion tokens contain the generated
row inputs; raw logits contain the full output row after each such input.

Identical complete bytes and deterministic execution imply equal continuation;
completeness and kernel/window behavior still need the GPU gate.
Adaptive timing/EWMA controllers, sampling, cross-device drafters, restricted
draft vocabularies for DSpark, mid-operation capture, fork/prefix sharing,
asynchronous disk writing, compression and one-shot phase-run session save/load
are unsupported. DSpark sessions use full draft vocabulary and a fixed policy
independently of legacy controller settings. There is no approximate fallback.

## CPU build and PC gate

In an x64 MSVC developer shell, without CUDA:

~~~powershell
New-Item -ItemType Directory -Force build-session-cpu | Out-Null
cl /nologo /std:c++20 /EHsc /W4 /D_CRT_SECURE_NO_WARNINGS /Iinclude tests/ds41_session_cache_test.cpp src/model_content_hash.cpp src/model_runtime_binding.cpp src/deepseek41_attention_state.cpp src/ds41_attention.cpp src/ds41_formats.cpp src/deepseek41_model.cpp src/gguf_reader.cpp src/model_manifest.cpp /Febuild-session-cpu/ds41-session-tests.exe /Fobuild-session-cpu/
.\build-session-cpu\ds41-session-tests.exe build-session-cpu/results
~~~

CMake target: strix-ds41-session-cache-tests, with BUILD_TESTING and the family
enabled, including in an existing CUDA-only build. Build the replica using
[standalone instructions](rtx-standalone.md).

Each invocation runs exactly one phase in one GPU window. Supply the same paths
and private run directory throughout. Supply a baseline binary and its source
revision label with required `--baseline-revision`. There is no fixed revision.
The label is checked against the executable's embedded `strix-build` JSON
`git_sha` when available, without executing the binary. Missing embedded identity
is recorded as unavailable, not verified. `run.json` records this result and both
executable SHA-256 values. The gate does not build either binary.

From Git Bash on the PC (native Windows paths):

~~~bash
CUDA_VISIBLE_DEVICES=0 bash tools/rtx-session-gate.sh baseline-0 \
  --exe build/strix-cuda-ds41-replica.exe \
  --baseline-exe build-baseline/strix-cuda-ds41-replica.exe \
  --baseline-revision BASELINE_SHA --draft-vocab full \
  --model assets/MODEL.gguf --receipt assets/MODEL.receipt.json \
  --seed fixtures/STATE.bin --draft assets/DRAFTER.gguf \
  --prompt fixtures/P-8192.ids --suffix fixtures/S.ids \
  --out ../qualification/rtx-session-gate --arena-gib 74.8 \
  --extra-flags "--host-margin-gib 1 --max-pinned-gib 70"
~~~

Repeat in separate granted windows, changing only the phase:

| Phase | Work | Prerequisites |
|---|---|---|
| baseline-0 | Labelled baseline reference, changed settings, reset | New run |
| baseline-1 | Same sequence starting with setting 1 | Same run inputs |
| candidate-0 | Cache OFF: repeat sequence and byte-compare native receipts | Both baseline phases |
| candidate-1 | Explicit cache admission, unnamed: compare receipts and fresh/reset raw dumps | candidate-0 and both baseline phases |
| save-off | Target-only L, A, disk checkpoint and parking | Both candidate phases |
| restore-off | Restarted B, A/L/B comparisons, CPU refusals | save-off |
| save-on | Fixed DSpark L, A, disk checkpoint and parking | Both candidate phases |
| restore-on | Restarted B, A/L/B comparisons, CPU refusals | save-on |
| auto-save-target | Default worker: ordinary baseline arms, launch-seed isolation, name-only history hit, non-prefix and eviction recomputation, explicit target-only D checkpoint | All eight preceding phases |
| auto-save-dspark | Separate default worker: ordinary baseline arms, name-only history hit, non-prefix and eviction recomputation, explicit DSpark D checkpoint | All eight explicit phases |
| auto-restore | New default worker: explicit restore of D by name, continuation equals ordinary L_D, unknown suffix turn is 404 in both modes | auto-save-target and auto-save-dspark |

Save/restore phases explicitly enable sessions with a fixed RAM budget, disable
automatic disk storage and use owner-managed checkpoint paths. They preserve the
launch seed before their interleaved ordinary imports. Candidate-1 explicitly
enables sessions to check unnamed-request identity. The three automatic phases use only operational
`--session-dir OUT/auto-tier --session-max-parked 1`; they pass no cache, disk or
RAM opt-in/override flags. These operational options do not enter compatibility.

All three automatic workers explicitly remove `STRIX_SESSION_CACHE` from their
child environments. Before requests, the driver asserts from `ready.json` that
session admission is enabled by default, without explicit opt-in, on
`rtx-tp1-decode-rows`. It rejects disabled, opted-in or differently routed reports.

Each automatic save phase repeats the plain/DSpark/reset ordinary arms and
compares native qualification records plus exact confidence bits, proposed IDs
and acceptance counts with the baseline. Each phase then exercises its
target-only or full-vocabulary DSpark mode:

- Ordinary L runs P + suffix without a session name or saved state.
- A receives only its name plus ordinary fields carrying full P, then is parked
  behind another name. A's full P + suffix request must report a history hit and
  equal L in all 64 generated IDs, all 64 x 129,280 FP32 logits, and complete
  attention and drafter bytes.
- A non-prefix full prompt must report a miss/recomputation and equal an ordinary
  fresh request. The request includes the launch seed through the ordinary contract.
- Two other names exceed `--session-max-parked 1`. Inventory must show A absent,
  with no managed checkpoint. A's longer full history (including its prior
  generated IDs) must then report a miss/recomputation and equal a fresh request.
- D repeats the first two name-only requests and matches A, including frontier
  bytes. Only an explicit `checkpoint` with durability disk makes D durable.
  Pointer, manifest, lineage, generation and every section must match its latest
  committed snapshot.

`auto-save-target` also imports the ordinary L oracle's complete attention state
in an unnamed request before the first explicit session create. That alternate
state has a later position than the launch seed. Explicit create and a shorter
full-history fallback must both equal the fresh launch-seed oracle in all generated
IDs, FP32 logits and attention bytes. The saved session history includes generated
rows, so the shorter request must report a history miss. The alternate import
needs no additional fixture. The session seed always belongs to the launch:
its nested host storage is copied lazily at first admission or immediately before
the first ordinary import replacement while sessions are enabled. Workers with
sessions disabled allocate no seed copy on replacement. A disabled launch cannot
enable sessions at runtime; guards refuse materialization from a replaced,
unpreserved import and refuse enabling sessions after a disabled launch.

Automatic prompt tails are capped at 192 IDs after the imported seed to fit the
additional full-prompt recomputation controls. Earlier phases retain P = 8,192.
The receipt binds this cap. Ordinary observers include prompt and rejected
verification rows; the driver selects committed generated rows from native window
acceptance records before comparing every FP32 byte. Ordinary controls have no
session frontier counters; those counters are checked separately between A, D
and the explicit checkpoint, without claiming identical full-prompt/turn metadata.

After worker exit and lock release, `auto-restore` verifies saved evidence without
changing managed entries and starts another default worker. An explicit
`--session D --session-op restore --session-durability disk` loads D by managed name. Its suffix continuation
must match the never-saved ordinary full-prompt L_D in all IDs, raw generated
logits, attention and drafter state. A suffix-only turn on an unknown name must
return `session_not_found`/404. Separate corrupt copies return
`session_corrupt`/422 before admission; RAM/active/eviction/disk stats and both
checkpoint copies remain unchanged by refusal. Continuing D retains its explicit
save point; another explicit checkpoint is required to save the newer state.

CPU tests cover exclusive ownership and release, stale/conflicting writers,
pointer-publication failures, replacement peaks, missing suffix continuations,
unavailable tiers and tombstone overflow. A restart regression saves D1 externally
and in the managed tier, restores external D1, advances to D2 in RAM, opens the
tier through other names and evicts D2. Suffix-only D returns 404; explicit managed
restore still returns D1 with identical sections, lineage and generation. Startup
allocation and session filesystem instrumentation begins at the start of `main`,
before Config or fixtures; it also checks that host RAM queries and resource
constructors remain unused until session admission.

Setting 0 runs a plain chat with prompt length 1. Setting 1 runs
`chats0,schats0f3` with prompt length 3 and 11 decode steps, providing its own
plain reference before speculative execution. The worker's cross-request
reference key includes imported state, sequences-file path, prompt length and
chat index; a reference from setting 0 cannot qualify setting 1. Both phase
orders therefore supply a matching plain reference in the same request.
For automatic phases, every speculative full-prompt oracle and name-only request
has a matching plain reference established in the same worker first. Equal prompts
reuse one fixture path; the canonical key matches fixture path, prompt length and
chat index. References cover the requested decode length. Named requests keep
only their name and ordinary fields. Up to eight additional plain arms are needed;
four decode 64 rows and four establish zero-step prompts. Their work is included
in the DSpark save phase budget.
Every ordinary baseline-comparison request supplies `--qual --draft-trace FILE` to both binaries.
For each speculative window the gate requires byte equality of normalized trace
records containing all five confidence bit patterns (`cb`), all five draft IDs
(`d`), accepted count (`a`), position, input, window shape and verified greedy IDs.
Missing traces or confidence bits fail the gate; decimal confidences and timing
fields are excluded. Candidate-internal A == L == B and parked-session raw
drafter-state comparisons remain separate evidence from this baseline comparison.
Every worker arm forces `--draft-vocab full`, also recorded in the run identity.

Phase receipts record the ordinary stdout/protocol/result byte contract against
the labelled baseline and the default-ON qualified-route policy. Their baseline
evidence compares normalized native qualification and drafter records, excluding
timings, process IDs and operational paths; it does not compare the entire stdout
stream, protocol envelopes or stderr log. The combined worker log preserves
stderr, including policy refusals; no broad stderr filter is applied.

Every worker disables residency loading and saving with `--residency-seed none`
and `--residency-save none`. After worker exit (including failure), the gate
asserts that no new entry appeared in the model directory. Budget extras accept
only `--host-margin-gib` and `--max-pinned-gib`; `SESSGATE_EXTRA` supplies the
default when `--extra-flags` is omitted. These flags apply identically to both
binaries and all phases, and are recorded in the run identity along with the
gate script hash. Use `--arena-gib` for the arena budget.

Session create, suffix ingestion, and the L control prefix share the ordinary
worker's prompt producer. It commits `max_rows` rows at a time and appends the
same drafter hidden rows. This imported-state replica disables standalone
prefill buffers; its admitted window is at most six rows with a drafter (eight
without one). Switching to standalone prefill would require new byte-identity
qualification. Prompt ingestion uses the ordinary logits policy; raw generated
logits are retained for qualification dumps.

Every phase prints its planning estimate: baseline/candidate **239 s**, save
**410.072 s**, restore **249 s** by default. Each includes one worker load at up
to 74 s, hashing 90 s and I/O/cleanup 30 s. The default cold R6 window estimate
is 64 ms: approximately 37.3 ms base plus 33 misses at 0.8 ms each. Two
6,144-token passes require 2,048 windows, or 131.072 s; save also allows 85 s
for decode/regression. These estimates do not establish local qualification.
Use `--window-ms`, `--load-seconds`, `--expected-hash-seconds`,
`--expected-prompt-seconds` and `--expected-decode-seconds` for local calibration.
The default absolute deadline adds 8% (at least 10 s), capped at 420 s:
baseline/candidate **259 s**, save **420 s**, restore **269 s**.
Automatic phases run after the prerequisite workers have warmed the identity
hash cache and allow 5 s for hashing by default. The automatic saves estimate
**299.296 s** for `auto-save-target` and **297.960 s** for `auto-save-dspark`,
including full-prompt controls, the target seed checks and matching DSpark plain
references. `auto-restore` estimates **164 s**. Each automatic phase launches one
default-config worker, checks enablement independently and has its own absolute
**420 s** deadline. At loads of **75 / 100 s**, the target save estimates are
**300.296 / 325.296 s**, DSpark save **298.960 / 323.960 s** and restore
**165 / 190 s**. Both automatic save estimates remain below **330 s** at a 100 s
load. Cold hashing or slower workloads require recalibration before running; progress never extends
the absolute deadline.
An explicit `--deadline-seconds` must exceed the estimate and remain within
420 s. A phase whose estimate cannot fit must be reduced before booking a window.

Session progress records include request, session, stage, completed/total work
and elapsed time. Prompt/decode progress appears every 16 windows or after
three seconds of completed work, plus completion. Hashing reports completed
bytes; restore reports read chunks and section verification. Capture, import,
and save have stage boundaries. These records do not manufacture heartbeats
while device work or I/O is blocked. `--stall-seconds` defaults to 20 s without
session progress. Each session request also has a total bound from its window
count, first-use hashing allowance, decode and I/O; progress cannot extend it.
Failure reports the last observed progress event. Startup has its own load
bound. The whole-phase watchdog remains armed through receipt publication.

No phase chains another phase. Missing prerequisite receipts or artifacts refuse
startup before model upload. Failed phases require a fresh run directory.

Set `CUDA_VISIBLE_DEVICES` to one GPU ordinal or UUID for every phase.
The gate records that selection and passes `--draft-vocab full` identically
to baseline and candidate workers; DSpark sessions require this launch route. P contains exactly 8,192 IDs extending seed
history. L processes P then S plus 64 generated rows in a single request without
capture/checkpoint/import at P. A captures/checkpoints at P and continues live.
B restores in a later invocation/window. Both DSpark modes require A == L == B
bytewise: 64 token IDs, all 64×129,280 FP32 logits, and complete canonical
target/drafter/frontier state. A→B→A parking is compared with A as well.

Cache-OFF and explicitly enabled unnamed baseline comparison uses the unmodified baseline's native --qual
JSON: fed/greedy IDs, logit FNV values, final state digests and chat tokens.
The driver extracts deterministic fields, retains raw event files, stores
canonical receipts and compares candidate receipts bytewise. Timing and process
metadata are excluded. These baseline receipts do not contain raw FP32 rows.
Candidate fresh/reset, interleaved and post-restore comparisons additionally use
--qual-dump raw IDs, every FP32 row and canonical target/drafter state. The raw
observer does not capture during a request and defaults empty.

Startup creates a suspended child with atomic job assignment outside the ownership
lock, then checks expiry and registers/resumes it under the lock. Expiry during
startup kills and reaps the child. The watchdog kills and reaps workers with two
seconds reserved, then terminates a blocked
driver at the deadline. After cleanup and all other writes, each successful phase
receipt is written to a temporary file, flushed and fsynced, then atomically renamed
as the final operation within the deadline. Later phases require this publication
version; older receipts require a fresh run. All evidence and checkpoints stay
outside Git. gate.json indexes the eleven phases; it establishes success only when
all eleven final phase receipts are present and valid. Automatic restore requires
both automatic save receipts and their artifacts before model upload. A nonzero
seed measures prompt work from that seed, not cold prefill from zero.

Session errors use `{"error":{"code","message"},"status":N}` in the spool
reply: missing explicit sources and missing suffix-only turns are `session_not_found`/404;
explicit managed-tier open failures are `session_tier_unavailable`/503 or 507;
a competing managed-tier owner is `session_tier_busy`/409,
stale or conflicting writers are `session_conflict`/409, incompatible checkpoints are
`session_incompatible`/409, corrupt checkpoints are `session_corrupt`/422 and
budget refusals are `session_budget`/507. Ordinary replies retain their existing
string error field. `resume` with `--session-path` loads a checkpoint; without a
path and without explicit durability disk it resumes RAM only. Explicit durability
disk selects the managed save point. Inventory reports active, parked and
disk states and durability.

## Measured qualification

On a single RTX PRO, the eight explicit-checkpoint phases passed
with P = 8,192 tokens, an imported 2,048-token seed, suffix S and 64 generated
rows. Ordinary target-only and DSpark arms matched the labelled baseline;
A == L == B == parked included every token, all 64 x 129,280 FP32 logits,
attention, drafter and frontier bytes. Corrupt, truncated and incompatible
checkpoint refusals also passed. DSpark used `--draft-vocab full` throughout.

An earlier session run measured save phases at **188-192 s** and restore at
**85-88 s**, with worker loads around **75-100 s** and identity-cache hits near
**1 ms/file**. The public-port rerun measured **210.32 s** target-only save,
**255.52 s** DSpark save, **110.42 s** target-only restore and **115.18 s** DSpark
restore. Baseline phases took **91.66 / 78.46 s** and candidate ordinary phases
**84.19 / 74.61 s**. These are workload-specific receipts for the eight explicit
phases.

The final candidate then passed all twelve phases in one GPU run: the eight
explicit phases plus `auto-save-target`, `auto-save-dspark`, `auto-restore` and
`auto-admission`, each with return code 0 against the labelled baseline.
