# Native full-model producer (draft)

`glm_native_service` is a persistent Windows CUDA worker for one sm_120 PRO.
It is explicit and unqualified; it changes no node engine defaults or auto route.
Build offline with MSVC, CUDA 13, CMake and Ninja using
`build.ps1 -Fresh -BuildDir BUILD_DIR -TierstreamSource TIERSTREAM_SOURCE -TierstreamBuild TIERSTREAM_BUILD -CoordinationFile COORDINATION_FILE`
under `pwsh`, with new GLM and dependency build directories. The dependency is
a clean detached tierstream checkout at `873b00453cec123ac9b7a8a8ebee7e60e20c643c`;
it is linked as a separately built static library, never copied into this tree.
Retain tierstream's MIT license (copyright 2026 hyukudan) with redistributed
linked binaries. The build waits while the last `GPU-BUSY start/end` coordination
marker is `start`, polling every two minutes before each build. The standalone
configure disables compiler launchers/ccache and builds at idle CPU priority
with at most two jobs. The build
compiles and disassembles only; SASS stays in RAM and only small evidence JSON
is written. Run CPU verification separately:

```powershell
ctest --test-dir BUILD_DIR --output-on-failure -j 1
BUILD_DIR/glm_native_verified_io.exe BUILD_DIR
python -B tests/glm_native/test_qualification_window.py
python -B tests/glm_native/test_producer.py --build BUILD_DIR
# Without any sockets/network:
python -B tests/glm_native/test_producer.py --cache-only --build BUILD_DIR
python -B tests/glm_native/test_shards.py --build BUILD_DIR
python -B tests/glm_native/test_shards.py --model MODEL_DIR --build BUILD_DIR
python -B tests/glm_native/receiver_loopback.py BUILD_DIR/glm_native_receiver.exe BUILD_DIR
```

## Model and cache

All 45 trunk layers, layer 45 MTP, embeddings, final norm, both HC sites,
shared/dense FFNs and every routed expert must be available. An incomplete
shard set fails before publishing any cache. The cache inventory validates
the full native tensor schema, including a separate `mtp.safetensors` omitted
from a trunk-only shard index, mul1 markers, rates, types, scales, finite
plain weights and the supported configuration. No vocabulary head is required.

```powershell
python -B src/glm/rtxpf/producer/cache.py --model MODEL_DIR --cache CACHE_DIR
python -B src/glm/rtxpf/producer/cache.py --verify --cache CACHE_DIR
```

This decoder already consumes the original compressed K16/N16 layout.
Persistent packing aligns original compressed tensor bytes to 256 bytes;
it does not decode or requantize them. The key binds the content-hashed
source manifest, layout, decoder, H128 transform and tile versions. Blob
digests are established before weight allocation. The worker holds read-only
Win32 handles denying writes and deletion through its entire lifetime. A cold
verification runs one worker per logical core, hashing independent whole files
in parallel with SHA-NI when available and two aligned 8 MiB reads in flight.
Whole-file SHA-256 values and the manifest format remain unchanged.

`%LOCALAPPDATA%/Strix/glm-verified-hashes-v2` persists computed shard digests and
successful blob verifications. Its protected directory DACL grants access only to the current
user and SYSTEM. A random 32-byte per-user key is created once, persisted using
Windows DPAPI (`CryptProtectData`, current-user scope), and authenticates every
record with HMAC-SHA256. The authenticated v3 canonical record binds format version,
volume serial, 128-bit file ID, size, last-write, creation and change times,
the file's last-change USN (`FSCTL_READ_FILE_USN_DATA` on the held handle), the
volume's USN journal ID (`FSCTL_QUERY_USN_JOURNAL`), and the computed digest.
Cache mode also checks that digest against the pinned expected value.
Byte writes invalidate records even after all basic timestamps are restored;
recreating the journal also invalidates records. Old timestamp-only records miss.
The accelerator requires a readable active NTFS journal. Non-NTFS volumes,
inactive/unavailable journals and failed queries always hash the complete file
and publish no record. Missing/wrong MACs, metadata/digest changes, and key
mismatches force a complete hash. Unusable keys are repaired under an exclusive
interprocess lock file, with a re-check under that lock; the repairing attempt
hashes fully and publishes no record. Any directory, ACL, key load, RNG, DPAPI,
key persistence or publication failure disables the accelerator for that attempt:
ignore records, hash fully, publish nothing. Record-write failure retains the
successful cold verification. Content mismatches and real read failures still reject.
The trust boundary includes NTFS journal semantics and Windows user isolation:
same-user code execution, privileged journal/volume manipulation, an
administrator, and offline filesystem/DPAPI compromise are out of scope.
Original shards are not
retiled or reread on a validated cache hit. Treat a cache directory as immutable and
model-specific; use another directory for a different model. Pin the complete
`manifest.json` SHA-256 in the launch command so tensor metadata is also bound.
Cache creation and validation are startup work, outside warm request timings.
The manifest is read once through its held deny-write/delete handle. Its pinned
SHA is checked on exactly those bytes, and the resulting parsed object is passed
to Runtime and the integrity tool without another pathname open. Prewarm or audit the same persistent cache without any
CUDA linkage:

```powershell
BUILD_DIR/glm_native_integrity.exe CACHE_DIR MANIFEST_SHA256
```

The CPU helper verifies the pinned manifest and all listed blobs; it does not
replace the runtime's full schema/geometry checks. Python `cache.py --verify`
remains an independent full-hash audit and does not use the metadata cache.
The helper reports `journal_unavailable_blobs` alongside `verified_hash_hits`;
zero hits on a volume without a journal means full verification, not failure.

### Direct shards (no disk cache)

Use `--model MODEL_DIR` instead of `--cache` and the cache manifest argument.
The worker opens the complete shard set read-only, holds Windows handles that
refuse writes/deletes, validates configuration/schema/markers, and computes or
recalls each source digest through the same authenticated authority as cache mode.
A matching record supplies its authenticated digest without reading the shard
payload. A miss hashes the complete held handle with two aligned 8 MiB reads in
flight, in parallel across files, then publishes the computed digest. Unavailable
journals or authority failures force full hashing without publishing records.
Configuration, shard headers and transform markers consume those retained handles.
Resident and file-tier payloads use the authenticated multi-location reader
described below. Replacing a model directory junction after inventory cannot
redirect payload reads: the adapter resolves the held object's final path and
locks that namespace for the reader's lifetime. The index
JSON is not consumed by this native reader; it inventories all shard files.
CPU regressions replace an ancestor junction after authenticated recall and
between file-tier requests, and compare synthetic packed blobs against the
previous pathname reader byte for byte, including zero padding and chunk tails.
The handles remain held until worker teardown. It derives the same content model
identity and cache key as the cache builder. No persistent packed or decoded files are created.

The reader builds an offset-sorted coalesced plan per layer/shard. Requests are
normally 32 MiB with 8--40 MiB tails. Source offsets align to 64 KiB; the EOF
tail length aligns to the reader's 4 KiB contract (a whole file smaller than
8 MiB is the size exception).
They read directly into the registered host memory used by `cudaMemcpyAsync`;
there is no payload window or staging memcpy. A scatter map supplies each
original tensor to its original packed device offset; device padding is zeroed.
Thus lexicographic tensor order, 256-byte padding, K16/N16 bytes, model identity
and cache key are unchanged. Cold base loads use bounded batches of 16 pinned
40 MiB slots. Per-volume reader pools are shared across shards, with 16 workers
by default (`--drive-readers 8..32`). Two file-layer ordinals read ahead,
independent of device parity. Host slots wait for their last H2D completion;
device slots wait for compute retirement. Finite F16/BF16/F32 validation runs
once at startup over immutable source bytes, including all file-tier layers.
In `--cache` mode the base and every expert blob are scanned too, at each
tensor's manifest offset and dtype, before startup succeeds: a pinned manifest
SHA does not prove the cache builder validated floats, so a cache holding
NaN/Inf is rejected at startup. Requests do not scan weights again. No floating point transformation is added.

### Authenticated multi-location I/O

`--weight-dirs 'DIR_A;DIR_B'` optionally supplies existing tierstream stripe
manifests; no stripe or model-data copies are created by the worker. The pinned
ABI has no callback authenticating its primary serving handle. The Windows GLM
local-drive adapter therefore retains the original HMAC/USN-verified deny-write/delete handle,
locks every ancestor of its final path against writes/rename/delete, refuses reparse
ancestors, and retains a second deny-write/delete guard with the same volume and
128-bit FileId. The canonical serving names reported by `ss_mfile` are checked
against these guards while the namespace remains locked. The library's own held
deny-write handles consequently refer to those immutable authenticated objects.
Any failed identity, namespace lock or direct-I/O check rejects startup.

Each admitted secondary manifest and data file is held and namespace-locked too.
The manifest is bound like the data file: the supplied directory and all its
ancestors are locked and must not be reparse points, the manifest leaf must be
a regular non-reparse file with the canonical `<primary>.stripe.json` name in
that same directory, and its deny-write/delete handle is retained for the
reader lifetime. The pinned ABI reparses the manifest by pathname and exposes
neither its text nor a held-handle admission, so after `ss_mfile_open` every
serving location's library-derived manifest name must be the admitted held
object of the tier owning that data, with bytes equal to the admitted text.
Another mapping over the same data FileId therefore cannot be served.
The pinned library lowercases every canonical path, so under a case-sensitive
NTFS directory its lowercase name could select an unlocked sibling (`tier`
beside the locked `Tier`). Every held source, manifest and weight directory and
all their ancestors are therefore queried with `FileCaseSensitiveInfo`, and a
set `FILE_CS_FLAG_CASE_SENSITIVE_DIR` (or any failure other than "unsupported
by this filesystem") refuses startup; model trees created case-sensitive by WSL
must be copied or have the flag cleared. Before `ss_mfile_open`, the exact
lowercased primary, directory, manifest and data names the library will build
must equal the bound names case-insensitively and open the same held FileIds.
The CPU test cannot set the flag without privilege/WSL; it then tests the
decision function and reports the namespace case as skipped. The full fix is
the library's bound-open API (held manifest + admitted bytes, held data handles)
on tierstream's reader/next; this adapter is retired on that bump.
Every placed logical chunk is compared byte for byte with the authenticated
primary before enabling that directory. An unauthenticated xxh64 cache is never
an authority for secondary admission. This extra verification is startup work.
The existing Windows/user-isolation trust boundary above still applies.

A future ABI can remove the namespace adapter with an options callback
`authenticate(user, location, native_held_handle)` invoked on every actual
serving handle before hashing, session reuse or any read. It must be invoked
again for every reused session; rejection must fail the open without issuing
payload reads, and the authenticated handle must remain held through close.
GLM would compare volume/FileId/size to its held verified primary and require
independent secondary admission. A borrowed-handle API may instead duplicate
the supplied deny-write handle and read exclusively through that duplicate.
The dependency checkout is not patched for either extension.

The pinned multi-location ABI serializes large entries within one preadv call.
GLM supplies eight concurrent calls to its shared per-volume pools and drains
all of them on errors. The usual 32 MiB ticket spans multiple 16 MiB stripe
chunks, permitting work on both homes. Actual drive use remains visible in
the receipts; an absent stripe cannot supply bytes from another drive.

Pending tierstream consolidation (on the tierstream I/O dependency bump, not
done here): `LayerReadAhead`'s `std::async` slot submission/draining and
`TierReader::read()`'s eight-call large-vector fan-out move to the library's
`DgReadPool` scheduler and a library-owned batch/asynchronous read interface.
GLM keeps tensor packing, dtype/finite validation, cache policy and CUDA
lifetime fences. A version bump alone is insufficient while the library still
serializes large vector entries.

`--file-tier-cache off|auto|on` controls optional pinned retention between
requests (default `auto`). Both admission modes require three physical/commit
availability samples, 50 ms apart, with variation at most 256 MiB each. For
rounded allocation size B, admission requires
`B + host_floor + 1 GiB <= min(all six availability samples)`.
The floor is rechecked after registration; failure releases the allocation.
Pressure below floor plus 1 GiB revokes caches before the next request, after
H2D drains. Admission is reconsidered between requests if RAM becomes available;
an impossible budget skips the sampling delay. Newly admitted buffers are filled
directly by normal read-ahead and become cache hits only after a successful
acquire. Failed fills remain misses. Explicit file-tier qualification bypasses retention. External
allocators may change RAM availability; the cache cannot reserve their floor.

Health's `startup_drive_io` and request `drive_io` report per-drive bytes,
library piece count and summed read busy milliseconds (not wall time).
Startup file-tier finite verification has separate payload-byte/time counters.
The file-tier budget/estimate describes uncached logical bytes; request timings
report actual reads and current retained-cache bytes. Payload byte
counters exclude coalescing overread; drive counters include physical reads.
The host-only `glm_native_io_bench SHARD BYTES_PER_ARM` uses ABBA, at most
1 GiB per arm (a 16 MiB multiple, so both arms read exactly the same bytes) and
4 GiB aggregate from one held object, without hashing or
writing its payload. It compares the legacy two-copy window with preadv at
depth 16 in aligned RAM. Both arms are prepared before timing: the shared ring
and staging are prefaulted, and each legacy arm gets a fresh prefaulted cursor
with no cached window, so setup and page-touch costs are outside both arms. It makes no pinned/H2D or model qualification claim.

`test_shards.py --build BUILD_DIR` runs small synthetic CPU source sets, checking
cold bytes read, restart recall without payload reads, timestamp-restored writes,
authenticated-record failures, and unchanged model identity/cache keys against
independent full hashing. With explicit `--model MODEL_DIR`, it hashes production
C++ virtual streams and the cache builder's production packing stream independently,
without creating a disk cache. Its
scope includes the complete base blob, two routed KDA and two routed DSA layers,
and the unique MTP layer. Source schema, layout sizes, canonical model identity
and cache key are checked too. CPU byte identity establishes compressed placement;
it does not qualify GPU arithmetic or model fidelity.

Health reports startup source hashing time, resident load time/bytes, file-tier
bytes/request and a host read-cost estimate extrapolated from resident loads.
Request timings include actual file read/placement time and bytes, staging wait,
graph binding time and total runtime. These host costs can overlap GPU work;
cold storage and actual admitted tiers determine the observed cost.

## Execution and residency

The driver embeds prefix `[0,N-1)`, executes each entire layer before the next,
then uses normalized final trunk hidden rows paired with input tokens `[1,N)`
to populate MTP. HC normalization/collapse/expansion, norms, routing and nonlinear
parts use FP32 arithmetic. The node's HC chains, 20 Sinkhorn iterations and
q/k/v-major convolution serialization are retained. Native projections,
KDA32, DSA64 and CUTLASS grouped MoE preserve their existing declared schedules.
Each layer completes before the next layer begins. Within a layer, attention,
HC, routing, shared/routed FFNs and MTP run in exact row tiles. KDA convolution
and B32 recurrence carry across tiles; DSA retains absolute positions and
complete pools. A tail contains only real rows, with no padded recurrence or
routing. State release occurs after the layer's final row tile.

Capacity accepts **131072 inputs**, including the node suffix token. Only the
four residual streams and final hidden rows span the prefix. All other activation,
dispatch and MTP scratch uses `--row-tile` (default `auto`: 256 below 32K
capacity and 2048 at or above 32K; numeric values fix a multiple of 32 no larger
than 32768). DSA's bounded 64-query score scratch still scales linearly with pool
capacity. At 128K/tile 256, activation accounting is under 11 GiB, before compressed
weights/rings, attention state and the shared native workspace. Admission still
uses actual allocation sizes; this is not evidence of successful device allocation.
The narrowed performance claim is fewer decode passes from larger row tiles,
not decoded-window reuse across a layer. Expert decoding still repeats per row
tile with two bounded decoded windows; at
32K the automatic tile reduces the number of complete expert decode passes
from 128 to 16. No cross-tile reuse or measured speedup is claimed. This changes scheduling only: KDA B32 carry, DSA absolute
positions, chronological K chains, H128/RN16 boundaries, route-slot order and
MMA family remain fixed. M-dependent hot/cold dispatch still needs the bitwise
execution gate. `--row-tile 256` retains the previous scheduling arm.
`admission_model.py` estimates activation/workspace and residual file layers
without CUDA; its physical and commit inputs are post-load observations.
Confirm estimates against health's actual allocation counters.

CUDA Graph templates capture the complete per-layer row sequence, including
CUTLASS decode-stream joins. At most two executable templates per layer are held:
the fixed full row tile and an exact tail. Changed positions, pointers, identities
or tail sizes are rebound through capture and `cudaGraphExecUpdate`, reinstantiating
if topology changes. No graphs accumulate per prefix position. Binding overhead
is reported, and `--no-graphs` provides the ordinary enqueue control arm. Graph
allocation is checked against the device cap. Graph execution, tiled numerical
identity and 128K admission remain GPU qualification gates.

The 20/19/14 device expert layers are upper targets for 8K/32K/128K
respectively. Pinned layers have no fixed count cap and are preferred to files
whenever physical RAM, Windows commit and pinned registration all permit.
Admission charges actual fixed allocations, existing device use, the 99.86 GB
cap, a 2 GiB device reserve, available physical RAM and available Windows commit.
Pinned admission leaves `--host-floor-gib` (default 8 GiB) available in both
physical memory and Windows commit; `--host-floor-gib 16` restores the old floor. A failed optional allocation demotes
to host or file immediately, with no admission retry or wait for resources.
There are two compressed layer rings, two decoded 16-expert buffers inside the
shared native workspace, one ordered H2D stream, event-protected reuse and at
most two file-layer read-ahead futures. Fixed rings that cannot fit cause startup
rejection. File-tier requests use the already-verified held handles with
unbuffered overlapped reads straight into 64 KiB-aligned registered rings,
without reopening or hashing. Immutable pinned descriptors are copied once at
startup. Ready/retired events permit read N+2, H2D N+1 and compute N to overlap;
host reuse waits for the last H2D read, device reuse for the last compute read.
`--max-device-layers N` and `--max-pinned-layers N` cap the respective expert
tier counts (0..43, defaults 43). Device admission also retains the capacity's
existing upper target; memory guards can demote further. All remaining layers
use the file tier. Caps, actual counts and the file-ring parity mask appear in
health and self-check reports. `--require-file-tier` requires at least two file
layers covering both ring parities and positive read bytes through each ring on
every request. Caps that guarantee any file layers also require positive file
reads. Requests reread their file layers, including consecutive requests when
only the last two layers are files. A failed coverage gate prevents sealing.
For a deterministic qualification arm at capacity 32768 use
`--max-device-layers 19 --max-pinned-layers 22 --require-file-tier`.
`--serial-stage` is the synchronization/control arm. Request bytes/read time
remain measured, and health reports hash hits, workers, floor and IO shape.
`GET /health` records actual admitted tiers and free-memory snapshots;
target counts are never reported as measured allocations.

## Service and wire

Set `CUDA_VISIBLE_DEVICES` to one **full** PRO UUID, then launch only during an
authorized device window:

```powershell
BUILD_DIR/glm_native_service.exe --model MODEL_DIR --key-file import.key --capacity 8192 --row-tile auto --bind 127.0.0.1 --port 8097 --allow-gpu
# Optional faster persistent cache, after its disk budget is approved:
BUILD_DIR/glm_native_service.exe --cache CACHE_DIR --cache-manifest-sha256 MANIFEST_SHA256 --key-file import.key --capacity 8192 --row-tile auto --bind 127.0.0.1 --port 8097 --allow-gpu
```

`GET /health` returns capacity, identity, draft qualification and residency.
`POST /v1/prefill` accepts this JSON shape:

```json
{"tokens":[1,2,3],"handoff":{"rank0":"127.0.0.1:9000","rank1":"127.0.0.1:9001","key":"64_HEX_DIGITS","node_identity":"r1b"},"identity":"glm-pf-rtx-native-v1"}
```

`key` is the 32 raw shared key bytes encoded as hex, compared against the
startup key file before opening either receiver connection. The service does
not log keys, bodies or endpoints. HTTP does not encrypt the request. Protect
the service and use an appropriate private deployment link. One request owns
the runtime; another prefill receives 503. Health remains available concurrently.
`STRIX_GLM_RTX_PREFILL=0` refuses startup. Unknown fields are refused.

The sender uses wire v1 begin/frame/commit/abort exactly: fresh receiver
challenge, HMAC-SHA256 with sequence, payload SHA-256 and ordered complete-frame
digests, raw bytes only. Per-layer state is released as soon as attention state
is final; bounded sender queues overlap the rest of the producer. KDA heads
are split 32 per rank; convolution selects each TP slice separately within
q/k/v. DSA histories/pools are replicated to both ranks. Pooled tails are omitted.

Both sender queues must drain before either commit is sent. On failure,
unsealed transactions abort or disconnect. ACK 1 is only a **local seal**:
the HTTP receipt is not evidence of the receivers' collective commit, apply
or continuation. Obtain both node receipts. The producer never supplies the
node's final prompt row, head, MTP seed or a qualification receipt. Node binary
hashes and node phase timings must come from the composed execution bundle.

The receiver accepts only the exact base native identity. Alternate indexer
IDs cannot be substituted in live handoff. For an offline diagnostic export,
use the same executable with `--export-state PRIVATE_OUTPUT --prompt PROMPT.u32
--identity INDEX_ID` instead of `--key-file`; all DSA index IDs from `dsa.cuh`
are selectable and also apply to MTP. These files/receipts retain that ID.

## Pending gates

CPU protocol tests establish byte transport into CPU staging, including both
ranks, malformed traffic and abort. They establish no CUDA execution or node
continuation. Compilation and per-object SASS checks cover native MMA/copy
loops, KDA blocked recurrence and DSA/MTP kernels. Complete-model allocation,
restart latency, pressure/demotion, deadline stalls, numerical/model fidelity,
node suffix, speculative oracle and endpoint qualification remain pending.
No on-disk cache is currently available. The `--cache` overlapped-reader and
H2D-overlap execution qualification is a separate pending gate. Direct-shard
file-tier runs and CPU byte tests do not establish that cache-reader gate.
The service contains no synthetic-state success path or CPU arithmetic fallback.

## Scheduling and FP32 index execution gate

The tiled FP32 index candidate uses 8-row/32-column projections with shared
operands and two complete independent K chains per thread. Scores use two
queries and 16 pools, with eight accumulators per thread, ascending K=0..127,
and the unchanged ascending 32-head fold. No split-K or new rounding point is
introduced. `--legacy-fp32-index` selects the original scalar index kernels.
All candidates retain draft status until native bitwise and workload gates pass.

`glm_dsa_bench --weights LAYER7.native --input CAPTURED_4097.f32 --rows 4097
--iters 1 --index-variant fp32 --self-check-exact --allow-gpu` runs OLD/NEW
kernels on the same captured normalized layer input. It compares raw ordinary
and trellis projections, valid score windows/ordered keys, full output/state,
selected IDs and cardinalities with host memcmp and differing-element counts.
4097 rows exercise both sides of the 512-pool cutoff, odd row tails and partial
pools. The captured FP32 input is 67,125,248 bytes. This mode requires real
captured input, and never substitutes another index identity.

`glm_native_service --self-check-schedule --prompt PROMPT.u32 ...` runs the
internal 256-row scalar-index/serial-stage arm and the candidate using the
same resident real weights and token IDs. It compares every exported state
component and the complete final hidden buffer using memcmp and byte counts;
per-rank SHA-256 covers payloads in release order, without wire headers.
Repeated `--prompt` options reuse one startup. Both arms include actual read
bytes/time and total prefill time. Mismatch exits 3 with counts. No state files
are written; baseline state is retained in bounded host memory for the check.

The main `qualification_window.py` mode compares HTTP handoff results to
private, independently captured `a6d126c4` receiver SEALED state digests. It
starts the candidate with `--model MODEL_DIR` (direct shards, no cache), capacity
32768 and default numerical/scheduling flags. Two local CPU test receivers use
node identity `r1`. Each resident service handles consecutive pairs of
1024/8192/32768-token requests. A second service repeats the same plan with
`--max-device-layers 19 --max-pinned-layers 22 --require-file-tier`. Every rank,
request and repeat must match the independent reference; every forced-file
request must read both rings. Missing evidence is failure, never a reference pass.

Keep reference results, provenance, prompt pairs and the private argument file
outside Git. The input file supplies `model`, `prompt_dir`, `reference_results`,
`reference_provenance`, `reference_results_sha256`, `reference_provenance_sha256`,
`reference_service_sha256` and `reference_commit` (`a6d126c4`). Pin the exact
private results/provenance files and the reference executable identified there;
that reference used cache input, capacity 32768 and row tile 256. The launcher
checks prompt u32 bytes against JSON IDs and provenance hashes, and binds both
candidate binaries, the reference model/manifest identity and expected complete
per-rank hashes in its private evidence. Reference provenance is an operator
attestation to the independent executable/run; this launcher does not rebuild
or execute that historical binary.

```powershell
# Plan only: no process, sockets, CUDA imports or device discovery.
python -B src/glm/rtxpf/producer/qualification_window.py --build BUILD_DIR --inputs PRIVATE_INPUTS.json
# CPU-only validation of private pins, all prompt pairs and candidate binaries.
python -B src/glm/rtxpf/producer/qualification_window.py --build BUILD_DIR --inputs PRIVATE_INPUTS.json --check-inputs
```

Add `--allow-gpu` only in an authorized PRO window, with one full PRO UUID in
`CUDA_VISIBLE_DEVICES`. Expected wall budget is **90--150 minutes** for two cold
source-hash startups and twelve HTTP requests; this is an estimate, not a measured
candidate latency. The hard aggregate deadline defaults to **180 minutes**,
can be shortened with `--timeout SECONDS`, and cannot exceed 10800 seconds.
An independent supervisor process enforces the absolute deadline, including input
validation, executable copying/hashing, startup and execution. Before any worker
code runs it is assigned, suspended, to a Windows job with `KILL_ON_JOB_CLOSE`;
all descendants inherit that job without breakaway. At the deadline the supervisor
terminates the entire job even if the worker blocks in validation or HTTP IO.
Cleanup waits at most fifteen additional seconds in total; no process-name kills
are used. Timeout is INCOMPLETE. Each run copies service and receiver executables
into a unique private run directory, hashes the copies and holds read handles
denying writes and deletion (and a directory handle denying replacement) until
all requests complete. Only those copies are launched. Every HTTP receipt must
name exactly the pinned service digest in `producer_executable_sha256`; rebuilding
the original candidate cannot change the running qualification identity. Logs,
receipts and results stay in a new immutable directory under the build directory;
no state payload files are written and the temporary handoff key is removed.

`--diagnostic-self-check` adds the internal OLD/NEW check after the independent
HTTP passes, under the same aggregate deadline, with both file rings required.
It needs a third startup and additional compute and may exceed the budget.
Internal agreement is diagnostic only: it shares candidate code. The independent
SEALED references cover exported state; independent final-hidden-buffer evidence,
other capacities (including the normal 8192 default), targeted hot/cold dispatch,
interleaved timing medians, model fidelity and complete node continuation remain
separate pending gates. No performance or production qualification follows from
CPU checks or from a reference plan that has not run.
The harness prints the separate pending `--cache` overlapped-reader/H2D-overlap
gate in its plan, input-validation output and run verdict; neither direct-shard
phase qualifies it, and no cache is created by this harness.
