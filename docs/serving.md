# Independent-session serving

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

## Explicit shared prefix cache

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

## Independent bounded PLE row caches

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
