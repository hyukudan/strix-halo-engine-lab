# Execution observability

The offline inventory and optional host dispatcher telemetry are the first
development increments for reconstructing effective execution and attributing
cost. They do not change the automatic optimization profile or numerical kernels.

## Offline inventory

`tools/qualification/runtime_inventory.py` reads existing CMake caches and JSON
outputs from the decoder, tensor audit, model manifest and platform probe. It can
hash the actual model and executable. It launches no decoder, probe or GPU job.

```bash
python tools/qualification/runtime_inventory.py --repo . \
  --cmake-cache build-host/CMakeCache.txt --receipt /private/decoder.json \
  --context-capacity-tokens 8192 --occupied-prefix-tokens 43 \
  --output /private/runtime-inventory.json
```

The optional receipt is an existing decoder stdout JSON object. Capacity and
occupied prefix are distinct; explicit declarations must agree with receipt
fields when present. Model file SHA-256 and the versioned model-content identity
are distinct and are both reported when the model is supplied.

A CMake cache attests configuration only. Tensor requirements attest neither
executable launcher support nor row/concurrency support. Missing effective
catalog, binary bindings and native qualification stay null. Profile resolution
comes from the actual decoder receipt, without duplicating the application
resolver. The collector separately surfaces reported prepared-submission,
shared-overlap, batched-activation and generation-output overlays. A selected
qualified base label does not qualify a combined opt-in configuration; such
receipts carry an explicit limitation, without fabricating a conflict with the
base recipe. Missing overlay metadata remains unknown.
Conflicting evidence produces a conflict list and exit code 3;
malformed inputs use exit code 2.
An inventory with unresolved fields is not a qualified baseline.

The decoder's optional `--execution-inventory` adds the actual selected compiled
steps and host launcher identities to each stdout receipt path. The exporter
checks execution, kernel-plan and launch-plan identities, dense steps and bound
functions without invoking any launcher. Variant precision, layouts, phase and
qualification are explicitly catalog declarations; native qualification remains
null. The selected decode geometry does not admit verify or prefill geometries.
Without this option the larger inventory is neither constructed nor emitted.

## Optional host instrumentation

`--host-dispatch-telemetry` binds one serialized thread-local observer before
model setup and retains it through GPU-owner teardown. Its schema-2 JSON goes
to stderr, including exception unwinding. It does not change profile selection.
`--execution-telemetry-trace` additionally retains the first 256 correlated events;
it requires the counter option and reports dropped events explicitly. Both
options are for diagnostic runs, and must be absent from speed measurements.

Fixed counters distinguish host launcher return/rejection/failure, actual HIP
copy/memset calls, allocation/free/registration calls, stream/event waits,
synchronization calls, capture, instantiate and replay API results. Successful
copy bytes are *accepted logical request bytes*, including queued async work,
not physical transfers or proof of completion. Failed calls retain their
requested byte count and original error code. Allocator bookkeeping and actual
freed byte counts remain unknown; no pointer-to-allocation shadow map is added.

Session-publication hooks run only after successful serial/prepared/window
commit. Proposed state generations remain a separate event and are never
promoted into token commits. Graph capture host calls and replay API calls have
separate counters. A trace `during_capture` labels the host-thread capture
interval, not the membership or execution count of graph nodes. Replay ordinals
identify API attempts; successful/failed replay counts remain distinct. GPU
kernel submissions/completions and physical transfer bytes stay null. A returned
launcher or synchronization alone does not invent such a count.

Correlation records request, execution, position, layer, operator, row and step.
The decoder's one-request process uses request id 1; `session_epoch` is explicitly
the state device-allocation generation, not an inferred semantic reset counter.
Joint R2 operations use an unknown individual-row sentinel and rows=2; R1 loops
retain their actual row index. Host steady-clock phase intervals cover admission,
prepare, prefill/token, selection and full-logit delivery; operator durations are
actual host launcher intervals. Nested phases overlap and must not be summed as
a disjoint request wall time. No GPU duration is derived from these clocks.

The HIP redirects are force-included only into runtime and decoder compilation.
The full SDK is parsed before the function-like redirects, and wrapper bodies
call the original API before macro definitions, avoiding recursive interception.
Function-pointer calls, other host threads and external-library/driver internals
are outside this coverage. With no observer the wrappers directly call HIP,
without timestamp reads, observer allocation, locks or additional HIP queries.
The observer owns only fixed arrays and saturating integers. This source change
has not been compiled or executed; target overhead remains unmeasured.

`summarize_gpu_trace` separately aggregates supplied profiler intervals in one
clock domain. It reports sum and busy union across all queues, rejecting invalid
ranges, mixed clocks and overflow. It neither subtracts a GPU sum from host wall
time nor infers dependencies or a DAG critical path. Hardware profiler/page-fault
capabilities are not discovered by these hooks and remain unobserved.

## Retained backing and bounded prefix snapshots

With execution inventory enabled, main consumes `HipModelResidency`'s actual
strong mapping/registration/copy owner inventory. Registered aliases reuse their
backing identity; copied allocations have distinct identities. The reported
reservation/peak values are declared virtual retained extents, not RSS/PFN or
physical-page measurements. No model bytes are duplicated for this inventory.

The explicit prefix-roundtrip path now stores its snapshot in a one-entry
`SerialPrefixCache`. Its retained byte budget is computed from the exact snapshot
plan before allocation; the capture candidate peak is outside the retained-cache
budget, and only this one capture is live. The aliasing lookup handle pins both
snapshot and actual experiment/allocator owners through completion or guarded
failure drain. Default decoding creates neither the pool nor a snapshot.

## Requested HIP process settings

With `--execution-inventory`, the decoder also emits the fixed allowlist
`hip_runtime_environment`: requested hardware queues, SDMA/architecture override,
kernel/copy serialization, launch blocking and device visibility. Values are
read from the actual decoder process; unset is null and an empty value remains
an empty string. The collector preserves this receipt instead of substituting
its own process environment. Effective hardware queues and observed concurrency
remain null: an environment request or two streams does not establish overlap.
No setting is modified. These source changes have not been executed yet.

## Planned PLE helper budget

`--ple-cpu-helper-threads 1..8` is an explicit per-planned-backend upper
bound, including its coordinator. It requires the resolved planned PLE backend
and does not enable that backend or alter the optimization profile by itself.
The default zero leaves legacy scheduling unchanged. Independent serving
sessions have independent backend budgets; this is not a process-wide CPU
budget, affinity policy or limit on custom source-internal threads. Receipts
report the requested cap separately from unknown actual helper counts.

## Opt-in device queue event trace

`--device-queue-trace` requires host telemetry and resolved eager D1. It does not
enable D1 or change any optimization flag. Graph, integer head, serving,
known-prefix, prompt lookup and prefix roundtrip are rejected for this initial
scope. Eager direct/prepared, sibling-stream overlap and GDN-island replay remain
explicitly profileable; internal graph nodes are not expanded into kernel spans.

The native observer preallocates2048 span slots (two timing events each) and an
origin event during setup. It forwards the existing host counters and receives
the actual dispatcher/TLS observer hooks. It retains the D1 stream/allocation
owner; binding lifetime ends before observer destruction. Event API errors are
sticky: the existing before-commit callback checks observer health before D1
publication, with no error clearing or new per-operator synchronization.

After normal existing producer completion, the decoder emits the standalone
JSON event trace on stderr. Export uses only event query/elapsed APIs and adds
no synchronization. Stored errors make the command fail rather than accepting a
healthy trace. Unknown/unowned/captured spans, dropped capacity and incomplete
queries stay explicitly labelled. Events perturb scheduling: these are launcher
queue intervals (including gaps/waits), not individual kernel busy times, GPU
submit counts or physical DRAM bytes. No benchmark tracing is automatic.
