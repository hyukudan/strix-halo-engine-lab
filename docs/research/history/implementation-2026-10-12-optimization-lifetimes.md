# Optimization lifetime implementation

This slice adds concrete prefix-state, staging and projection consumers while retaining
native model/transport/performance qualification as separate gates.

## Retained arithmetic history

`execute_serial_token` stages an arithmetic-row interval before dispatch and publishes
it only after successful state completion. Consecutive identical serial rows coalesce;
canonical prefill chunk cuts remain distinct. A history records numerical, reduction
and state-update identities through the resume position. Unknown earlier state is
kept unknown, and failed mutation taints history.

Prefix capture requires complete history. Matching/`try_restore` treats different
tokens, chunk cuts, arithmetic identities, taint or runtime applicability as an ordinary
cache miss before any transfer. Checked prefix restore installs validated producer
history after completed transfer. Raw snapshots and unqualified verification windows
clear unavailable lineage rather than inventing serial provenance. Transfer/dispatch
failure poisons state and history.

## Current graph eligibility

Compiled serial/verification wrappers recheck current resource bindings, allocation
generations, weight residency, unsafe/dynamic/host steps and shape. Token/position
arguments require an explicit stable device-data ABI; baked eager scalar arguments
are rejected. Checked memory covers all sessions, shape/eligibility instances,
workspace, executable storage, capture peak and retained dynamic inputs.

This is a current-resource eligibility guard, not a native graph implementation or
physical memory reservation. Existing eager whole-model plans remain ineligible.
Native replay qualification is always false in portable results. The packed projection
lab separately implements experimental HIP capture/replay of one fixed-shape operator
and changes its device input values to check against an independent reference.

## Event-owned host uploads

`HostUploadPool` owns source bytes and allocation generations through prepare, submit,
poll and cancel. Each pool has a process-unique nonwrapping ticket namespace. Pending
slots cannot be reused; cross-pool/stale tickets, overlapping destinations and binding
drift reject. Count, bytes and poll progress are bounded. Only explicit completion or
cancelled-quiescence proof retires an operation. Errors/timeouts retain sources and
prevent new work; pool destruction does not drop backend-owned pending sources.

`HipHostUploadBackend` uses pinned host memory, explicit legacy-stream copies and
events, with device-identity checks and a quiescence fallback after recording failure.
Failed teardown retains the operation/source/event until process exit. Destination
allocation lifetime remains the caller's responsibility; the destination must remain
owned through quiescence. Progress requires one serialized owner.

The PLE launcher consumes this service before dependent projections. Its default is
the existing synchronous copy. Explicit owned opt-in has a one-slot 16 KiB pinned
source maximum, exposed for admission. Charge this additional storage plus backend
metadata and existing staging in the actual host memory observation/manifest. Injected
pools require their own explicit bounds and ownership. No service is automatically
enabled by discovery or graph eligibility.

## Projection experiment and validation

IQ4_XS projection already reuses each decoded weight across C1/2/4/8 logical rows.
The new candidate preserves its 256-thread FP32 FMA/reduction tree and performs the
last five tree levels in 32-lane registers, avoiding five workgroup barriers. A host
reference and arithmetic-sensitive negative controls distinguish alternate reduction
and contraction choices. Native fixtures require exact reference bits and untouched
guard outputs before timing eager or graph replay. Tolerance checks reject nonfinites.

Portable strict Release tests exercise real serial producers, cache transfer/miss
behavior, stale graph bindings/memory and delayed upload/cancellation/failure lifetime.
The full optional HIP build and Wave32/Wave64 labs compile offline using ROCm 7.2.4.
Seven focused lifetime/reduction/state tests also pass with AddressSanitizer and
UndefinedBehaviorSanitizer, including leak detection. GPU fixtures,
model fidelity, complete costs, graph executable memory and actual throughput remain
unmeasured; no experimental path is promoted by offline compilation.
