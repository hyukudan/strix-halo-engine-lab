# RTX prefill host placement

The prefill service accepts `--host-gib G --host-pin-order layer|deadline|early`.
`early` is the legacy ascending fill without the layers the idle preload already holds: the pinned
layers lead the schedule and later FILE reads overlap their compute.
`layer` is the default and preserves the ascending host-budget allocation.
`deadline` selects the longest affordable suffix of the streamed layer schedule,
starting at the last layer and stopping when the next streamed part will not fit.
It excludes the first device-slot-count streamed layers, which the idle preload
already holds on the device. Fully resident layers consume neither a host budget
nor a streaming slot. A partially resident layer costs only its streamed bytes.
Unused budget is intentional: an earlier host burst would precede a later file
deadline. Explicit whole-layer tiers retain their supplied placement.

The policy targets an ascending single-pass request after idle preload. It keeps
file reads together before the host suffix; the loader has four bounded staging
buffers, rather than an independent queue of future file layers. It does not
reschedule requests, increase read-ahead, or change slot counts. Bounded passes
and repeated layer schedules reuse this placement but have different deadlines;
the policy is not an optimizer for those schedules. Cold or back-to-back requests
can consume an unfinished preload.

The per-expert planner first constructs the same resident expert IDs and virtual
order as `layer` with the same budgets and ranking, then changes only the streamed
source. Tensor bytes, group boundaries, device residency, kernels and arithmetic
remain unchanged. Pinned-allocation failures retain the existing file fallback.
The selected host layers are filled in ascending order at startup; under memory
pressure, inspect the effective placement rather than assuming all planned pins
succeeded.

`GET /v1/describe` includes `describe.host_pin_order` and
`describe.host_layers` (absolute layer IDs after pin-allocation fallback).
Each prefill reply includes the same fields under `prefill`. The existing `tiers`,
`resident_experts`, pinned-memory and pin-failure fields provide the remaining
placement and allocation details.

The CPU-only `strix-ds41-host-pin-plan-tests` target checks selection, preload
exclusion, partial-layer costs, slot clamping and budget boundaries. CPU checks
and compilation do not establish device exactness or workload performance.
Qualification requires matching complete token IDs and logits/state digests
between `layer` and `deadline` with the same model, capacity, residency budget,
ranking and numerical identity, followed by interleaved workload measurements.

## Service defaults and memory admission

For service loads with `--pass-rows 0` (the normal mode) and no explicit
`--tiers`, `--host-gib-auto 32` (opt-in) resolves to a 32 GiB cap,
`--host-commit-floor-gib 20` and `--host-pin-wait-s 90`. Explicit values win.
`--host-gib 0` disables optional expert pins and streams those bytes from FILE.
Bounded loads and explicit tiers do not inherit these service defaults; other
binaries and decode-only library objects do not inherit them either.

`--host-gib-auto G` selects the **implicit, opportunistic** policy (omitting both host flags keeps the file-only tier). The
32 GiB budget is a cap, attempted in whole streamed-layer parts, rather than
a guaranteed allocation. A partially resident layer costs only its remaining
experts; a whole-layer provider allocates complete layers. There are no partial
pins within a selected layer part. Unused budget is possible.

Before optional pins, the service allocates mandatory device workspaces,
attention/batch contexts, expert residents and device streaming slots, pinned
FILE staging, every configured context's maximum-capacity Engram rows, and the
maximum-capacity capture buffer when `--fast-export` is enabled. This also
moves normal-mode Engram allocation from the first request to load time.
Request token uploads, pageable hashes, export/serialization copies, queued
pipeline states and runtime/driver allocations still consume additional room.

For each optional allocation of `n` bytes, admission requires both:

- free physical memory >= `n + floor + 4 GiB` (24 GiB reserve at the default);
- free commit >= `2n + floor` (20 GiB floor at the default).

The implicit policy uses current telemetry and one nonthrowing pin attempt.
Physical shortage, commit shortage, allocation failure or unavailable telemetry
immediately demotes that layer to FILE, reading the same tensor bytes. It never
issues a commit-growth probe, retries or sleeps, even if an explicit floor of
zero or an explicit wait was supplied. Its effective wait is always zero; the
resolved wait option remains 90 seconds unless overridden.

An explicit `--host-gib` retains the explicit policy: with a positive floor it
may issue a Windows commit-growth probe and retry against one shared deadline,
including charge-settlement waits. Once that deadline expires, later layers
are demoted without starting another allocation/probe. Explicit wait zero
permits one attempt per layer. Explicit floor zero keeps the legacy allocation
path, including its provider allocation exception behavior. Unknown telemetry
declines floor-governed optional allocations. Floor and wait must be finite
and nonnegative. The wait bounds retries, not total startup time: model reads
and OS/CUDA calls can take additional time.

Successful pins remain held until service destruction; startup telemetry is
not a reservation for future services and pins are not reclaimed dynamically.
A near-full 32 GiB expert tier plus staging holds roughly **35 GB pinned for
the service lifetime**. Mandatory Engram and fast-export buffers can increase
that total depending on capacity and batch contexts. Set `--host-gib` lower,
or use `--host-gib 0`, when Qwen/MiniMax PRO services share the PC. Qualify
shared-service launch orders and peak concurrent workloads before deployment.
Loading/filling and releasing pins can add startup/shutdown latency.

`GET /health` and `GET /v1/describe` expose a `host_tier` snapshot: policy,
budget, requested/effective expert bytes, effective absolute host-layer IDs,
demotions with `physical`, `commit`, `alloc` or `telemetry` reasons, floor,
resolved/effective wait, current free physical/commit bytes (`"unknown"` when
unavailable), and pressure (`ok`, `low`, `unknown`). Pressure reports whether
the configured reserves remain available; it does not reserve future memory.
`total_service_pinned_bytes` includes expert staging, Engram and the
service-owned fast-export capture buffer. Mutable pinned accounting is
published atomically at completed GPU stages (also on failure), so health can
read the last synchronized snapshot while a request runs. Placement is
immutable after load. Driver `describe` also exposes byte accounting and
demotion reasons; the service's `service.pinned_gb` includes capture memory.

`strix-ds41-host-pin-admission-tests` is a CPU-only decision and fault-injection
gate for admission boundaries, unavailable telemetry, allocation failure,
implicit versus explicit retries, zero-floor fallback and shared deadlines.
It does not qualify native execution, memory coexistence or numerical fidelity.

Status (2026-10-27): opt-in. Measured exact, 1K -3.8..-3.9 s (-24..-30 %), 8K -28 %, 30K -12 %; the service
default stays file-only until batch formation, slots/subsets and coexistence with the Qwen/MiniMax PRO services are
qualified. Deployments set the budget (e.g. the hyperloom preset) with `--host-gib-auto 32`.
