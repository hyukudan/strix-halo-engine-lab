# Two-host experimental GPU transport commissioning

`strix-direct-gpu-commission` is a real Linux/HIP executable for the project-owned
experimental DMA-BUF stream extension. It invokes native HIP allocation, DMA-BUF
export, GPU kernels/copies/events and the documented extension ioctls. It does not
substitute a fixture for an absent driver. Stock `ENOTTY`, absent loaded-module
GNU build ID, unsupported export/import geometry or mismatched identities fail.
No GPU, controller DMA, driver installation or network run was performed while
developing this executable. Offline compilation and injected tests do not qualify
the physical path.

The separate `LinuxDirectGpuStreamCommissioning` facade cannot be passed as a
production backend, IO provider, staging provider or qualification declaration.
Its privately owned provider always reports unavailable native readiness and no
production current binding. A private serialized dispatch permits only the
experimental lifecycle after exact same-FD binding. No positive qualification
object is created. Every config/receipt records `production_qualified=false`.

## Operator prerequisites

- Two compatible Linux/HIP hosts with independently verified GPU UUID, architecture
  `gfx1151`, runtime/driver versions, kernel release and actual loaded
  `thunderbolt_stream` GNU build ID. Build/source receipts remain private.
- A deliberately installed and booted kernel/module containing the extension in
  [kernel/README.md](kernel/README.md). Compilation of its object is insufficient.
  Use the supported software connection manager, interrupt mode and ConfigFS
  `gpu_direct` selection before opening; this program does not configure them.
- Exactly reconciled local/peer endpoint UUIDs, route/hop/ring configuration,
  physical-link label and a fresh increasing peer-agreed epoch. Ring windows
  must match `--ring-frames` on each endpoint. The whole exported allocations
  must meet the kernel's mapped-SG, reservation-object and DMA-mask restrictions.
- A working authenticated SSH tunnel established by the operator. The harness
  TCP adapter binds/connects **only `127.0.0.1`**. It has no LAN listener, TLS
  implementation or security claim. Do not expose its loopback port through an
  unauthenticated forwarding service.

No prerequisite is inferred from allocation flags, successful queries, local
event completion or receipt agreement. Dedicated uncached HIP allocations only
record the requested policy. Cache/system-fence coherence, peer/controller
behavior and reset/disconnect recovery still need independent physical evidence.

## Check configuration without hardware work

Preparation can collect metadata before any pool allocation or TCP setup:

```text
strix-direct-gpu-observe-driver --configured-gpu-direct \
  --observe-driver /absolute/configured/device \
  --kernel-release EXPECTED_RELEASE --driver-build-id EXPECTED_GNU_BUILD_ID
strix-direct-gpu-commission --observe-gpu INDEX
```

The driver observer requires explicit acknowledgment that a privileged,
serialized ConfigFS owner selected `gpu_direct` and excludes mode changes from
preflight through close. It verifies the actual loaded GNU build ID and kernel
release against expected values **before opening** the route, then issues only
the existing QUERY and checks exact FD identity before/after. Its raw ABI,
per-open instance, endpoint/route/hop identities, ring limits and state are
observations; QUERY still reports unavailable native readiness. A later
commissioning process gets a different per-open instance and binds its own FD.
Unknown/replaced descriptors are retained rather than closed as another owner.

This mode acknowledgment is not qualification. Opening an arbitrary stock or
copied-mode stream may itself allocate/start rings and enable paths before QUERY
returns `ENOTTY`. Do not open an unknown stream to infer compatibility. The
operator must preserve exact configured-route association and exclude ConfigFS
changes across the preflight/open gap; this program does not guess a ConfigFS
path or change driver configuration. The commissioning runner also checks
expected module/kernel identity before opening stream routes.

GPU observation selects the requested HIP device and reads its actual UUID,
architecture/features and runtime/driver versions. It rejects architecture
overrides and performs no explicit payload allocation, copy, export, GPU kernel,
controller command or socket operation. The HIP runtime may initialize its own
bookkeeping while answering metadata queries. Neither observation mode was run
against a target GPU/stream during development; tests use injection and the
driver syscall negative uses `/dev/null` with explicitly synthetic preflight
identity. No stock stream is opened by that test. Metadata is not native GPU
execution, import/coherence compatibility or production qualification.

The Linux/HIP target has `--help` and `--check-config` paths that invoke no HIP,
driver or socket operation. `--check-config` validates bounds and emits a JSON
config with exact expected identities, the descriptor fingerprint and the full
expected payload SHA-256. These are operator expectations, not observations.

Supply all required identity arguments shown by `--help`; append `--check-config`
to the same arguments intended for the real run. Unknown/duplicate options,
missing opt-in, unsupported roles, zero identity, noncanonical UUID/build ID,
invalid ring capacity and excessive budgets are rejected before native work.
Keep configurations and receipts outside Git. The program emits JSON on stdout
and does not create a qualification/admission record.

## One-way run

The default profile performs one transfer, sequence 1, with opaque bytes. Both
processes use the same campaign/run IDs, seed, byte length and descriptor IDs.
The SHA-256-derived shape fingerprint binds those pattern/run fields; existing
v2 kind4 control messages also check every descriptor field, endpoint UUID,
physical-link label and epoch before tensor submission. Each process checks its
actual newly opened device instance and actual module/GPU observations locally.

Start the sender with `--role sender --control-mode listen` and its loopback port.
On the receiver, arrange an authenticated SSH local forward to that sender port,
then run with `--role receiver --control-mode connect` and the forwarded local
port. `--local-endpoint`/`--peer-endpoint` are reversed between roles; GPU UUID,
runtime/driver and kernel/module expectations are local to each host. Both select
their explicit local `--dma-device`, `--physical-link` and `--epoch`. Match the
ConfigFS ring geometry; the executable neither edits nor recreates the endpoint.

The sender fills a deterministic pattern on the GPU, packs it into the imported
TX pool and publishes the terminal record after the owned stream's system-fence
event. The receiver posts actual controller RX frames before peer-ready control.
It validates every completed frame, acquires the entire payload into a separate
owned GPU allocation, unpacks to another retained GPU buffer, and waits for the
consumer event. It then copies **all bytes** to a retained pinned readback buffer,
waits for its system-fence event and compares each byte independently of SHA-256.
Only a complete matching payload permits consumed credit and physical RX repost.
The sender waits for controller completion, transport ACK and consumer credit
before retirement. ACK alone does not report successful payload verification.

## Reuse and bounded duplex sessions

`--iterations N` selects 2–128 complete transfers on the same owned FD, exported
pool, import generation and ARM. The epoch is fresh for the whole invocation;
physical sequences and operation handles increase through the session. Run IDs
are `RUN.s0`, `RUN.s1`, and so on. `--payload-sizes 1,4097,5120` supplies exactly
one size per transfer, with the first equal to `--payload-bytes`. Without this
option each transfer uses the initial size. Total scheduled payload is bounded
to 128 MiB; native source/acquire/unpack/readback allocations use the largest
size, and actual allocation extents remain independently admitted.

The default session is sequential. `--alternate-directions` reverses roles for
every successive transfer; both peers keep reciprocal local roles. Every TX
waits for controller completion, ACK and consumer credit, and every RX completes
acquire/unpack/full verification/credit/repost. Retirement finishes before the
next producer or RX geometry replacement. The driver's next RX PREPARE safely
drains/rearms old standby frames; unsolicited standby traffic remains an error.

`--duplex-pairs` instead requires an even count and permits one TX and one RX
operation at a time. It is mutually exclusive with `--alternate-directions`.
Adjacent transfers have opposite roles; both peers publish their TX producer
record, post RX, exchange exact ready messages and only then submit TX. Owned
controller completions are demultiplexed by the two exact handles/descriptors.
RX acquire, consumer verification and repost progress before a blocking TX
ACK/credit read, including TX-first completion. Both operations retire before
the next pair. No third operation, unbounded queue, concurrent caller, larger
pipeline, automatic recovery or production full-duplex capability is enabled.
The two-operation capacity and successful paired completion do not prove
simultaneous physical DMA. The current receipt has no matched controller DMA
start/end intervals. The independent checker explicitly reports physical
overlap as `UNKNOWN`; host submission order or the order of polling completions
cannot replace target timing/traces or qualify overlap.

Session config schema `strix-direct-gpu-commissioning-session-config-v2` contains
`campaign_id`, `session_id`, `schedule` (`sequential`, `alternating` or
`duplex_pairs`), `concurrency` (1 or 2 respectively), `executed:false`,
`production_qualified:false` and ordered `transfers`. Transfer configs retain
the exact config-v1 fields. Session receipts use
`strix-direct-gpu-commissioning-session-v2` with the same session identity,
`requested_transfers`, `completed_count`, stable `pool_binding_identity` and
`import_generation`, final `cleanup`, `error`, `production_qualified:false`, and
ordered `transfers`. Each emitted transfer has the complete legacy receipt
fields but explicit schema `strix-direct-gpu-commissioning-transfer-v2` and
`cleanup:"import_held"`; only the outer cleanup describes final release. A
failed step/pair aborts the suffix, whose unattempted records are absent. Missing
coverage or any retained resource cannot pass paired review. Legacy v1 schemas
remain unchanged and reject session-v2 records.

The one-way v1 profile reverses roles in separate processes using a new increasing
epoch/run/request. Session-v2 changes direction only through its declared schedule.
Do not reuse an exposed epoch after a failed attempt. Production C1 capacity is
one global bulk operation; the isolated duplex profile is one TX+RX pair.
Payload frames are contiguous without wrap, with a dedicated
terminal frame; the final payload frame may be partial. Maximum payload is
`(ring_frames - 1) * 4096`, at most 16 MiB minus one frame. Imported extents are
bounded to 128 MiB; three complete native execution allocations are admitted by
the GPU helper within 64 MiB, plus one requested payload-size pinned readback
buffer and the helper's two 32-byte metadata slots. Allocation rounding must fit
those admission limits. Fixed event and stream counts are retained through work.

## Results, deadlines and cleanup

`strix-direct-gpu-commissioning-v1` receipts contain actual observed endpoint,
GPU/runtime/module identity, exact descriptor/pool/import generations, expected
and observed payload hashes, separate controller/acquire/consumer/repost/ACK/
credit flags and released/retained cleanup. Sender-only receipts never attest
receiver verification. `executed` means payload execution was attempted,
including an enqueue attempt that failed; success requires all role-specific
stages, complete verification on the receiver, matched peer receipts and released
cleanup. Initialization errors without complete observed identity use a separate
error schema and are not valid run receipts.

TCP connect/listen and each frame have wall-clock deadlines and syscall budgets.
Controller progress, GPU event polling and full readback are bounded too. The
GPU helper waits cooperatively using an independent monotonic event deadline
and pause, checking before and after queries and after pauses. Native CLI
`--gpu-timeout-ms` (default 10000), `--gpu-maximum-polls` (default 100001) and
`--gpu-pause-ms` (default 1) apply to producer, acquire, consumer and full readback
events. They are separate from controller `--maximum-polls`; both GPU time and
call-count bounds must permit completion. Clock regression, unavailable clocks,
late success or a pause/callback error fail closed with queued owners retained.
These are configured limits, not observed performance or cache-coherence proof.
The total runner deadline is checked around trusted callbacks and ioctls; it cannot
preempt a stuck kernel ioctl or arbitrary callback. An operator watchdog may
bound process supervision, but killing a process is not a completion proof.

Cancel/error/timeout cannot authorize reuse. STOP proves only controller drain;
actual GPU stream quiescence is independently required before RELEASE and owned
allocation cleanup. Pending/unknown resources remain retained, potentially for
process lifetime. Failed frees are not blindly retried. A verification mismatch
or D2H/event error publishes no consumed credit and issues no RX_REPOST. The
program does not recover/rearm a failed endpoint or select production fallback.
Native qualification, performance characterization, repeated workload fidelity
and reset/recovery campaigns remain separate work.

## Further reading

See the [lab index](../README.md) for prerequisites and related experiments,
the [qualification ledger](../../docs/qualification-ledger.md) for current gates,
and [independent oracles](../../tools/oracles/README.md) for semantic checks.
