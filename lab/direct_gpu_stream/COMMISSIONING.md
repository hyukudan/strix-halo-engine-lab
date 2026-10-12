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

The initial profile performs one transfer, sequence 1, with opaque bytes. Both
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

Swap roles only in separate processes using a new increasing epoch/run/request.
Do not reuse an exposed epoch after a failed attempt. Native C1 capacity is one
global bulk operation, contiguous payload frames without wrap, with a dedicated
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
total runner deadline is checked around trusted callbacks and ioctls; it cannot
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
