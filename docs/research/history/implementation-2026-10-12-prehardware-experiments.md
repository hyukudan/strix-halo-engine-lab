# Pre-hardware experiment implementation

## Integer projection

The opt-in [integer projection lab](../../../lab/integer_projection/README.md) now
implements Wave32 IU8/IU4 kernels, group-local activation preparation and ordered
scale application. Q4_0 can use A8/IU8 and A4/IU4; IQ4_XS uses its nonlinear
codebook decoded to signed I8, never raw nibble indices as IU4 values. Lossless
staging preserves original weight codes/scales. Activation quantization defines
a different numerical plan from the floating control.

The portable reference checks actual K640/K2560 storage geometry, integer products,
group offsets, row independence, exact padding, overflow, nonfinite values, tampered
staging and the floating environment. Native fixtures compare group integers and
binary64 output bits, including native solo-versus-batched execution. Staging and
reference operation storage is bounded by an explicit 256 MiB laboratory ceiling.
Full-model integration and precision selection remain separate development work.

## Trusted driver adapter

`DirectTransportDriverBackend` is a concrete consumer of the existing native-session
interface over `DirectTransportDriverIo`. The latter is a trusted normalized boundary,
not a kernel UAPI. The default provider reports unavailable and performs no ioctls.
No supported GPU export/import UAPI was supplied for the target driver, so its
syscall/GPU provider still needs implementation against the selected real ABI.

The adapter owns an import token before activation, retains it across partial errors,
and releases only after exact device/import/lease/pool quiescence proof. It bounds
payload, imported bytes and operation metadata; fences device/ABI/peer/epoch drift;
binds events to full descriptors and allocations; and retains per-direction sequence
high-water marks across same-epoch reopening. Producer release, RX acquire and
completed consumer release require distinct operation-bound proofs. RX completion
alone cannot grant credit or repost. One serialized progress owner is required.

Portable tests use an explicitly synthetic provider to exercise this real adapter.
They do not implement kernel DMA. A provider must atomically fence operations at
the actual driver boundary and pin every allocation referenced by an import token.
On unproved teardown the bounded token/metadata remain quarantined until process
termination; a failed runtime must stop rather than create replacement adapters.

Backend pool release now returns a typed result. Session close, failed-open cleanup
and destruction release the host registry lease only when the backend reports this
exact pool released. Otherwise the registry lease stays held; explicit close throws
and fences new work while permitting a later close to obtain real quiescence proof.
Destruction retains the registry entry rather than allowing an ambiguous DMA owner
to be replaced. Backend implementations must adopt this source-level API change;
a void callback return cannot be interpreted as release proof.

## Character-device control baseline

`DirectStreamControlByteStream` implements the existing byte-stream interface over
trusted synchronous FD I/O. Its concrete POSIX owner opens a caller-selected canonical
character device nonblocking and close-on-exec, validates the opened inode/device
against the explicit observation, and fences descriptor open generations. Other
platforms reject the concrete opener while retaining portable injectable framing.

The stream uses the existing TCP-compatible U32 length prefix and composes with the
v2 framed control wire. One absolute deadline and a bounded I/O-call budget cover
prefix and payload together, including partial progress and interrupted/would-block
operations. Identity or time drift after readiness or I/O rejects before more work;
partial, malformed, closed or expired incoming frames permanently fail the stream.
Tests include real socketpair interoperability and synthetic fault injection. The
character-device fixture is not a two-host link proof.

This is a CPU-copy control channel, not a tensor-payload backend, device discovery
authorization or GPU DMA implementation. Both peers must explicitly use compatible
control framing. Port/peer reconciliation, actual driver stream behavior and useful
two-host workload measurements remain target gates.

## Dynamic graph inputs

`graph_inputs` defines a bounded little-endian ABI for U32 token IDs and aligned
U64 absolute positions. Packing validates all rows, vocabulary/context bounds and
consecutive positions before creating a packet. All padding is initialized. Binding
requires an exact allocation extent, execution identity and nonzero generation.

The separate experimental `launch_hip_graph_embedding` consumes that ABI for one
row on an explicitly supplied stream. It supports the existing F32, F16, BF16, Q8_0
and IQ4_XS embedding decoders and FP32/BF16 output. Its portable preflight validates
encoding geometry, resident weights, byte extents, alignment, allocation generations
and disjoint weight/input/output/status ranges. A device U32 status is overwritten on
each execution: token and position errors are flags 1 and 2. Invalid inputs produce
zero output without dereferencing an invalid embedding row; callers must treat a
nonzero status as failure rather than accept that output.

The native fixture captures the real F32 gather into a HIP graph and changes tokens
and positions in the same input allocation on successive replays. It checks exact
independent embedding rows, output guards, invalid device inputs and cleared status
after a later valid replay. The fixture uses its own nonblocking stream and retains
its bounded device resources on teardown without stream-quiescence proof.

This API is not wired into the existing eager serial dispatcher and does not make
that dispatcher's scalar token arguments dynamic. PLE host lookup, position-dependent
attention and recurrence, changing launch geometry and whole-model state commit still
need their own device-input/capture implementations. Existing whole-model graph-safe
flags remain unchanged. Allocation owners must retain every captured resource through
completion; portable binding checks do not reserve or own physical GPU memory.

## Evidence boundary

Portable tests and offline HIP compilation can establish contract correctness and
compiler compatibility. They cannot establish native fragment correctness, graph
replay quality, driver DMA visibility, model fidelity or performance. Native fixtures
must run on the actual configured device before qualification. Capture/executable
memory remains unmeasured, so no experimental graph is admitted as a measured
whole-model production plan.
