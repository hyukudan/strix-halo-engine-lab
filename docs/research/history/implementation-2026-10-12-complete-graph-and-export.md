# Complete C1 graph and experimental GPU transport

## Explicit complete graph mode

`ExecutionBindingRequest::device_graph_inputs` selects a separate C1 decode
topology and requires `require_graph_safe`. Its fixed maximum launch geometry
is distinct from the eager dynamic topology. Every selected GPU variant must
support capture. The complete catalog's matching
`HipLauncherCatalogConfig::enable_device_graph_inputs` enables device-input
embedding, resident-table PLE, complete GDN and complete indexed attention.
Static operators use the same explicitly owned stream. Mode mismatches fail.
An integer LM-head override is blocking and cannot enter this graph.

`HipSerialGraph` captures the actual compiled DAG through the builtin catalog,
instantiates it and replays it on an owned nonblocking HIP stream. It retains
the compiled plan, catalog/configuration and a trusted resource owner. The
owner retains model, scratch and state allocations and excludes concurrent
rebinding/writes. This ownership declaration does not grant model qualification
or resource admission. All model tensors must be resident/repacked local device
allocations; host lookup/prefetch resources are excluded.

Capture binds source content/layout, exact execution/kernel identities,
addresses, sizes and allocation generations. Actual HIP pointer attributes and
allocation extents fence every GPU resource. Writable scratch and session state
cannot overlap one another or model weights; read-only weight sharing remains
permitted. Source owners are rechecked before and after replay. Changing a
binding rejects replay before dispatch. The device-input scope validates its
ABI and disjoint four-byte status; it is thread-local, non-nestable and restored
on unwind.

One 128-byte auxiliary allocation contains token/position, status and bounded
previous/next n-gram history views. Input staging/readback buffers are members
of the retained graph object, never temporary asynchronous sources. Requested
workspace admission includes the entire HIP-reported scratch allocation plus
auxiliary capacity. The 16-byte argument view is separately charged by the
existing graph accounting. Persistent model/state allocations remain part of
their owner admission. Executable/capture-memory bounds are explicit inputs;
HIP/driver internal physical peak remains a native profiling gate, not an
observation supplied by this constructor.

PLE uses the existing hash arithmetic and gathers BF16 rows directly from a
resident table. The graph produces next n-gram history on device. Host position
bookkeeping performs no write while capturing; its dedicated catalog consumer
is validated. The owner uploads committed history and dynamic token/position
before replay, waits for actual stream completion, checks sticky device status
and rechecks bindings before publishing host histories. Final logits are
checked for nonfinite values. The existing serial-session lifecycle publishes
token counters, state generations and arithmetic lineage only after the entire
replay succeeds. No capture advances state or token metadata.

Late arithmetic/submission failures can already have changed device state.
They poison the session and graph; they are not transactional rollback.
Unproved stream quiescence or cleanup retains the entire graph, async sources
and resource owners through process teardown. Failed graphs cannot be reused.
Recovery still requires explicit state restoration and a new compatible graph.

The unchanged eager catalog remains the default. Its optional integer LM-head
override is installed through an explicit runtime configuration/bundle; the
default floating bundle rejects another kernel/profile identity. Eager PLE
on a nonblocking explicit stream fences prior arena consumers before its
synchronous host upload. File-backed PLE remains eager only.

## Concrete local DMA-BUF export

`DirectGpuExportOwner` allocates bounded TX/RX pools through the existing pool
owner and retains their actual allocator. `HipDirectGpuExportIo` checks local
device allocation extents and calls `hipMemGetHandleForAddressRange` with
`hipMemRangeHandleTypeDmaBufFd` and flags zero. The exported objects retain full
native ranges, view offsets, generation/physical-link bindings and distinct
CLOEXEC file-descriptor identities. Partial exports and failed cleanup retain
their pools and descriptors until exact cleanup succeeds or process teardown.

An export is a local prerequisite. It neither imports buffers into a USB
controller nor establishes cache visibility, reset generations, an uncached
mapping, transport events or native readiness. The default driver provider
continues to report unavailable. No opaque/VMM FD import is substituted for a
DMA-BUF contract, and no undocumented ioctl is issued.

`HipDirectGpuPoolAllocator` is a separate, explicitly enabled device-only
allocator. It requests `hipDeviceMallocUncached` through
`hipExtMallocWithFlags`; it does not change ordinary model allocations. Two
fixed allocation slots bound local TX/RX ownership. Whole observed allocation
capacities, including alignment/rounding, count against the admitted byte
limit. Failed allocation/export/free callbacks preserve partial ownership;
unknown native extents report incomplete accounting. A successful native free
is recorded before post-call checks, so a later device mismatch cannot trigger
a second free of a reused address. Its typed release witness prevents an
unproved `free_device` return from authorizing export-owner cleanup.

The uncached allocation request expresses a software policy. Whether the
exporter, IOMMU, host controller and GPU establish the required external-DMA
visibility on a particular pair remains a target qualification gate.

## Experimental kernel and userspace path

The project now supplies an actual opt-in extension to a pinned Linux driver,
a documented UAPI and `LinuxDirectTransportDriverIo`. See the
[kernel preparation and lifecycle guide](../../../lab/direct_gpu_stream/kernel/README.md).
Its enabled driver object compiles against the real source with strict warnings.
The complete kernel tree, build outputs and receipts remain private. The public
patch preserves upstream notices and includes the applicable GPL/exception text.
No module was loaded and no controller transfer was executed.

DMA-BUF attachments map into the actual controller DMA device. Checked SG ranges
supply frame bus addresses; split frames, bounce mappings, overlapping backing
objects and invalid extents are rejected. The ABI fences device/open instance,
local/peer identities, route/hop IDs, epoch, pool and import generation. Payload
starts are aligned; a partial final payload frame is supported. A separate final
frame carries the commit record. The kernel treats that record as bytes.

The userspace provider translates IMPORT/ARM/PREPARE/SUBMIT/EVENT/CANCEL/STOP/
REPOST/RETIRE/RELEASE into typed engine operations. It checks every completion
frame, including terminal offset/length and CRC/protocol errors. Cancellation
does not imply drain. Failed callback, reply, path teardown or release retains
exact owners. Controller stop and GPU quiescence must both precede import release.
Stock drivers reject this experimental ABI; there is no fallback guessed ioctl.
The native provider observes the loaded `thunderbolt_stream` GNU module build ID
and rechecks it at the FD-bound syscall boundary. A missing/malformed note or
changed build rejects the binding. The build ID is an observed identity, not
hardware qualification; source/binary provenance belongs to the independent
target evidence chain.
Because the kernel assigns a fresh instance UUID on open, preflight observes and
binds that identity on the same retained FD. `bind_observed` is allowed only once,
before any import or operation, against the cached and freshly queried exact
OPEN endpoint. Identity adoption leaves readiness false. A subsequent query
still needs the separate current qualification authority; reopening a probe FD
cannot carry the old instance identity into a new provider.

`HipDirectGpuStreamGpu` retains an allocator/import use lease, a complete GPU
execution owner, one explicit stream, fixed pinned terminal metadata and bounded
events. Imported pools cannot be freed or drained while that lease is live.
Actual D2D copies pack TX sources. RX acquire copies the entire payload to a
separate registered GPU destination before terminal readback and completion.
Actual D2D copies then unpack ordered model buffers. System-fence GPU events
authorize producer release and consumer completion; a CPU callback return or
terminal-only read cannot provide either proof. Poll budgets fail while retaining
resources when completion is unknown. This path includes local GPU copies;
only terminal/control metadata crosses host memory.

A dedicated bounded control channel uses explicit wire-v2 kind 4 with its own
contract version. Exact peer readiness precedes TX submission. Received ACK and
consumer credit bind the complete physical descriptor, commit and endpoint/link
identities. Existing wire kinds retain their encoding; the ordinary control
server rejects GPU-control messages on its channel. No downgrade is allowed.

`DirectGpuBoundaryStaging` connects these operations to the existing native
distributed backend. It checks the exact placement/request and physical lease,
packs/unpacks the boundary's ordered execution buffers, retains their owners and
waits for typed GPU completion. Fixed C1 staging requires local DMA completion,
remote transport ACK and remote consumer completion before returning from send.
The first userspace implementation permits one bulk operation globally per
provider, contiguous windows and one physical slice. It does not advertise
simultaneous full duplex, ring wrap or striping. Separate physical providers
remain separate resources.

The native factory supplies no positive target qualification. A separate trusted
qualification authority must bind the exact observed driver/GPU/pool path and
remain current. Device discovery, an uncached request, QUERY results, successful
compilation and synthetic fixtures cannot activate a qualified link. Hardware
import/coherence, complete peer lifecycle and model continuity remain execution
gates.

## Verification

Portable tests exercise whole-target integer tiling, GPU export ownership and
failure cleanup, graph control/PLE ranges and compiler mode isolation. Optional
HIP host-only tests exercise the complete catalog, TLS scope and injected HIP
export calls with real POSIX descriptors; these tests execute no GPU work.
Additional injected tests cover actual HIP copy/event paths and Linux ioctl/control
translation, partial failures, delayed drain, replay rejection and stock ENOTTY.
Shared kernel helpers have independent randomized SG/lifetime tests; actual
LP64/ILP32 compilation pins ABI layout and ioctl sizes.

The complete synthetic native fixture contains actual embedding, PLE, GDN,
indexed attention, hyper mixing, routing, routed/shared experts and LM-head
kernels. It compares changing/EOS/repeated-token replays against eager execution
for full logits, every persistent state byte and metadata, checks host history
against the CPU hash, and includes binding/extent/alias/admission/nonfinite
negative controls. Individual GDN/indexed fixtures also cover dynamic counts and
state errors. These native fixtures are compiled offline, not executed here.
The full fixture's `--validate-fixture` mode checks both actual catalogs and
tensor-access coverage on the host, without model/GPU execution.

Native model quality, state fidelity, actual graph memory, preparation-inclusive
latency, accepted-token cost and physical transport behavior remain execution
gates. No optimization is promoted by these software checks.
