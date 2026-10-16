# Cooperative controller recovery and safe domain retirement

This is an additive source change over lifecycle patch 0008. It adds patch 0009
and updates the multi-slot engine delivery. It does not install drivers, reset
hardware, change the IOMMU, run GitHub Actions or declare production readiness.

## Domain retirement without an open-FD reference cycle

GPU contexts now register in a module-owned registry. After the connection
manager stops and its queued removal work drains, `tb_domain_remove()` calls a
lifecycle notifier **outside the domain mutex, before CM deinitialization**.
The GPU consumer snapshots one referenced stream device and immutable registry
sequence at a time; it never holds the registry mutex while waiting for that
stream's mutex. Under that mutex it revalidates that the context is still the
original registered owner and retires it only after proving ring stop, callback
completion and absence of its original DMA paths.

An ordinary open FD no longer has to keep the entire NHI domain alive after
those proofs succeed. Its context/rings/imports and extra quarantine references
are released; the old FD becomes error-only. Its users count is NOT set to zero,
so a replacement context cannot be opened while that original FD still exists.
Exported fences retain their independent lifetime. Poll uses an RCU read-side
section and context storage is released with `kfree_rcu`, preventing a concurrent
poll from dereferencing freed context storage.

After domain shutdown is gated, the cleanup path does not call the path-mutating
CM callback through possibly retired router objects. It consults the retained
CM inventory instead. An empty path inventory is not, by itself, a DMA-stop
proof: ring/callback completion is checked separately. A stale peer or reused
HopID does not transfer ownership to a newly attached endpoint.

If stop or path removal cannot be established, the resources remain retained.
In particular, `going_away`, an elapsed timer, a failed MMIO read, process exit,
and missing PCI enumeration are NOT invented evidence of DMA quiescence. This
safety condition can still delay removal for a malfunctioning controller. No
software can safely reclaim DMA-visible memory merely by renaming uncertainty
as completion. The implemented retirement covers software-known quiescent
contexts, including still-open FDs, rather than promising to cure every hardware
failure.

## Exact, bounded local controller groups

`SGU_IOC_RESET_GROUP` lends 1..16 GPU-stream FDs, each with its exact current
`sgu_fence`. A group is rejected before reset for duplicate devices/FDs, an
incorrect operation identity, unsupported modes, active/import-ambiguous
sessions, mixed NHI domains or lock contention. Every GPU session must already
be stopped, with its paths removed. User copies and final `fput()` calls occur
outside the acquired stream/DMA mutexes. A nonblocking device-lock check pins the
original NHI driver binding/devres against unbind; a plain device reference is
not treated as sufficient.

The core independently checks the entire inventory: every non-control allocated
ring must be a unique, caller-owned member, stopped and empty. All DMA tunnels
must be absent. Unknown consumers are a veto, not implicitly consenting clients.
The existing AMD allowlist, supported NHI version, runtime power checks, control
channel restart and post-reset register checks remain. No PHY-training register
is changed. Admission stays blocked after an unproved destructive reset.

Once the destructive sequence starts, every participant is marked failed and
requires `RECOVER` followed by a strictly newer peer-agreed epoch. That remains
true if copying the successful reply to userspace fails. Failed admission does
not secretly stop another user's GPU session. The userspace adapter conservatively
invalidates all participants after any attempted ioctl whose outcome could be
ambiguous, including a thrown post-call identity check.

## USB4NET is a cooperating participant, not an unknown ring

The explicit `SGU_RESET_GROUP_NETWORK` flag permits a temporary administrative
outage of the original controller's USB4NET interfaces. The network module's
registered recovery callback holds RTNL across its entire transaction, first
preflighting every candidate, then saving the previously-up devices. It invokes
normal `dev_close()`/`ndo_stop()` (NAPI/work cancellation, logout, path/ring cleanup)
—not a fabricated ring-pointer reset—before executing the core continuation.

Only previously-up interfaces in that domain are restored with normal
`dev_open()` and normal peer login. Interfaces that were down, or belong to a
different controller, are untouched. A nondestructive core rejection restores
what was safely closed. A failed stop or failed destructive reset leaves the
interfaces down; it never restarts traffic onto untrusted controller state.
All restoration attempts are made after a successful reset, while the first
restoration error is reported. This may drop packets and is not transparent
packet delivery or replay of interrupted application work.

Network recovery and domain retirement use **separate notifier chains**. Sharing
a writer-preferring notifier semaphore between a callback that acquires stream
mutexes and a reset issued under those mutexes would risk an unregister/reset/
retirement deadlock. Network callback registration spans the entire close/reset/
restore continuation, so the module cannot be removed between the phases.

Third-party drivers and copied-mode stream FDs not explicitly represented by the
GPU group remain vetoes. Automatically taking over their ownership would not be
safe generic recovery. The supported cooperating set is GPU-stream groups plus
USB4NET; it is deliberately not a promise to reset arbitrary unknown consumers.

## Engine integration and peer protocol

`LinuxGpuQueue::reset_controller_group()` validates the local set, requires each
member to have no unretired operations, checks every pool's independent GPU
quiescence callback, borrows only owner-retained native FDs and executes the group
ioctl. Multiple members can be coordinated by a serialized engine owner in one
process. It neither guesses descriptors from `/proc` nor sends them over a peer
connection.

The existing window coordinator and native `strix-gpu-queue-visibility` runner
can select a reset at the existing `--recover-every N` boundaries with
`--controller-reset`. It drains the completed window, stops local DMA, verifies
GPU idle, exchanges peer-ready, resets locally, exchanges the reset outcome, and
only after matching success recovers/reimports with an incremented epoch. The
native CLI owns one GPU FD per host; other same-controller GPU sessions must be
coordinated by the multi-member library API rather than silently omitted.

`--include-network --independent-control` explicitly authorizes the network
outage and acknowledges a separate authenticated control path. It MUST NOT use
USB4NET on the controller being reset, even if tunneled through SSH. The code
cannot infer the physical route of an arbitrary caller-supplied control socket;
the acknowledgment is an operator obligation, not an automatic route proof.

Recovery flags are negotiated in the canonical wire frame and native campaign
fingerprint. Old peers reject the nonzero extension field. Wrong flags, stale
identities, failed reset replies or timeouts abort without replaying a window.
There is no automatic reset loop or retry of ambiguous destructive ioctls.

The multi-slot receiver now runs its raw GPU verifier **before** any acquisition
or payload/terminal copy. The consumer destination is checked separately after
acquisition/unpacking. This extends the native C1 pre-copy test to the multi-slot
path; it does not remove the subsequent GPU-to-GPU copies. Events in the native
slot memory helper explicitly request system-scope release. These API choices
are the tested software contract, not proof of real cache behavior.

## Offline validation boundaries

Portable tests execute the full materialized GPU implementation, reset backend,
network callback, queue adapter and peer coordinator with explicitly substituted
kernel/HIP/MMIO boundaries. Two-process peer tests communicate over real local
sockets, but their DMA completions remain injected. Partial source-context tests
verify the core hooks and apply/reverse the complete new patch; the contexts are
clearly labeled and are never installed as full upstream files.

Kernel API probes compile against installed 6.12.107 headers with W=1 and -Werror,
with declared private layout/version stand-ins. They are object-only API probes,
not the pinned 7.3-rc5 module link. No native HIP compilation, full pinned kernel
build, MMIO/DMA execution, GPU instructions or hardware measurements occurred.
`VALIDATE_NATIVE.py` automates the full offline build once the actual pristine
pinned source and ROCm development toolchain are supplied. It never boots,
installs, signs, loads or resets the result, and records success only after
commands actually return successfully.
