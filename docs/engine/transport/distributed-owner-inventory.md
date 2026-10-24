# Stage-owned storage inventory

`HipDistributedStageResources::owner_inventory(actual_owner)` reports live
selected-weight backing, scratch, device/host state pools and the separate D1
record. `HipDistributedTokenParticipant::owner_inventory(actual_owner)` adds the
participant's actual full or typed local snapshot pools. The latter is limited
to real stage-owned participants; a borrowed legacy allocator cannot acquire
invented ownership through this API. Both require the same shared object and
control block that actually retain the allocations. A same-address no-op pin
is rejected.

The `HipDistributedOwnerInventory` result contains `retained`, an explicit
`snapshot_included` indicator and `unobserved_categories`. It is a declared
storage scope, not a process-total or hardware-fit result. Transport/external
pools, PLE backend/cache internals, stream/event/driver storage and host object
or container overhead remain unobserved. A stage without its attached
participant cannot include that participant's snapshot. No estimate or unknown
owner is assigned a fabricated live pin.

Allocator introspection uses the live allocation record. Device and pinned
host records include the exact requested alignment reservation, aligned view
offset and logical payload; pageable host records include their actual API
payload request. Driver page rounding and heap bookkeeping are not observed.
An allocation lease is minted when allocation occurs and is independent of
virtual addresses. Backing keys combine that process-local issuer/lease with
the real binding generation. Owners save their original descriptors and reject
changed bindings, generations, freed blocks or replacement allocation leases.
Snapshot plan rebinding between inputs preserves the original storage lease.

Views from weight registration reuse the existing actual mapped-backing lease;
registered host/device aliases are counted once. Copied weights and independent
state/scratch/D1/snapshot blocks retain distinct leases. A shared monotonic view
ID issuer lets separately collected inventories be joined without view-ID
collisions; repeated backing keys deduplicate only the same retained allocation.
Addresses remain explicit view evidence and never establish a backing key.

The result pins actual owners. Participant snapshot pins keep the participant,
its stage and allocator alive; stage pool pins keep the stage alive. Inventories
are returned, never stored in their own owners, so this creates no self-cycle.
Callers serialize collection with allocation release and model progress and
keep metadata stable. Collection performs no GPU allocation, payload copy or
completion; source/device validation can query the original HIP binding.

The actual callable model fixture opts in using
`qualify_hip_distributed_stage_inputs(..., include_owner_inventory=true)`.
It calls the real participant inventory before and after the known-input or
streaming transactions, checks live pool bindings/leases/padding, snapshot
storage, repeated-inventory deduplication and same-address false-pin rejection.
The returned receipt exposes the final inventory and retains its real owners.
The default fixture route does not collect it. Source fixtures are present;
compilation, native/model execution and qualification remain pending.

Ordinary runtime memory has a separate retained-owner seam:
`RuntimeExecutionMemoryOwner::allocate(..., shared_allocator, generations)`
retains the actual allocator through pool release. The original overload with
an allocator reference keeps its borrowed lifetime contract. Moving an owner
transfers its allocator pin and original binding/lease snapshots. A live
inventory requires the owner to stay in place while its views are used.

`hip_runtime_memory_owner_inventory(actual_shared_memory_owner)` requires the
actual memory and HIP allocator control blocks. It compares all original
binding fields, generations and allocation leases before exposing the three
scratch/device-state/host-state pools. The allocator's optional lease hook
defaults to opaque; that allows retained lifetime without claiming inventory
evidence for custom allocators. Known HIP allocations report real reservation
padding. The returned views retain the memory owner and its allocator without
storing that result in either owner. The CLI can join these views with model
views through the same backing-key accounting; this is independent of D1 and
PLE owners.

Neither scope aggregates serving caches, separate ordinary D1 allocations,
all PLE/custom backends or external transport into a global inventory. Those
consumers need their own actual allocator/owner seams. Declared storage must
not be presented as complete process memory, RSS, PFNs or coherent DMA
capacity. Optional supplied memory observations stay separate from storage
reservations. All new fixtures remain source-only pending final validation.
