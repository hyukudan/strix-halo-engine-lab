# Conservative MoE fork/join arena

`prepare_moe_fork_join_arena` prepares memory metadata for one specific C1/R1
calendar. After `mlp_hyper_mix` completes, shared may overlap the serial sequence
`moe_route` then `moe_experts`. `moe_combine` waits for completion of both branches.
Combine completes before the following step, and only one layer can be active.
Outside those forks, the original serial calendar remains mandatory.

The planner consumes the existing compiled serial DAG, IO and arena. It verifies
the exact contiguous group, its dependencies, layer labels, IO roles and absence
of branch persistent-state effects. It reconstructs allocation lifetimes from IO
and rejects missing generations, malformed extents, stale lifetime annotations
and aliasing between simultaneously live source allocations. Unsupported input
is rejected; it does not receive a best-effort layout.

Every shared output and shared temporary moves to a dedicated appended region.
The output and temporary ranges are distinct and each is sized for its maximum
across layers. Their lifetime annotations cover branch admission through combine.
The read-only input and every routed/base allocation keep their existing offsets.
The input remains live throughout route and experts; the required join prevents
combine or any subsequent base writer from overwriting it while shared is active.
The original shared ranges become unused holes. Keeping those holes makes the
incremental memory charge simple and avoids changing the serial allocator.

The complete allocation is the retained serial arena plus the aligned maximum
shared output and aligned maximum shared temporary. `budget_bytes` caps this
whole allocation. Insufficient budget and arithmetic overflow fail during host
preparation, before any device submission. The reported additional bytes are a
planned allocation cost, not measured UMA peak or physical memory traffic.

The returned plan exposes const views and a layout identity bound to its complete
source snapshot, offsets, extents and admitted calendar. Admission budgets and
timings do not enter that identity. A physical binding must carry the new layout
identity, a nonzero generation, enough bytes and an aligned, nonoverflowing base.
The original serial binding is rejected. Preparation does not mutate the source
compiled execution or manufacture a compatible launch plan.

`plan.rebind_execution(source)` first compares the entire source with a private
deep snapshot: DAG, dependencies, IO, bindings, flags, bounds, counters and arena.
A different source reusing the same identity is rejected. It returns a new
compiled copy with the expanded arena and layout identity as execution identity.
The kernel plan identity stays unchanged because kernels and their numerical
contract stay unchanged. Whole-plan graph safety is set false; preparation rejects
device-graph-input topologies. `plan.validate_execution(rebound)` checks the
complete expected transformed snapshot when constructing a future dispatcher.

Integration must perform this transformation before allocating resources,
compiling the launch plan and creating sessions/prepared executions. Scratch,
launch plan, session execution identity and receipts then consistently use the new
identity. Recompile the launch plan with the actual launcher registry; copying
its old identity is insufficient. Regular resource and eager-preflight admission
still apply. Replacing just the serial arena while retaining its identity, or
changing context identity around a mismatched launch plan, is outside the contract.

The caller must retain the plan and actual allocation owners through verifiable
device quiescence. Validation of a binding cannot prove ownership, GPU visibility,
absence of aliases with other allocations, or completion. A late event, partial
enqueue, error or cancellation must prevent reuse until all submitted users have
drained. Graph executables and callbacks holding addresses also retain that duty.
No serial first/last-step number is a substitute for the required device joins.

This is a memory-planning component. It adds no decoder option, stream, event,
launcher, completion ticket, graph qualification or changed default. Ticket and
owner retention, independent branch status, deterministic error arbitration and
the HIP dispatcher remain separate work before enabling concurrent execution.

The accompanying test source uses real serial planning components and retained,
aligned host allocations. It covers all six linear extensions of shared
begin/end and routed route/experts, poisons live buffers, checks actual outer
guards, traverses every layer and exercises structural, budget and binding
rejections. These simulations are neither native concurrency evidence nor model
arithmetic validation. The new source has not yet been built or executed.
