# Execution optimization gates

Status: Draft. These requirements extend the existing optimization strategy.
They authorize investigation, not default activation or native qualification.

## Workload and topology applicability

OPT-001: Every optimization records the exact backend/architecture, weight codec,
projection shape, context, phase, per-expert row occupancy, concurrency, numerical
profile and memory-residency mode. Discrete GPU PCIe assumptions must not be applied
automatically to a unified-memory GPU or to a network boundary.

OPT-002: Compare one device, local multiple devices, independent requests and a
whole-model companion before selecting segmented execution. A stronger last stage
can change head/draft/verification cost; device order is part of the measured plan.
Capacity fit alone does not establish the fastest plan.

## Launches and graph lifetime

OPT-003: Investigate staging inputs once, batching compatible small projections and
reusing expert weight loads across verification rows. Include gather/scatter, routing,
layout, preparation, scaling and synchronization in complete cost. Total request rows
are not a substitute for the occupancy of each expert.

OPT-004: A graph without host residency decisions is eligible only while every tensor
it assumes resident remains correctly bound. Residency changes, pool generations,
addresses, canonical chunk plans and state bindings invalidate or select a different
graph. Persistent host decisions cannot be removed merely because the previous
window happened to touch cached experts.

OPT-005: A captured graph belongs to its exact session resource bindings or to an
explicitly qualified relocatable contract. A new stream may not reuse stale capture
addresses. Capture/preparation peaks and cold first-use costs are included in
admission and measurements. Failure must stop execution before stale graph replay.

## Cache, prefill and fairness

OPT-006: Prompt workspaces may borrow cache capacity only under an explicit ownership
transition. Cache restoration is measured, including multi-device restoration and
the subsequent decode. Compare retained per-device workspaces with temporary loans;
neither is universally preferred. Shared physical memory is charged once.

OPT-007: Scheduling can yield between canonical prompt chunks without changing their
arithmetic decomposition. Warm-up must yield to foreground work without releasing
buffers still in use. Fairness tests include long-prefill arrivals, active decoders,
cancellation and bounded resource retention.

OPT-008: Prefix reuse binds checkpoint, tokenizer/template, numerical/prefill plan,
state layout and history. Shared state pools expose exact capacity, reservation,
eviction and isolation rules. Persistent state adds crash-safe record validation;
neither an opaque file hash nor successful deserialization proves replay equivalence.

## Numerical and recovery boundaries

OPT-009: A different reduction grouping or peer-assisted prefill can change state
even when final text appears coherent. Such modes need their own numerical identity,
quality and replay gates. Exact speculative verification cannot silently change its
authoritative target by selecting a faster kernel or cache policy.

OPT-010: Nonfinite logits/state stop authoritative generation and produce an explicit
failure. Replaying from a clean prefix is a separately bounded recovery policy with
fresh state and recorded outcome; it cannot silently mask a persistent correctness
defect or bypass admission. Never commit a token sampled from nonfinite logits.

OPT-011: Adaptive overlap, cache policy and speculation depth are selected from
qualified complete-cycle observations. Calibration separates sampled/greedy modes,
warm/cold phases, routing distributions and concurrent/solo workloads. A single
global timing average may not contaminate an incompatible workload's choice.

OPT-012: Dropping a selected expert because its weights are absent, steering routing
toward resident experts, or changing routing with window-wide cache state defines a
different numerical target. Record the approximation, routing rule and quality gates
explicitly. Exact execution must retain every contribution required by the canonical
router regardless of cache residency, batch composition or expert rank.

## Acceptance and implementation order

OPT-013: Retained state records its arithmetic lineage through the resume position:
canonical produced-row intervals, complete prompt chunk cuts, reduction plan and
state-update identity. A valid current endpoint alone does not establish that earlier
chunks matched. A lineage mismatch is a cache miss. Serial-only state must not be
relabeled as chunked-prefill state when that path is introduced.

OPT-014: Values that vary per step, including token IDs and absolute positions,
are device data or part of an explicitly qualified graph-update mechanism. Capturing
an eager call with scalar arguments does not make those arguments dynamic. Graph
instance count, workspace and executable memory are charged across all sessions,
shapes and eligibility classes.

OPT-015: An optional tensor-parallel numerical profile may define a fixed reduction
tree over canonical slices. Rank reassignment preserves exactness only if the slice
partials and tree arithmetic remain identical. Define zero-padding/sign rules and
account for extra partials, scratch and communication. This investigation is separate
from whole-block pipeline execution and does not establish mixed-vendor equality.

OPT-016: Nonfinite detection taints the affected state lineage and derived cache
entries. Restore can use only an independently valid earlier state. Row-local failure
containment requires proven independence of quantization, routing, scratch and state;
otherwise poison the shared execution scope. Inject failures to check that no affected
tokens or snapshots are committed.

OPT-017: Online timing adaptation excludes cold capture/prefill transition samples
or records them as separate costs. Keep runtime estimates separate from qualified
calibration evidence and bind mode/context/concurrency. Speculation selection retains
the plain-depth-zero path and periodically rechecks its cost under bounded policy.
Performance adaptation never changes the approved authoritative arithmetic profile.

Start with portable stale-binding, ownership and applicability regressions. Next
prove native scalar/serial correspondence and retain disassembly. Finally compare
alternating complete-workload trials with the same model/profile, memory caps and
approved policy, counting committed tokens, latency tails, preparation peaks and
quality. Record regressions as well as improvements.

Prioritize projection/verification launch reduction and resource-bound graph replay,
then preparation-inclusive expert grouping and prefill workspace policy. Optional
model mutation or online training requires a separate product/model identity and is
outside this inference optimization draft.
