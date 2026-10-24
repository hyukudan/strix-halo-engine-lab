# Complete-model experimental decode recording

`strix-hip-decode-token` can drive the existing complete C1 BF16 model either
through eager dispatch or the explicit device-input HIP graph. A separately
selected integer LM head replaces only the final projection in eager mode.
Neither mode grants model, device or performance qualification. `--help` is
host-only and requires no model, GPU, stream driver or network connection.

## Explicit routes and memory ceilings

The default remains eager floating decode with the existing residency policy.
Graph selection requires `--execution-mode graph --residency copy` and all four
positive memory ceilings: `--graph-workspace-bytes`, `--graph-executable-bytes`,
`--graph-capture-peak-bytes` and `--graph-budget-bytes`. Their complete declared
peak must fit the budget. Actual native backing capacities are checked by the
graph implementation; internal graph allocations still require hardware profiling.
Graph mode explicitly makes the entire model resident, including the potentially
large PLE table. It cannot use planned host PLE lookup, registered host views or
the blocking integer head. Failure to fit is an error, never an automatic change
to the numerical or execution plan.

The eager alternate head is selected with `--integer-lm-head a8-iu8`, `a4-iu8`
or `a4-iu4`, `--residency copy` and both `--integer-internal-bytes` and
`--integer-retained-bytes`. Storage is derived from the actual authoritative
rank-two local output tensor. Original Q4_0 and IQ4_XS blocks must exactly match
K/N and their byte extent; no padded K, requantization or code reinterpretation
is implicit. IQ4_XS uses its nonlinear INT8 lookup and cannot select raw IU4.
Only the output head quantizes activations; describing the whole model as W4A4
or W4A8 would be incorrect. The remaining model uses its existing BF16 route.

Compilation binds the alternate head before building the complete execution
plan. The actual launcher catalog then consumes the bound native runtime bundle.
Before each token the diagnostic scheduler advances the writable arena's access
epoch; original checkpoint weights remain read-only. The integer implementation
includes widening, repacking, activation preparation, group scaling, tiled
projection, FP32 conversion and the status-gated whole-logits commit.

## Ownership and recording

The graph and integer allocation pins retain strong owners for the mapped model,
residency allocation, runtime allocator, scratch/session allocations and copied
resource metadata. A retained failed native object cannot keep references to
caller stack metadata. Graph tensor access is compiled from its actual topology,
which omits eager host lookup steps. Eager and graph plans retain independent
execution, kernel, layout and runtime compatibility identities.

`--record-raw` records actual full FP32 logits, executed input history, emitted
greedy tokens and all seven state components using the same recorder for each
route. `execution_mode`, `evidence_kind`, actual BF16 activation storage and the
exact numerical identity are explicit and mutually checked. A floating arithmetic
descriptor does not erase the different graph kernel plan; an integer descriptor
applies only to the alternate LM head. Missing or conflicting producer fields
are rejected by the independent importer rather than inferred from CLI labels.

Recording and full-state copies are correctness diagnostics. No record grants
performance evidence or production qualification. Source/compiler provenance,
tokenization and decoded text remain independent bindings.

Native setup copies and fresh-memory initialization also bind the original
device index, UUID, architecture and HIP runtime/driver before submitting work.
If a copy or initialization fails, normal allocation/mapping cleanup proceeds
only after synchronization and another matching original binding. An unproved
outcome terminates the diagnostic process without ordinary owner destruction.
This protects borrowed mapped sources and state destinations during setup; it
does not promise automatic device recovery or bound a blocked kernel-driver call.

## Snapshot correction and validation boundary

KV and sparse-index allocations use context-capacity strides between layers.
A component-wide `prefix_tokens * aggregate_bytes_per_token` copy misses later
layers for a partial prefix. Snapshots now preserve each complete valid component,
including inactive capacity. This deliberately increases snapshot admission cost
and changes the snapshot identity; the full-component format is explicitly v2.
Capture completion and original-device identity are checked before any intervening
mutation, including runs without RAW recording. Continuations must advance the
state before restore and compare complete logits as well as tokens and stop state.
A compact per-layer format must be implemented
as a separate checked format; old partial snapshots are not silently reused.

Portable regressions cover partial-prefix restoration of every component byte,
memory ceilings, original output-block geometry, nonlinear-code restrictions,
unknown modes and incompatible execution combinations. The native graph fixture
also contains an actual same-plan snapshot/restore/replay comparison of complete
state, logits and arithmetic lineage. That fixture must still execute on the
target; compiling it or checking its host catalog does not pass the hardware gate.
