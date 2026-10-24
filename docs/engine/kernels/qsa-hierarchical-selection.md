# Hierarchical QSA selection diagnostic

The shared selection backend implements fixed top-512, compression-four block
selection for up to 32768 blocks and 131072 visible tokens. The diagnostic target
and the decoder's explicit `--qsa-hierarchical-selection` path link the same
implementation. The runtime flag defaults off and does not alter automatic
profile selection. Its source integration remains unqualified.

For sparse selection, a validation kernel first rejects nonfinite scores through
caller-owned sticky status. Tiles sort 1024 candidates cooperatively and retain
512; pairwise merges retain the best 512 until one list remains. Ordering uses
score descending, then original block ID ascending, including signed-zero ties.
Padding IDs are unique across source tiles, so merge ranks and writes remain
distinct. The final kernel sorts retained block IDs chronologically, expands each
to four tokens, and appends the original partial tail. Input scores are read-only.

When at most 512 complete blocks are visible, the direct route emits every token
without reading scores. This includes 512 blocks plus a partial tail. The empty
case submits no work. Status is never cleared by the wrapper; nonzero status
suppresses selection and scratch writes. The caller must complete preceding
producers and retain all views and the explicit stream until quiescence.

Each ping/pong workspace has at most 16384 eight-byte items: 256 KiB total of
requested scratch across both buffers. Local sorting requests 8 KiB LDS; the
chronological phase requests 4 KiB. These are source layout sizes, not observed
resources or physical memory measurements.

The five source kernel symbols are `validate_scores`, `direct_tokens`,
`tile_top512`, `merge_top512` and `chronological_expand`. An empty visible range
submits zero kernels; a nonempty direct range submits one. For sparse selection,
let `L = ceil(blocks / 1024)`. The wrapper submits validation, local selection and
chronological expansion, plus one merge kernel per level: `3 + ceil(log2(L))`
launches, from three to eight in this scope. Captured-node fixtures check these
shape-dependent counts. Compiled symbol inventory, ISA and resource usage remain
unobserved until the deferred compiler checks.

Source fixtures compare every ordered token against an independent CPU oracle
for boundaries, tails, ties and long inputs, including eager and captured reuse,
finite extrema and subnormals, status/nonfinite rejection, read-only scores and
guards. Boundary cases include 2048 complete blocks without padding and 32767
blocks with a three-token tail. Sticky-status plus nonfinite input must produce
the exact ORed status while leaving output and both workspaces unchanged;
all-nonfinite input and a nonfinite first score exercise the same write barrier.
Admission fixtures also reject legacy/per-thread streams and malformed empty,
null and misaligned workspace views before capture submits any node.

These fixture additions have not been compiled or executed in this stage.
Compilation, execution, ISA/resource checks and timing are deferred. Production
source integration and complete-attention fixture source are present; their
execution and long-context model qualification remain open.

The decoder option requires eager R1/C1, the owned deferred-completion stream,
compression four, budget 2048 and a session context of at most 131072. Whole-graph,
integer-head and prefix-roundtrip modes reject the option. Prepared submission,
GPU token selection, shared overlap and projection organization options remain
independent. No default flag is enabled or disabled by this option.

With the option off, workspace offsets, bytes, kernel IDs and implementation
identities remain the incumbent values. With it on, the arena appends aligned
ping/pong regions and a separate four-byte local status. Each region is bounded
at 128 KiB; the local status is separate from the eight-byte deferred first-error
record. Positions that use all visible tokens keep the incumbent direct path.
For sparse selection the caller resets only its local status, computes the
unchanged block scores, and invokes the shared backend. A final publisher records
nonfinite rejection in the distinct `qsa_selection` error domain while preserving
any earlier first error. On rejection it writes safe zero indices before later
attention gathers. The token completion fence rejects the token before session
publication; this does not promise rollback of partially written device state.
No helper clears or aliases the caller's existing first-error record.

Selected variant IDs contain `.hierarchical-top512-v1` before projection suffixes;
compiler and implementation identities and opt-in workspace identity bind the
organization. Arithmetic in score producers, projections, softmax and value
accumulation stays unchanged. Sparse selection submissions are counted by layer
only after the complete helper and status-publisher enqueues succeed. Post-loop
receipts emit the total and 48 layer counters with `not_assessed` qualification.
These counters prove host submission, not device completion or measured speed.

The added production fixture source compares the complete ordered indices,
attention output and persistent state at direct/sparse boundaries, with tied and
varied histories, two deterministic repetitions and read-only weights. Separate
nonfinite cases check the QSA error domain, earlier-error preservation and safe
indices; metadata and actual-capture negatives require no submitted nodes or
owner writes. These new fixtures have not been executed.
