# Hierarchical QSA selection diagnostic

The independent testing target `strix_qsa_selection_diagnostic` implements the
fixed top-512, compression-four block selection contract for up to 32768 blocks
and 131072 visible tokens. It is not linked into the decoder and does not add a
runtime option or default selection.

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
Compilation, execution,
ISA/resource checks and timing are deferred. Decoder workspace/status integration,
complete attention equality and long-context model measurements remain open.
