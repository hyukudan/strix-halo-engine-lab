# Step-major tiled prefill

The explicit `--known-prefix-prefill-rows T` route admits 16 through 8192 known
inputs in one transaction. It preserves the existing closed two/four/eight-row
window options and defaults. It is a separate compiled organization with its
own arena, execution identity, producer lineage and output-row mapping.

Execution completes each source step for all T rows before advancing to the
next source step or layer. Eligible shared/GDN/expert projections reuse their
existing exact eight-row entries. All-Q8 GDN retains its two-row entry. Remaining
rows and unsupported steps use the original one-row launcher, without padding.
This is tiled multi-row matrix-vector execution; it is not a general GEMM
implementation, and it does not run a complete decoder forward every eight rows.

GDN recurrence and convolution still update rows in increasing position order.
PLE and QSA retain the original row-local operations and history transitions.
The PLE tail after the key/value projections (key/query gate, gated value,
grouped norm, causal convolution and residual) runs as one rows-form sequence of
three launches per T-row block instead of four launches per row. Each row keeps
its own reductions, gate, norm and tap order; a row's convolution taps read the
previous state or the BF16 of an earlier row's normalized value, which is what the
per-row state shift stores, and the state after the block is written once every
row has read it. Rows whose inputs or outputs are not at one uniform stride use
the per-row launches. `STRIX_PREFILL_PLE_ROWS=0` selects the per-row launches;
`STRIX_PREFILL_PLE_ROWS_CHECK=1` runs both forms from the same inputs and state and
reports the number of differing output and state values per block.
Every declared position-history reader receives its own absolute position;
future-row host position values are not reused. There is no temporal scan or
parallel recurrent update. The routed expert workspace is retained for an
eight-row tile, reused only by ordered work on the same owned stream. Its latest
summary describes the last eligible layer's last complete tile, not all T rows.

The compiler requires explicit arena and host-arena-metadata payload limits.
Eligible terminal-row and state-only owners allocate headless row stripes plus
one logits-capable stripe per owner. Only a produced terminal head resolves that
stripe, including the actual terminal row of a partial block. The compiler
validates every mixed extent, offset and row view before mutation. State-only
owners reserve the stripe for complete native launcher construction but produce
no logits. All-rows scoring, segmented/HTTP batch owners, MTP prompt capture,
closed verify windows, graph inputs and complete-step dump diagnostics retain
the uniform arena.

Eligible QSA row prefill keeps selected token IDs in all original row workspaces,
but scores and selection maxima live in a ring of two slots owned by each tiled
owner. A slot holds G 64-row groups (rows per slot = 64 G). The default starts
at 512 rows and halves while the two slots exceed 1 GiB at the owner's maximum
positions: 512 / 512 / 256 rows per slot at 262,144 / 524,288 / 1,048,576
(320 / 640 / 640 MiB per owner). `STRIX_QSA_SCORE_RING_ROWS=64|128|256|512|1024`
overrides the geometry; it is fully charged and may exceed 1 GiB.

A stage call publishes once for the whole call (RoPE table, norm/RoPE, raw/KV
store, pool, key ranges), then scores and selects one slice of up to G groups
per slot: one score launch with the call-wide rows-per-CTA policy, one batched
selection, the slice's trace hashes, then the slot's consumer event. No deferred
score reader survives reuse. In ring mode the stage batch is not limited to
65,536 blocks (the maxima live in the bounded ring). The scorer and selection
operations, ties, causal bounds, token-zero multiplicity and full-batch
attention union/tile order are unchanged. The score-free, headless row size is
324,352 bytes for the native model geometry; the terminal stripe adds 993,280
bytes per owner. Ring geometry follows the workspace's maximum positions rather
than the current prompt length.

Admission charges every retained owner separately: the main owner and each
retained tail owner carry their own ring and terminal stripe (at 262,144 with
the default geometry about 341 MiB for a 64-row tail owner and 326 MiB for a
16-row tail owner). Receipts report `qsa_score_ring_rows` and
`qsa_score_ring_workspace_bytes`. Under explicit YaRN the owner inventory names the rings
(`qsa_score_ring`) and terminal stripes (`terminal_logits_stripes`) once, split out of the
primary and tail workspaces, and charges no embedded selection or stage-batch maxima for
compact owners (their maxima live in the ring).

Both memory policies default on for eligible routes and follow the qpf2k
defaults policy: `--baseline`, `STRIX_BASELINE=1`, `STRIX_QPF2K_DEFAULTS=0` and
GSQ checkpoints compile the legacy layout. `STRIX_QSA_SCORE_RING=0` and
`STRIX_TERMINAL_LOGITS_ARENA=0` independently restore their legacy layouts;
`STRIX_CTX_EXACT_MEMORY=0` disables both. With both off, physical plans,
offsets, allocation sizes and printed byte receipts retain the legacy values.
Admission and effective configuration use the compiler's arena and ring
formula. GPU gates: the ring and stage split are byte-identical to the legacy
layout in fixtures up to 262,144 blocks and in the 262,143-token QSA trace.
The reported host payload covers arena vector elements; allocator overhead and
the fixed closed-eight selection/catalog metadata are separate host margin.
The allocation consumer must also admit actual retained model/state/repack/
staging owners plus the new workspace, expert scratch and host margin before
allocating. These declared limits are not observations of free physical UMA.

The native owner retains one launcher catalog/plan rather than T cloned decoder
plans. Runtime context storage is a fixed pack of eight. Complete source,
resource, row mapping and every row's pure launcher admission finish before PLE
starts or any model state changes. A failed dispatch cancels outstanding PLE,
drains before releasing owners and poisons the session without partial metadata
publication. Successful completion publishes all T input positions and serial
state-generation counts once. There is no rollback guarantee on failure.

All-rows output retains every configured head. Terminal-row output computes only
the last head and is the intended generation frontier. State-only T prefill is
admitted only with zero requested generation; combining it with generation is
rejected before native execution. Incomplete prompt suffixes use explicit R1
calls through the existing chunk cursor. No consumed input is replayed to create
a missing head.

The actual closed-entry dispatch counters identify reused tiles. The legacy
logical weight-byte/launch-saving totals are not populated for this new T route;
zero in those legacy totals is not a measured traffic estimate. Full-model
serial/chunk fidelity, native/compiler gates, memory peaks, TTFT and throughput
remain to be assessed. No T size or speed gain is qualified by source admission.
