# Step-major tiled prefill

The explicit `--known-prefix-prefill-rows T` route admits 16 through 2048 known
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
Full T row stripes are allocated, including logits capacity even when some heads
are omitted. No alias of an omitted output is invented to bypass admission.
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
