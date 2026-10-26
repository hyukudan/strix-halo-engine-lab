# Experimental exact prefill schedules

Both experiments are off unless their environment switch is exactly `1`.

`STRIX_DS41_PREFILL_SLABS=1` permits attention calls of up to 256 rows. The
default call limit stays 64. A prefill without an explicit nondefault window
selects 64, 128 or 256 rows from its row bound and projection chunk size; the
selected window divides the chunk and is reduced further to bound total
attention-lane scratch to 2 GiB. This bound includes scores, keys, query prep,
partials and tickets; row buffers and candidate masks are separate.
`STRIX_DS41_PREFILL_WINDOW` overrides the
selection. The selected window also defines row ownership in distributed
prefill and is fixed before scratch allocation. The decode window remains eight
rows. Compression, key visibility, key tiles, chunk boundaries, softmax and
chunk combination retain their existing per-row arithmetic.

`STRIX_DS41_PREFILL_TILES=1` enables resource-based candidates for dense and
routed prefill. These are provisional candidates, not a measured tuning table:

- Dense Q8/A8 uses 16 outputs per block for widths at least 4096, at least 512
  outputs and at least 16 input rows; other shapes keep 32 outputs. The smaller
  tile has the same 16 input rows and four class waves.
- Routed batches use 32 rows only when both input widths are at most 1280 and
  the mean entry count is at least 32; other shapes retain 16 rows. The existing
  sparse expert selection remains active.
- Q8 gate/up and Q8/Q2 down use the stage kernels to avoid the split variants'
  register spills. Short down shapes with at least 512 outputs also use stage
  tiles. IQ2 gate/up retains the split form for 16-row batches.

`STRIX_DS41_PREFILL_DENSE_TILE=16|32` and
`STRIX_DS41_PREFILL_MOE_TILE=16|32` override the tile selection only when
`STRIX_DS41_PREFILL_TILES=1`. Invalid override values throw. Existing stage and
MoE decoder-form controls still apply. Qualification of the automatic policy
uses the default decoder form.

The integer group products, 32 class chains, separate floating-point scale
products/additions, reduction tree, clamped SwiGLU and A8 rounding are unchanged.
An output tile changes which block owns an output, not its arithmetic sequence.
Neither switch selects a lower precision numerical contract.

Before enabling either switch by default, compare complete workload results and
timings on the target device. Slab scratch is proportional to the selected rows
times the context bound and attention lane count; increasing the slab is not a
free increase in batching. Export/import and continuation gates must retain the
same live attention state.
