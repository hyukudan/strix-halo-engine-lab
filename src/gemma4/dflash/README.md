# DFlash device interface

The opt-in device path implements `g4-dflash-v1.1`, declared in
`dflash.hpp`. It consumes the strict target primitives and immutable caller
tables. Activations and context K/V are FP32; no activation quantization,
approximate transcendental functions, or alternative reduction order is used.
This module alone does not establish complete-workload qualification.

`DraftModel` validates and uploads immutable drafter coefficients and RoPE tables
once for a loaded model. Construct multiple `DeviceDrafter` streams from the
same `shared_ptr<const DraftModel>`; each stream owns only its tagged context,
unpublished append and reusable execution workspace. The compatibility
constructor creates a private model for explicit copy-path comparisons.
`reset()` discards pending work, clears the context payload/tags/frontier and
timings, and retains allocations. It cannot recover a poisoned publication.

Embedding/head views borrow the target's original packed resident allocation.
They must outlive the model and its streams. The engine verifies source identity,
geometry, stride and format before accepting an alias; otherwise it owns one
immutable packed copy. No embedding scaling, head cap or coefficient conversion
is applied to this view.

One instance owns one stream's independent rings and bounded scratch. Calls
are synchronous at validation/publication boundaries and must be serialized
per instance. The fixed block always has eight rows, including for a requested
window of one proposal. The seven proposal rows are normalized mask rows 1..7.
Block KV never enters the committed context rings.

`append` accepts 1..8 contiguous committed rows; an initial retained prefill
tail may start after zero. Chunk longer inputs in the scheduling owner. An
empty append is a no-op. Absolute tags detect gaps, duplicates and stale slots.
The rotary adapter copies each supplied pair into a fixed interleaved
128-pair row without recomputing trigonometry. Unused pairs are `(1,0)`.
Context visibility is query-relative and strict on the left; partial sums
follow absolute 64-position boundaries, including across the context/block
boundary.

`run_device_cycle` accepts any `Target`/`KvTxn`, including `DeviceTarget`.
The owner begins the target transaction, calls the helper, and destroys the
transaction on failure to discard its overlay. The helper drafts only for an
admitted nonzero window, verifies one target pass, applies `verify_greedy`,
stages exactly the surviving processed taps, commits the target prefix, then
publishes the draft context. It returns the next unprocessed anchor. EOS and
budget truncation occur before either publication. A target commit failure
discards the staged draft append; a device publication failure poisons the
drafter, and the owner must fail/discard that request's stream. Acceptance
counts are updated only after both publications succeed.

The owner calls `admit_window` for automatic/selection-policy admission.
Unmeasured auto remains zero; a nonzero auto window requires a conservative
measured gain of at least 5%. The helper also enforces `STRIX_G4_DFLASH=0`,
budget and full-block position reservation. A zero window skips block/head
work but still appends committed taps for possible future draft cycles.
`STRIX_G4_SHARED_WEIGHTS=0` selects the exact separate-row linear fallback;
the shared primitive's norm switches remain available. There are no draft
graph or fused-SiLU optimizations in this implementation.

`strix-gemma4-dflash-device-tests` uses tiny native tensors and frozen rotary
fixtures. Full-model work belongs only in the explicit diagnostic CLI:

```sh
strix-gemma4-dflash-device-probe MODEL.gguf TARGET_ROPE_DIR DRAFT.gguf DRAFT_ROPE.g4dr PROMPT.npz 128 all --bytes
```

The NPZ supplies `prompt_ids` as little-endian int32. The probe prints private
JSON receipts containing raw IDs, hashes, acceptance observations, timings and
canonical target state digests. It first generates a serial reference, then
compares auto and every fixed window. `--bytes` additionally compares context
K/V, normalized block rows, all seven head-logit rows and proposal IDs with the
host draft recipe. Timings distinguish GPU block/head/append work from target
verify and publication wall time; model loading, diagnostic host work and
prefill are outside decode timings. Keep model-specific receipts outside Git.
