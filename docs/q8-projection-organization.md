# Experimental Q8 projection organizations

The decoder accepts explicit eager C1 overlays for existing BF16 activations and
Q8_0 weights:

- `--prefetch-q8-lm-head` groups the existing eight loads while retaining one
  wave per output row.
- `--prefetch-q8-lm-head --wave8-q8-lm-head` assigns eight output rows to a
  256-thread block.
- `--wave8-q8-gdn-input` uses eight output rows per block for Q8 GDN input
  projections. The GDN operator still runs its original recurrence and output
  projection, with their existing shared-memory requirements.

These options preserve automatic base-profile resolution. They do not enable
themselves by default or inherit the base profile's qualification. A manual or
fallback configuration must supply the existing virtual Q8 prerequisites;
GDN also requires its existing prefetch option. Graph and integer-head modes
are excluded. The launchers reject actual stream capture before enqueueing.

Each row retains its coefficient expression, leaf accumulation, reduction tree,
operand order and rounding points. Changing the load organization or block
geometry still requires compiler, ISA, native execution and model-equivalence
checks. The supported projection width is a nonzero multiple of 32, at most
10240, with at most 248320 output rows. GDN requires Q8_0 qkv and z tensors in
every selected layer; other supported input tensors keep their original format
handling. LM candidates require BF16 input and Q8_0 weights.

Before residency, admission checks the file offsets, exact shapes and extents of
every selected weight across all layers, including Q8 beta and alpha. Every
actual device address is checked again after residency and before any token
launch. Projection views retain 16-byte alignment; GDN workspace head counts
must also keep its adjacent FP32 beta and gate arrays aligned. A late-layer
alignment error therefore rejects setup before earlier layers can mutate state.

Selected kernel IDs, implementation identities and compiler tags distinguish the
overlays. The receipt declares all three options and marks their qualification
as `not_assessed`. Inventory collection preserves that distinction when an
automatic base profile is also selected.

Source fixtures compare actual production-launcher output, GDN scratch and
persistent state, and registration/admission behavior. Compilation, execution,
complete-model fidelity and paired performance are pending. No speed increase
or default adoption follows from these source changes.
