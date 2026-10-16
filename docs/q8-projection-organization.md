# Experimental Q8 projection organizations

The decoder accepts explicit eager C1 overlays for existing BF16 activations and
Q8_0 weights. The input and LM overlays are:

- `--prefetch-q8-lm-head` groups the existing eight loads while retaining one
  wave per output row.
- `--prefetch-q8-lm-head --wave8-q8-lm-head` assigns eight output rows to a
  256-thread block.
- `--wave8-q8-gdn-input` uses eight output rows per block for Q8 GDN input
  projections.

Two further independent output overlays use the existing internal FP32
activation slices and store BF16 results:

- `--wave8-q8-gdn-output` requires the resolved virtual GDN output path.
- `--wave8-q8-qsa-output` replaces the output projection's 256-thread LDS
  reduction with eight independent output waves per block.

Both keep the original 256-leaf DAG. Their separate add helpers preserve each
consumer's ordered operands, including NaN payload priority. No intermediate
BF16 conversion is added to their FP32 input. The width comes from value heads
or attention heads (6144 in the canonical model), rather than hidden size 2560.

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

Output admission checks every GDN and attention output weight, with no
per-layer precision fallback. After scratch allocation and before persistent
state initialization, it binds each output tensor to its unique compiled step,
resolves the actual workspace source and destination, and checks dtype-specific
extents, owners, alignment and aliasing. The existing BF16-input view contract
is kept separate from the FP32-input/BF16-output contract. Launchers repeat the
constant-time view check before enqueueing. The output overlays retain the
compound operators' existing conservative workspace and shared-memory bounds.

Selected kernel IDs, implementation identities and compiler tags distinguish the
overlays. The receipt declares all five options and marks their qualification
as `not_assessed`. Inventory collection preserves that distinction when an
automatic base profile is also selected.

Source fixtures compare actual production-launcher output, GDN scratch and
persistent state, and registration/admission behavior. Compilation, execution,
complete-model fidelity and paired performance are pending. No speed increase
or default adoption follows from these source changes.
