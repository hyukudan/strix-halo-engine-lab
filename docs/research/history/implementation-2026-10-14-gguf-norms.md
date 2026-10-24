# Serialized normalization scales

GGUF normalization tensors used by the hyper mixer, PLE and indexed-attention
operators contain multiplicative scales. Their source zero-centered parameter
has already been folded into `gamma = 1 + parameter` during export. These HIP
operators now multiply by the stored scale directly. Adding one again changed
the model computation. GDN's direct SSM normalization multiplication is unchanged.

The three launcher/compiler identities advance to v2 so previous numerical
plans and cached artifacts cannot silently reuse the old interpretation.
The GGUF representation follows the maintained checkpoint conversion contract
in [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/tree/master/conversion).
No exporter implementation is incorporated into these kernels.

Mathematical fixtures retain source parameters for independent expected values
and fold them only for serialized device bindings. Hyper-mix fixtures include
nontrivial per-stream scales, zero and negative resulting scales. PLE checks
distinct key/query/convolution scales against FP64 normalization and gate
calculations, including stored convolution state. Attention graph fixtures use
unit serialized scales for identity normalization and retain their overflow and
no-partial-state-commit checks.

The complete 174-entry suite passed on gfx1151. A full-checkpoint, 16-token text
smoke now starts with the correct completion for a factual prompt. This narrow
smoke does not establish full logit/state fidelity, long-context quality,
workload throughput or distributed continuity.
