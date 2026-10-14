# Routed experts with K-quant and affine blocks

The HIP routed-expert launcher now accepts Q4_K, Q5_K and Q5_1, in addition to
its existing encodings. Selected gate/up projections and weighted down
accumulation read the original packed weights without a whole-model expansion.

Q4_K and Q5_K encode 256 values per block, with eight 32-value groups. Each
group has unsigned six-bit scale and minimum factors. The value is
`d * scale * code - dmin * minimum`; Q5_K carries a separate high bit per value.
Q5_1 encodes 32 values with an affine scale and offset, `d * code + m`. Row
widths must contain complete blocks. Checked runtime tensor geometry and
binding validation precede dispatch.

The public GGUF encodings belong to the ggml format maintained by
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). This implementation
uses the specified packed layouts; no upstream dequantization implementation
is incorporated into the launcher.

The native tests independently generate weights and their expected values
before packing, then compare complete routed-expert outputs with CPU matrix
products and SiLU. They cover every code and six-bit factor, multiple blocks
per row, noncontiguous selected experts and all three encodings individually,
plus a mixed Q4_K gate, Q5_K up and Q5_1 down configuration. These fixtures
passed on gfx1151. They do not qualify full-checkpoint logits, model quality,
throughput or distributed continuity.
