# Strix Halo Packed IQ4_XS MoE Reuse Lab

This lab tests a concrete grouped-MoE hypothesis: rows routed to the same expert
can share packed IQ4_XS weight reads and dequantization work across explicit
1/2/4/8-row chunks.

The lab executes the complete routed-expert sequence:

1. packed gate projection;
2. packed up projection;
3. SiLU(gate) times up;
4. packed down projection;
5. route-weight multiplication;
6. scatter-add into the original logical row.

It compares two schedules:

- direct: one one-row batch per routed assignment;
- grouped: deterministic expert-major batches decomposed as 8/4/2/1 without padding.

Both schedules are checked against a CPU reference built from the same IQ4_XS
byte contract.

The fixture uses compatible synthetic geometry:

- hidden size 512;
- expert intermediate size 256.

This is deliberate. The pinned production geometry is hidden=2560 and
intermediate=640. IQ4_XS has a 256-element physical block, so it can represent
the 2560-wide gate/up input directly but not the 640-wide down input under the
current GGML row contract.

Therefore this lab proves packed-weight reuse mechanics. It does not select
IQ4_XS as the canonical format for every expert tensor.

Build on native ROCm:

    cmake -S lab/moe_iq4_xs -B build/moe-iq4-xs -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build build/moe-iq4-xs

Example:

    ./build/moe-iq4-xs/strix-moe-iq4-xs --rows 8 --experts 8 --top-k 4 --iterations 100

Qualification must report modeled weight bytes, direct/grouped timing and
reference error. A grouped result only becomes preferred after real gfx1151
measurement shows that reduced packed-weight traffic survives launch,
accumulation and routing overhead.

## Further reading

See the [lab index](../README.md) for prerequisites and related experiments,
the [qualification ledger](../../docs/qualification-ledger.md) for current gates,
and [independent oracles](../../tools/oracles/README.md) for semantic checks.
