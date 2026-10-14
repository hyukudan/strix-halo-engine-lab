# Strix Halo MoE Expert Execution Lab

This lab validates expert scheduling and output restoration before packed-weight reuse is introduced.

The fixture uses small FP32 matrices so the execution contract remains easy to audit.

Two GPU paths are compared:

- direct assignment order: one workgroup per row/top-k assignment;
- expert-major grouped order: the same assignments reordered by expert.

Both paths perform gate projection, up projection, SiLU(gate) times up, down projection, route-weight multiplication and accumulation back into the original row.

The required invariant is:

direct output == grouped output == CPU reference

within the declared FP32 tolerance.

The grouped path is intentionally only a scheduling baseline. It does not yet claim weight reuse or lower traffic. The next optimization step is to combine this grouping contract with packed low-row projection so multiple rows assigned to one expert can share packed expert weights.

Build on native ROCm:

    cmake -S lab/moe_experts -B build/moe-experts -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build build/moe-experts

Example:

    ./build/moe-experts/strix-moe-experts --rows 4 --experts 16 --top-k 4 --iterations 200

The fixture dimensions are hidden=64 and intermediate=32. They are correctness dimensions, not production model dimensions.

## Further reading

See the [lab index](../README.md) for prerequisites and related experiments,
the [qualification ledger](../../docs/qualification-ledger.md) for current gates,
and [independent oracles](../../tools/oracles/README.md) for semantic checks.
