# Strix Halo Q8_0 Target-Dimension MoE Reuse Lab

This lab closes the geometry gap left by the synthetic IQ4_XS reuse experiment.
It keeps the routed-expert dimensions pinned to the current target model:

- hidden size: 2560;
- routed intermediate size: 640;
- top-k default: 10;
- explicit logical row classes: 1/2/4/8.

Q8_0 uses 32-element blocks, so both the 2560-wide gate/up input and the
640-wide down input are represented without padding or a private repack. The
fixture defaults to 16 experts rather than materializing all 512 expert tensors,
keeping the standalone lab practical while preserving the real per-expert
projection geometry. `--experts 512` remains available for a full storage-
footprint run on a sufficiently large target.

The complete routed-expert path is executed from packed bytes: fused gate/up
projection, SiLU(gate) times up, packed down projection, route-weight
multiplication, and scatter-add into the original logical row.

Two schedules use the same deterministic route fixture and CPU packed-weight
reference:

- **direct**: one C1 batch per routed assignment;
- **grouped**: expert-major batches decomposed into 8/4/2/1 rows without
  padding.

Inside every grouped projection kernel a packed Q8 weight scalar is decoded
once and applied to every active logical row in that batch. This makes the
weight-reuse mechanism concrete rather than merely reducing a host-side byte
estimate. Row identity and route weights stay on device during expert
execution; grouping metadata is still prepared on the host, so fully
device-resident grouping remains a separate runtime step.

Build on native ROCm:

```bash
cmake -S lab/moe_q8_0 -B build/moe-q8-0 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/moe-q8-0
```

Run the target-dimensional default fixture:

```bash
./build/moe-q8-0/strix-moe-q8-0 \
  --rows 2 --experts 16 --top-k 10 --iterations 20
```

Repeat with `--rows 1`, `2`, `4`, and `8` for row-class qualification.
The JSON result reports direct/grouped batch counts, modeled packed-weight
traffic, HIP-event timing, speedup, correctness errors against the same CPU
reference, and the exact target geometry.

This experiment does **not** make Q8_0 the canonical expert checkpoint format.
It establishes a target-compatible baseline that can be compared with mixed
per-projection formats or a future explicitly defined repack once quality and
memory trade-offs are measured.
