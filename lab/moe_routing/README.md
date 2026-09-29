# Strix Halo MoE Routing Lab

This lab isolates MoE routing from expert execution.

The first candidate follows the pinned routing semantics:

1. FP32 softmax over all expert logits;
2. top-k expert selection per logical row;
3. optional renormalization of selected probabilities;
4. output selected expert indices and weights.

The fixture generator avoids cutoff ties because upstream top-k tie order is not
yet a canonical semantic contract.

## Initial geometry

Defaults:

- 512 experts;
- top-10;
- logical row classes 1/2/4/8;
- one workgroup per row.

This is a correctness/performance baseline, not the intended final selection
algorithm. The first kernel uses cooperative max/sum reductions and a simple
single-lane top-k scan after softmax.

Later candidates may replace only the selection mechanism while preserving the
same output contract.

## Build

```bash
cmake -S lab/moe_routing -B build/moe-routing -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/moe-routing
```

Run:

```bash
./build/moe-routing/strix-moe-routing \
  --rows 4 --experts 512 --top-k 10 --iterations 500
```

Supported explicit row classes are 1, 2, 4 and 8.

## Output

The executable reports:

- device and architecture;
- rows / experts / top-k;
- average routing time;
- max weight error;
- exact selected-index agreement;
- distinct expert count across the round;
- mean adjacent-row route overlap;
- kernel resource metadata.

## Next candidates

After this baseline is qualified on real `gfx1151`:

- Wave32 and Wave64 top-k reduction candidates;
- hierarchical top-k without materializing all probabilities;
- direct production of expert-group metadata;
- fused route histogram / touched-expert bitmap;
- route sorting/grouping only when expert reuse justifies it.

Expert GEMMs are a separate lab because route selection and grouped expert
execution have different shape and memory bottlenecks.
