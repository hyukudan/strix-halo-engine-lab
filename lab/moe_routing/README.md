# Strix Halo MoE Routing Lab

This lab isolates MoE routing from expert execution and keeps the routing
contract identical across host and device candidates:

1. FP32 softmax over all expert logits;
2. top-k expert selection per logical row;
3. optional renormalization of selected probabilities;
4. deterministic cutoff ties: equal probabilities prefer the lower expert index;
5. output selected expert indices and weights.

The deterministic fixture injects a cutoff tie when the requested geometry
permits it, so both correctness paths exercise the canonical tie rule.

## Target geometry

Defaults:

- 512 experts;
- top-10;
- logical row classes 1/2/4/8;
- one workgroup per row.

Two HIP candidates are built for `gfx1151`.

### Serial-selection baseline

`strix-moe-routing` keeps the original correctness/performance baseline:
cooperative softmax max/sum reductions followed by a single-lane top-k scan.

### Wave32 hierarchical candidate

`strix-moe-routing-hierarchical` maps one thread to each of the 512 target
experts and uses sixteen 32-lane groups. For every top-k slot, each group
reduces to its best remaining expert, then the first Wave32 reduces those group
winners to the global winner. The selected expert is removed and the process
repeats.

At the pinned 512-expert/top-10 geometry this mirrors the host hierarchical
contract exactly:

- group width: 32;
- groups: 16;
- local candidate scans: 5,120;
- group-reduction candidates: 160.

The device candidate requires Wave32 and validates exact selected indices plus
selected-weight error against the independent CPU reference before reporting
timing.

## Build

```bash
cmake -S lab/moe_routing -B build/moe-routing -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/moe-routing
```

Run the baseline:

```bash
./build/moe-routing/strix-moe-routing \
  --rows 4 --experts 512 --top-k 10 --iterations 500
```

Run the hierarchical candidate with the same geometry:

```bash
./build/moe-routing/strix-moe-routing-hierarchical \
  --rows 4 --experts 512 --top-k 10 --iterations 500
```

Supported explicit row classes are 1, 2, 4 and 8.

## Output

The baseline reports routing latency, exact selected-index agreement, weight
error, route-union/overlap metrics and kernel resource metadata.

The hierarchical candidate additionally reports group width, group count,
local candidate scans and group-reduction candidate count so native
measurements can be checked directly against the host plan.

## Qualification still required

The code now has both the serial-selection baseline and the Wave32
hierarchical device candidate. The remaining gate is native Strix Halo
qualification:

- build and run both on the same ROCm/toolchain revision;
- compare latency for row classes 1/2/4/8;
- inspect VGPR/LDS/spill occupancy;
- confirm the hierarchical candidate remains exact under the cutoff-tie fixture;
- decide whether removing full probability materialization is worthwhile;
- only then promote a routing path into the production kernel registry.

After routing selection is qualified, the next MoE work is direct/grouped
packed expert execution, device-side grouping metadata and route-distribution
qualification on the canonical checkpoint.

Expert GEMMs remain a separate lab because route selection and grouped expert
execution have different shape and memory bottlenecks.
