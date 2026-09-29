# Strix Halo Recurrent GDN Lab

This lab isolates the recurrent-state update used by the target Gated DeltaNet
path.

It does not implement the surrounding projections or convolution. Inputs are
already in the numerical form required by the recurrent rule:

- normalized key;
- normalized and scaled query;
- exponentiated decay;
- beta;
- value.

The goal is to measure the cost of recurrent **state traffic** independently
from projection cost.

## Kernel decomposition

The target recurrent matrix is conceptually:

```text
[value head][key_dim=128][value_dim=128]
```

A workgroup owns one value head and one value tile.

Current candidate geometry:

```text
128 threads
32 value columns per tile
4 tiles per 128-wide value head
```

Each lane owns one key row and keeps the tile's state values in registers while
processing a compile-time window of logical tokens.

For each token:

1. apply decay to retained state;
2. reduce `k^T state` across the 128 key lanes;
3. compute the delta vector;
4. update retained state;
5. reduce `q^T state` to produce the output.

The state tile is written back only after the full window.

## Why this shape

A full 128×128 state matrix is too large to keep in one workgroup's registers.

Tiling the value axis:

- preserves exact sequential token order;
- gives disjoint state ownership across workgroups;
- keeps approximately 32 FP32 state values per lane;
- uses a bounded LDS reduction buffer;
- allows windows of 1/2/4/8 tokens without cloning candidate state.

## Build

On native ROCm:

```bash
cmake -S lab/recurrent_gdn -B build/recurrent-gdn -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/recurrent-gdn
```

Run:

```bash
./build/recurrent-gdn/strix-recurrent-gdn --tokens 4 --iterations 200
```

Supported token windows are 1, 2, 4 and 8.

## Comparison

The executable compares:

- **serial**: the one-token candidate launched once per token, forcing state to
  global memory between tokens;
- **retained window**: one kernel processes the whole window and writes the
  recurrent matrix once.

Both paths are checked against the same CPU reference.

The benchmark reports:

- serial and retained-window time;
- speedup;
- estimated recurrent state bytes read/written;
- bytes avoided by retention;
- maximum output/state error;
- kernel resource metadata when available.

## Qualification

This lab is not target-qualified until run on real `gfx1151` hardware.

Promotion requires:

- CPU-reference match;
- serial versus retained-window equivalence;
- fresh versus resumed state equivalence;
- repeated long-window stability;
- no register spilling that defeats the state-traffic reduction;
- end-to-end benefit once surrounding projections are integrated.
