# GDN prefill: two-stream segment overlap (CUDA Qwen MoE)

Home: shared plan `include/strix/gdn_prefill_overlap.hpp` (backend-neutral) + CUDA schedule in
`src/cuda/models/qwen/qwen_gdn.cu` `launch_core_overlap` (the GDN prefill core of the CUDA Qwen prefill driver).

## What it changes

One GDN prefill call (`launch_core`, one layer, one block of rows) runs four steps: gates (K3) and conv (K4) are
row-parallel, the delta-rule recurrence (K5) walks the rows in order with one CTA per (value head, 32 columns), and the
gated norm (K6) is row-parallel again. K5 keeps 192 CTAs of 128 threads busy for the whole call: about one CTA per SM
on a 188-SM GPU, latency bound on the row chain. Serially, nothing else runs during it.

The call is cut into segments (`gdn_overlap::plan`: calls of 4096 rows or more, n = min(4, rows / 2048) balanced
segments of at least 2048 rows, starts on multiples of 256 rows; shorter calls stay serial). The recurrence of segment
i runs on the caller's stream while an auxiliary stream of the calling thread runs the norm of segment i-1 and
gates + conv of segment i+1. Events order the chains:

```
aux:     K3 K4 (0) | K3 K4 (1) | wait K5(0), K6 (0) | K3 K4 (2) | wait K5(1), K6 (1) | ...
stream:  wait (0), K5 (0) | wait (1), K5 (1) | wait (2), K5 (2) | ...  | wait aux (join)
```

## Why the bits do not change

- Same kernels, same per-row arithmetic. Segment starts are multiples of 256, so the conv chunks (32 rows), the gates
  row groups (8) and the A8 delta slots (256) of a segment are the whole call's, and each segment uses exactly its part
  of the whole call's scratch.
- The conv history and the FP32 recurrent state pass from segment to segment exactly as they pass from block to block
  between calls today (the continuation tests of `tests/cuda/qwen_gdn_tests.cu` check block splits bitwise).
- A 16-bit stored state is narrowed once per call, so a split would add narrowings: the overlap runs only with the FP32
  state. A capturing stream, a one-segment plan or `STRIX_QRTX_GDN_OVERLAP=0` keep the serial order.

## Tests and counters

- `tests/gdn_prefill_overlap_tests.cpp` (host): the plan tiles every count up to 70,000 rows (and large counts) in
  order, aligned, 2 to 4 segments of at least 1280 rows, serial below 4096.
- `qwen_gdn_tests` (CUDA): `core overlap T=4096 / 6000 A8 / 8269 A8 ... == serial`: every output row, the state, the
  conv history and the A8 codes / deltas, bitwise. `--bench-core-overlap ROWS` times the serial and the overlapped
  call; `--bench-overlap ROWS` is the co-residency counter (recurrence of one segment against gates + conv + norm of
  another).

## Segment size (measured, RTX PRO 6000, no model)

`qwen_gdn_tests --bench-core-overlap`, serial vs overlapped call, medians of 5, bits equal in every case:

| rows | segments | serial | overlapped | change |
|---|---|---|---|---|
| 1024 | 2 x 512 | 0.587 ms | 0.610 ms | -3.9 % |
| 2048 | 4 x 512 | 1.122 ms | 1.132 ms | -0.9 % |
| 4096 | 4 x 1024 | 2.207 ms | 2.059 ms | +6.7 % |
| 8192 | 4 x 2048 | 4.315 ms | 3.761 ms | +12.8 % |

A linear fit gives ~0.13 us saved per overlapped row and ~86 us of fixed cost per extra segment, so the plan keeps
segments of at least 2048 rows and leaves calls below 4096 rows serial.

## Switch

`STRIX_QRTX_GDN_OVERLAP` (default on; `0` = the serial K3..K6 order).
