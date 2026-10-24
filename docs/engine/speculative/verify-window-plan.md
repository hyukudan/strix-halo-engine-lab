# Multi-row verify window: design and status

Started 2026-10-17; status updated at `f429bd6`.

## Why

Baseline on one Strix Halo node (Ryzen AI Max+ 395, `gfx1151`), 43-token prompt, 256 generated tokens:

| Path | tok/s | cost per forward |
|---|---|---|
| serial decode, 1 token/step (D10 flags) | 20.9 | 47.9 ms |
| serial decode without `--model-copy-tier experts` and `--staged-moe-router-logits` | 24.6 | 40.7 ms |
| MTP with a fixed R2 window (before this work) | 15 | ~130 ms/cycle |

Both removed flags were slower on this node. The A8 numerical profile
`hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup-v1` is worth ~12 ms/token (54 ms without it). It needs
prepared-eager + GDN islands. The MTP/window paths excluded all three (MTP's serial base ran at ~59 ms/step),
so the window has to reproduce the A8 profile arithmetic per row. Speculative decoding only pays if a T-row
verification forward costs far less than T single-row forwards.

rocprofv3 on the old R2 path. The GPU was busy 86% of wall time, so the cost was GPU work, not host gaps:

| kernel | per window | note |
|---|---|---|
| `batched_typed_hyper_projection_kernel` | ~17 ms | HC runs per row |
| `known_moe_q51_csr_down_kernel<2>` | ~11 ms (276 us/layer) | union down is slow |
| `known_moe_csr_kernel` | ~6 ms (139 us/layer) | CSR built by a single thread |
| `prefetch8_wave_q8_lm_head_kernel` | 3.1 ms per row | full-vocab head re-read per row |
| `project_csr<2>` | ~6.5 ms | union gate/up |
| ~5.4 ms host gap per cycle | | checkpoint + syncs |

## Design

A window has T rows at consecutive positions: the pending token plus T-1 drafts, all 48 layers. Every
multi-row kernel performs each row's arithmetic exactly as the 1-row kernel does, so greedy acceptance is
exact and the emitted tokens are bit-identical to serial decode.

1. Weights are read once for all T rows: HC read/write, dense Q8/Q4K/BF16 projections, router GEMV + top-k,
   shared expert, PLE key/value projections, full-vocab LM head. **Done.**
2. Routed experts: a device-side plan with one thread per (row, k) entry, grouped by distinct expert with rows
   ascending. Grouped gate/up/down kernels then decode each expert once and apply it to all its rows. **Done**
   (`--mtp-window-expert-union`, A8 profile union byte-exact against serial A8 rows).
3. QSA: append K/V and indexer keys for all T cells first; each row attends with n_kv = pos+1. There is no KV
   rollback: stale cells are overwritten later. The indexer partial-block tail is snapshotted before the window;
   at commit it is restored and only the accepted keys are re-appended. **Done** (`--mtp-window-qsa-rows`).
4. GDN (`gdn_window_enqueue` / `gdn_window_commit_enqueue`): the window never commits its own rows. It first
   applies the previous window's accepted rows (count and slot read from a device `GdnWindowControl`) to the
   recurrent + conv state and writes it once. It then runs its T rows in registers from that state and saves
   per-row pre-conv BF16 qkv, normalized k, v, beta and gate (52.5 KiB/row/layer, two slots). The state is read
   once and written once per cycle, with no checkpoint copy. A standalone commit applies pending rows before a
   serial token. Windows of different sizes share one deferred record. **Done** (`--mtp-window-prefix`).
5. PLE history: per-row snapshot inside the window; the commit selects row n_keep-1. **Done.**
6. Argmax for every row on the window stream into host-mapped memory, one stream sync per window, acceptance
   prefix computed on the host, commit launched asynchronously. **Done** (`STRIX_MTP_GREEDY_ROWS=0` keeps the
   older per-row selectors).
7. One hipGraph per T plus a commit graph. **Dropped:** a measured per-kernel cost of 1.98 vs 1.82 µs did not
   pay for capture and the input plumbing. Independent kernels use any-order launches instead (see
   [docs/decisions/decision-register.md](../../decisions/decision-register.md)).
8. MTP teacher forcing from the window's final residual rows, catching up only the accepted rows; Q4_0 copies
   of the MTP draft projections; window length chosen from draft probability (`--mtp-spec-min-p`) and an online
   tokens-per-ms cost model. **Done.** Q4_0 draft copies are on by default (`STRIX_MTP_DRAFT_Q4_ALL=0`
   disables them).

## Result

Adaptive windows `{2,4,8}`, `--mtp-spec-min-p 0.85`, draft vocabulary subset with Q4_0 draft copies. On the
same node, UD-Q4_K_XL, 256 tokens: 48.4 tok/s stable on a Python-code prompt (49.5 with a smaller draft
vocabulary) and ~34-37 tok/s on prose. Output is identical to serial greedy decode in every run.

An R8 cycle is dominated by the routed-expert union, which is inherent to verifying eight exact rows.

## Strix Halo specifics used

- Unified LPDDR5X (~210 GB/s effective): decode is bandwidth bound. Multi-row kernels amortise the weight read
  across T rows.
- 32 MB Infinity Cache (MALL): small weights re-read by row loops hit the cache, so multi-row work targets the
  large tensors first (expert union, LM head 0.67 GB, GDN/QSA projections).
- Host-mapped coherent memory: inputs, outputs and flags move without hipMemcpy; the host polls mapped words.
- 16 Zen 5 cores share the memory bus, so the host stays thin (no CPU experts).
- RDNA 3.5 wave32 and `v_dot4` int8 (A8 paths).

## Kernel work packages

Each package is independent and has a bit-exact test against the 1-row kernel.

- KP1 MoE union: parallel device CSR/plan build (replacing the single-thread kernel), grouped down (Q5_1 / Q8_0
  A8) and gate/up (Q4K/Q5K A8G32) for T = 2, 4, 8.
- KP2 HC multi-row: `hyper_mix_known_window_rows_*` (mode `hyper_mix_rows`, variants `hyper_mix_rows{2,4,8}_v1`,
  request `rows_hyper_mix`, CLI `--mtp-window-hyper-mix`) covers the typed prefetched Q8 selection with split-K
  FP32 or v8 projection and lane-chain or Q8 batched output. The lane-chain output is compiled for 8 rows; split-K
  writes 2 outputs per block by default.
- KP3 LM head multi-row + per-row device argmax to mapped memory.
- KP4 GDN window (read-only state, saved inputs) + commit replay kernel.
- KP5 QSA multi-row (append-all-then-attend) + indexer tail snapshot/commit; parallel exact top-k block selection.
- KP6 router multi-row + top-k (lane per (expert, row), 128 coefficients in flight per lane step).
- KP9 Q8 rows: `src/hip_q8_rows_gemv.hpp` reads each Q8_0 block once per wave for several outputs and all T rows.
  It keeps every row's R1 wave8 leaf order and tree operands, and serves GDN qkv+z/out, QSA input (five matrices,
  one launch)/output and shared gate+up/down in the existing rows modes (2040 bitwise cases against the R1
  kernels in `strix-hip-q8-rows-projection-tests`). Opt-in `known_window_a8_projections` (per-32 int8
  activations, dot4) is numerically distinct and not part of any mode yet.
- KP12 adaptive window: draft early stop, shared deferred owners across window sizes, per-owner cost model.
