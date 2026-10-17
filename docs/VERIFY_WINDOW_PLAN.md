# Multi-row verify window: plan (2026-10-17)

## Why

Measured on cluster2 (Bosgame M5, gfx1151), same 43-token prompt, 256 generated tokens:

| Engine | tok/s | cost per forward |
|---|---|---|
| ours, 1 token/step (D10 flags) | 20.9 | 47.9 ms |
| Strata `--spec 2`, no drafter | 22.5 | ~47 ms/round |
| Strata MTP (`--spec 4 --mtp --lookup-chain 3`) | 36.2 | ~72 ms/round, 2.63 tokens/round |

The single-token forward is already at parity. Update (same day, ablation on cluster2): dropping
`--model-copy-tier experts` and `--staged-moe-router-logits` (both slower on this box) gives
**40.7 ms/step = 24.6 tok/s**, byte-identical 256 tokens, ahead of Strata's no-draft 22.5. The A8 numerical
profile `hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup-v1` is worth ~12 ms/token (54 ms without it) and requires
prepared-eager + GDN islands; the MTP/window paths currently exclude all three (MTP's serial base runs at ~59
ms/step), so the window must reproduce the A8 profile arithmetic per row. The ~2x gap is the multi-token forward:
Strata verifies a 4-7 token window in ~1.5x a 1-token step and commits the longest accepted prefix.

Our MTP R2 window is slower than plain decode (15 vs 19.4 tok/s; ~130 ms per cycle).
rocprofv3 on the R2 path (GPU busy 86% of wall, so this is GPU work, not host gaps):

| kernel | per window | note |
|---|---|---|
| `batched_typed_hyper_projection_kernel` | ~17 ms | HC runs per row |
| `known_moe_q51_csr_down_kernel<2>` | ~11 ms (276 us/layer) | union down is slow |
| `known_moe_csr_kernel` | ~6 ms (139 us/layer) | CSR built by a single thread |
| `prefetch8_wave_q8_lm_head_kernel` | 3.1 ms per row | full-vocab head re-read per row |
| `project_csr<2>` | ~6.5 ms | union gate/up |
| ~5.4 ms host gap per cycle | | checkpoint + syncs |

## Target design (modelled on Strata's, adapted to our engine)

Window of T = 1..8 rows at consecutive positions (pending token + T-1 drafts), all 48 layers.
Every multi-row kernel performs each row's arithmetic exactly as the 1-row kernel does,
so greedy acceptance is exact.

1. Weights read once for all T rows: HC read/write, dense Q8/Q4K/BF16 projections, router GEMV + top-k,
   shared expert, PLE key/value projections, full-vocab LM head.
2. Routed experts: device-side plan (one thread per (row,k) entry, grouped by distinct expert,
   rows ascending), then grouped gate/up/down that decode each expert once and apply it to all its rows.
3. QSA: append K/V and indexer keys for all T cells first; each row attends with n_kv = pos+1.
   No KV rollback (stale cells are overwritten later). Snapshot the indexer partial-block tail
   before the window; at commit restore it and re-append only the accepted keys.
4. GDN (implemented: `gdn_window_enqueue` / `gdn_window_commit_enqueue`): the window never commits
   its own rows. It first applies the previous window's accepted rows (count and slot read from a
   device `GdnWindowControl`) to the recurrent + conv state and writes it once, then runs its T rows
   in registers from that state and saves per row pre-conv BF16 qkv, normalized k, v, beta, gate
   (52.5 KiB/row/layer, two slots). The state is read once and written once per round; no
   checkpoint copy. A standalone commit applies pending rows before a serial token.
5. PLE history: per-row snapshot inside the window, commit selects row n_keep-1.
6. Argmax for every row on the device into host-mapped memory; one stream sync per window;
   acceptance prefix computed on the host; commit launched async.
7. One hipGraph per T (inputs through small pinned buffers copied by the graph's first nodes),
   one commit graph, MTP round/step graphs on a high-priority stream.
8. MTP teacher forcing from the window's final residual rows (device-to-device), catch up only the
   accepted rows; Q4 copy of MTP projections; window length chosen by draft probability
   (spec-min-p) and an online tokens-per-ms cost model.

## Strix Halo specifics to exploit

- Unified LPDDR5X (~210 GB/s effective): decode is bandwidth bound; both engines sit at ~50% of
  the ~24 ms/token byte floor. Multi-row kernels amortise the weight read across T rows.
- 32 MB Infinity Cache (MALL): small weights re-read by row loops hit the cache; prioritise
  multi-row for large tensors (experts union, LM head 0.67 GB, GDN/QSA projections).
- Host-mapped coherent memory: inputs/outputs/flags without hipMemcpy; poll mapped words.
- 16 Zen5 cores share the memory bus: keep the host thin (no CPU experts).
- RDNA3.5 wave32, v_dot4 int8 (A8 paths), WMMA for T>=4 where useful.

## Kernel work packages (independent, each with a bit-exact test against the 1-row kernel)

- KP1 MoE union: parallel device CSR/plan build (replace the single-thread kernel), fast grouped
  down (Q5_1 / Q8_0 A8) and gate/up (Q4K/Q5K A8G32) for T = 2,4,8.
- KP2 HC multi-row: typed hyper projection + lane-chain output + prepare norm for T rows.
  Status: `hyper_mix_known_window_rows_*` (mode `hyper_mix_rows`, variants `hyper_mix_rows{2,4,8}_v1`,
  request `rows_hyper_mix`, CLI `--mtp-window-hyper-mix`) covers the typed prefetched Q8 selection with split-K FP32
  or v8 projection and lane-chain or Q8 batched output; component bit-exactness only, no model-level gate yet.
- KP3 LM head multi-column + per-row device argmax to mapped memory.
- KP4 GDN window (read-only state, saved inputs) + commit replay kernel.
- KP5 QSA multi-row (append-all-then-attend) + indexer tail snapshot/commit.
- KP6 router multi-row + top-k.
- KP9 Q8 rows: `src/hip_q8_rows_gemv.hpp` reads each Q8_0 block once per wave for several outputs and all T rows,
  keeping every row's R1 wave8 leaf order and tree operands; it serves GDN qkv+z/out, QSA input (five matrices,
  one launch)/output and shared gate+up/down in the existing rows modes (2040 bitwise cases against the R1
  kernels in `strix-hip-q8-rows-projection-tests`). Opt-in `known_window_a8_projections` (per-32 int8
  activations, dot4) is numerically distinct and not part of any mode yet.
