# Calibrated low-bit packs

The low-bit resident row storages `strix.q4_g64` and `strix.q6_g64` (`include/strix/weight/host_lowbit.hpp`) are
normally produced at load time from BF16 matrices: one FP16 scale per 64 weights, chosen by a clip search on the
weights alone, then round-to-nearest codes. A calibrated pack holds the same storages for selected matrices, coded
with the statistics of the rows that actually feed them. The run-time format, the device kernels and the bytes read
per token do not change; only the codes and scales do.

## Pieces (shared, host side, off by default)

- `include/strix/weight/calib_moments.hpp`: named input statistics. An engine hands the rows that feed a set of
  matrices (one named input, for example the normalised attention input of a layer) to a `MomentSet`, which keeps
  `H = sum x x^T` over every row and lists the input's consumers by checkpoint tensor name. File format
  `strix-calib-moments/v1`.
- `include/strix/weight/host_lowbit_gptq.hpp`: the calibrated encoder.
  - *importance*: the clip search of each 64-weight group weights the squared error of each column by `H[j][j]`.
  - *feedback*: error-compensated rounding. Columns are coded in ascending K, and the rounding error of a column moves
    onto the columns not coded yet through the damped inverse-Hessian factor. This minimises the output error
    `sum ||(W - Q) x||^2` over the calibration rows. A group's scale is chosen when its first column is reached.
  - Neither technique: the load-time bytes, bit for bit (host test).
- `include/strix/weight/lowbit_pack.hpp`: the pack sidecar `strix-lowbit-pack/v1`. It holds the coded matrices with a
  fingerprint of each BF16 source tensor. `ResidentSlicePlan` loads a matrix it holds instead of converting it, and
  refuses a pack whose format, shape, byte count or source fingerprint does not match.
- `tools/lowbit_calib_pack.cpp` (`strix-lowbit-calib-pack`): BF16 checkpoint + statistics in, pack out. It also
  writes a table of the relative output error of every matrix for every technique, measured on a second statistics
  file collected from other text.

## Procedure

1. Collect statistics with an engine that binds the taps (Inkling 2 Mini: `STRIX_CALIB_MOMENTS=FILE` on an EXACT
   prompt run). Run it twice, on two disjoint texts: a fit set and an evaluation set. Neither may overlap the held-out
   windows of the quality gate.
2. Pack the matrices:
   `strix-lowbit-calib-pack --model MODEL.safetensors --moments fit.moments --eval eval.moments --storage SUBSTRING=q4|q6 ... --technique both --out PACK --report table.tsv`
   Add `--compare` to measure the other techniques and the other format for every matrix too.
3. Load it with the low-bit weight policy (Inkling 2 Mini: `STRIX_LOWBIT_PACK=PACK`). The receipt line
   `strix-lowbit-pack/v1` names the pack identity (SHA-256 over the header and every payload byte, so packs coded with
   other statistics or settings differ) and how many matrices came from it. Without a pack it reads
   `none (load-time round-to-nearest)`.
4. A decision row that admits a calibrated mix names the pack identity it was measured with; the engine refuses
   another pack for that row (its stated quality is that pack's). Without the pack file the row does not apply: the
   load keeps BF16 and the plan says the pack is absent.

## Inkling 2 Mini rows (gfx1151, HIP 7.1)

Pack: technique `both`, damp 0.1, q4_g64 for the attention projections (q, k, v, o of every layer) and the head,
statistics from 6,722 calibration rows (evaluation set 2,194 rows, both disjoint from the gate windows). The engine looks
for it as `inkling2-mini-q4.lowbit-pack` next to the checkpoint (or `STRIX_LOWBIT_PACK`), identity `6c2dfa3505a9e239`.

Offline relative output error on the evaluation rows (mean over the 24 layers, ratio to the load-time round-to-nearest
pack of the same format): importance-weighted clip alone 0.81-0.90, error feedback 0.30-0.43, both 0.29-0.42; damp 0.1
a further -6 % against 0.01.

| profile | mix | gate v2 vs BF16 (16 held-out prose windows, prefill / decode) | served B=1 / C8 decode, 8K prefill (16 slots, context 16384) |
|---|---|---|---|
| EXACT | BF16 | bitwise | 20.81 / 44.15 tok/s, 18.95 s |
| FAST | head q4_g64, every other role BF16 | PASS: KL 0.0023 / 0.0023, top-1 0.970 / 0.976, dNLL +0.0011 / +0.0005, route flips 0 | 21.27 / 50.36 tok/s, 18.95 s |
| TURBO | attention and head q4_g64 | below the FAST bar (route flips 4.07x the null): KL 0.0064 / 0.0063, top-1 0.958 / 0.963, dNLL -0.0016 / -0.0025 | 24.09 / 52.50 tok/s, 19.01 s |

The same mixes packed by the load-time round-to-nearest converter measure KL 0.0217 / 0.0210, top-1 0.912 / 0.919,
dNLL +0.0223 / +0.0191 (attention and head q4) and fail FAST for the head alone (5 of 16 windows); the calibrated pack is
what makes the head row pass. TURBO prints its quality in the `weight_status` receipt line (`turbo_quality ...`).

Quality is measured with the gate (`tools/strix-gate/gate_v2.py` through the engine's gate kit) on the loaded pack,
never inferred from the offline table.
