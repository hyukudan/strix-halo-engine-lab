# Calibrated low-bit packs

The low-bit resident row storages `strix.q4_g64` and `strix.q6_g64` (`include/strix/weight/host_lowbit.hpp`) are
normally produced at load time from BF16 matrices: one FP16 scale per 64 weights, chosen by a clip search on the
weights alone, then round-to-nearest codes. A calibrated pack holds the same storages for selected matrices, coded
with the statistics of the rows that actually feed them. The run-time format, the device kernels and the bytes read
per token do not change; only the codes and scales do.

## Pieces (shared, host side, off by default)

- `include/strix/weight/calib_moments.hpp`: named input statistics. An engine hands the rows that feed a set of
  matrices (one named input, for example the normalised attention input of a layer) to a `MomentSet`, which keeps
  `H = sum x x^T` over every row and lists the input's consumers by checkpoint tensor name. It also carries the
  provenance of its rows (`include/strix/weight/calib_provenance.hpp`): the identity of every calibration window (SHA-256
  of its token ids) and the hash of every run of 32 consecutive ids. Rows are accumulated per registered window
  (`add_window` returns the window, `add(window, ...)` takes it; there is no add without a window), and each input
  stores its row count per window. Writing or reading a file in which an input's rows are not all attributed to
  registered windows is refused. File format `strix-calib-moments/v4`: each window record carries its token ids, and
  reading recomputes every window identity and every 32-id run hash from them (a file whose stored identities differ,
  or whose ids are missing, is refused). An accumulation that fails leaves the moments unchanged. v1 (no provenance),
  v2 (rows not attributed to windows) and v3 (identities without their ids) files are refused. The packer refuses a
  fit or evaluation input that no attributed row fed, and reports a zero or non-finite proxy signal as unavailable.
- `include/strix/weight/host_lowbit_gptq.hpp`: the calibrated encoder.
  - *importance*: the clip search of each 64-weight group weights the squared error of each column by `H[j][j]`.
  - *feedback*: error-compensated rounding. Columns are coded in ascending K, and the rounding error of a column moves
    onto the columns not coded yet through the damped inverse-Hessian factor. This minimises the output error
    `sum ||(W - Q) x||^2` over the calibration rows. A group's scale is chosen when its first column is reached.
  - Neither technique: the load-time bytes, bit for bit (host test).
- `include/strix/weight/lowbit_pack.hpp`: the pack sidecar `strix-lowbit-pack/v2`. It is bound to the full SHA-256 of
  the checkpoint file it was coded from (the statistics depend on every tensor, packed or not) and each coded matrix to
  the full SHA-256 of its BF16 source payload. Opening it for another checkpoint, or finding a matrix whose source bytes
  differ anywhere, fails with `stale_source`; framing, bounds, the identity and the resident slices are read from one
  open file. `ResidentSlicePlan` loads a matrix it holds instead of converting it.
- `tools/lowbit_calib_pack.cpp` (`strix-lowbit-calib-pack`): BF16 checkpoint + statistics in, pack out. It refuses
  statistics without provenance, and statistics that share a window or any run of 32 ids with a held-out gate manifest
  (`--heldout`, required, every manifest the qualification uses). The evaluation statistics (`--eval`) must carry
  provenance too and are held to the same rule against the held-out manifests and against the fit windows
  (`require_separation` in `calib_provenance.hpp`), so no held-out row enters a reported error. It also writes a
  table of the relative output error of every matrix for every technique, measured on the evaluation statistics.

## Procedure

1. Collect statistics with an engine that binds the taps (Inkling 2 Mini: `STRIX_CALIB_MOMENTS=FILE` on an EXACT
   prompt run). Run it twice, on two disjoint texts: a fit set and an evaluation set. The tool enforces that neither
   overlaps the held-out windows of the quality gate, nor each other.
2. Pack the matrices:
   `strix-lowbit-calib-pack --model MODEL.safetensors --moments fit.moments --eval eval.moments --heldout GATE_MANIFEST --storage SUBSTRING=q4|q6 ... --technique both --out PACK --report table.tsv`
   Add `--compare` to measure the other techniques and the other format for every matrix too.
3. Qualify the pack with `STRIX_LOWBIT_PACK=PACK` and `--qualify-weights POLICY` on Inkling 2 Mini.
   `--qualify-decode-weights POLICY` instead retains BF16 beside the compact copies and uses compact weights on the
   selected decode widths. The `strix-lowbit-pack/v2` receipt gives the pack identity (SHA-256 of its header and
   payloads) and the matrix count.
4. A registered row requires its measured pack identity. The CLI and server account for the resident copies before
   admitting the requested slots and verify the opened pack again after upload. A missing, stale or mismatched pack,
   or insufficient slot capacity, produces a decline receipt and retries the incumbent BF16 load.

## Inkling 2 Mini rows (gfx1151, HIP 7.1)

Pack: technique `both`, damp 0.1, q4_g64 for the attention projections (q, k, v, o of every layer) and the head,
statistics from 4,290 calibration rows (evaluation set 1,324 rows; the packer checked both against the gate's held-out
windows: no shared 32-token run). The engine looks for it as `inkling2-mini-q4.lowbit-pack` next to the checkpoint (or
`STRIX_LOWBIT_PACK`), identity `b0bd0103dcdd9ffd`. Offline relative output error on the evaluation rows: 0.362 of the
load-time round-to-nearest pack of the same format (mean over the matrices).

The historical TURBO row requires `STRIX_INK_LOWBIT_TURBO=1` with `--profile turbo` (the command line and
`strix-serve`, whose TURBO preset forwards the engine's TURBO column). It is not a TURBO default: since the HIP greedy
pick screens the BF16 head, the q4 row decodes slower than the BF16 image at B=1 and within noise at C8 / C16 (table
below, last row), so the ladder rule keeps TURBO on the tier below. Configured context and lane capacity must be known
and at most 16,384 tokens and 16 lanes (a server needs `--slots 1..16`); outside them, on another backend, with lookup
or drafter speculation, or without the pack (absent or other content), the incumbent weights load and a
`strix-lowbit-pack/v2 declined (...)` line says why. Compact weights run on decode widths 1, 8 and 16 using the
measured FP16 tile arithmetic. Other widths, prefill and verify use the resident BF16 image; a served lane takes its
final prompt token as a BF16 prefill row (emits-first prompt route) on every served route. EXACT and FAST use the
incumbent weights. The final calibrated q4 head lacks a passing FAST receipt.

| profile | mix | gate v2 vs BF16, decode (39 held-out windows, prose / code / structured, 8K-64K context, 2,457 rows) | served B=1 / C8 / C16 decode, 8K prefill (16 slots, context 16384) |
|---|---|---|---|
| EXACT, FAST | BF16 | bitwise | 23.36 / 53.76 / 66.74 tok/s, 14.15 s |
| (FAST candidate) | head q4_g64, every other role BF16 | every class and the pooled bars pass (KL 0.0010, top-1 0.993, dNLL +0.0011, route flips 0), but two prose windows fail dNLL: not a FAST row | 23.96 / 64.55 / 81.52 tok/s, 14.15 s |
| TURBO opt-in | attention and head q4_g64 | scored on the production route (BF16 prefill of the teacher prefix, q4 decode steps): KL 0.0033, top-1 0.980, dNLL +0.0027, route flips 0.00x the null, every window inside the TURBO window bar; a TURBO row (its q4 head fails the FAST bar on two prose windows) | 27.03 / 68.48 / 86.60 tok/s, 14.38 s |

The decode figures above predate the screened BF16 head of the HIP greedy pick. Against a build that screens it (served,
16 slots, context 16384, 256 tokens per request, ABBA, aggregate decode tok/s over the decode window): B=1 BF16 23.10 /
23.70, TURBO opt-in 20.08 / 20.10; C8 47.60 / 49.02 against 48.80 / 48.80; C16 55.74 / 58.19 against 58.79 / 59.02;
8K prefill equal (563 / 595 against 581 / 594 tok/s). The q4 row projects the full q4 head and gives up the screen.

TURBO prints its quality in the `weight_status` receipt line (`turbo_quality ...`).

Quality is measured with the gate (`tools/strix-gate/gate_v2.py` through the engine's gate kit) on the loaded pack,
never inferred from the offline table. `STRIX_INKLING_GATE_ROUTE=decode-arm` scores the two-image route via
`--qualify-decode-weights`, including when collecting evidence for a new receipt. Use `=row` to score the admitted
TURBO row within its caps, or `=arm` to score one compact image for the entire run.
