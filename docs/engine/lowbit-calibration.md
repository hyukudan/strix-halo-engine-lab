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
3. Load it with the low-bit weight policy (Inkling 2 Mini: `STRIX_LOWBIT_PACK=PACK`). The receipt line
   `strix-lowbit-pack/v2` names the pack identity (SHA-256 over the header and every payload byte, so packs coded with
   other statistics or settings differ) and how many matrices came from it. Without a pack it reads
   `none (load-time round-to-nearest)`.
4. A decision row that admits a calibrated mix names the pack identity it was measured with; the engine refuses
   another pack for that row (its stated quality is that pack's).

Quality is measured with the gate (`tools/strix-gate/gate_v2.py` through the engine's gate kit) on the loaded pack,
never inferred from the offline table.
