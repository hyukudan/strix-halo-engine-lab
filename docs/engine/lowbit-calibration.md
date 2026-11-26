# Calibrated low-bit packs

The [Strix Pack v1 specification](strix-pack-v1.md) defines SP2, SP3, SP4, SP4T and the v2 sidecar
container. It links to backend conformance fixtures and model-independent size and error measurements.

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
  the digest of its original matrix bytes, including the scale plane for a paired encoding. Opening it for another checkpoint, or finding a matrix whose source bytes
  differ anywhere, fails with `stale_source`; framing, bounds, the identity and the resident slices are read from one
  open file. `ResidentSlicePlan` loads a matrix it holds instead of converting it.
- `tools/lowbit_calib_pack.cpp` (`strix-lowbit-calib-pack`): checkpoint matrices and optional calibration statistics in, pack out. It refuses
  statistics without provenance, and statistics that share a window or any run of 32 ids with a held-out gate manifest
  (`--heldout`, required, every manifest the qualification uses). The evaluation statistics (`--eval`) must carry
  provenance too and are held to the same rule against the held-out manifests and against the fit windows
  (`require_separation` in `calib_provenance.hpp`), so no held-out row enters a reported error. It also writes a
  table of the relative output error of every matrix for every technique, measured on the evaluation statistics.

- Strix pack SP4 with the fixed table (`strix.sp4t`, `strix/weight/host_lowbit.hpp` `sp::`): 68 bytes per 128 weights
  like `q4_g64`, sign + magnitude-index codes on the table {1,3,5,8,10,14,18,24}, one scale byte per 32 weights and a
  per-tensor base. `--storage SUBSTRING=sp4 --table fixed --out PACK` writes it; its pack line carries a tenth field, the
  16-byte tensor header in hex (part of the identity; a header with another table is refused). There is no load-time
  encoder: a `strix.sp4t` matrix comes from a pack. The base reaches the consumer as `ResidentMatrixView::outer_exponent`.
  HIP consumer: the small-M schedule's `Sp4Operand` (`core/f16_tile_gemm_small.hpp`; table values as exact FP16
  operands, one FP32 multiplier per 32-weight group, 2^base at the store), FP16 form only
  (`hip.sp4t.f16-group32-scale-chain@1`). Inkling 2 Mini binds it as the head arm `weights.head.sp4t@1`
  (`--qualify-decode-weights`, or the pending lever `STRIX_INK_LOWBIT_HEAD=sp4t`, command line only).

## Host Strix packs for grouped experts

`--model` accepts a single safetensors file, an indexed checkpoint directory, or its safetensors index JSON.
The shared source binds rank-2 BF16 matrices, packed MXFP4 (`I8`, low nibble first, plus `F8_E8M0` per 32
columns), and E4M3 FP8 with an F32 scale per 128×128 block. Paired scales use the matrix name plus
`_scale_inv` and may be in another shard. Decoding produces FP32 using the shared weight codecs.

- The checkpoint fingerprint is SHA-256 of the sole file, or SHA-256 of the literal
  `strix-pack-sharded-checkpoint/v1` followed by the 32-byte SHA-256 of the manifest and the 32-byte SHA-256
  of each complete shard in lexical filename order. A BF16 entry retains its payload SHA-256. Paired entries
  hash the literal `strix-pack-source/v1`, one byte (`m` for MXFP4, `f` for FP8), then the 32-byte code hash and
  the 32-byte scale hash. `--probe-source TENSOR` indexes the checkpoint and reads just the named matrix and
  its scale plane, printing the decoded FP32 byte digest on a little-endian host.
- `--technique rtn` can pack without calibration inputs. `--table fixed` keeps the existing `strix.sp4t@1`
  output for SP4. Fitted SP4 and both SP3/SP2 table modes write `strix.sp4@1`, `strix.sp3@1` or
  `strix.sp2@1` with the complete 16-byte header in each entry. Use `LowbitPack::open_host` for these host
  storages. The grouped CPU projection consumes the expert-major plan and can mix bit widths in one call.
  The model runtime requires its own explicit binding and a qualified device tile.
- With `--out`, the writer stages the payload of each matrix into an adjacent file as it is coded and retains
  only entry metadata. It publishes the completed sidecar to a new name through a hard link on the same
  filesystem. `--report` and the diagnostic `--dequant-out` also take new paths. The dequant diagnostic
  accumulates its selected BF16 output in memory. Use `--out` for whole-model work.

### Layer moments and memory

A capture binding can register the calibration windows once, declare the inputs for its current layer, and
call `MomentSet::flush_layer(path)`. The write releases the layer's input buffers and invalidates its input
handles. Replay the same windows or register more before capturing another layer. Pass one `--moments FILE`
per layer to the packer, with `--eval FILE` in the same order if used. The packer opens one layer pair at a
time, validates its provenance and checks all fit windows against all evaluation windows. Each selected
tensor must occur once across the supplied layers. A full layer can be captured and packed separately if
disk space limits the number of full moment files kept at once.

Select `MomentLayout::diagonal` when capturing to write `strix-calib-moments/diagonal-v1`. It keeps the v4
provenance and attribution records, followed by one FP32 value per column of each input. `importance`
uses those entries for clipping and the printed output proxy uses a diagonal covariance. `feedback` and
`both` require full fit moments. An evaluation file may independently use a diagonal proxy.

Let `K_i` be the width of input `i` in the active layer, and `S_i = K_i` for diagonal storage or
`K_i²` for full storage. Capturing bounds the FP64 moment arrays, including update scratch, by
`8 × (sum S_i + max S_i)` bytes. Pass a byte limit to the `MomentSet` constructor to enforce that component.
A fit/evaluation pair uses `8 × sum S_i` bytes across both files, enforced by `--moments-bytes N`.
Serialization uses up to one additional `4 × max K_i` byte row. These bounds cover the moment values.
Token ids, attribution, provenance, coder factors, worker scratch and one matrix's weights have separate
storage. For 512 experts with two K=2048 inputs each, one diagonal fit layer uses 16 MiB of moment arrays
with 16 KiB update scratch. Its on-disk diagonal payload is 8 MiB. Checkpoint headers and file handles are
shared by all layers.

### Pack plan

`strix-lowbit-pack-plan --model-description FILE --budget-gib N --cold sp2|sp3|sp4` prints TSV columns
`tensor`, `format` and `bytes` to stdout. The summary on stderr includes resident weight bytes and exit
code 2 denotes an over-budget plan. `--reserve-gib N` reserves space for KV, activations and scratch.
The plan accounts for resident matrix payloads. Sidecar framing and alignment are on-disk allocations.

The JSON schema `strix-pack-model/v1` accepts optional `retained_bytes` for aggregate weight estimates,
`retained` entries with `name` and `bytes` for individually listed source tensors, and `expert_groups`
with `first_layer`, `layers`, `experts` and `projections`. Each projection declares `name` (with
`{layer}` and `{expert}`), `rows` and logical `columns`. For a mixed plan, add measured `routes`
triples `[layer, expert, count]` and pass `--cold sp2 --hot-experts N`. Candidates with positive counts
are considered by count descending, then layer and expert ascending. A promotion to SP3 includes every
projection of that expert and must fit under the remaining budget. Feed the TSV back to the packer with
`--plan FILE` for exact tensor selection.

The [ERNIE sizing description](../../examples/ernie-pack-description.json) uses 42 routed layers, 512
experts per layer, three 2048×2048 matrices per expert, and an estimated 22.19 GB of retained weights.
At 108 GiB, all SP2 uses 91.5 GiB and SP2 with 3,225 experts promoted to SP3 uses 96.3 GiB. The mixed
footprint can be reproduced with synthetic positive route counts for any 3,225 experts. Actual selection
requires a route capture. These are model weight estimates.

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
| (explicit option) | head Strix pack SP4, fixed table (`strix.sp4t`), every other role BF16 | decode-arm route: pooled KL 0.00067, top-1 0.994, dNLL -0.0002, route flips 0; every class passes; FAST: 37 windows pass, prose-13 fails dNLL (+0.01709, limit +0.01704), structured-14 unresolved (top-1 0.905, dNLL -0.020): not a FAST row; TURBO limits: 39 / 39 windows pass | not served (command line only); CLI B=1, 127-token prompt, 512 steps, three interleaved rounds: 33.20 ms per step against 34.26 (BF16 head) and 35.28 (q4_g64 head) |
| TURBO opt-in | attention and head q4_g64 | scored on the production route (BF16 prefill of the teacher prefix, q4 decode steps): KL 0.0033, top-1 0.980, dNLL +0.0027, route flips 0.00x the null, every window inside the TURBO window bar; a TURBO row (its q4 head fails the FAST bar on two prose windows) | 27.03 / 68.48 / 86.60 tok/s, 14.38 s |

The decode figures above predate the screened BF16 head of the HIP greedy pick. Against a build that screens it (served,
16 slots, context 16384, 256 tokens per request, ABBA, aggregate decode tok/s over the decode window): B=1 BF16 23.10 /
23.70, TURBO opt-in 20.08 / 20.10; C8 47.60 / 49.02 against 48.80 / 48.80; C16 55.74 / 58.19 against 58.79 / 59.02;
8K prefill equal (563 / 595 against 581 / 594 tok/s). The q4 row projects the full q4 head and gives up the screen.

TURBO prints its quality in the `weight_status` receipt line (`turbo_quality ...`).

SP4 head pack: the same calibration moments and held-out manifests as the q4 pack, technique `both`, damp 0.1, fixed table, head only, identity `9566197bac8cd738`, base -36, offline relative output error 0.359 of round-to-nearest. It is not a TURBO row yet: the TURBO row carries q4 attention and a q4 head under a served receipt; swapping its head needs the combined arm gated, the SP4 arm served and a served receipt at widths 1, 8 and 16. The SP4 arm used small-M in the recorded comparison, and the q4_g64 arm used the 128-row tile with no bound `lowbit_rows`. The `strix-halo-128` profile now supplies points for the q4_g64 / q6_g64 group-scale form. Inkling identity and timing with the profile points are pending.

Quality is measured with the gate (`tools/strix-gate/gate_v2.py` through the engine's gate kit) on the loaded pack,
never inferred from the offline table. `STRIX_INKLING_GATE_ROUTE=decode-arm` scores the two-image route via
`--qualify-decode-weights`, including when collecting evidence for a new receipt. Use `=row` to score the admitted
TURBO row within its caps, or `=arm` to score one compact image for the entire run.
