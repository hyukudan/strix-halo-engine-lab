# Numerical identity ds41-trellis-k5-v1 (QTIP-style trellis dense decode sites)

Status: declared (round 4). Opt-in: `STRIX_DS41_TRELLIS=<pack dir>` (DeviceForwardOptions::trellis), classes
`STRIX_DS41_TRELLIS_SITES=` (q_a, q_b, kv, o_a, o_b, shexp, head, all; each class acts only where its hook is in the
build). Unset: the current identities, bit for bit (the trellis code is not reached).
Default class list: **empty**. STRIX_DS41_TRELLIS alone selects no site (DeviceForward / BatchForward print
"ds41 trellis: STRIX_DS41_TRELLIS is set but no site class is selected" at startup); STRIX_DS41_TRELLIS_SITES must name
the classes (e.g. `shexp`, or `all` for A/B). Reason: on v12 no class wins in-app yet (q_b / o_a / o_b slower in-app,
round-4 attn; q_a / kv lose at every R, round-4 qakv; shexp +0.5 ms at R1 and -1.1 ms per R6 window, round-4 core;
head: its group's section).

## Weights

- Pack: `ours-K5` (dense.L0..L39.safetensors + dense.head.safetensors; SHA256 manifest `ours-K5.sha256` next to the
  pack on both nodes, 43 files). Every dense linear is K = 5 bits per weight, codebook mul1 (marker 0x83DCD12D), no bias:
  `<name>.trellis` I16 [I/16][N/16][16 K], `<name>.suh` F16 [I], `<name>.svh` F16 [N]. The pack's index.json is
  partial and not used: the reader (ds41_trellis.cpp) reads every shard header.
- Effective weight W = diag(suh) H_I Q H_O diag(svh) (H = normalized 128-block Sylvester Hadamard).
- Name map (ds41_trellis.cpp): blk.L.attn_q_a -> layers.L.attn.wq_a, attn_q_b -> wq_b, attn_kv -> wkv,
  attn_output_a -> wo_a.slice.<g> (one linear per group), attn_output_b -> wo_b, ffn_gate_shexp -> shared_experts.w1,
  ffn_up_shexp -> w3, ffn_down_shexp -> w2, output.weight -> head.
- The Q8_0 / c-u6 copies of these tensors stay loaded (prefill and the default identity use them).
- `ours-K4` (also on both nodes) is NOT this identity (it would be ds41-trellis-k4-v1).

## Library

- strix_trellis C ABI 1, fused **v12**: `libstrix_trellis_fused_v12.a` SHA256
  97b4af73c40ffd86a7656ad864eac1c485b3004f60ce457c5019a8340d5224cd, header `strix_trellis.h` SHA256
  2ef2604f1b9cc2a55dc315ebbc039723071128d5fd9314ccf6e5edd4de04a3a6 (an external prebuilt library, located with
  CMake `-DSTRIX_DS41_TRELLIS_LIBRARY=` / `-DSTRIX_DS41_TRELLIS_INCLUDE_DIR=`; option `STRIX_DS41_TRELLIS` defaults ON
  only where both are given and exist).
- Every call is `strix_gemv` (never `strix_gemm`: its WMMA path has another reduction), alpha = 1, beta = 0, input and
  output contiguous F32 [rows][I] / [rows][N] in the ORIGINAL basis. The library applies suh, H128, the trellis core,
  H128 and svh itself (its stages: RN32(x * suh) -> RN16 -> H128 in F32 -> RN16 core input; FP16 x FP16 products with
  FP32 accumulation, split over K by the library's fixed per-shape policy; partials summed in the output transform;
  H128 F32; * svh). No A8 at a trellis site: X is the site's F32 activation.
- Row chunking (part of the identity): R rows go to the library in consecutive chunks of at most
  `trellis_chunk_rows` rows (ds41_trellis.hpp), one strix_gemv each. Declared R-invariant for chunk sizes 1..C (see
  "R-invariance" below): a row's output does not depend on the other rows of its call.
- Strided rows / accumulate (enqueue_trellis_linear): non-contiguous rows are copied into a contiguous staging buffer
  (bit copies); accumulate is y = y + L(x) with one FP32 add per value after the library call.

## Rank slices and sums

A site may use a 128-aligned slice of a linear (ds41::TrellisSlice):
- rows [a, a + n): the tiles' columns [a/16, (a+n)/16) and svh[a, a+n) — a linear of its own (the library may choose
  another split for the smaller N, so a row slice is NOT bitwise the corresponding rows of the whole tensor; within
  the identity every path uses the same slice).
- columns [a, a + c): the tiles' rows and suh[a, a+c); its output is a partial product, the site group declares how
  the rank partials are summed (order and precision).
Per-site slicing and sums: see the site sections below (appended by each site group).

## Reference

The library's GPU output equals its CPU ordered references only numerically (the library's own validation: NRMS ~1.6e-6 against the
ordered FP32 CPU core and the FP64 core; not bitwise: the GPU splits K). The identity's reference is therefore the
**device run at R1** (DeviceForward, one row per step) with this identity; verify windows (R2..R8, with rejections /
rollback through commit_window), BatchForward steps and the two ranks are gated bitwise against it. The CPU host forward
has no trellis path.

The exact gate stays **GPU against GPU** with the submission order the identity fixes (row chunks, padding, slices and
sums): the library's DOT2ACC accumulation was later shown on hardware not to follow its CPU oracle's order, so no CPU
model of the dot products is a bitwise reference for any K or mix.

## Announced identity and memory accounting

The identity an engine announces comes from the packs it actually prepared, not from this document's name:

- Step identity (`PlanStep::identity`): `ds41-trellis-k<K>-v1:<site>` with the K of the pack that step reads
  (`trellis_step_identity`); the shared expert reports `ds41-trellis-mix-v1{shexp_w13=K<a>,shexp_w2=K<b>}:shexp` when
  its gate / up and down come from packs of different K (`trellis_shexp_step_identity`); `:shexp-tp` with SHEXP_TP.
- Configuration identity (`TrellisContext::announced_identity`, `BatchForward::trellis_identity()`, the rank app's
  storage line, serve's startup line and `/status` memory): one K over every enabled site class ->
  `ds41-trellis-k<K>-v1:<sites>`; otherwise `ds41-trellis-mix-v1{<mix class>=K<k>,...}:<sites>`, followed by
  `(= ds41-trellis-mixA-v1)` or `(= ds41-trellis-mixB-v1)` when the per-class K are those declared below; then
  `:shexp-tp`. Sites are listed in the order q_b, o_a, o_b, shexp, head (q_a and kv have no engine hook and are never
  listed). HEADOUT and OA_GROUP are adapter forms with the same bits and do not appear.
- `trellis_weights_namespace()`: the configuration identity plus, per mix class used, its pack directory and a
  fingerprint (FNV-1a 64 of the directory's SHA256 manifest `<dir>.sha256`, else of its file names and sizes). It is the
  input for namespaces that must not mix states of different weights (the prefix cache's model id; serve
  `--describe-json` "trellis").
- Memory: the dense Q8_0 / c-u6 copies stay loaded, so the trellis bytes are **added** to the dense weights.
  `BatchForward::footprint().trellis` = prepared weights (trellis + suh + svh of every distinct tensor or slice,
  `trellis_site_bytes`) + the context's x / y staging + the HEADOUT row buffers + the o_a group runner's staging; the
  constructor checks it against the allocations. The library workspaces are only known after the library's queries at
  load (`BatchForward::trellis_library_bytes()`; reported by serve, not in `footprint()`).

## R-invariance

Checked with strix-hip-ds41-trellis-tests on real ours-K5 tensors (one node, 2026-10-21):
per row, GEMV with R = 1..8 rows (two row offsets), GEMM R = 1..32, repeat determinism.

- R1 repeat: deterministic everywhere.
- GEMV R1..R8 one class (every row bitwise the same for any R): q_b (1280 -> 32768), o_b (8192 -> 5120) and its row
  half (2560) and K half (4096), q_a (5120 -> 1280), kv (5120 -> 512), head (5120 -> 129280) and both 64640-row rank
  slices.
- Several classes (the library's per-row-count K split / kernel mode, v12 gemv_policy.hpp, and the R1 fused-input path):
  o_a group (4096 -> 1024): R1 = R4 = R8, R2/3/5/6/7 differ (split 16 vs 32); shared w1 / w3 (5120 -> 2304) and w2
  (2304 -> 5120): R1 differs from R2..R8. R-class run (layers 0/21/39, o_a groups 0 and 7.3; rows
  independent within a call everywhere): w1/w3 classes R1 | R2..R7 | R8; w2 R1 | R2..R8; o_a R1,R4,R8 | R2,3,5,6,7;
  q_b, kv one class.
- GEMM (strix_gemm, WMMA) differs from GEMV for every R: never used.

Hence the per-shape submission sizes of `trellis_submit_sizes` (ds41_trellis.hpp / hip_ds41_trellis.hip): a chunk of
n rows is submitted with the smallest allowed count >= n, padded with copies of its last row; R > max allowed is cut in
chunks of the max. Every allowed count of a shape is in one R class, so a row's bits never depend on the window or
batch it is in. Table (v12, K5): o_a group 4096x1024 {1, 4, 8}; shared w1/w3 5120x2304 {2..7} (R1 padded to 2,
R8 = 7 + 1 padded to 2); shared w2 2304x5120 {2..8} (R1 padded to 2); every other shape {1..8}. The test checks the
entry point (enqueue_trellis_linear, R = 2..8, 13, 32, strided + accumulate) bitwise against every row alone through the
same entry point: all bitwise on the 12 tensors of the run.

## Sites

### shexp (shared expert; group core)

- Placement: replicated on both ranks (the default form today); each rank runs the whole gate / up / down. The split
  form (STRIX_DS41_SPLIT_SHARED=1, STRIX_DS41_BATCH_SHARED=split) is refused with a trellis shared expert. The
  overlapped form (enqueued between the start and the finish of the half-sum exchange, two ranks without op hashes)
  runs the same calls on the same stream and buffers.
- x = the FFN site's normalized F32 input (m.x, [R][5120]); the routed experts keep its A8 codes as before.
- gate = L(w1, x), up = L(w3, x) ([R][2304] F32), act = swiglu_clamped(gate, up, swiglu_limit) in F32 per value
  (ds41_moe.hpp, the model's clamps; no A8), shared = L(w2, act) ([R][5120] F32).
- Combine unchanged: y = (shared + low) + high (enqueue_canonical_sum).
- Step identity string: `ds41-trellis-k5-v1:shexp`.
- Gates (two-node pair, 2026-10-21, c-u6 all on, --tp-attention --engram-direct, Madrid+8):
  STRIX_DS41_TRELLIS unset: op hashes IDENTICAL to the CPU host references on both ranks (c-u6 p9-host-cu6-L40 and base
  p7-host-L40: 21 steps, 14,406 op hashes, 0 differ). SITES=shexp: R1 run = the reference; fill windows 2..8 and
  rejected windows (--fill-window 6,8,5 --fill-keep 2,5,1,3) IDENTICAL to the R1 fill on both ranks (14,406 hashes);
  the overlapped form (no op hashes) gives the same tokens and logits FNV as the hashed R1 run; the shared_expert hashes
  of rank 0 and rank 1 are identical (840).
  BatchForward (SITES=shexp, with the footprint fix 5f9921b): batch == alone for N = 1, 2, 4, 8 (662 rows, tokens +
  full logits, 0 differ) and the external reference: 256 batch rows of prompts p0 / p2 equal to DeviceForward alone at
  R1 in the same identity (fed / argmax / full-logits FNV).
- Timing (two-node pair, c-u6 all on): shared-expert bytes 1.150 -> 0.887 GB/token/node. Shared expert alone
  (STRIX_DS41_EP_OVERLAP=0 op timing, rank 0): R1 6.57 -> 6.97 ms (175 -> 127 GB/s), R6 window 8.39 -> 6.99 ms.
  Untraced walls (4 reps each): R1 40.63/40.56/40.54/40.45 -> 41.94/41.12/41.10/41.04 ms (+0.5..0.6 ms; the R1 row is
  padded to 2 rows and 5120x2304 at R2 is slower than v12's R1 fused form); R6 67.54/67.19/67.16/67.26 ->
  67.91/66.42/66.13/66.14 ms (-1.1 ms in reps 2-4). Not in the default list (R1 loses).

### shexp-tp (shared expert split over the ranks; declared revision, round 4b)

Opt-in **STRIX_DS41_SHEXP_TP=1** (DeviceForwardOptions::trellis_shexp_tp; BatchForward reads the same variable) on top
of the shexp class (STRIX_DS41_TRELLIS + STRIX_DS41_TRELLIS_SITES containing shexp; the flag without them is refused).
Unset: the replicated shexp above, bit for bit (code path unchanged). Step identity string `ds41-trellis-k5-v1:shexp-tp`.
A NEW numeric identity (its outputs differ from :shexp); it has its own gates below.

- Weights per rank r (r = 0, 1; hI = 1152 = 9 x 128): gate = row slice rows [hI r, hI r + hI) of w1, up = the same rows
  of w3 (TrellisSlice rows: 5120 -> 1152, tiles' columns [72 r, 72 r + 72), svh of those rows, suh whole), down = column
  slice columns [hI r, hI r + hI) of w2 (TrellisSlice columns: 1152 -> 5120, tiles' rows [72 r, +72), suh of those
  columns, svh whole). 3 x 3.7 MB per rank and layer (~423 MiB per node instead of 845 MiB). World 1 (one process)
  prepares both ranks' slices and runs both partials, so its bits are the pair's.
- Partial: P_r = L(down_r, act_r), act_r = swiglu_clamped(L(gate_r, x), L(up_r, x), swiglu_limit) in F32 per value,
  x = the FFN's normalized F32 input (the routed experts keep its A8). The three calls go through enqueue_trellis_linear
  with contiguous rows of the exact widths (ds41_shexp_tp.hpp, src/hip_ds41_shexp_tp.hip).
- Sum: rank r's half sum S_r (enqueue_half_sum, unchanged) then T_r = S_r + P_r (one FP32 add per value), T_r goes
  through the existing half-sum exchange (same size, R x 5120 F32), and y = T_0 + T_1 (one FP32 add per value, rank 0
  first) on both ranks. That is y = (S_0 + P_0) + (S_1 + P_1), instead of (shared + S_0) + S_1. No overlapped form
  (P_r must be in the message); the partial is computed in the shared_expert step.
- The shared_expert op hash is rank-local (rank r: P_r; one process: P_0), like q / o / o_low under TP attention.
- R-invariance (v12, layers 0 / 20 / 39, both rank slices of w1 / w3 / w2): GEMV R = 1..8 is
  ONE class for all 18 tensors (rows independent within a call), so the submission sizes are {1..8} (no padding) and the
  entry point is bitwise vs every row alone for R = 2..8, 13, 32, strided + accumulate.
- Reference: the device run at R1 with this identity (as for the whole identity).
- Gates (round 4b): single node 4 layers: flag off == c9c76f5 binary
  (single + emulate, 2465 op hashes); replicated shexp == c9c76f5; shexp-tp R1 == windows 2..8 with rejections == world 1
  == world 2 emulate == tp-attention (419 rows, 0 differ; control vs :shexp differs); op-hash path == fused path (single,
  emulate); BatchForward --check N 1/2/4/8 (single + emulate) and windows, external N 4/8 == DeviceForward alone (364
  rows). Pair, 40 layers, c-u6 all on: unset == CPU host c-u6 both ranks (14,406 op hashes);
  shexp-tp fill windows 6,8,5 keep 2,5,1,3 == R1 fill (pos lines, both ranks, rank 0 == rank 1); BatchForward N 1/2/4
  PASS + external 170 rows == DeviceForward alone.
- Timing (pair, c-u6 all on): op timing (STRIX_DS41_EP_OVERLAP=0 for :shexp) shared_expert R1
  7.02 -> 4.07 ms, R6 window 7.07 -> 4.86 ms (0.887 -> 0.444 GB/token/node). Untraced walls (2 reps): R1 unset
  40.56/40.63, :shexp 41.07/41.23, :shexp-tp 40.53/40.71 ms; R6 window unset 67.07/67.11, :shexp 66.13/66.12, :shexp-tp
  68.39/68.39 ms. The overlapped replicated form hides its shared expert in the half-sum exchange wait at R6; the TP
  partial must be in the message, so it sits on the critical path: R1 -0.5 ms vs :shexp (= unset), R6 +2.3 ms.

### q_b, o_a, o_b (attention projections; group attn)

Pack linears: `layers.L.attn.wq_b` (1280 -> 32768), `layers.L.attn.wo_a.slice.<g>` (g = 0..7, each 4096 -> 1024, the
group's input is columns [4096 g, 4096 g + 4096) of the attention output), `layers.L.attn.wo_b` (8192 -> 5120).
Every path (tensor-parallel attention or not, DeviceForward and BatchForward) uses the SAME per-rank linears below; a
process without TP attention runs both ranks' linears itself, so its bits equal the TP pair's.

- **q_b**: rank r's linear is the row slice rows [16384 r, 16384 r + 16384) of wq_b (tiles' columns [1024 r, 1024 r +
  1024), svh[16384 r, +16384)), x = the q RMS-normed F32 q_lora row ([R][1280]), y = q heads [32 r, 32 r + 32)
  ([R][16384]). No sum. (Measured on L10: these rows are also bitwise the corresponding rows of the whole tensor; not
  part of the identity.)
- **o_a**: rank r runs groups g in [4 r, 4 r + 4), each its own linear wo_a.slice.<g>, x = columns
  [4096 (g - 4 r), +4096) of the rank's F32 attention output o ([R][16384], strided: copied into staging), y = columns
  [1024 (g - 4 r), +1024) of o_low ([R][4096]). No sum. Submission sizes (v12's split count for 4096 x 1024 depends on
  the row count: 16 K-splits at 1, 4, 8 rows, 32 otherwise): every o_a call has 1, 4 or 8 rows; a chunk of 2-3 rows is
  padded to 4, 5-7 rows to 8, with copies of its last row (outputs of the pad rows discarded). Rows of 1-, 4- and
  8-row calls are bitwise equal (bench, both rank slices, L10).
  Staging is not part of the identity: `STRIX_DS41_TRELLIS_OA_GROUP=1|2` (default off, ds41_trellis_group.hpp) submits
  the same library calls with the same row counts and pad rows, but a 1-row chunk reads o / writes o_low in place and a
  longer chunk stages all groups (and pad rows) with one kernel into [G][m][4096] and scatters the real rows back with
  one kernel (mode 2: the groups' calls on concurrent streams). Same bits (bench: 296 rows, R 1..16,
  4 (layer, rank) sets, 0 differ). A future grouped library GEMV is used
  only if its rows equal this path bitwise at R 1..8 (startup self-check).
- **o_b**: rank r's linear is the K-half columns [4096 r, 4096 r + 4096) of wo_b (tiles' rows [256 r, 256 r + 256),
  suh[4096 r, +4096), svh whole), x = the rank's o_low ([R][4096], F32, no A8), output the partial product h_r
  ([R][5120]). Sum: y = h0 + h1, one FP32 add per value in that order (the existing add_halves after the same 20 KB
  half-sum exchange; one process: h0 into y, then y = y + h1 by accumulate, the same bits). The partials are
  numerically, not bitwise, the whole tensor's product (L10: max |h0 + h1 - whole| 7.6e-6 at max |y| 7.4).
- Under the flag the A8 quantization of these sites' inputs is skipped (STRIX_DS41_ATTN_A8's A8 of o / o_low is unused
  by a trellis o_a / o_b; a Q8 o_b after a trellis o_a quantizes its input itself).
- Step identity strings: `ds41-trellis-k5-v1:q_b`, `:o_a`, `:o_b`.

### head (LM head; group head)

- Weights: the pack's `head` (5120 -> 129280, K5, 414 MB). Split vocabulary as today: rank r computes the logits of
  vocabulary rows [r * 64640, (r + 1) * 64640) (ds41::vocab_slice(129280, 2, r); 64640 = 505 x 128, so the slice is
  128-aligned and H_O's blocks stay inside it) from its own **row slice** of the pack's head (TrellisSlice rows: tiles'
  columns [r * 4040, (r + 1) * 4040), svh of those rows, suh whole; 207 MB per rank instead of 269 MB c-u6 / 352 MB
  Q8_0). World 1 (one process, both ranks' work) prepares both slices and runs one call per slice, so its logits are the
  two ranks' bits.
- x = the final hc collapse + RMSNorm output (m.h, F32 [R][5120]; the same values in the fused hc site and the
  unfused / op-hash path); no A8. logits[row][slice] = L(head slice, x[row]) through enqueue_trellis_linear (rows in
  one strix_gemv, R <= 8; the row stride of the logits is 129280, so the output goes through the staging copy).
- Argmax, candidate exchange, logits gather (gate runs) and device_logits (requant gate) unchanged: they read the
  same [row][vocab] buffer.
- R-invariance (v12): per row identical for GEMV R = 1..8, rows reversed, rows past R holding NaN, both rank slices
  and the whole head (a slice's rows are bitwise the matching rows of the whole tensor for this shape), repeats
  (strix-hip-ds41-trellis-head-bench, 1276 comparisons; the R-invariance run above agrees). Submission sizes {1..8}.
- Not covered: DSpark's draft head (its own Q8_0 copy of output.weight, CPU / HIP drafter) stays Q8_0: it only proposes
  tokens; verification uses this head.
- Step identity string: `ds41-trellis-k5-v1:head`.
- Adapter form `STRIX_DS41_TRELLIS_HEADOUT=1` (default off; same identity, same per-row bits; DeviceForward and
  BatchForward): R1 writes the library output straight into logits (one row is contiguous whatever the stride); R > 1
  writes a per-slice rows buffer [slice][8][64640] (contiguous rows, no staging copy) that argmax, the batch slice
  download and `DeviceForward::device_logits_rows` read (device_logits() then holds no rows of that step; batch gate
  runs with gather / world-1 kept logits copy the rows into logits); R5..R7 are submitted as 8 rows in place (rows
  R..7 of m.h and of the rows buffer are scratch). Exact because the head is one R class and its rows are independent
  within a call, rows past R holding anything (NaN included).

## Mixed K per class (round 4c): STRIX_DS41_TRELLIS_MIX

- Spec: `STRIX_DS41_TRELLIS_MIX="class=dir,...[,default=dir]"` or a manifest file (one `class=dir` per line, `#`
  comments). Classes: q_a, kv, q_b, o_a, o_b, shexp_w13 (shared gate + up), shexp_w2 (shared down), shexp (both), head.
  STRIX_DS41_TRELLIS, if also set, is the default for the classes the mix does not name; a class with no pack and no
  default fails at load. STRIX_DS41_TRELLIS_SITES still selects which hooked sites run trellis (q_a / kv have no hook:
  they keep today's form even when the mix names a pack for them). The storage line prints the mapping.
- Every K is the same library (v12), the same calls and the same per-shape submission sizes, keyed on (I, N, K).
- R classes per (shape, K), v12 (one node; layers 0 / 20, rows independent within a call everywhere):

| shape (I -> N) | site | K3 | K4 | K5 |
|---|---|---|---|---|
| 5120 -> 1280 | q_a | 1..8 | 1..8 | 1..8 |
| 5120 -> 512 | kv | 1..8 | 1..8 | 1..8 |
| 1280 -> 16384 (rank rows) | q_b | 1..8 | 1..8 | 1..8 |
| 4096 -> 5120 (K half) | o_b | 1..8 | 1..8 | 1..8 |
| 4096 -> 1024 | o_a group | 1..8 | 1..8 | {1,4,8} / rest |
| 5120 -> 2304 | shared w1/w3 | 1..8 | 1..8 | R1 / R2..7 / R8 |
| 2304 -> 5120 | shared w2 | 1..8 | 1..8 | R1 / R2..8 |
| 5120 -> 1152 (TP rows) | shared w1/w3 TP | 1..8 | 1..8 | 1..8 |
| 1152 -> 5120 (TP cols) | shared w2 TP | 1..8 | 1..8 | 1..8 |
| 5120 -> 64640 (rank rows) | head | 1..8 | 1..8 | 1..8 |

  So only K5 needs padding (the table of trellis_submit_sizes); the entry point is bitwise vs every row alone for
  R = 2..8, 13, 32 and strided + accumulate on all 46 tensors of the run. Microbench (entry point, warm): K3 / K4 run
  at about the K5 times (e.g. shared w1 R1 32.6 / 32.1 us vs K5 38.3 padded; q_b rank rows R1 49.1 / 48.2 vs 51.6;
  head rank slice R1 830 / 889 us vs 1014): v12 has no K3/K4-tuned kernels, so fewer bytes give little time.

### ds41-trellis-mixA-v1

Mapping (manifest mixA.txt, SHA256 7c0af09f...): q_b, o_a, o_b, shexp_w2 = ours-K3;
shexp_w13, q_a, kv = ours-K4; head = ours-K5 (pack SHA256 manifests ours-K3/K4/K5.sha256 on both nodes). In the engine
with STRIX_DS41_TRELLIS_SITES=q_b,o_a,o_b,shexp,head: q_b / o_a / o_b / shared down K3, shared gate + up K4, head K5;
q_a / kv stay c-u6 (no trellis hook). Everything else as ds41-trellis-k5-v1 (F32 activations at the trellis sites,
GEMV only, the same rank slices and sums, device R1 = reference).

### ds41-trellis-mixB-v1 (production candidate)

Mapping (manifest mixB.txt, SHA256 f4ce1bf8...): every dense class = ours-K4 (`default=`), head =
ours-K5. With STRIX_DS41_TRELLIS_SITES=all (+ STRIX_DS41_TRELLIS_HEADOUT=1, STRIX_DS41_TRELLIS_OA_GROUP=1): q_b / o_a / o_b
/ shared gate, up, down K4, head K5; q_a / kv keep today's form. Every K4 shape is one R class (table above), so no
padding at the K4 sites; the K5 head is one class too. Otherwise as ds41-trellis-k5-v1.
Equivalent spec without a file: `STRIX_DS41_TRELLIS_MIX="default=<K4 dir>,head=<K5 dir>"`, or
`STRIX_DS41_TRELLIS=<K4 dir> STRIX_DS41_TRELLIS_MIX="head=<K5 dir>"`.
- Gates (pair, LAYOUT 1, c-u6, P16 defaults incl. expert TP, binary of c4d142c):
  no trellis env == the expert-TP CPU host reference tp-cu6-L40 on both ranks (21 steps, 14,406 op hashes, 0 differ);
  mix-B with SITES=shexp,head (+HEADOUT) and SITES=all (+HEADOUT, +OA_GROUP): fill windows 6,8,5 with rejections
  (keep 2,5,1,3) give the same pos lines (tokens + logits FNV, 21) as the R1 fill on both ranks; op-hash lines differ
  by 2,520 between the 1-row and the window fill for every set INCLUDING unset (the window-buffer artifact, not trellis).
  BatchForward SITES=all mix-B: batch == alone N = 1,2,4,8 (662 rows, tokens + full logits) and EXTERNAL PASS (256 rows
  of p0 / p2 == DeviceForward alone R1, same identity). Storage (both ranks): 201 trellis sites, 1997 MiB.
- Timing (pair, c-u6, P16 defaults, untraced walls, 12 windows of 1 and of 6 rows, 36 decode
  steps; two reps): no trellis R1 36.85 / 36.94 ms, R6 63.67 / 63.49, decode 36.6 / 36.8 ms/token.
  shexp,head + HEADOUT: K5 R1 37.35 / 37.24, R6 62.41 / 62.44, decode 36.9 / 36.9; **mix-B R1 36.33 / 36.18, R6 63.82 /
  63.83, decode 35.9 / 36.1** (no R1 padding at K4: -1.1 ms vs K5 at R1, +1.4 at R6).
  all + HEADOUT + OA_GROUP: K5 R1 38.00 / 38.34, R6 65.27 / 65.44; mix-B R1 36.94 / 36.85, R6 68.30 / 68.29 (R6 +3 ms vs
  K5, +4.7 vs no trellis: the v12 K4 forms of the attention sites are slower at R6 than K5's; e.g. o_a group K4 R6
  29.2 us vs K5 20.7 us (K5 pads R6 to its fast 8-row split-16 form)).
