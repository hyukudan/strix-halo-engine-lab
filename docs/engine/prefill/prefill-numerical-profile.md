# Prefill numerical profile

`--prefill-numerical-profile NAME` selects the arithmetic of the prompt prefill separately from
`--numerical-profile`, which keeps selecting the arithmetic of decode: serial target rows, MTP target
windows and their checkpoints. Without the flag the prefill runs `--numerical-profile`, and every route,
token and existing receipt field is unchanged. An explicit value equal to `--numerical-profile` is the
same single-profile run.

## Admitted pairs and routes

A split pair is admitted only where it runs with existing kernels:

| Decode (`--numerical-profile`) | Prefill (`--prefill-numerical-profile`) |
| --- | --- |
| `hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup-v1` | `hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1` |
| `hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1` | `hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup-v1` |
| `hc-deferred-norm-f32+q51-q80-a8-down+q5k-a8-gateup-v1` | `hc-deferred-norm-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1` |
| `hc-deferred-norm-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1` | `hc-deferred-norm-f32+q51-q80-a8-down+q5k-a8-gateup-v1` |

The two profiles of a pair differ only in the dense Q8_0 projections (GDN, QSA, shared expert): FP32
activations or int8 activations per group of 32. Their HC, MoE and LM-head kernels, arena layout and
state layout are the same, so a state produced by one is an input state of the other. The decode
profile must be explicit. One of three prefill routes is required:

- the tiled known-prefix prefill: `--known-prefix-window --known-prefix-full-rows
  --known-prefix-prefill-rows T`. Every complete T-row block runs the prefill profile. The unpadded
  suffix of fewer than T rows and generation run the decode profile, as serial rows;
- the MTP prompt (`--mtp`): every prompt row runs the prefill profile. MTP replay, draft cycles,
  target windows, checkpoints and generated rows run the decode profile;
- the tiled MTP prompt (`--mtp` with the tiled prefill flags above), the route of `--http-serve`, its
  batched slots (`--http-batch`) and cohort requests on the tiled prompt. Every complete block (and tail
  block) runs the prefill profile. The calibration rows, the serial suffix, MTP replay, draft cycles,
  windows and generated rows run the decode profile.

Model copy tiers, GDN islands, prepared or overlapped serial execution and the MTP
metadata diagnostic are rejected with a split pair. Every other admission rule of `--numerical-profile`
applies to both profiles. Other pairs fail at option parsing.

## Execution

The engine compiles a second serial execution for the prefill profile from the same binding and buffer
requests with only the dense A8 flag changed. Its dense projection kernel IDs carry (or lack) the
`.dense-a8-v1` marker, so the tiled prefill plan derived from it selects the matching known-window
verify IDs, and its launchers are configured from the prefill profile. The two executions are checked
to be siblings (same steps, dependencies, state access, buffer IO, arena layout and geometry; different
execution and kernel-plan identities).

A session is owned by one executor at a time. The prefill execution runs on the serial scratch arena of
the decode execution under its own execution identity (the layouts are identical and the two never run
concurrently), so the split needs no additional device memory; the extra cost is host-side: a second
compiled plan, and on the MTP route a second launcher catalog and launch plan. Before a prefill block (or
the MTP prompt) the session is handed to the prefill execution and afterwards back to decode
(`hand_off_serial_session`): only the executor identities change. The arithmetic lineage keeps the
producer identity of every row, so a session prefilled under a split pair can never be taken for a
decode-only serial history (prefix caches require one).

On the MTP route the target producer runs the prompt rows on the prefill execution. After the replay the
pending prompt row is re-minted with the handed-off session; the MTP chain check admits exactly that one
executor change and still proves the exact lineage continuation. MTP exactness holds from the handoff
state: replay against a golden dumped with the same pair has error 0, and every generated token is
verified by the decode target from the state the prefill produced. It does not reproduce the tokens of
a single-profile run from the start.

On the tiled MTP prompt the target producer never holds the prefill execution. The calibration replay
runs its rows on the decode execution, so its golden records the decode profile for both phases (the
golden of a single-profile run with the decode profile is the same file). After the calibration the
prompt starts from a fresh state: each block and tail backend hands the session to the prefill execution
around its block, exactly as on the tiled route; the serial suffix runs through the target producer on
the decode execution, and when a block ends at the last prompt row the target adopts that row (hidden
state and logits) on the decode execution. Every generated token is verified by the decode target from
the state the fast blocks produced, with the same tokens as serial decode from that state. Every batched
slot builds its own sibling view of its serial scratch, its own prefill owners (prefill launcher
configuration over its own state, PLE and launchers; the prefill arenas are shared, one prompt runs at a
time) and its own handoffs. Each slot's block owner holds a copy of the plan metadata (about 286 MB at
T = 8192); the setup's UMA growth estimate counts one per slot.

## Identity

- Receipt: `prefill_numerical_profile`, `prefill_numerical_profile_source` (`explicit` or
  `numerical-profile`) and `numerical_profile_pair_identity` in every run; with a split pair also
  `prefill_dense_a8_configured`, `prefill_serial_execution_identity` and
  `prefill_serial_kernel_plan_identity`, and `optimization_profile_selected` gains `+prefill:NAME`.
  MTP receipts add `target_prefill_numerical_profile`, `target_decode_numerical_profile`,
  `target_prefill_execution_identity` and `mtp_golden_target_profile_pair_recorded`; the tiled MTP
  prompt's `mtp_tiled_prompt` object adds `block_numerical_profile` and `serial_numerical_profile`.
- `numerical_profile_pair_identity` of a single-profile pair is that profile's identity; a split pair
  has a distinct identity that also differs from the reversed pair.
- MTP goldens (version 2) record the prefill and decode profile identities; see
  [the golden format](../speculative/mtp-forward-golden-file.md). A golden dumped under one pair is rejected under
  another before any target work; a version-1 golden is admitted only for a single-profile run.

## Fast prefill profile (draft)

Quality at long context (D-036): against the exact FP32-dense profile, the default components are closer than
dense A8 at 16K and 32K. KL is 15-21 % lower than dense A8 on wiki and code documents, in every position bin.
Long-document retrieval at 8K/16K/32K is unchanged: 116/120 correct, the same as FP32 dense and dense A8.

`fast-prefill-f16wmma-v1` is a prefill-only profile (rejected as `--numerical-profile`). It is admitted
only with the decode profile `hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1`, on the same two
routes and with the same rejections as the pairs above. Its prefill execution is compiled from the decode
profile's binding and buffer requests (same kernel ids, steps, arena and state layout), so the handoff
state is an ordinary decode input state; `prefill_arithmetic_identity` folds the pair identity into the
prefill execution's kernel-plan and execution identities, so the two executions remain distinct siblings.
The prefill launchers are configured with the profile's components, recorded as `prefill_fast_components`
and selected by `--prefill-fast-components hc,dense,moe,qsa,router,dense8` (any non-empty subset, default hc,dense,moe,qsa,router; a subset has
its own pair identity):

- `hc`: the wide prefill HC steps run FP16 WMMA projections: down (10240 -> 320) with the injection
  sharing the input read, and up (320 -> 10240) with the sigmoid and four-stream mix in the epilogue.
  Effective Q8_0 weights (scale x code, exact in FP32) are rounded to FP16 by a pack kernel per call;
  the normalized rows and the low values are rounded to FP16 as operands; accumulation is FP32 over the
  whole width. The FP32 low values, the injection (FP32 inputs and weights, FP32 FMA chains) and the mix
  (stream order) keep FP32 arithmetic. Steps that do not take the wide path keep the decode profile's
  kernels. The READY `hc_output_wmma` default selects OUT16 on admitted wide calls;
  see the [guards and restoration controls](../../optimizations.md#bit-changing-prefill-options).
  `--prefill-exact-fp16`, `--prefill-disable hc_output_wmma` or the corresponding config/environment
  controls restore the FP32 lane-chain output on the projection's FP32 low values, including
  when `STRIX_HCW_OUTPUT=2` requests legacy FP16 WMMA output. The legacy selector is governed
  by the same enabled set. Blocks of up to 8192 rows. Diagnostic
  `STRIX_HCW_NORM16`: 1 = a fast-profile HC norm also writes the normalized rows in FP16 (padded pitch)
  for the projection; 2 = it writes only the FP16 rows and the lane-chain output's stream mix reads them
  too (the injection then runs on the FP16 values widened to FP32). Off by default.
- `dense`: GDN, QSA and shared-expert dense projections of prompt launches of 32..256 rows run
  FP16 x FP16 -> FP32 WMMA: activations rounded to FP16 into a scratch slot (finite values beyond the
  FP16 range are counted), Q8_0 blocks dequantized to FP16 in LDS by the projection. Launches of other
  row counts keep dense A8. The GDN output projection does not use the fused A8 quantizer.
- `dense8` (opt-in, not in the default set, exclusive with `dense`): the same projections, on 256-row
  launches where dense A8 takes its 4 x 2 LDS form, run A8-G64: activations in int8 per (row, 64-feature
  group) with the dense A8 quantizer's rules; each Q8_0 block pair requantized once per weight matrix to
  one scale `dp = max(|d0|, |d1|)`, `q' = clamp(rint(q * d_b / dp), -128, 127)` (exact where the two
  scales are equal), stored as 68-byte pairs (FP32 `dp`, 64 codes); int32 accumulation over each pair and
  one FP32 step `acc += (dp * delta) * dot` per pair, NaN canonical. Other launches keep dense A8.
- `moe`: the routed MoE of every complete tiled prefill block (the T-row CSR path) runs FP16 WMMA. Each
  Q4_K/Q5_K gate/up value `d*sc*q - dm*m`, Q5_1 down value `d*q + m` and Q8_0 down value `d*q` is formed in
  FP32 and rounded once to FP16 inside the kernel (no FP16 copy of the weights in memory). The BF16 input
  row is scaled by `2^s` (one `s` per prompt row, putting the row maximum in `[2^13, 2^14)`) and converted
  to FP16; `h = silu(g) * u` is computed and kept in FP32 and rounded to FP16 by the down kernel with a
  power-of-two scale per (row, slot) from its largest `|h|`. `v_wmma_f32_16x16x16_f16` accumulates over the
  whole K in FP32 (K steps of 64 ascending) with no per-group epilogue; the scales are undone exactly on
  the FP32 sums. Merge (route weights in slot order) and the BF16 commit are unchanged. A non-finite `g`,
  `u` or `h` fails the row closed (status 8). Each (row, output) sums in the same order whatever CSR batch
  (up to 64 assignments of one expert) it lands in, so results do not depend on T or on the other rows.
  Serial rows (the unpadded suffix, every MTP prompt row) and windows keep the A8 MoE. The kernels reuse the
  A8 routed-MoE scratch views. Receipt: `prefill_moe_f16wmma_rows` (prompt rows times layers that ran it).
- `qsa`: the sparse attention of the prefill groups (16, 32 or 64 rows of the tiled prefill; head size 256)
  runs over the union of the selections of `t` consecutive rows (`STRIX_QSA_FAST_TILE`, 4 by default). A
  union kernel lists, in ascending order, the 4-token blocks any of the `t` rows selected, each with a row
  mask, and the multiplicity of tokens 0..3 in each row's selection (the selection repeats them when fewer
  blocks than the budget are eligible). For each KV head and each 16-token chunk of the union,
  `v_wmma_f32_16x16x16_f16` forms the scores of the `t` rows times the query heads of that KV head: K is
  converted from BF16 to FP16 (exact in the FP16 normal range; larger magnitudes saturate at 65504), Q is
  rounded once to FP16 after a power-of-two scale per (row, head) that puts its largest `|q|` in
  `[2^13, 2^14)`, undone with `1 / sqrt(256)` on the FP32 scores. A (row, token) pair outside the row's
  selection or causal prefix gets score `-inf`; a repeated token's weight is multiplied by its multiplicity.
  The softmax is the online one in FP32 and base 2 (running maximum and sum per (row, head); the weights
  `2^(s - m)` are rounded once to FP16 and multiply V, converted like K, with FP32 accumulation; the
  accumulators are rescaled when a maximum grows). Slices of the union run in separate workgroups and are
  merged in FP32; the merged sum is divided by the merged denominator and the exact path's BF16 roundings
  and BF16 sigmoid gate follow. Unlike the exact path, the softmax weights are not rounded to BF16 and the
  score and value sums are in hardware order. The indexer, block selection, KV store and output projection
  are unchanged; serial tiles and MTP prompt rows keep the exact attention. The kernels use the rows'
  attention-score workspace as scratch; a shape it does not cover keeps the exact kernels.
  `STRIX_QSA_PREFILL_SELFCHECK=1` reports, per prefill call, the error of the attention outputs against the
  serial exact tiles.
- `router`: the router logits of the tiled prefill blocks (BF16 activations) run `v_wmma_f32_16x16x16_bf16`.
  Each FP32 router weight `w` is split into two BF16 terms, `h = bf16(w)` and `m = bf16(w - h)` (round to
  nearest even; `h + m` keeps 16 significant bits, relative error at most `2^-17`); the BF16 activations are
  exact operands, so each product is exact, and the dot products accumulate in FP32 (K steps of 32, term
  `h` then `m`). Top-k selection, normalisation and everything after the logits are unchanged; windows,
  serial rows and FP32 activations keep the exact router.

The components act only in complete tiled known-prefix blocks: the serial suffix of the tiled route and every prompt
row of the MTP prompt route run the serial kernels, where the profile changes the identities but not the arithmetic.
On the tiled MTP prompt (the server) the blocks run the fast arithmetic and the suffix the decode profile, as on the
tiled route.
QSA indexer and selection, GDN recurrence, PLE and LM head keep the decode profile's arithmetic (and the
QSA attention and router logits without `qsa` and `router`).
Tokens differ from a dense-A8 run; exactness is defined from the handoff state as for every split pair.

Qwen `hc_output_wmma` is default ON. `hc_output_wmma` passed a pre-declared confirmation holdout; its NLL endpoint is measured against a rounding-equivalent control whose per-case movement is capped at +-0.02 nats/token (a bounded control allowance, not an unbiased estimate of the uncapped difference); the absolute NLL change vs the exact reference also passed (-0.37 % NLL/token, upper bound +0.38 %). The off switch restores the exact output path.
Controls are server-level only: `STRIX_PREFILL_HC_OUTPUT_WMMA=0`,
`--prefill-disable hc_output_wmma`, or config `prefill_disable = ["hc_output_wmma"]`;
master `STRIX_PREFILL_EXACT_FP16=1`, `--prefill-exact-fp16`, or config
`prefill_exact_fp16 = true`. Restart the server; request-level controls return HTTP 400.
Fast HC with this option is excluded from MTP and nested admission pending qualification;
ordinary MTP/HTTP profiles retain their existing arithmetic. OFF restoration compares logits
and seven state payloads to public main; compiled, lineage and cache identity bytes may differ.

Validation-only split prefill can export `--validation-final-state DIR` after eager
generation; the manifest records the prefill execution and option identity. This copies
all seven allocations and invalidates timing. For unmodified public-main references,
`tests/qwen_prefill_public_controls_gate.py --state-format session` uses `--session-save`:
all committed KV/index rows plus the five complete fixed components and token frontier.
It verifies every payload hash and the envelope root while allowing identity metadata to differ.
Unused cache capacity is outside that committed-state comparison; the final-state export
checks complete allocations separately.
