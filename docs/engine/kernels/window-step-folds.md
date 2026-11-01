# Verify-window step folds

Opt-in folds of short dependent steps of a verify window (2, 4 or 8 rows) into the launch of a neighbouring
step. Each removes one or two dependent dispatches per layer; each defaults off and is selected by its own
environment switch. Prompt prefill, single-row decode and the MTP draft are unchanged.

| switch | steps | launches per fold |
| --- | --- | --- |
| `STRIX_WINDOW_INJECT_NORM=1` | inject rows → HC rows (attention→MLP mix, MLP→next attention mix, last MLP→final mix) | inject + HC norm → one |
| `STRIX_WINDOW_COMBINE_INJECT=1` | MoE combine → MLP inject → next HC rows | combine + inject + HC norm → one |
| `STRIX_WINDOW_COMMIT_COMBINE=1` | routed experts → MoE combine | routed commit + combine → one |

The window executor's step hook launches the fold at the first step and records the folded steps as
completed without a launch. A fold whose views do not chain, or alias in a way the fused kernel cannot
order, is declined before anything is enqueued and the steps launch separately.
`STRIX_WINDOW_FOLD_VERBOSE=1` prints each decline and its reason.

## Arithmetic

Every value keeps the expression of the separate launches:

- MoE combine: `shared + routed`, stored in the activation format.
- Inject: `streams + hidden * gate` (the stored combined row when the combine is folded), stored in the
  activation format. The HC step reads exactly these stored streams.
- HC normalisation: the 256-leaf sum of squares per (row, stream), the same reduction tree,
  `rsqrtf(sum / hidden + epsilon)` and `value * inverse * weight`. The split-K projection and the output then
  run as before.
- Routed commit: unchanged; the combine reads the committed routed value (or, after a routed failure, the
  routed row as it is, as the separate combine would).

Every two-operand FP32 step of the folded kernels is operand-pinned (inline `v_add_f32` / `v_fmac_f32` /
`v_mul_f32`) to the instruction of the separate launch, as read in the gfx1151 ISA: the combine adds
`shared, routed`; the inject is `v_fmac hyper += gate * hidden`; the norm leaves `v_fmac sum += value * value`;
the tree adds the upper operand first; the normalised value multiplies `inverse, value` (FP32 streams:
`value, inverse`) and then the weight; the mean divides by the runtime width. Operand order matters only when
two NaNs meet (which payload survives), so the folded bytes equal the separate launches for any input, NaN
payloads included.

## Ordering and aliasing

- Inject + norm: one workgroup per (row, stream). The streams written may replace the same row's current
  streams element by element; any other overlap between a written row and a folded input declines.
- Combine + inject + norm: one workgroup per row. Stream 0's threads compute the combined row before a
  workgroup barrier, so the streams and the combined row may replace the same row's shared or routed rows.
  Overlaps across rows decline. With the canonical window arena this fold is mostly declined (another row's
  streams overlap a row's shared output); the inject fold then still applies.
- The HC workspace (normalised rows, low values) is dead after its step. When the arena gives it the memory of
  a folded input, the fold uses an owner-held per-stream workspace instead; the outputs are unchanged and only
  the dead workspace bytes differ.
- Commit + combine: each output element's thread reads its shared and routed values and the accumulation
  before storing the combined value. A combined row may replace its own row's shared or routed row exactly.
  When the shared expert runs on the second stream, the owner's stream joins it just before the commit
  instead of before the combine.

## Fixtures

- `strix-hip-known-window-tests --window-folds` (ctest `strix-hip-known-window-fold-tests`): inject + HC and
  combine + inject + HC folds at 2/4/8 rows in the canonical window arena equal the separate rows launches
  over the whole arena (the HC workspace masked). A declined combine fold is checked through its fallback.
  The same cases then run over NaN-payload corpora (one value in eight special, every value a NaN with its own
  payload, NaNs of different payloads in different leaves of one stream, NaN norm weights; NaN gates, shared and
  routed values) and compare every byte, the HC workspace included wherever the fold put it; then
  `known_moe_commit_combine_selftest` checks the commit + combine kernel against the commit and combine kernels
  on rows of any bits, row and header failures and in-place combined rows (every byte, CSR header and D1 record).
- `strix-hip-moe-known-rows-tests` (all profiles): the routed rows launch with the combine folded into its
  commit leaves the arena as the routed launch does and writes BF16(shared + routed) per element.

## Status

Recommended together: `STRIX_WINDOW_INJECT_NORM=1 STRIX_WINDOW_COMMIT_COMBINE=1`. With the canonical arena the
combine + inject fold is nearly always declined and its attempts only add host work; leave it off.


Opt-in. Tokens and MTP replay (error 0) are identical on complete decode runs with each fold and with all
three. The throughput effect is small (a few dispatches per layer) and is to be measured on the complete
workload before any becomes a default.

## Combinations

- `STRIX_HC_PERSISTENT=1 STRIX_HC_PERSISTENT_INJECT=1` (persistent HC with the inject folded in) and the inject
  folds above both want the inject step. A window fold that has already taken an inject step (combine + inject)
  leaves it to the fold; otherwise the persistent fusion is tried first and the norm fold only when it declines.
  Each is qualified alone; the combination is exact by construction (each path equals the separate launches)
  but not measured.
- The commit + combine fold keeps the verifier's active rows (`STRIX_MTP_ACTIVE_ROWS=1`, D-041): rows outside
  the expert union commit a zero routed row, and the folded combine stores `shared + 0` for them as the separate
  combine would.
