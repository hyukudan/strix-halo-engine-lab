# Teacher-forced scoring over the tiled prefill

`--teacher-forced-score` (with or without `--teacher-logits-file`) runs on the serial eager route, or on the
tiled known-prefix prefill when the run also selects it:

```
--known-prefix-window --known-prefix-full-rows --known-prefix-prefill-rows T
--known-prefix-output-policy all-rows          (the default; terminal-row and state-only are rejected)
```

plus the usual tiled-prefill budgets (`--known-prefix-workspace-budget-bytes`,
`--known-prefix-host-metadata-budget-bytes`, `--known-prefix-uma-budget-bytes`,
`--known-prefix-host-headroom-bytes`). Any other known-prefix selection with teacher forcing is rejected
at option parsing. `--generate 1` and every other teacher-forced restriction are unchanged. As for any
known-prefix route with a numerical profile, the run states the full v5 organization flags explicitly
(a bare `--numerical-profile` is rejected there); the serial route resolves the same organization
automatically, and both produce the same logits.

## Execution

- Every complete T-row block runs with every row's LM head (no block is headless). The head of a block
  runs as the multi-row prefetch Q8_0 head in packs of eight rows when the serial head is the prefetch
  Q8_0 head (`--prefetch-q8-lm-head`, resolved by the automatic profile): one weight read per pack, and
  per row the leaf products, lane tree and local tree of the R1 head, so every row's logits are the R1
  head's bits. Other heads keep the R1 launcher per row. The receipt names the choice
  (`teacher_forced_tiled_lm_head`).
- After a block commits, its T full FP32 rows are downloaded with one strided copy from the block arena
  (the logits stripes already exist there; no device memory is added) into one pinned host staging
  buffer of T x vocabulary floats, retained for the run and counted in the prefill runtime payload. No
  prompt-sized buffer exists on either side.
- The rows are scored like serial rows: full-vocabulary finiteness check and the same requested-logprob
  arithmetic (`compute_requested_logprobs`, shared with the serial consumer). Rows are scored in parallel
  host threads; each row's arithmetic is unchanged. They are appended to the score ledger in position
  order before the next block starts; the block's spool append (position order, one writer) runs on a
  background thread while the next block computes, and is drained before the staging buffer is
  overwritten, before any serial row and before the receipt. A final teacher input inside a block
  completes the spool without a row. `STRIX_PREFILL_TIMING=1` prints the download, scoring and spool
  times of every block.
- The unpadded suffix (fewer than T inputs) runs as serial rows through the same consumer as the serial
  route.

The logits file and the `requested_logprobs_rows` have exactly the serial route's format, order and footer
(`TMLLOG01`/`TMLEOF01`, see the teacher-forced section of
[optimization-profile.md](../decode/optimization-profile.md)).

## Numerical profiles

Rows of complete blocks carry the arithmetic of the prefill profile (`--prefill-numerical-profile`, default
`--numerical-profile`); serial suffix rows carry the decode profile, as on every tiled route (see
[prefill-numerical-profile.md](prefill-numerical-profile.md)). With a single profile the two are the same.
With a split pair a file holds both unless T divides the number of inputs. The receipt records
`teacher_forced_route`, `teacher_forced_tiled_input_rows`, `teacher_forced_serial_input_rows`,
`teacher_forced_tiled_rows_numerical_profile` and `teacher_forced_serial_rows_numerical_profile`, next to
the existing prefill/decode profile identities.

## Exactness

Within one numerical profile the tiled prefill reproduces the serial rows bit for bit, and the multi-row
head reproduces the R1 head per row, so the logits file is byte-identical to the serial route's file for
the same inputs. This is checked on full-model runs, not inferred; see docs/decisions/decision-register.md (D-034).
