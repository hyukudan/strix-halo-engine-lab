# Prompt groups: several requests' prompt blocks in one prefill run

Opt-in (`STRIX_PROMPT_BATCH=1`) for the batched tiled MTP route (`--http-batch N`, cohort requests on the tiled
prompt with `STRIX_BATCH_INTERLEAVE=1`). Without it every prompt unit runs alone, as before.

## What runs together

A request's prompt runs as a sequence of units: T-row blocks, tail blocks (`--known-prefix-tail-rows`), then serial
rows. The units of one request and their boundaries do not change. When the oldest running prompt's next unit is a
tiled block, the next blocks of the other running prompts join it, oldest first, while their rows fit the row
budget: the rows of the T-row block owner, or `STRIX_PROMPT_BATCH_ROWS` when it is smaller. With
`STRIX_PROMPT_BATCH_ROWS` above T, setup builds one more tiled owner of that many rows (its own arena) that only runs
groups. A group of two or more blocks runs as one segmented prefill; a lone block or serial rows run alone. The
decode cycle of the requests past their prompt follows each prompt step as before (prefill and decode are not fused).

## Segmented prefill

`execute_known_prefix_prefill_segments_eager` runs the blocks of several sessions over consecutive rows of one tiled
arena. Segment s owns rows `[offset_s, offset_s + rows_s)` (multiples of 8) with its own session, state memory,
positions, tokens, PLE lookup and history, terminal head and lineage. Every step runs in the step-major order of a
tiled block:

- Steps whose arithmetic per row depends only on that row's input and the weights run once for the rows of every
  segment through their full-T entry, inside a cohort scope (per-segment session and position rules): HC mixers,
  router, routed expert union (one CSR over all rows; each (row, output) sums in slot order whatever batch it lands
  in), inject and MoE combine. `STRIX_PROMPT_BATCH_SHARED` lists them (`hc`, `router`, `moe`, `elementwise`,
  `shared` for the shared expert's dense projections, `all`, or `none`; default `hc,router,moe,elementwise`).
- Every other step runs per segment exactly as that segment's block alone: the same entry, row count, tile
  boundaries and order (GDN projections and recurrence, QSA projections, KV, selection and attention, PLE, the
  shared expert, row loops and the terminal head of the segment's last row).

Each segment is admitted as its block alone before anything is enqueued; a failure after that poisons every
segment's session. The lineage of a segment records the producer of the session's own block of that size (the
per-row arithmetic it repeats). The routed union of a group of R < T rows uses its own CSR workspace of R rows.

Exactness: every request's state after each block, its terminal logits, the predictor rows and every generated
token are those of the request alone (`STRIX_PREFIX_CACHE_DIGEST=1` adds a SHA-256 of the final prompt row's logits
to the prompt receipt next to the state digest).

## Switches

| Switch | Effect |
| --- | --- |
| `STRIX_PROMPT_BATCH=1` | Group the running prompts' blocks (default off) |
| `STRIX_PROMPT_BATCH_ROWS=N` | Row budget of a group; above T, a dedicated group owner of N rows (multiple of 8, up to 8192) |
| `STRIX_PROMPT_BATCH_SHARED=list` | Families run once for all segments (default `hc,router,moe,elementwise`) |

Receipts: each request's tiled prompt receipt counts `grouped_blocks`; the cohort receipt adds `prompt_batch`
(groups, grouped blocks and rows, group and prompt time).
