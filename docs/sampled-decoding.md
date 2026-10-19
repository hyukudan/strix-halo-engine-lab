# Sampled decoding with speculative drafts

Status: opt-in on the explicit MTP route. Temperature 0 (the default) is the unchanged greedy path: no sampler
object exists, no logits are copied to the host and every selection is the finite first-max token.

## Options

| Option | Meaning |
| --- | --- |
| `--mtp-sample-temperature T` | `T > 0` enables sampling (identity `sampling-v1`); 0 keeps greedy |
| `--mtp-sample-top-k K` | keep the K most likely tokens (0 = off) |
| `--mtp-sample-top-p P` | keep the shortest most-likely prefix holding mass P (1 = off) |
| `--mtp-sample-min-p P` | keep tokens with probability at least P times the maximum (0 = off) |
| `--mtp-sample-seed S` | seed of the counter-based random stream |
| `--mtp-sample-speculation on\|off` | `off` proposes no drafts: plain sampling from the target, one token per cycle |
| `--mtp-sample-schedule fixed\|adaptive` | `fixed` (default) chooses windows and draft lengths from draft probabilities only; `adaptive` also uses measured cycle times (faster on average, not reproducible) |
| `--mtp-sample-repeats N` | N independent requests over one loaded model, seeds S..S+N-1, one receipt each (distribution tests) |

Speculative sampling needs `--mtp-draft-chain`. The receipt has a `sampling` object: the policy, its identity,
and the number of drafts, plain decisions, accepted and rejected drafts of the request.

## Distribution (`sampling-v1`)

For one row of FP32 logits, in this order: `z = (logit - max) / T` in double; drop `z < -40` (at most
`vocabulary * e^-40` of the mass, about 1e-12 here) and, with min-p, `z < ln(min_p)`; top-k by `(z desc, id asc)`;
weights `exp(z)` summed in token id order; top-p keeps the shortest `(weight desc, id asc)` prefix reaching
`top_p` of the total; normalize. Sampling walks the kept tokens in their stored order (id ascending, or the top-p
order) against one uniform. The draft distribution `q` is the same transform over the draft head's rows; with a
draft vocabulary subset `q` is zero outside the subset.

## Decisions

Every emitted token is one decision at an absolute position `n`:

- no draft for `n`: sample `p` with uniform `U(n, plain)`;
- draft `d ~ q` (drawn with `U(n, draft)`): accept when `U(n, accept) * q(d) < p(d)`; otherwise sample
  `normalize(max(0, p - q))` over the target's kept tokens with `U(n, residual)`. A rejected decision never
  returns `d`, and the residual reaches tokens outside the draft vocabulary.

Uniforms come from the existing counter-based generator keyed by seed, policy identity, position and draw class,
so a position draws the same values whether it is decided in a window row, a serial step, or the serial replay of
a restored window. The output at each position is distributed exactly as `p` given the preceding tokens, for any
draft distribution.

## Where decisions happen

- Draft chain: with a sampler every draft step completes, its compact logits are copied to the host, the draft is
  drawn from `q` and written into the device token slot that feeds the next step. Draft `k` of a cycle proposes the
  token at `base + k + 2` (`base` is the cycle's pending row).
- Target rows: the finite first-max pass still validates every row. A serial row then copies its logits and
  decides. A window attaches a row selection to its capture: after the window completes and before any row is
  published, rows are decided in order from row 0 until the first row without a draft, a rejection, EOS or the
  budget. Prefix acceptance is unchanged: row `r + 1` is kept only if its input (the draft) equals row `r`'s
  decision, which for a drafted row means the draft was accepted.
- Prompt: only the final prompt row decides by sampling; earlier rows keep first-max selection because their
  predictions are never emitted.

## Device decisions

Sampled rows are decided on the GPU by default (`src/hip_token_sampling_device.hip`): target rows (window rows and
serial rows) and the draft chain. `STRIX_SAMPLE_PATH=host` (or `STRIX_SAMPLE_HOST=1`) restores the host path
everywhere, `STRIX_SAMPLE_PATH=device` forces the device path, and `STRIX_SAMPLE_DRAFT_HOST=1` keeps only the draft
chain on the host. The default (`auto`) is the device path for every policy: it decided faster than the host for
each kind of decision (window rows, serial rows, draft steps) with every filter combination measured.

- Same transform: the finite check, `z = (logit - max) / T` in double, the cutoff, top-k by `(z desc, id asc)`
  (radix selection on the logit's order key; a tie at the k-th key takes the lowest token ids), weights `exp(z)` in
  double, top-p over the `(weight desc, id asc)` prefix and the inverse CDF in the same entry order.
- Small kept sets: when the cutoff and top-k keep at most 2048 entries (any min-p or top-k filter in practice), the
  kept entries are compacted in token id order and one workgroup decides the row with the host's own sequence:
  the total summed in token id order, the top-p prefix and its mass, the cumulative probabilities, the acceptance
  test and the residual walk.
- Top-k up to 64: one pass over the logits. Each of the 32 workgroups of a row keeps its own k largest entries
  (radix selection in local memory, ties by lowest id), and one workgroup merges the lists, which contain the
  row's top-k, applies the cutoff and decides as above.
- Large kept sets: the weights are summed in token id order over a fixed two-level partition (per thread, then per
  workgroup). With top-p, the prefix boundary, the sample and the residual sample are mass quantiles in
  `(weight desc, id asc)` order: four radix rounds over the order key with per-digit masses, then a walk over
  the entries of the final key in token id order. The masses are summed in 80-bit fixed point (two 64-bit integer
  accumulators, resolution 2^-80), so every sum is exact and independent of the order of the atomic additions,
  and no candidate limit applies.
- Same uniforms: the host computes each row's `U(n, plain|accept|residual|draft)` with the counter-based generator
  and passes them to the device; only tokens and decision flags come back.
- Rounding: the device `exp` may differ from the host's in the last bit, and large kept sets sum in a different
  order (or in fixed point), so a decision can differ from the host path only when its uniform lies within a few
  double ulps of a cumulative boundary.
- Draft chain: each step's sampling kernels run after the draft head and overwrite the step's device token slot,
  which the next step's embedding reads, so the chain never waits for the host. The draft head's entries are
  visited in token id order through an index map uploaded when the entry list grows. Each step's `q` stays in
  device memory (probabilities over the head's ids, ascending) and the target rows read `q(d)` and the residual
  from there. Every policy runs the draft chain on the device.

`strix-hip-token-sampling-device-tests` compares device and host decisions (with and without drafts, the draft
distribution in pinned host or device memory, every filter combination, top-p up to 0.99 without top-k, top-k on
both sides of the one-pass limit) for rows of 4099 to 248320 tokens, and
device draft steps (token, support and probabilities) over entry lists shaped like the draft head's.
`STRIX_SAMPLE_TIMING=1` prints the host time of sampled decisions per kind and window size, and the device
sampler's kernel time, at exit.

## Reproducibility

Which positions carry a draft, and which draft, depends on the window and draft-length choices. The `fixed`
schedule derives both from draft probabilities only (the smallest owner covering the confident drafts, no
cost-model continuation check), so a run is a function of the seed, policy, prompt and configuration. `adaptive`
keeps the measured-cost choices of greedy decoding; it remains correct in distribution but two runs with one seed
can differ.

## Cost

On the device path a window's rows are decided together by the sampler's kernels after the window, and a draft
step adds its sampling kernels to the chain; only the decisions are read back. The host path copies one FP32
vocabulary row (about 1 MB) per decided row and builds its distribution there, and each draft step copies the
draft head's rows and completes before the next step; rows after the first rejection are not decided.

## Validation

- `strix-token-sampling-tests` checks the transform, a single speculative decision (draft on a vocabulary subset)
  and whole sequences generated in the shape of the engine cycle (sampled draft chain with early stop, windows with
  fewer rows than drafts, ordered row decisions) against exact target probabilities with chi-square tests, for
  temperature, top-k, top-p and min-p. Accepting every draft, or replacing the residual by `p`, fails them.
- On hardware (Qwen3.8-Flash-Next, production MTP flags): temperature 0 output is identical to the previous greedy
  binary with replay error 0; one seed reproduces its tokens; 300 speculative against 300 plain sampled requests
  (T = 1, six tokens, 1,059 draft decisions) pass a two-sample chi-square test at every position.
- Not yet measured: throughput. Single smoke runs at T = 0.6-0.7 produced 2-2.5 times the tokens per second of
  plain sampling.
