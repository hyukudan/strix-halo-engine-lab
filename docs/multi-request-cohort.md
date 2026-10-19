# Several requests in one decode loop (cohort windows)

Opt-in CLI benchmark mode for aggregate throughput with several independent requests. Every request
produces exactly the tokens it produces alone; the receipt reports aggregate and per-request rates.

```
strix-hip-decode-token MODEL.gguf TOKEN_ID ... (request 0: positional prompt, --mtp-golden ...)
  --mtp-cohort-request TOKENS_FILE,GOLDEN_FILE,GOLDEN_SHA256   (requests 1..3, repeatable)
  [--mtp-cohort-mode lockstep|sequential]                      (default lockstep)
```

On the serial prompt route (the default) `TOKENS_FILE` holds decimal token IDs separated by commas or whitespace;
`GOLDEN_FILE` is that prompt's MTP
golden under the run's numerical profile (dumped with `--mtp-golden-dump` like any other golden). The rest of
the command line is the ordinary speculative configuration of the README; the cohort requires prefix windows
over the row set `2,4,8`, the draft chain and one numerical profile for prompt and decode (no
`--prefill-numerical-profile`). At most four requests in total on the serial prompt route (eight on the tiled
prompt, below); `--generate` applies to each request.

## Tiled prompt (batched route)

On the tiled MTP prompt (`--known-prefix-window --known-prefix-full-rows --known-prefix-prefill-rows T`, the serving
route) cohort requests take `--mtp-cohort-request TOKENS_FILE` without a golden: the predictor is qualified once by
the calibration replay of the positional prompt's first `--mtp-calibration-rows` pairs, exactly as for the server,
and every request then runs its whole prompt through the tiled prefill (T-row blocks, tail blocks, serial rows) like
a served request. Up to eight requests in total (seven besides the positional prompt); prompts are bounded by the
context only. `--mtp-sample-*` applies to every request (request i uses seed + i).

Each request has a slot (see [serving.md](serving.md#batched-decoding)): its own state, session, launch plan, PLE
backend, tiled prefill owners and backends (the prefill arenas are shared and used by one prompt at a time), prefix
cache view, predictor, target producer and window owners on one shared window stream. `lockstep` runs every
request's prompt (one after another) and then decodes them together; `sequential` admits each request after the
previous one ended, each exactly as alone. With more than four requests a cycle forms two or more cohort windows (two
rows per request, groups of four requests per 8-row window).

The receipt (`mode` `native-mtp-cohort-tiled-prompt-target-verified-greedy|sampled`) has the fields of the serial
cohort receipt plus each request's tiled prompt receipt, its prompt time and restored tokens; the replay errors are
those of the shared calibration replay. `STRIX_BATCH_INTERLEAVE=1` interleaves the prompts with the decode cycles
(then the cycle wall includes the prompts).

## Per-request state

Each request has its own serial scratch and persistent state (KV cache, GDN recurrent and convolution
state, PLE history), PLE lookup backend and prefetch state, launch plan, session, MTP target capture,
D1 record, MTP runtime with its own draft state and draft vocabulary (`auto:DIR` picks the pack from that
request's prompt), and its own 2/4/8 window owners with their deferred GDN record, PLE snapshots and greedy
replies. Weights, the MTP sidecar assets, the manifest and the state layout are shared. Every request runs
its own prompt, independent replay against its own golden (the declared error policy, 0 by default),
qualification and activation before generation, exactly as the single-request route does. In cohort mode
the window owners of all requests use one stream.

## Cycle

`lockstep` runs one cycle of every active request at a time (`hip_mtp_greedy_cohort_cycle`):

1. Each request drafts its chain as in an ordinary cycle, up to the first draft below `--mtp-spec-min-p`
   (the cost-model early stop is not used in a cohort cycle), and at most to the per-request row cap
   minus one: the cap is 8 / requests in the cycle (4 rows with two requests, 2 with three or four), so
   that one cycle fills one 8-row window (`STRIX_COHORT_MAX_ROWS=2|4|8` overrides it).
2. Each request gets the prefix owner the verifier would size for its drafts: 1 + the leading drafts at or
   above the minimum probability, at least 2, rounded up to 2, 4 or 8 and limited to the cap, if that
   owner fits its budget, context room and lineage history. Requests are packed into cohorts of 4 or 8
   rows, larger windows first; a cohort that does not add up is rounded up with padding rows of a smaller
   window (padding rows repeat the last draft and are never accepted), else its last request leaves it. A
   request alone, one that needs eight rows, or one with no fitting owner verifies with its own verifier
   as usual.
3. A cohort runs as one window (`HipKnownPrefixWindow::execute_mtp_cohort`): every request's own window is
   admitted exactly as on its own, then the steps advance in lockstep on the shared stream. Steps that read
   only weights and per-row activations run once for the rows of all requests on a primary owner (an idle
   owner of the cohort's row count): routed expert union, HC mixers, router, LM head (each request's
   argmax rows into its own replies), the shared expert projections (on the primary's side stream, joined
   before the MoE combine, as in one window) and the input and output projections of GDN and QSA. Inside
   the GDN and QSA steps the recurrence and the KV append/selection/attention run per request from that
   request's state (`gdn_window_cohort_enqueue`, `indexed_attention_known_window_cohort_launch`), and the
   shared-expert activation runs per request. Every other step (embedding, PLE, MoE combine, injections,
   the observed last MLP injection) is each request's own launch.
4. Each request then accepts its longest prefix, commits its deferred GDN rows and PLE snapshot, rewinds
   its frontier and publishes its rows through its own target capture, as after its own window.

`sequential` runs the requests one after another, each exactly as alone; it is the in-process reference
for the aggregate rate.

## Exactness

The shared steps perform each row's arithmetic exactly as that row's own window does (per-row pointers,
the same kernels and reduction trees; the multi-row kernels are row-count independent per row). The dense
A8 output codes of QSA are written per request at the request's rows of one slot whose layout does not
depend on the row count. Fixtures compare cohorts byte for byte against each session's own window:
`strix-hip-gdn-deferred-window-tests` (sessions with their own state, pending rows and commits; cohorts of
2 to 4 sessions, 2 to 8 rows) and `strix-hip-known-window-tests` (QSA cohorts of two and three sessions at
mixed positions, FP32 and dense A8). On the full model each request's generated tokens are compared with
its own single-request run.

## Switches

| Switch | Effect |
| --- | --- |
| `--mtp-cohort-request` | Enables the mode; absent, every route is unchanged. |
| `--mtp-cohort-mode sequential` | Requests one after another (reference). |
| `STRIX_COHORT_SHARED=0` | Lockstep cohorts with every step launched per request (no shared steps). |
| `STRIX_COHORT_SHARED_EXPERT=0` | Shared steps as above except the shared expert, which runs per request. |
| `STRIX_COHORT_MAX_ROWS=2\|4\|8` | Per-request row cap in a cohort cycle (default 8 / requests). |

The receipt (`mode` `native-mtp-cohort-...`) reports `mtp_cohort_wall_ms`, `mtp_cohort_generated_tokens`,
`mtp_cohort_aggregate_tokens_per_second`, cohort window counts and, per request, its replay errors, cycles,
acceptance, window histogram, finish time, rate and generated tokens.

## Limits

- Serial prompt route: CLI benchmark mode, at most four requests of up to 4097 prompt tokens each. The tiled
  prompt route serves through the same slots (`--http-batch`, [serving.md](serving.md)).
- PLE and the elementwise steps run per request (PLE weights are small; elementwise steps read no weights).
- Drafting is sequential per request.
