# Joint serving windows (experimental)

`STRIX_MTP_BATCH_JOINT=1` enables joint window scheduling on the native HTTP
batch route. It is off by default. A single decoding request continues to use
the ordinary request controller and its existing execution path.

The batch controller searches the requests' R2/R4/R8 choices together under
`STRIX_COHORT_ROWS=16|32` (32 if unset). It measures a whole cohort topology,
rather than attributing its wall time to any one request's window owner. Each
slot has independent acceptance history, cleared at admission. Initial bounded
probes and periodic exploration prevent a first-use cost from permanently
excluding deeper windows. The controller can skip drafting for a request whose
expected accepted tokens do not repay the draft cost.

An undrafted request normally uses an R2 owner, verifies its pending target token
and commits exactly one row. A larger owner can pad a valid cohort. Padding is rolled back by the
existing prefix commit. Window admission still checks the request's own token
budget, context capacity and history capacity. Context and budget tails use the
existing fallback. Six-row plans are excluded because they cannot use a four-
or eight-row primary and would otherwise execute two independent chunks.

With `STRIX_COHORT_WIDE=all`, cohorts of 9–32 rows also share the HC projections,
router and shared expert, in addition to the existing GDN/QSA projections, head
and routed expert union. Every request retains its own recurrent state, KV,
capture, token selection and prefix commit. Padding is never published. These
paths retain the selected arithmetic expression for every row; native identity
qualification is required before promotion.

Diagnostics (off unless explicitly enabled):

- `STRIX_MTP_BATCH_LOG=1`: cohort geometry, emitted tokens, wall time and steady
  clock boundaries. Joint cycles also report draft and target phase times.
- `STRIX_COHORT_WIDE_LOG=2`: every cohort's wide, chunk-shared and own step counts;
  `=1` retains the sparse log.
- `STRIX_WINDOW_ROUTE_LOG=1`: per-session routes, including shared cohort steps.
  This synchronizes and must not be used for throughput measurements.

Logical weight sharing is not a hardware DRAM-byte measurement. Expert kernels
may tile assignments, caches may reread weights, and state/KV traffic grows with
the independent requests. End-to-end qualification must measure C1 and multiple
concurrent requests with diagnostics disabled and compare each request's tokens,
full committed vocabulary rows and authoritative state against a single run.

Identity validation covers greedy requests with a 1,024-token prompt and the
selected dense A8 profile, min-p 0.7, independent slots at concurrency 1/2/3/4,
64/128 output tokens and cohorts through 32 rows. The kernel qualification and
final scheduling check total 31 requests, 3,328 complete FP32 vocabulary rows and
217 canonical state components identical to C1. Occupied KV/index bytes and all
fixed state components are compared; unused cache tails and scratch are outside
the committed-state contract. The final C3 exploration check pads R2+R2+R2 to
R4+R2+R2 when admission allows it. Context/budget tails retain their smaller valid
chunks when they cannot be padded.
