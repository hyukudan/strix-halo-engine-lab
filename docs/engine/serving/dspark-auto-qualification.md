# DSpark auto qualification

Auto defaults to the frozen, language-independent `structural-v1` prior. Eligible C1 JSON,
code and long prompts enter the existing legacy automatic controller. All other prompts
(including ordinary English and Spanish prose/story) use plain with no neural probe.
Node qualification (pre-registered not-slower rule below): Serve, two Strix Halo nodes over 2x USB4 Gen3, PerfOpt kernel, one load with mode switched per request (off / auto / auto / off, twice per class), 256-token greedy C1 chat requests: generated token IDs (ids_fnv) and text identical in every visit of every class; wall time auto vs off: JSON -34.6 % (13.57 -> 8.87 s, 95 % interval -4.99..-4.40 s), code -17.5 % (-2.67..-1.90 s), Spanish prose plain (no drafts) +0.2 % (-0.08..+0.13 s), English story plain -0.1 % (-0.04..+0.03 s). Long prompts were not measured on serve (prompts above max_rows stay plain). No universal no-slowdown claim follows from these host features for unmeasured content.

`STRIX_MM_DSPARK_PRIOR=unqualified-v1` or `--dspark-prior unqualified-v1` selects the
conservative prior with no promoted cells. CLI wins over environment; invalid versions
fail host preflight on either app/rank. The default is structural-v1 in both rank and serve;
`--print-effective-config` prints the resolved version. The serve OFF switch remains unchanged.

### Frozen structural-v1 features

Rank 0 alone decodes existing prompt IDs to bytes with the host tokenizer (special tokens
skipped), then scans ASCII markers without language detection, model inference or GPU work.
Long prompts use token count alone and skip byte decoding. Features are O(prompt bytes),
with a delimiter stack bounded by prompt bytes; no history or fitting occurs at runtime.
Thresholds are constants in `minimax_auto.hpp`; changing them requires a new prior version.

- **Long:** at least **6551 prompt tokens**, including the boundary fixture. This length rule
  supersedes the earlier design's proposal that length alone never promotes. Serve's existing
  grouped-prefill guard still takes precedence: prompt tokens > max_rows stays
  `plain:grouped_taps_unqualified`; do not enlarge max_rows for this policy.
- **JSON:** at least **2 matched delimiter pairs** and **2 quoted-key/colon patterns**,
  with `100 * (2 * pairs) >= 2 * byte_count` (2% delimiter density). Curly/square delimiters
  must be typed, balanced and outside double-quoted strings; backslash escapes are honored,
  and unterminated strings or crossed/unmatched delimiters reject this JSON feature.
  Keys are closing double quotes followed by ASCII whitespace then a colon. This is a marker
  heuristic, not a JSON validator.
- **Code:** any triple-backtick fence; **or** at least **8** bytes from `;{}()=` with
  `100 * marks >= 6 * byte_count`; **or** at least **3** nonblank lines starting with a tab
  or four spaces, at least **40%** of nonblank lines indented, and at least **2** occurrences
  of whole ASCII words from `def return class import for while if else function const let
  var fn struct public void int package func`. Keywords are programming syntax, never natural
  language labels. Code punctuation counts all bytes, including strings/comments.

Precedence after the safety guards is long, JSON, code, then plain prose. Route reasons are
`auto:structural_v1_long`, `auto:structural_v1_json`, `auto:structural_v1_code`, and
`plain:structural_v1_prose`; the fallback prior reports `plain:prior_unqualified`.
Both ranks follow rank 0's existing agreement/control messages. C1 promotion uses legacy
window/gate behavior (including its exploration), since request-local net-cost ACTIVE and
negative-saving exit to PLAIN_FINAL are not implemented. Concurrency/policy retirement is
permanent for that request; no stale-ring reactivation occurs after a burst drains. Serve
checks queued competitors as well as live/parked work before promoting. Inflight=0 disables
admission; raising the cap never enables automatic C2 speculation.

Keep target model/profile, kernels, transport, capacity, slots, max_rows, tokenized prompts,
stops and all unrelated defaults fixed. Never enlarge max_rows only for the speculative arm.
Grouped passive trailing taps need a separate memory and byte-identity gate; current auto uses
plain for grouped prompts. No recomputed prefill, extra target reduction or altered target row
schedule is admitted by this implementation. Full asset-content hash agreement and extended
policy/action identity checks remain qualification work; startup currently agrees mode, asset
presence, prior version, inflight cap and protocol, after target load and before drafter work.

## Exactness before timing

Build both ranks from one revision; record executable hashes and compiler, kernel/PerfOpt,
`STRIX_DIST_USB4_WEIGHTS`, model and drafter hashes. Run the host tests, golden/G4 L5,
decode-check, verify/exchange, rows==alone N1..8, DSpark host/HIP and candidate payload gates.
Use the same target identity for all comparisons. Run ordinary and grouped/tail resume fixtures
at cuts 1/7/60 and nested cuts 1/3/7/17/31/45/59 with whole and bounded children, 256 continuation
tokens, on both ranks and with taps ON/OFF. The existing resume oracle alone does not cover
new retirement actions or HTTP load; extend it before declaring that coverage complete.

Under C1/C2/C4, compare full token vectors per request with OFF and with each request alone on
both ranks. Include ties, rejection at every depth, K=0, capacity edges, 6551-token ring wrap,
640-token prose-to-code, code/JSON-to-prose, multilingual and Unicode structure, cancellation,
slot reuse, mode flips during active/parked prefill and decode, and C1-to-C2-to-C1. Compare prompt
outputs, continuation logits and committed K/V/index digests; exclude uncommitted suffix storage.
Check version/checkpoint/stale actions fail both ranks and never mutate open-task global taps.
Missing or skipped coverage is unqualified. `--require-identity` still fails on existing open gates.

## Paired measurement

Use turns of at most 480 seconds including startup, warmup, checks and drain. One binary and one
loaded drafter serve both arms; switch only `dspark_mode` at quiescent boundaries through
`/admin/sched`. Reset contexts and selector state symmetrically. Keep trace/feature settings equal;
primary timing is unprofiled. Report separate cold load and real peak memory, including tap banks.

Freeze structural-v1 and all fixtures before holdout timing. Use original and held-out JSON, code,
long-context/ring-wrap, English story and losing Spanish prose fixtures, including Unicode/Spanish
JSON and code. C1 generates 256 tokens; C2/C4 bursts generate 128 per request with identical EOS.
Include homogeneous and fixed mixed bursts, plus short-output/long-prompt latency cases.

Freeze eight independent load/turn-level blocks per content/concurrency cell. Each block contains
the original fixture and a rotating holdout in balanced AB/BA order; alternate ABBA and BAAB across
turns. Obtain separate A/A drift controls. Inspect blinded pilot variance before freezing n; no
optional stopping. Aggregate within each load before computing intervals (tokens are not independent
samples). Keep slow samples. Freeze exclusions; report incomplete pairs and replace whole pairs.

Report decode and completion tok/s per class, C2/C4 burst aggregate throughput, original-fixture
intervals, TTFT, ITL tails, makespan, startup/probe/draft/verify/tap/append/control costs and memory.
With d = auto throughput minus OFF throughput, use mean(d) +/- t(.975,7) sd(d)/sqrt(8).
Reject any selected regime whose upper bound is below zero. A straddling interval means only
"not demonstrated slower"; there is no 3% minimum gain veto. No aggregate may hide a losing class
or reproducible original fixture. Auto fallback must show zero residual draft/context work.

The existing `tools/minimax/serve_bench.py --strict` accepts `cell` scripts with policies
`{"dspark_mode":"off"}` and `{"dspark_mode":"auto"}` and an `order` such as
`["off","auto","auto","off"]`. Use its serial references and cancellation operations.
It is not a complete both-rank byte-state oracle; its strict token/digest checks do not replace one.
Do not use streamed chunk counts as generated-token counts.

`tools/minimax/dspark_pairs.py MANIFEST.json PAIRS.jsonl` evaluates the frozen eight-block intervals.
The manifest lists required keys `[content, concurrency, endpoint, population]`, including both
original and pooled populations. Each JSONL record contains `key`, `block` (0..7), `off`, `auto`,
`complete:true`, `exact:true`; values are independently aggregated paired throughput for that load.
The tool rejects missing/duplicate blocks or cells, incomplete/inexact pairs and nonfinite values.
It does not certify input provenance, choose exclusions, aggregate requests, or establish exactness.
A larger preregistered n requires a corresponding reviewed statistical change before unblinding.
