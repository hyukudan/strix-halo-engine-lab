# Automatic decoder optimization profile

Selective stored-byte model backing is documented in
[model-weight-residency.md](model-weight-residency.md). It is an explicit
experiment and does not change the automatic default.

The decoder defaults to `--optimization-profile auto`. The low-level launcher
configuration defaults remain conservative; the application resolves a content-admitted
profile after reading GGUF metadata and identifying the immutable mapped file's
content, before choosing kernels or allocating device execution storage.

The initial allowlist contains one exact model-content identity:
`3726d6f0d9ba30b9d655ba303595118941baeca5e6945fc8785f938e0f5455e4`.
The loader's existing full content hash is reused, rather than hashing the large
model again. A filename, matching architecture or matching dimensions do not
admit another checkpoint to this profile. The grouped A8 path has its own numerical
identity and its existing validation scope; automatic selection does not establish
universal quality for other weights or contexts.

For that content, compatible model/router contracts, an A8-enabled build, eager
execution, registered residency and context 8192, the automatic profile selects:

- parallel/staged/fused HC with typed projection and typed-row prefetch;
- exact Q8 HC output prefetch and lane-chain HC output;
- virtual Q8 LM head, GDN input/input prefetch and GDN output;
- exact QSA input prefetch and shared-expert prefetch;
- deferred MoE first-error status publication;
- grouped A8 gate/up and exact grouped Q5_1 down;
- parallel wave router selection, replacing ranked routing unless ranked was
  explicitly requested.

When `--context` is omitted, a successfully selected profile uses its admitted
context 8192. An explicit context is never changed. Other explicit contexts retain
the caller's configuration and emit `context-not-qualified`. The current source
supports wider contexts, but the automatic profile does not claim those timings
or numerical gates. Without automatic selection the previous implicit context128
is retained.

With no low-level optimization flags, `auto` remains the application default.
If any kernel/PLE/coalescing optimization option is supplied without an explicit
`--optimization-profile auto`, the resolver preserves those options and uses
`explicit-low-level-options`. This keeps legacy commands and A/B arms from gaining
any automatic kernel or status additions implicitly. Context, token count and prefix-roundtrip options
alone do not count as low-level optimization selection. An explicit `auto` permits
composition; explicit ranked routing then keeps ranked and adds the remaining bundle.

Roundtrip prefix capacity is checked after profile resolution and before HIP setup
or enqueue. Thus an implicit context can first resolve to 8192, admitting a prefix
longer than the legacy default of 128. Generation retains its existing behavior:
it can stop at `context_capacity` before the requested token count. An explicit
context or an automatic fallback is validated using the context actually retained.

`--optimization-profile manual` disables all automatic additions and preserves
explicit options. Use it for controlled A/B comparisons and for selecting the
legacy configuration. Ranked plus implicit auto stays explicit; ranked plus an
explicit auto request retains ranked routing in the automatic bundle.
Explicit graph, integer-head, copied residency, replay-reference and planned PLE
IO modes fall back to the explicit configuration; their existing admission guards
still apply. Missing build capability, another checkpoint or incompatible metadata
also produce a reasoned fallback without automatically selecting A8. Speculative
execution has separate APIs and is unaffected by this decoder policy.

The adopted v3 policy includes QSA prefetch, shared-expert prefetch and deferred
status alongside wave routing and lane-chain HC output. Its own native, full-model
and clean automatic workload gates passed at the qualified source revision below.
Earlier manual bundle evidence is retained separately; individual speedups are not
added to infer the combination.
Arithmetic candidates use the separate typed option below. Explicit experiment
flags can compose with an explicitly requested automatic profile, but this does
not qualify their combined cost or numerical behavior.

All three decoder receipt paths add:

| JSON field | Meaning |
| --- | --- |
| `optimization_profile_requested` | `auto` or `manual` |
| `optimization_profile_selected` | `manual`, `canonical-a8-wave-qsa-hc-shared-deferred-v3`, `canonical-a8-ranked-qsa-hc-shared-deferred-v3` or `canonical-a8-wave-qsa-hc-shared-output-v5` |
| `optimization_profile_reason` | `qualified-checkpoint` denotes content admission, not bundle qualification; otherwise explicit ranked override or a specific fallback reason |
| `optimization_profile_automatic_applied` | Whether the automatic bundle was added |

`manual` in the selected field means the caller's explicit configuration is used,
including existing default values. Existing per-kernel mode/identity/dispatch fields
continue to describe effective execution. The qualified content/context slice is
adopted; other configurations retain their existing admission. Portable tests
exercise the actual application parser and resolver with runtime capability present
and absent; they do not execute HIP kernels or measure token latency.

Observed for code revision `dee811a391485b894dbda5ea1d7fd29efc1bd1ec`:
GNU 13.3/WSL Release, HIP OFF and warnings as errors, full build and 159/159 CTest
entries passed. Later documentation-only changes do not change that tested code.
Earlier decoder/catalog/catalog-fixture C++20 syntax checks with private HIP
declarations covered capability present and absent; they do not establish fresh
native compilation or GPU execution for v3 at that historical checkpoint.

## Ordinary-generation concurrency profile

For the qualified checkpoint, ordinary CPU-output generation selects
`canonical-a8-wave-qsa-hc-shared-output-v5`. It retains the v3 arithmetic
bundle and adds slot-parallel Q5_1 down projection, batched MoE activation,
paired wave8 Q8 GDN input/output, GDN islands, prepared eager submission and
MoE/shared-expert overlap, prefetch8/wave8 Q8 LM head and wave8 Q8 QSA output.
Explicit overlap arena budgets are preserved.

All existing content, model, capability, context, residency and execution
allowlist guards still apply. Manual mode and implicit low-level overrides keep
their behavior. MTP, serving, prompt lookup, known-prefix windows, prefix
roundtrip, RAW, replay reference, target-input export, GPU scalar output,
requested logprobs, device queue tracing, online MoE scheduling, ranked routing and additional
numerical organizations retain the prior policy and their explicit flags.
Post-request final-state validation and host observation remain permitted.

The qualification slice covers sixteen consumed rows, all 248320 FP32 logits
per row, and all seven initial/final state payloads, compared bit for bit.
Complete request measurements exclude model setup and retain every scheduled
sample. This finite slice does not establish all-position or full-context
coverage. Automatic-profile integration, measured adoption and its final build checks are recorded
separately from the historical v3 adoption below.

## Qualified adoption (2026-10-15)

Source `265fd78762884d18eb32aa60294408f080088d52`, source archive SHA-256
`f0c87566c37fa5ca2e76caacd02c5af94a0080bb3b08911d377e7058a90db155`,
passed independent source checks, 225/225 native tests and a complete 14-module
diagnostic linkage check. All 245 linked kernels matched the preceding qualified
integration under the compiled code/descriptor/FP checks; the 225 legacy kernels
also passed their preservation audit. Seven actual CLI cases exercised automatic,
manual, explicit-ranked, generation and prefix-roundtrip behavior.

The canonical context 8192/eager/register workload passed every one of 554 complete
FP32 logit arrays against the preserved reference. Clean automatic generation then
matched all 512 immutable reference IDs in all eight runs (two warmups and six timed
samples). Actual receipts report 19,944 QSA projection submissions, 79,776 shared
projection submissions and 554 completed tokens from the deferred owner. These are
host submission/owner counts, not inferred per-layer GPU execution counters.

On the same gfx1151 node, the three-pair descriptive automatic comparison against
the preserved baseline measured a median paired ratio of 1.2599223 (+25.99%);
post-first-token wall-rate medians were 12.1007 and 15.2520 tokens/s. A separate
direct comparison against the preceding wave+HC-output automatic profile measured
1.1555118 (+15.55%), with wall-rate medians 13.1692 and 15.2226 tokens/s. Both arms
of this direct comparison used clean `--optimization-profile auto`; its baseline
was compiled on the other host and actually revalidated on the execution host.
Each campaign retained all samples, used two warmups and three alternating pairs,
and stopped under its declared descriptive screening rule. These limited prompt,
context and host observations are not statistical significance or broad model
quality claims. Detailed immutable receipts remain outside Git.

The measured workload used wave routing; explicit-ranked CLI checks do not
extend these measurements to the ranked variant.

The primary source was fast-forwarded to that exact qualified revision. This is
source adoption; it does not assert deployment of a user-facing remote binary.
The low-level/manual defaults, explicit overrides and fallback contracts above
remain in force. Additional graph, speculative, integer-head, model/context and
projection-prototype combinations need their own qualification.


## Explicit numerical recipes

`--numerical-profile` selects an arithmetic overlay independently of automatic
organization and submission. Its default, `exact-v5`, preserves the existing
selected arithmetic; it does not mean that every manual configuration or every
checkpoint has the qualification of the ordinary automatic profile.

| Value | HC projection | Routed down projection |
| --- | --- | --- |
| `exact-v5` | Existing selected path | Existing selected path |
| `hc-splitk-f32-v1` | FP32 split-K | Existing selected path |
| `q51-a8-down-v1` | Existing selected path | A8 activation and integer Q5_1 dot |
| `hc-splitk-f32+q51-a8-down-v1` | FP32 split-K | A8 activation and integer Q5_1 dot |
| `q51-q80-a8-down-v1` | Existing selected path | A8 activation and integer Q5_1 plus eligible Q8_0 dots |
| `hc-splitk-f32+q51-q80-a8-down-v1` | FP32 split-K | Both eligible A8 down formats |
| `hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup-v1` | FP32 split-K | Adds unsigned-five-bit A8 gate/up at layer2 and its Q8_0 A8 down |
| `hc-deferred-norm-f32+q51-q80-a8-down-v1` | FP32 split-K with deferred per-stream RMS | Both eligible A8 down formats |
| `hc-deferred-norm-f32+q51-q80-a8-down+q5k-a8-gateup-v1` | Deferred per-stream RMS | Both down formats plus layer2 Q5_K A8 gate/up |

These alternatives change rounding or reduction association. They are explicit
candidates with `not_assessed` numerical qualification, not aliases for exact
baseline outputs. The FP64 HC control remains confined to diagnostic harnesses;
there is no FP64 command-line recipe.

The arithmetic option does not count as a low-level organization option. It is
applied after the existing selector: ordinary automatic generation retains v5's
prepared dispatch, graph islands, fork/join and projection organizations. The
receipt retains `optimization_profile_base_selected`, appends the nondefault
recipe to `optimization_profile_selected`, and reports `numerical_profile`.
No prerequisite flag is enabled or disabled by this overlay. A manual or
teacher-forced diagnostic must supply a full compatible organization when the
base selector does not provide it.

Nondefault recipes require the exact admitted content, canonical model geometry,
context8192, registered weights, eager floating C1 execution and the full v5
organization. Graph execution, integer heads, scalar-only output, MTP,
known-prefix windows or CSR, serving/cohort execution, prompt lookup, prefix
roundtrip, alternate recurrent layouts, GQA organizations and offline selection
are rejected. Other checkpoints and missing prerequisites fail before device
allocation. The HC metadata pass checks all97 selected mixer bindings and exact
tensor formats/extents before residency; live owner and dispatch preflight still
runs. Prepared launcher plans are built from the selected numerical kernel IDs,
not reused from a baseline plan.

`--prefill-numerical-profile` selects the prompt prefill arithmetic separately
(default: `--numerical-profile`); only the dense-A8 sibling pair is admitted, on
the tiled known-prefix prefill or the MTP prompt. See
[prefill numerical profile](prefill-numerical-profile.md).

The admitted routed census is43 Q4_K/Q5_1 layers, four Q4_K/Q8_0 layers and one
Q5_K/Q8_0 fallback layer. The Q51-only recipe preserves the four Q8_0 fallback
layers; the Q51/Q80 recipe changes only those four additional eligible down
projections. Those original six profiles preserve the Q5_K gate/up layer.
The explicit `+q5k-a8-gateup-v1` atom instead admits only canonical layer2,
using the original unsigned-five-bit qh bit-plane with the versioned integer
A8 gate/up epilogue. Its down uses the existing Q8_0 A8 recipe. The canonical
43/four/one classification remains unchanged; selected execution then has48
quantized gate/up layers,43 Q5_1 A8 down layers and five Q8_0 A8 down layers,
with no legacy Q5_K submission. Missing or different census/content fails closed.
Compact RAW
identities distinguish the new down arithmetic and do not claim the incumbent
exact-grouped Q5_1 reduction. Integer activation codes, FP32 scales, tiny-group
handling, integer dot order and ordered route merge are versioned contracts.

Receipts distinguish configured recipes and model eligibility/fallback lists from
`hc_splitk_f32_submissions`, `q51_a8_down_submissions`,
`q80_a8_down_submissions` and `q80_a8_down_legacy_fallback_submissions`.
`hc_materialized_norm_submissions`, `hc_deferred_norm_submissions` and
`q5k_a8_gate_up_submissions` separately report the additional actual producers.
These are successful host enqueues. They are not graph replay counts,
physical traffic estimates or performance measurements.

Deferred HC is restricted to all97 canonical mixers, including the final mixer.
Every norm is F32[10240], every up projection is admitted Q8_0, and every
selected binding uses the deferred lane-chain recipe. No lane-chain-off variant
is registered or chosen for that profile; partial eligibility fails before any
HC launch. The existing materialized path and its mixed fallback stay intact
for older profiles. The new split-K computes per-stream inverse RMS in the
existing workspace, and the lane-chain output reloads raw BF16 input and F32
gains. Live range/generation checks preserve disjoint input, output, workspace
and weight owners. Injection consumes the original expanded input and its
separate projection result, not a fictitious normalized buffer. The arithmetic
association has its own identity; no equivalence to the previous path is claimed.

## Teacher-forced scoring diagnostic

`--teacher-forced-score --generate 1` consumes a bounded sequence of at least two
known input IDs. For input position `p < N-1`, it requests exactly the supplied
ID at `p+1`, using the full finite vocabulary and the existing FP64
log-sum-exp reduction in ascending vocabulary-ID order. The last teacher input is
consumed once with ordinary full logits and creates no label row. The normal
single pending greedy output remains separate from the teacher labels.

`requested_logprobs_rows` must contain exactly `N-1` rows, each with its actual
input token, position and one next-input ID/log probability. All labels and
capacity bounds are checked before inference. Prefix state-only requests delegate
to this same full-output consumer, so a score never skips its head or consumes
an input twice. A post-consumption presentation failure poisons and drains the
request instead of publishing a partial healthy result.

This mode rejects static `--logprob-token` requests, RAW/state exports, replay
references, scalar output and alternate controllers. It is a diagnostic scoring
run: complete-request timing is disabled, and its score ledger is not a speed
measurement. Arithmetic candidates retain their explicit recipe and unassessed
qualification; computing scores alone does not approve their quality thresholds
or change defaults.


`--teacher-logits-file PATH` optionally writes complete finite FP32 logits for
all N-1 scored rows, in ascending vocabulary order, to an exclusively created
bounded diagnostic file. It requires teacher-forced scoring. The header binds
vocabulary and all teacher IDs; per-row input/position fields and a completion
footer prevent accepting an interrupted stream. The file is capped at16GiB
before inference. Appending a row or finishing the file after target consumption
uses the same poison/drain failure path as the score ledger. Full-logit exports
are for independent KL/top-1 analysis, not a performance run; no metric or
qualification is inferred merely because a file was created.

Both outputs can also be produced through the tiled known-prefix prefill
(`--known-prefix-window --known-prefix-full-rows --known-prefix-prefill-rows T`
with the all-rows output policy), with the same file format, row order and score
rows; see [teacher-forced-tiled-prefill.md](teacher-forced-tiled-prefill.md).

## Experimental recurrent island layout

`--gdn-recurrent-layout transposed` can explicitly compose with GDN islands in
manual ordinary eager execution with key-value recurrent storage. Baseline stays
the default; stride32 and stride33 remain excluded from islands. The existing
layout marker and island organization receipt identify this unqualified candidate.
The explicit split-K f32 Q51/Q80 recipe and its Q5K gate/up extension may request
this composition only with their complete manual organization flags. Automatic
profile selection, deferred-normalization recipes and other controllers retain
their existing admission. Whole-state and whole-model bitwise validation and complete-request performance
qualification are required before adoption; source admission is not qualification.
