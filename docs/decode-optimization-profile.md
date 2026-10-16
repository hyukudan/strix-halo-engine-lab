# Automatic decoder optimization profile

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
The HC projection prototype adds no decoder option. Explicit experiment flags can
compose with an explicitly requested automatic profile, but this does not qualify their combined cost
or numerical behavior.

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
