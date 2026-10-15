# Automatic decoder optimization profile

The decoder defaults to `--optimization-profile auto`. The low-level launcher
configuration defaults remain conservative; the application resolves a qualified
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
- grouped A8 gate/up and exact grouped Q5_1 down;
- parallel wave router selection, replacing ranked routing unless ranked was
  explicitly requested.

When `--context` is omitted, a successfully selected profile uses its qualified
context 8192. An explicit context is never changed. Other explicit contexts retain
the caller's configuration and emit `context-not-qualified`. The current source
supports wider contexts, but the automatic profile does not claim those timings
or numerical gates. Without automatic selection the previous implicit context128
is retained.

With no low-level optimization flags, `auto` remains the application default.
If any kernel/PLE/coalescing optimization option is supplied without an explicit
`--optimization-profile auto`, the resolver preserves those options and uses
`explicit-low-level-options`. This keeps legacy commands and A/B arms from gaining
wave, HC output or A8 implicitly. Context, token count and prefix-roundtrip options
alone do not count as low-level optimization selection. An explicit `auto` permits
composition; explicit ranked routing then keeps ranked and can add HC output.

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

The QSA prefetch, deferred-status and shared-expert prefetch increments remain
explicit experiments pending their own performance qualification. Lane-chain HC
output now joins the automatic profile after its separate qualification; this
policy change prepares the combined wave/HC configuration for review and validation.
Separate measurements do not establish a combined speedup or justify adding their
individual gains.
The HC projection prototype adds no decoder option. Explicit experiment flags can
compose with an explicitly requested automatic profile, but this does not qualify their combined cost
or numerical behavior.

All three decoder receipt paths add:

| JSON field | Meaning |
| --- | --- |
| `optimization_profile_requested` | `auto` or `manual` |
| `optimization_profile_selected` | `manual`, `canonical-a8-wave-hc-output-v2` or `canonical-a8-ranked-hc-output-v2` |
| `optimization_profile_reason` | Qualified selection, explicit ranked override, or the specific fallback reason |
| `optimization_profile_automatic_applied` | Whether the automatic bundle was added |

`manual` in the selected field means the caller's explicit configuration is used,
including existing default values. Existing per-kernel mode/identity/dispatch fields
continue to describe effective execution. This is a source-policy draft requiring
review and fresh integration/native qualification before adoption. Portable tests
exercise the actual application parser and resolver with runtime capability present
and absent; they do not execute HIP kernels or measure token latency.

Observed host verification: GNU 13.3/WSL Release, HIP OFF, warnings as errors,
full build and 159/159 CTest entries passed. The actual decoder, catalog and
catalog fixtures also passed C++20 syntax checks with private HIP declarations,
with the A8 runtime capability both defined and absent. This is not a native HIP
compiler or GPU execution result.
