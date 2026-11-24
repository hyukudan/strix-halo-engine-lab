# Parallel candidate drafter: host contract

Status: host reference and admission only. No native backend is bound, no model
fidelity or throughput qualification is claimed, and every profile defaults OFF.
The method and checkpoint schema are attributed to Inco AI's DFlash 2; model
assets retain their model-card Apache-2.0 license. No external implementation
source or model asset is included in this change.

## Shared components

| Stage | Shared home | Family binding |
|---|---|---|
| Header admission, tensor shapes, explicit BF16 read and digests | `weight/block_draft.hpp` | Config data |
| Scalar mathematical forward and candidate selector | `spec/block_draft_reference.hpp`, `src/core/block_draft_reference.cpp` | Target embedding, raw head, feature taps |
| One-pass proposal adapter | `spec/block_candidate.hpp` | Accepted feature frontier callback |
| Window selection and speculative cycle | Existing `ParallelBlockDrafter`, `BlockDrafter`, `BlockSpeculator` | Existing verify-rows target and prefix commit |
| Lever | `spec/block_draft_policy.hpp`, common profile resolver | Profile-table row and common option parser |

No new model-directory file is needed. The initial family binds the shared lever
in its existing profile table and common CLI/server parser. Shared implementation
and the family binding are separable changes; shared work lands first.

## Admission and arithmetic

`index_block_draft` reads the eight-byte little-endian header length and bounded
JSON through the existing safetensors reader. It requires every tensor exactly
once, BF16 storage, matching config shapes, contiguous nonoverlapping payload
ranges and supported arithmetic. Admission caps the config at 16 KiB and 256
JSON values, and the safetensors header at 64 KiB with a value budget derived from
the expected tensor schema and up to 16 metadata entries. The tensor count
cannot exceed the expected schema size. Version one admits an eight-row block
with up to eight draft layers and eight target taps with 16-bit IDs. Duplicate
JSON keys and invalid geometry fail without replacing the caller's previous
result. An explicit shard namespace prevents implicit full-file hashing during
header admission. Header and config
SHA-256 digests identify metadata only: they do **not** attest payload contents.
`digest_block_draft_payload` is an explicit, separate payload operation.

The scalar oracle uses FP64 intermediate arithmetic on decoded BF16 weights.
It is a mathematical reference, not a bitwise emulation of a device library's
BF16 reductions. It reconstructs draft K/V from committed target features on
each call. Device implementations should retain their own K/V rings, append
only committed features, and check against this oracle under a declared
rounding/error contract. The target's own K/V is never borrowed by this schema.

For each block the oracle performs one pass, with the anchor embedding followed
by mask embeddings. Context features are concatenated in declared tap order,
linearly fused and RMS-normalized. Each layer has pre-normalized GQA attention
and a SwiGLU MLP, each wrapped in pre/post dynamic two-tap causal convolutions
and a residual. Dynamic coefficients are generated from the normalized input;
post coefficients are reused after the attention/MLP. Convolution history is
zero at the start of each draft block. Draft attention is bidirectional with
per-query sliding visibility; target verification remains causal.

The embedding callback returns one row for each position in the full block,
even if the controller asks for fewer proposals. The head callback returns one
raw-logit row per requested proposal over the **real** target vocabulary.
Padded storage ids must be excluded. The target binding declares that live
output width in `CandidateTargetHooks::live_vocab` (0 < width <= the checkpoint
config's padded `vocab_size`, and at least `selector_top_k`); the adapter
refuses a head result whose column count differs from it (truncated or padded)
and an anchor id outside it, before the selector runs. The selector first keeps top-k unary
candidates per position, then chooses a path sequentially using the previous
selected token's code and a projected hidden-state/next-token interaction.
The config determines whether softcap is applied before or after that
correction. Ties use ascending token id. This is a sequential conditional
selector, not beam search or a globally optimal path solver.

## Greedy cycle and tiers

Verification inputs are `[anchor, draft_1, ..., draft_k]`. Row `i` predicts the
token following input row `i`. Accept the longest prefix equal to those target
argmax ids; emit that prefix and the correction/bonus target token. Commit
exactly `accepted + 1` input rows and discard every later speculative row,
including all associated convolution/recurrent state. The last emitted token
is the next unconsumed anchor. EOS, remaining output budget and context room
truncate the prefix using the existing controller. Stop the outer generation
loop after EOS. The target must produce the same row logits/tie rule as serial
decode; this remains a native identity gate, not a consequence of host tests.

The shared lever accepts `off`, `dflash2:8:greedy` and `dflash2:8:sampled` through
`--block-drafter` or `STRIX_BLOCK_DRAFTER`. CLI and server call the same profile
resolver and validation. Defaults and OFF fingerprints are preserved in EXACT,
FAST and TURBO. EXACT admits only the greedy rule as a candidate. The sampled
rule is a FAST candidate (inherited by TURBO), but **execution is refused in
all tiers** until the conditional proposal/rejection hook is qualified. Greedy
native execution is also refused while the backend is unbound.

For eventual sampled verification, proposal q must be the selector's actual
conditional sparse distribution given the accepted prefix. Acceptance is
`min(1, p(token)/q(token))`; first rejection samples normalized `max(p-q, 0)`
over the entire real vocabulary. Full acceptance samples the target bonus
distribution. Sparse support is zero outside the candidate ids. Temperature,
target transforms, top-p/top-k and RNG ownership belong to the target request
contract. The host selector returns sparse q, but no production rejection hook
or distribution-preservation claim is added here.

## Device handoff

Use shared dense rows for fusion, Q/K/V/O, MLP and coefficient projections;
shared RMS/RoPE and prepared GQA for normalization and attention; shared
`ShortConvolution` with dynamic groups for the two taps (adapt coefficient
layout, reset history per block). Existing bidirectional attention policies
must be checked for the **per-query** sliding boundary: a block-wide context
window is not automatically equivalent. The candidate head needs raw top-k
logits, gathered predecessor/successor codes, and a small conditional selector.
Use existing shared head/lookup families with traits, not a private model kernel.

Bind target capture ids through its existing tap plan, including the final-norm
tap. Preserve the real vocabulary separately from padded weight dimensions.
Measure register/spill/occupancy and no-model fixtures first, then validate
accepted and rejected prefixes against serial target logits, all state payloads,
EOS, context pressure, cancellation, slots, CLI/server and rank agreement.
Multi-device planning and exchanges remain in the existing shared planner;
there is no automatic multi-device admission or hard-coded hardware cost here.

Build `strix-block-candidate-tests` and run it with the existing
`strix-spec-block-parallel-tests` and `strix-spec-block-drafter-tests`.
Optional metadata-only checkpoint validation:

```text
strix-block-candidate-tests --header-only DRAFT/config.json DRAFT/model.safetensors
```

Synthetic host tests cover admission without payload access, tensor loading and
digests on synthetic bytes, analytic forward fixtures, selector conditioning and
softcap order, target hook wiring, every greedy acceptance length, EOS and both
caps. Full checkpoint payload execution and native identity remain pending.
