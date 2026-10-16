# Native MTP source implementation

Status: source implementation only. No build, native execution, model replay,
performance qualification or default activation accompanies this increment.

## Assets and arithmetic

`HipMtpAssets::load` consumes a caller-selected local sidecar, explicit conversion
metadata, the expected full file SHA-256, and the actual retained target mapping
and residency. It hashes the sidecar once, preserving both the raw file digest
and the existing framed model-content identity. Bounded GGUF metadata inspection
cannot authorize a live model binding. Only an explicit shared-target metadata
contract admits borrowing the target embedding and output table.

Semantic revision 3 normalizes the entire HC-wide hidden vector before reshaping.
The embedding projection is computed once and broadcast; the hidden projection
is independent. Packed storage does not imply a single concatenated reduction.
Converted effective norm gains and raw offset parameters have separate encodings.
Normalization/projection cast choices are explicit native budget inputs and enter
the arithmetic profile. Native entries require qualification and activation for
that exact profile; portable qualification with profile zero is insufficient.

The shifted pair is hidden at position i and input token at i+1. The predictor
attention/cache position is i, while the logical seed position remains i+1.
Old contracts and unshifted/stale pairings are rejected.

## Actual ownership and execution

`HipMtpResources` retains the two actual weight residencies, its own stream,
independent KV/raw-index caches, checkpoints, wide state, operator workspaces,
logits and sticky error record. Payload budgets are checked before allocation.
These are proposal allocations, without manufactured target session identities.
The native runtime preflights all eleven stages before submitting the first one.
Raw operators consume actual typed sidecar views and an explicit owned stream.

`HipMtpTargetHidden::execute_serial` runs the actual target through a direct
launcher dispatcher. It copies the final MLP injection output before the terminal
head mixer can reuse the arena. Native completion and finite first-max selection
precede session commit. Only then is a retained target-hidden ticket published,
including the consumed token, actual pending prediction, content identity,
position, session history and live snapshot allocation lease. A live ticket
prevents slot reuse. Native failure drains before cleanup and poisons the session.

`HipMtpRuntime` provides actual native forward dispatch, cycle inputs and token
selection. Its diagnostic forward obtains candidate replay evidence without
minting a qualified plan. Production priming/proposals/reconciliation require
externally obtained qualification for the same plan and numerical profile.

## Qualified consumer and reconciliation

`HipMtpGreedyRuntime` is an explicit **serial target verification** consumer.
It does not implement batched verification or establish a speed improvement.
Its fresh `prefill_prompt` streams adjacent target tickets and known prompt tokens
into predictor priming instead of retaining all prompt hidden rows. Prompt
execution itself remains serial in this consumer.

A cycle starts with an already emitted, unconsumed pending target prediction.
Only subsequent actual target predictions are returned. Candidate matches extend
verification; they never replace target selection. EOS and output budgets leave
the final emitted token unconsumed, so continuation does not emit it twice.

Matching accepted tokens do not prove that speculative draft KV is canonical.
The runtime restores its proposal checkpoint and replays every newly committed
hidden/token pair from actual target tickets. Reconciliation requires complete
ordered coverage and the exact final target session; emitted token IDs alone
cannot authorize cache reuse. Lifecycle admission distinguishes active proposal
and reconciliation phases without synthesizing a synchronized state copy.

The retained target slot count must cover maximum proposal depth plus three:
previous primed row, cycle base, and newly committed rows. The consumer caps depth
to this reservation and remaining target/predictor capacity. Callers release old
cycle results/tickets before continuing; additional externally retained tickets
may exhaust slots and are never overwritten.

## Independent replay and remaining validation

`replay_hip_mtp_forward` consumes independently supplied complete wide-hidden and
full-vocabulary logit goldens, with a pinned payload digest and content/profile
identities. Actual target inputs are compared before predictor mutation; each
complete output is captured before reuse. The current replay producer covers a
fresh sequential forward only, not resumed or rejection/reconciliation scenarios.
Callable HC/routed/shared component fixtures also require independent goldens.
The small payload fixture does not exercise a model forward.

Still required after source integration is complete: compile/object preservation
and new-instantiation checks; component and full checkpoint replay, including
rejection, continuation, EOS/budget and native failure paths; native measurements;
and the final requested external review. A separate batched target-capture and
verification integration is needed before this consumer can use known-row target
parallelism. No current speed, acceptance rate, native correctness or readiness
claim is made by the source receipt.
