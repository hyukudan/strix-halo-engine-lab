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

The target capture factory accepts the actual deferred-status configuration.
When D1 is selected it owns a sticky eight-byte device record and transactional
host-history shadow. The record is initialized once at setup; status readback,
finite selection and host publication precede session commit. A failed owner is
never reused, and enabling MTP does not silently turn the automatic D1 path off.

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

`HipMtpGreedyRuntime` supports serial target verification and an explicitly
prepared closed known-row verifier. `use_window_verifier` attaches the actual
same-target R2/R4/R8 owner before the first cycle. This source does not establish
a native speed improvement.
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

`HipMtpWindowVerifier` prepares two reusable capture pools and a retained target
checkpoint before execution. A cycle uses at most one closed target window and
caps candidate depth to rows minus one. Every speculative input must match the
preceding finite target prediction. An accepted chunk publishes its real final
session and its complete row snapshots; there are no fabricated intermediate
sessions. A distinct `known_window_chunk` lineage binds actual row launch plans
and promoted operator recipes. The ordinary uncaptured window behavior remains
unchanged.

A mismatched input or consumed EOS rejects the provisional chunk. The verifier
restores actual state bytes and the exact recorded metadata/history, then selects
and verifies the serial continuation again. No rejected snapshot reaches MTP
reconciliation. Short depth/budget/capacity uses the serial path. Capture payloads
remain pinned while any row ticket is retained; pools are reused only after the
previous readers release them. The checkpoint allocates device/host pools once.
Its normal capture/restore/discard waits only on the owned stream, with global
recovery reserved for uncertain native failure.

The retained serial target slot count must cover maximum proposal depth plus three:
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
Callable HC/routed/shared/window/cycle component fixtures also require independent
goldens. Cycle traces can require accepted-window, restored rejection, EOS, budget
and continuation coverage and compare complete final target and reconciled MTP
states. They retain host receipts rather than keeping obsolete capture leases.
The small payload fixture does not exercise a model forward.

Still required after source integration is complete: compile/object preservation
and new-instantiation checks; component and full checkpoint replay, including
rejection, continuation, EOS/budget and native failure paths; native measurements;
and the final requested external review. Closed-window capture, whole-chunk
promotion and rejection/replay are implemented in source and still require their
native independent fixtures. No current speed, acceptance rate, native correctness or readiness
claim is made by the source receipt.
