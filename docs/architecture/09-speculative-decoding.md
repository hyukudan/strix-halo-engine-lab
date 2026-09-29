# 09 — Speculative Decoding

## Decision

Speculative decoding is an execution optimization, not a semantic mode.

For a fixed engine configuration, accepted speculative output MUST match the corresponding serial decoding contract.

## Required state

A speculative step may affect:

- logits;
- KV;
- recurrent state;
- short history;
- sparse index state;
- n-gram history;
- RNG state.

Verification and rollback MUST cover every affected state class.

## Draft depth

Draft depth SHOULD be bounded and may adapt to measured acceptance and cycle cost.

The controller SHOULD optimize completed-token time, not acceptance rate alone.

A deeper draft chain is useful only if the expected accepted work exceeds its additional proposal, verify and rollback cost.

## Greedy mode

Greedy speculative decoding SHOULD be qualified first because it provides the simplest exact-replay contract.

## Sampling

Sampled speculative decoding MUST NOT be enabled until RNG semantics are fully specified.

The random stream MUST be stable with respect to:

- seed;
- absolute token position;
- request identity;
- rejected drafts;
- batching.

Rejected speculative work MUST NOT consume randomness that changes the serial result.

## Verification

Verify kernels SHOULD evaluate candidate rows in parallel while preserving each row's serial arithmetic.

Verification MUST have a deterministic first-mismatch rule.

## Commit

Only accepted tokens may mutate committed state.

The runtime SHOULD prefer minimal snapshots plus accepted-path replay/recompute over full-state cloning when recurrent state is large.

## Metrics

The runtime SHOULD expose:

- proposed tokens per cycle;
- accepted tokens per cycle;
- acceptance distribution by depth;
- proposal cost;
- verify cost;
- rollback cost;
- net tokens per second.

Speculation MUST be disabled automatically or by policy when it loses against serial decode.
