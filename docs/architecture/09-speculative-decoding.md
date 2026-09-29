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

Depth zero MUST be an explicit candidate. Speculation SHOULD disable itself when the estimated full-cycle cost is worse than qualified serial decode.

A useful decision model is equivalent to:

```text
cost_per_committed_token(depth) =
    (proposal
     + verification
     + commit_or_replay
     + uncovered_communication
     + control_overhead)
    / expected_committed_tokens
```

Timing categories MUST be defined so overlapped work is not double-counted.

A deeper draft chain is useful only if the expected accepted work exceeds its additional proposal, verify and rollback cost.

Controller state SHOULD be conditioned on factors that materially change cost or acceptance, such as context class, concurrency, verify width, routing distribution, numerical plan and distributed topology.

The controller SHOULD use bounded exploration and hysteresis so noisy measurements do not cause rapid oscillation between depths.

## Greedy mode

Greedy speculative decoding SHOULD be qualified first because it provides the simplest exact-replay contract.


## Greedy acceptance contract

Greedy verification is now a pinned semantic contract.

For a draft of length `N`, the target evaluates `N + 1` greedy token
selections:

1. compare each draft token with the target-selected token at the same position;
2. accept only the contiguous matching prefix;
3. append exactly one target-selected bonus token at the first mismatch position;
4. if all `N` candidates match, append the target-selected token at position `N`;
5. crop the resulting continuation to the remaining length budget;
6. crop inclusively at the first EOS token when applicable;
7. recompute accepted candidate matches as `valid_token_count - 1`;
8. discard target-side speculative state for
   `candidate_length - accepted_matches` candidate positions.

With zero matches, the cycle reduces to one normal target token.

The verification contract is independent of the proposal mechanism. An MTP
predictor, prompt lookup or another qualified drafter may propose tokens, but
the unchanged target owns acceptance.

## Sampling

Sampled speculative decoding MUST NOT be enabled until RNG semantics are fully specified.

The random stream MUST be stable with respect to:

- seed;
- absolute token position;
- request identity;
- rejected drafts;
- batching.

Rejected speculative work MUST NOT consume randomness that changes the serial result.

## Proposal semantics

A proposal mechanism MAY be cheaper or more restrictive than the target model, including a reduced proposal vocabulary or approximate predictor, provided the unchanged target verifies every emitted token under the declared sampling contract.

A proposal optimization MUST NOT restrict the target model's own vocabulary, logits or sampling distribution.

Proposal-specific caches and weights MUST be included in memory admission and in the speculation cost model.

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
- net tokens per second;
- serial baseline cost;
- estimated and measured cost per committed token;
- state bytes written or retained for rollback;
- routed-expert overlap where MoE materially affects verification cost;
- exposed communication time in distributed mode.

Speculation MUST be disabled automatically or by policy when it loses against serial decode.
