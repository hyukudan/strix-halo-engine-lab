# Prefix fork: follow-up reuse of a retained batch slot

The batched Qwen MTP HTTP server answers a follow-up turn without prefilling the part of
the prompt that a finished request already consumed. `STRIX_SERVE_PREFIX_FORK=0` (or the
baseline mode) keeps every admission cold.

## What it does

When a greedy, unconstrained request finishes with reason `stop` or `length`, its slot keeps
its target state, session and predictor rows instead of leaving them to be reset at the next
admission. The server records the consumed history H (the prompt plus every emitted token
except the last) and the last emitted token, which is pending and not part of the state.
The frontier |H| is announced and sealed in a `PrefixDirectory` entry bound to that slot.

At admission of a new prompt P the directory is asked for a sealed entry whose frontier is a
prefix of P. The slot is reused when all of these hold:

- H is a strict prefix of P, with at least one token left after it;
- `P[|H|]` equals the retained pending token;
- the slot is free and its entry is still sealed;
- the request is greedy and unconstrained, and the server is not in nested service mode.

The slot then skips the target reset. The predictor runtime is rebuilt over the slot's durable
rows and restored to frontier |H|-1, the held hidden row h[|H|-1] is the one copied at the
end of the previous request, and the prompt prefill resumes at position |H| over `P[|H|:]`
(the session-suffix route of `TiledMtpPrompt`: serial rows up to the next block boundary, then
tiled blocks, or serial rows only when the remainder is short). The restored token count is
reported as the cached prompt tokens of the response.

## Fallbacks (cold admission)

A request is admitted cold, exactly as without the switch, when there is no matching entry,
the entry is stale, the pending token differs from `P[|H|]`, the request carries a constraint
or a sampler, or the server is in nested service mode. A resumed admission that the prompt
class refuses before any device work is re-admitted cold on the same slot.
A slot is not retained when the request was cancelled, hit the context limit, failed a
diagnostic or constraint check, asked for no retention, or any frontier check fails (session
health, arithmetic lineage, committed position, predictor frontier, pending token). Using a
retained slot cold, cancelling it, or abandoning the engine withdraws its entry. A cold
admission prefers a free slot without retained state, then the least recently retained one.

Frontiers shorter than 256 tokens are not retained (the directory minimum).

## Diagnostics

- Start-up prints `serve lever name=prefix_fork env=STRIX_SERVE_PREFIX_FORK active=.. value=..`.
- Each admission prints `prefix fork slot=N restored=T prompt=P hit=0|1 reason=<...>`.
- The end-of-serve stats JSON gains `prefix_fork_hits`, `prefix_fork_restored_tokens`,
  `prefix_fork_retained` and `prefix_fork_refused`.

## Exactness

Token ids equal a cold run of the same prompt (device gate). The state of a resumed slot was
produced partly by decode rows and partly by prefill, so its arithmetic lineage differs from a
cold prefill; the gate compares generated ids and the final-state digests
(`STRIX_FINAL_STATE_DIGEST`) against a cold run.
