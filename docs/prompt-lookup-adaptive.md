# Observed prompt lookup depth

`--prompt-lookup-adaptive` explicitly enables a bounded depth policy for the
existing prompt lookup target. Fixed depth remains the default. Adaptive mode
uses depths zero, one and two, capped by the actual request budget and context.
It cannot be combined with an explicit fixed depth.

The runtime starts a steady clock at `cycle` entry. The caller closes the scope
with `finish_adaptive_cycle` after consuming the pending bonus and publishing
the cycle's output/bookkeeping. The one wall interval includes proposal,
snapshot, target verification, restore/replay, target selection, bonus
consumption and intervening caller work. It is host elapsed time, not GPU busy
time. There are no separately added component estimates or overlap corrections.
Disabled mode adds no clock reads.

A cycle is eligible for cost selection only after successful consumption of
every emitted token. Its denominator is the actual change in committed target
tokens, checked against the emitted count and position change. A final cycle
that stops at EOS, budget or context with an unconsumed bonus is censored. In
particular, selecting the cached depth-zero frontier does not create a zero-cost
serial baseline. Failed cycles and invalid observations do not enter cost data.
Cost eligibility does not grant numerical or native qualification.

Within each cost context the controller first observes two completed cycles per
available depth, starting at zero. It compares total complete wall time divided
by total consumed tokens. A different preferred depth must improve this ratio
by more than three percent. Every 32 completed observations it explores one
available depth in rotation. Repeated exploration can select a slower depth;
this is an explicit opt-in policy with no promised speed improvement.

Cost history resets at 256-position buckets and whenever proposal identity,
effective maximum depth, the full EOS list or snapshot alignment changes.
Execution, numerical and resource bindings belong to the retained runtime;
their existing admission checks remain active. Re-priming also resets cost
history. A poisoned proposal state limits selection to depth zero without
relaxing target admission or suppressing the original proposal exception.

Receipt arrays report request-wide observations and selections, including work
across cost-context resets. They are not current-context cost estimates. An
unfinished observation must be closed before the next adaptive cycle. The
closure uses fixed storage and does not throw after a target commit. This source
integration has no new numerical/native/full-model qualification claim.
