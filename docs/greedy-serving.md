# Greedy serving

## Explicit shared-projection cohorts

`--serve-tensor-cohort-rows 2|4` optionally groups currently ready independent C1 targets, bounded by `--serve-concurrency`. It never waits for a full cohort. Only the three shared-Q8 projections use tensor rows; the rest retain their per-request R1 arithmetic and state. Unsupported member combinations use the scheduler's explicit R1 fallback. Default rows=1 retains independent workers.

The cohort staging payload is charged to the shared reservation before the cohort is constructed. Stream/event driver overhead remains in the explicitly supplied serving headroom. No request snapshot, cache or state is shared across sessions. Telemetry wrappers forward the request binding around actual cohort work; the cohort reports its own actual shared/fallback counters. Prepared mode validates retained metadata but cohort execution uses direct layer-major dispatch, not the prepared run hot path. These source candidates remain unqualified until final validation.
