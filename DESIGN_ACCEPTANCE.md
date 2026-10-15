# Design acceptance actions

These project requirements consolidate the design review actions recorded on
2026-09-28. Original review inputs and correspondence are archived privately.
The dated actions below are historical acceptance criteria; current implementation
and qualification are recorded in [the ledger](docs/qualification-ledger.md).

| Area | Acceptance requirement |
| --- | --- |
| Model semantics | Establish canonical QSA/PLE/GDN, hyper-connection, output-gate and positional semantics using independent oracles before promoting optimized execution. Dense attention remains a separately identified baseline until applicability is established. |
| Complete memory | Account for KV, draft state, indexer, FP32 recurrent state, rollback, lookup caches, page tables and preparation peaks. Distinguish physical admission from mapping limits and preserve host headroom. |
| UMA residency | Measure allocation/registration/mapping/residency behavior on actual hosts. Discovery and apparent capacity do not prove no-fault execution or stable residency. |
| Numerical optimization | Separate checkpoint encoding, arithmetic, reduction order and persistent-state semantics. Select a faster path only after correctness, quality and complete-workload evidence. |
| Backend applicability | Bind exact compiler/runtime/device/library artifacts and unsupported paths. A library build or instruction emission does not establish full-model qualification. |
| Speculation | Bind authoritative target, RNG/acceptance and committed state. Retain serial/reference baselines and require independently verified rollback/commit. |
| Host control | Preserve explicit privileged-operation boundaries, reversible platform configuration and effective telemetry. Ordinary inference and browser-origin model output must not acquire administrator authority. |
| Publication | Preserve accurate provenance for actual adopted components and official references. Comparative inputs and review correspondence remain outside Git. |

The original 2026-09-28 review actions must not be interpreted as a current list of
unimplemented features or as experimental measurements. Acceptance is determined
by the exact implementation tests and target evidence for each current path.


D-016 / E-051: deferred eager MoE status remains an opt-in experiment. See
[the completion and recovery contract](docs/deferred-moe-status.md). Source and
portable tests do not qualify native HIP execution, linked arithmetic identity,
model arrays/token publication or complete-workload latency. Those gates must
pass before adoption; first-error merge and host-exception precedence are explicit.
