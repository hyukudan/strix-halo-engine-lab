# Committed QSA key service

The native MTP CLI and explicitly opted-in HTTP routes can retain one derived QSA key service per
session. Twelve target entries serve every admitted R2/R4/R8 verifier and serial
fallback. A separate thirteenth entry serves the predictor, including batched
predictor dispatch. Each HTTP slot owns its own service.

`STRIX_QSA_MTP_POOL=0` restores the historical MTP key paths, admission bytes and
inspection image. Otherwise the service defaults on when the route and memory
admit it on the qualified CLI routes in the table below. HTTP slots, nested
scheduling, session and prefix restore/reuse require `STRIX_QSA_MTP_POOL=1`.
Baseline, GSQ, defaults suppression, explicit FP32 pool storage, graphs,
hierarchical selection, wide-copy writers and explicit cohort requests retain
their existing paths. The explicit YaRN MTP CLI route can use the
retained table-backed service and rolling PLE by default for factor 2 through
524288 positions and factor 4 through 1048576 positions. Both `=0` off switches
remain authoritative. YaRN HTTP remains closed. The ordinary serial pool is a separate
policy. Failed service admission or allocation before prompt mutation retains the
historical native route and reports a decline reason. YaRN refuses that failure
before mutation; rerun with `STRIX_QSA_MTP_POOL=0` for historical consumers.

For capacity C and B = floor(C / 4), packed key payload is 13 * B * 128 * 2 bytes.
There is no persistent FP32 mirror or per-window service. One shared selector
scratch envelope reserves eight rows of max(ceil(B / 4), 16384) FP32 words. A
16 KiB host metadata envelope covers bounded entry, lease and event-handle state;
driver bookkeeping remains subject to the independent physical headroom reserve.
The reservation is charged once before window construction and per HTTP slot.

Canonical raw BF16 keys and KV remain authoritative. Each entry retains a
committed complete-block frontier and at most three provisional complete blocks.
A writer invalidates every block intersecting its first rewritten row before
publishing raw bytes. Accepted prefixes promote only complete, computed blocks.
Predictor proposals remain provisional until actual teacher/fast replay certifies
the cache rows. Restores, imports, replay, cancellation and prompt replacement
invalidate the appropriate entries. Derived keys are absent from snapshots. The prefix-cache restore/import hook
invalidates both domains before restoring canonical state. Fixed-session registry
restores and live boundaries start a fresh service epoch before state reuse;
ordinary request resets also start a fresh epoch. These boundaries drain prior
consumer events and preserve the admitted allocation.
Per-entry completion events order writers and readers across streams; leases
retain the service through queued consumers. Flushing deferred canonical state
retains certified keys.

Packing preserves the upper sixteen bits of the existing final BF16-rounded key;
integer widening restores exactly those FP32 bits before the existing products,
reduction tree and head accumulation. Selection and attention are unchanged.
`STRIX_QSA_MTP_POOL_CHECK=1` checks target and predictor score bytes against fresh
raw-key scoring, compares selected IDs and mutated scores, restores captured
outputs, and fails on any mismatch. Summary statistics report rebuilds,
invalidations, promotions, comparisons and differences.

`STRIX_MTP_LOGITS_DIGEST=1` captures actual physical window logits before slot
reuse and emits committed-step and terminal SHA-256 receipts. It also hashes the
actual final hidden ticket. Prompt/final state diagnostics require the component
registry (`STRIX_PROMPT_STATE_DIGEST=1`) and its reporting controls
(`STRIX_PREFIX_CACHE_DIGEST=1`, optionally `STRIX_FINAL_STATE_DIGEST=1`). These
checks synchronize and invalidate timings.

Host frontier/admission tests and device fixture sources accompany the service.
The default scope is limited to eager CLI rolling PLE, native MTP CLI, YaRN
factor-2 MTP CLI through 524K, factor-4 MTP CLI through 1M, and serial YaRN.
The 1M factor-4 prose gate matched all 256 tokens, every committed full-vocabulary
logit digest, prompt/final target state and predictor state across serial YaRN,
pooled MTP and pool-off MTP. This evidence does not qualify composed HTTP
scheduling or restore ownership. Host tests and builds add no GPU qualification.
Final-prefill seeding
and selector algorithm changes are not part of this service.

The shipped route policy is:

| Route | Rolling PLE | QSA MTP pool (`STRIX_QSA_MTP_POOL`) |
|---|---|---|
| Eligible eager tiled CLI | Default on | Native MTP CLI default on when admitted |
| Adaptive R2/R4/R8 + min-p | Leased CLI enabled | Enabled when admitted |
| HTTP, slots, nested scheduling | Off (unsupported) | Opt-in: `STRIX_QSA_MTP_POOL=1`, per-slot owner |
| Session load/live/follow-up | Off | Opt-in: `STRIX_QSA_MTP_POOL=1`; fixed-session boundaries invalidate |
| Prefix restore/import/reuse | Plain prompt-wide; MTP warm | Opt-in: `STRIX_QSA_MTP_POOL=1`; restore invalidates both domains |
| YaRN | Serial default on; MTP f2 through 524K and f4 through 1M default on | MTP f2 through 524K and f4 through 1M default on |
| Baseline / DEFAULTS=0 / GSQ | Suppressed | Suppressed |

Explicit opt-in never bypasses unsupported writers, rotary admission, capacity
or master off switches. A session save of the initial plain request can finish
its current rolling lease; subsequent loads/continuations use the warm route.
HTTP receipt counters for paid drafts and chosen window rows are emitted only
for an actual QSA MTP pool owner. With the pool off the historical receipt has no extra fields.

Automatic PLE setup failure, including unsupported direct I/O or a direct read
probe error, destroys provisional readers/leases and drops all rolling credit.
A fresh complete resident owner census checks total UMA, observed free memory,
workspace, metadata and GTT before warming the table or mutating prompt state.
If that envelope does not fit, execution refuses clearly before prompt mutation;
MTP is never silently disabled. Required reclaim refuses instead of substituting.
Post-admission read failures poison the source and exit nonzero with a clear
runtime error. There is no mid-run resident restart, even if tokens have not
been published: prompt/calibration may already have mutated target and predictor
state, and no pristine replay checkpoint or transactional restart owner exists.
Restart with a fresh invocation to obtain independent zero-credit admission.

The explicit YaRN CLI keeps the target's immutable table for all thirteen roles.
`STRIX_QSA_MTP_POOL_CHECK=1` reports per-role score comparison coverage with
`qsa-mtp-pool role=R checks=N scores=N differ=N`. Require all thirteen roles,
nonzero compared scores and zero differences; the aggregate alone cannot prove
that coverage. Common serial/MTP generation ledgers and target-state digests are
described in the [CLI contract](speculative/mtp-native-cli.md#explicit-yarn-mtp-cli).
These diagnostics and CPU tests do not extend the GPU qualification scope above.
