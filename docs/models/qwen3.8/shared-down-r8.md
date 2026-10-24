# Shared down R8 organization (draft)

`STRIX_QWEN_SHARED_DOWN_NARROW=1` selects a fixed-width LDS chain for the shared
expert's dense A8 down projection in an eight-row window with input width 640.
The default is off. Other window sizes, arithmetic profiles and cohort entries
retain their existing organization.

This specialization uses the existing A8 projection kernel. It preserves each
output's integer dot products, ascending Q8-block FMA chain, canonicalization,
scalar-gate multiply and BF16 store. Weights, expert routing and the window's
commit protocol are unchanged. The wave retains two outputs and stores only the
twenty real Q8 blocks of each chain, with one padding slot. Native performance
qualification is separate.

The projection fixture exposes explicit reference and narrow organizations,
independent of the environment. `STRIX_SHARED_DOWN_R8_ONLY=1` runs their bitwise
checks in `strix-hip-q8-rows-projection-tests`. Adding `STRIX_Q8_ROWS_BENCH=1`
also runs the short isolated warm/cold diagnostic.

## Prefix and state qualification

The test executable `strix-hip-mtp-model-qualification` accepts the normal
greedy MTP decode arguments plus `--qualification-prefix-gate NEW_DIRECTORY`.
This mode requires the three deferred-state owners R2/R4/R8, the min-p verifier,
32 free context rows and at most 1 GiB of session-state payload. It runs after
the actual prompt and exits before ordinary generation. The destination must
be new; qualification reference/export options are mutually exclusive with it.

The gate builds a serial target oracle from that prompt state, then calls the
actual verifier with forced rejected drafts at each possible position, complete
acceptance and owner transitions including serial fallback. It compares full
logits for committed rows, confirmed tokens, the seven state components after
materializing GDN, and the committed frontier. Generations count all physical
writes, including rejected rows, and are checked against that write schedule.
The receipt records these actual counters, every physical window row's logit
digest and all seven full physical state digests for comparison across builds.

KV and sparse-index cells at or after the committed frontier are outside the
logical state comparison. Only their host snapshot copies are canonicalized;
the live caches are never cleared to manufacture equality. All earlier cells,
recurrent and convolution state, PLE and host history remain covered. This
scope is explicit in the receipt. Discarded speculative cache tails are compared
only across baseline/OFF/ON runs with the same physical schedule, using the
separate full physical digests. Timing with this diagnostic is invalid.

Before restoring the initial test session, the runner materializes the previous
prefix and discards the owners' non-authoritative caches. Consecutive owner
transitions retain their caches and use the actual prior accepted ticket. The
runner optionally uses the system OpenSSL SHA-256 provider for large snapshots,
checks it against the portable provider, and keeps the portable fallback. This
dependency is private to the diagnostic executable.

Run the same gate against the baseline implementation, the candidate with the
flag off and the candidate with it on. Compare the complete receipts. A passing
diagnostic does not establish whole-workload fidelity or a performance gain;
those require the independent generation and measurement gates.
