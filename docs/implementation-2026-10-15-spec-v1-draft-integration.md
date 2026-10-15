# Spec v1 reversible draft integration (2026-10-15)

This branch combines the wave router, Q8 indexed-attention input prefetch,
lane-chain HC output, deferred MoE status and Q8 shared-expert prefetch increments.
This initial integration preserved all low-level modes OFF by default. The later
[automatic profile policy](decode-optimization-profile.md) selects a content-bound
v3 candidate with wave, QSA, HC output, shared-expert prefetch and deferred status
after exact content/mode admission, pending its own qualification; manual mode preserves the
original explicit configuration. The independent HC whole-projection
prototype is diagnostic only and adds no decoder command-line selection.

The implementation preserves separate per-layer/per-step fallback IDs and separate
QSA/shared statistics owners. Final decoder receipts retain both actual projection
submission counters, wave admission counts, HC output admission counts and the
actual deferred owner's completed-token count. Deferred status still rejects graph
and non-C1 execution; composing options does not weaken that admission.

The feature decisions are D-015 through D-020 and evidence entries E-050 through
E-055. Source history retains each increment and follow-up. Shared integration
points include CMake, execution binding/compilation, launcher catalog, command-line
options, receipts and documentation. Each feature's dedicated fixtures remain.

Observed integration checks:

- Fresh GNU 13.3/WSL Release build with HIP OFF and warnings as errors; 157/157
  CTest entries passed, including the combined host-contract fixture.
- The combined fixture selects all five decoder modes in one serial plan, checks
  independent mixed QSA/shared/HC fallback IDs, verifies base execution identity
  with extra candidate registrations and checks the deferred graph rejection.
- C++20 syntax checks passed for the decoder, launcher catalog and catalog fixtures
  using private HIP API declarations. This checks host syntax, not HIP semantics.
- All 112 baseline device/global definitions across the five affected HIP units,
  plus the frozen increments' device/global bodies, remain literal matches after
  line-ending normalization. This is a source preservation check, not linked ISA.

The integrated source remains a draft. It requires review of the combined changes,
actual HIP compilation and linked legacy/candidate ISA, execution of native fixtures,
full-model array/token fidelity and paired preparation-inclusive workload timing.
Portable success does not establish compiler equivalence, GPU exactness or benefit.
No source-level result in this document qualifies a default change or performance
claim. Private review and comparative evidence remain outside Git.
