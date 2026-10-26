# Shared-expert down cache-policy pilot

`STRIX_EXPERIMENTAL_MALL_SHARED_DOWN_NT` is a CMake option, disabled by default.
It selects non-temporal Q8_0 weight reads only in the shared-expert dense-A8
down projection for one through eight decode rows. Gate/up, other projection
sites, Q6 copies and wide prefill retain their existing read policies.

On gfx1151 the selected builtin emits SLC+DLC, not DLC alone. The switch
preserves loaded bits, activation reads, arithmetic, reduction order and
output conversion. It changes neither the numerical profile nor the tensor
format. Record the executable hash and CMake option when comparing builds.

Native OFF/ON checks passed on the 48-layer model for code with a 1K prefix
and prose with an 8K prefix: 64 generated tokens per prompt and 44 window
acceptance/rejection cases, including every physical logit row and all seven
physical state components. The replay hidden/logit errors were zero. These
checks preserve the existing numerical identity; they are not a quality change.

This remains an experimental execution policy. Complete workload performance
and interaction with other experimental changes require separate gates;
enabling the option is not evidence of a model speedup.


## Native production-base port

The production-base port retains the OFF-by-default CMake option. Native
Release gfx1151 builds pass warnings-as-errors and the CPU A8 contracts.
A production prose prompt of 8192 tokens has identical complete prefix receipts
and terminal full-vocabulary logits with the option OFF and ON. The retained
OFF receipt was reused only after checking its executable hash, full arguments
and production environment. An additional 1024-token code prompt qualifies the
narrow-R8 specialization with narrow enabled in both OFF and ON builds.

Each model run executes the serial oracle and 22 prefix acceptance, rejection,
owner-transition and fallback cases, comparing all physical logit rows and all
seven physical state components, including the initial prefill frontier and
discarded tails. The existing Qwen decode owners are R2/R4/R8 with R1 fallback.
The numerical profiles and arithmetic order remain unchanged.

The production shared module has 395 functions: 385 are byte-identical OFF/ON.
Only ten shared-down dense-A8 kernels change: the original eight and two newer
narrow-R8 variants that inherit the site policy. The changed kernels are wave32
with zero register-spill counts, and their load instructions carry SLC+DLC.
No application speedup or default promotion follows from these exactness gates;
the integrated workload campaign remains a separate decision.
