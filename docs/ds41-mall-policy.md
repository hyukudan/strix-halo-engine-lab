# Routed Q2_K down cache-policy pilot

`STRIX_EXPERIMENTAL_MALL_Q2_DOWN_NT` is a CMake option, disabled by default.
Enable it in a separate HIP build to select non-temporal reads of routed
Q2_K down weights, including packed TP half tails and held-group variants.
The Q2_K branch of the standalone format decoder uses the same read policy.

On the inspected gfx1151 compiler, these reads emit SLC+DLC. Recheck the ISA
after compiler changes. This switch
does not mean DLC alone. Codes, scales, activation reads, route selection,
floating-point operations and reduction order retain their existing contracts.
Other weight formats and the prefill implementation keep their existing policies.
The numerical profile identity is unchanged; record the executable hash and
CMake option when comparing builds.

Native format tests and a four-layer, two-rank emulated model gate passed
with identical full logits and serialized state for OFF/ON, including
window rejection. These checks do not qualify forty-layer distributed
performance or interaction with other experimental changes. The option
remains experimental and disabled pending those gates; no model speedup is
established by the cache policy itself.
