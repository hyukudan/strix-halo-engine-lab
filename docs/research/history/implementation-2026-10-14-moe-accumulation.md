# FP32 routed-expert accumulation and validation

Routed-expert gate/up and down reductions use FP32. The output accumulator now
also remains FP32 across every selected expert. Conversion to activation storage
happens once, after the final route; BF16 partial sums are no longer repeatedly
rounded. The launcher identity and compiler identity advance to v2.

The shared MLP scratch plan reserves two FP32 intermediate vectors, one FP32
hidden-width accumulator and a four-byte validation word. The existing shared
expert requirement is retained when larger. Buffer admission, variant scratch
accounting and launcher bounds use the increased requirement.

A device kernel checks every selected expert ID and route weight before any
expert computation. Invalid IDs or nonfinite weights set a validation flag;
subsequent expert kernels and final output publication are suppressed. Eager
C1 reads the flag after draining its stream and throws on failure. This adds
a stream synchronization to eager dispatch. Captured execution writes the
existing aggregate graph status and leaves checking to the graph completion
boundary, without a host read during capture. Scratch ownership extends to
actual completion in either mode.

Native tests use unequal hidden/intermediate widths, independently generated
packed weights and FP32/BF16 activation storage. BF16 expected output is rounded
only once. Invalid IDs and NaN route weights reject eager execution and captured
graph replay, with committed output bytes unchanged. These are operator checks;
full-model fidelity, workload performance and distributed continuity remain
separate qualification gates.
