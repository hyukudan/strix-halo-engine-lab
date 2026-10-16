# Generation request timing

The ordinary generation receipt reports `generation_request_elapsed_ms` from
two `std::chrono::steady_clock` readings immediately around the single greedy
request controller call. Both CPU and GPU token selection use this same outer
boundary. It includes prefix processing, generation execution, device readback,
token selection, and controller work. Model loading, residency setup, backend
construction, receipt serialization, and final recording/export are outside it.

`generation_request_output_tokens` is the actual emitted token count, including
an emitted EOS token when present. `generation_request_output_tokens_per_second`
is that count divided by the elapsed seconds; it is zero if the measured duration
is zero. The denominator includes the prefix, so this is whole-request output
throughput rather than a steady decode rate. Early EOS or context exhaustion use
the actual count instead of the requested maximum.

The receipt names this boundary in `generation_request_timing_scope`. These four
fields and both clock readings are omitted during RAW recording or final-state
validation export. Those paths have additional validation work and do not provide
performance evidence. Existing per-step timing fields retain their original
boundaries; CPU and GPU per-step sums cover different output work and are not a
substitute for the common request measurement.

This source change has not been compiled, tested, or measured. It changes neither
the request controller nor token/state publication or numerical operations.
