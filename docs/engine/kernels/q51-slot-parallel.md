# Slot-parallel Q5_1 down projection

This experimental eager execution path separates each selected expert's Q5_1 down
projection from the ordered weighted merge. It retains the eight virtual leaves
per output, the original feature order and reduction tree. The merge starts at
positive zero and consumes slots in ascending order, using route weight as the
first fused multiply-add operand and the unweighted dot as the second operand.
Intermediate dots are stored as FP32 without conversion.

Enable with `--q51-slot-parallel-down` together with the grouped exact Q5_1 down
and Q4_K A8 gate/up options. It requires eager single-request, single-context
execution. Graph capture and integer projection combinations are rejected.
The option defaults to false. Its implementation marker precedes the grouped
and gate/up terminal markers so dispatch identity checks remain unambiguous.

The admitted model geometry has 48 layers, hidden width 2560, intermediate width
640, 512 experts and 10 selected slots. Exactly 43 layers use the paired format;
layers 2, 4, 30, 46 and 47 retain the existing fallback. Preflight checks the
complete classification before device residency or state initialization.

Each MLP workspace gains 102400 bytes of aligned raw dots, increasing its total
from 65280 to 167680 bytes. All matching workspace classes grow, including
shared and fallback allocations. Fork/join planning reserves the same additional
102400 bytes for its independent concurrent tail, in addition to changes in the
serial arena. Arena totals are calculated by the planner. Old layouts cannot be
bound to a request using the new path. Raw dots require neither clearing nor a
separate finite scan; existing output checks and commit rules remain in force.

The new path submits a dot grid across outputs and selected slots followed by
one ordered merge. It adds one launch and approximately 204800 bytes of logical
scratch traffic per eligible layer. The observed submission counter is reported
separately from the grouped-path counter. It increments after both kernels are
successfully enqueued, including calls later rejected by existing status checks.
Additional parallel work does not by itself establish improved occupancy or
end-to-end throughput.

Source implementation and host/native fixtures are present. Qualification still
requires source review, compiler arithmetic and resource inspection, native
execution, complete model logits and state checks, and clean paired timings.
No performance improvement or default promotion is claimed at this stage.
