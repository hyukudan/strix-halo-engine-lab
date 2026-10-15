# Typed hyper-connection row prefetch probe

Status: compile/ISA probe only. This variant has not passed device, model or
performance qualification. No speed or occupancy result is claimed.

`--prefetch-typed-hyper-rows` requires the existing typed, fused, staged and
parallel hyper-connection options. It selects a new kernel ID, implementation
hash and compiler identity `hip-gfx1151-hyper-mix-v8-typed-prefetch-rows8`.
All qualified prior identities and device definitions remain unchanged in source.
This replaces the unexecuted rows8-v1 compile probe with a new experimental
rows8-v2 identity; its earlier snapshot and artifacts remain separate.

Each producer lane loads its normalized feature and eight independently
decoded output-row coefficients into an unrolled register array before storing
to shared memory. The complete-row path removes per-row load branches. In a
partial row block, excess reads repeat the last valid row; only valid row
coefficients are stored. Invalid logical producer rows also target the last
valid LDS row with the identical coefficient; these repeated same-thread
stores finish before the unchanged consumer barrier and touch no invalid row.
Full/partial selection precedes each complete tile loop. The grid guarantees
at least one valid row in every
launched block, and the feature-tail predicate still guards every load.
Final mixers launch no injection blocks and do not dereference null injection
pointers.

Coefficient decoding calls the unchanged Q8_0/F32 helper. Consumer feature
order, positive-zero accumulator, FP32 operations, epilogues, two uniform
barriers per tile,128 threads,128 feature values and8Ã—129 coefficient pitch
are copied unchanged. The4,640-byte shared allocation and existing4,864-byte
declaration, global workspace and type preflight are retained. There is no
cross-tile prefetch, new launch bound, scheduler builtin, floating-point option
or cache transformation.

Source ordering does not guarantee independent hardware issue: the compiler
can resink loads or eliminate repeated partial-row reads. Before device use,
inspect actual gfx1151 code generation for overlapping valid coefficient
loads, preserved materialization/ascending accumulation/barriers, zero private
memory and spills, matching descriptor modes, and unchanged prior function
bytes and constant-pool targets. A source-level register array is not evidence
that private memory is absent.

Only after those gates pass should expanded full/partial-row operator fixtures,
poison/no-op/mutation controls, model logit/token fidelity and matched timing
be run. Current typed-projection qualification does not qualify this probe.
