# Experimental fused-launch hyper injection

```sh
strix-hip-decode-token MODEL.gguf TOKEN_ID \
  --parallel-hyper-mix --staged-hyper-projections --fused-staged-hyper-mix
```

The optional fused selection requires the staged projection path. It defaults
to disabled and duplicate or inconsistent selections are rejected. The v4
staged path and the original parallel/default paths remain available.

The candidate appends injection workgroups to the low-rank projection grid.
The grid has `ceil(lowrank / 8) + ceil(hc_count / 8)` workgroups per row when
injection is present, and only `ceil(lowrank / 8)` for the final mixer. Every
workgroup chooses one uniform role: low-rank or injection. That role selects
the unchanged weight type, output count, ascending scalar accumulation and
epilogue. Canonical layers have 40 low-rank workgroups and one injection
workgroup. The final mixer has no injection workgroup and passes no injection
pointer to a consuming role.

Projection uses 128 threads with the same compiler launch-bound policy as v4;
no explicit launch-bound annotation is added. The normalization and v4 output
kernels are unchanged. Producer
and consumer barriers protect the existing 128-feature FP32 shared tiles;
inactive output lanes still participate. There is no dot-product reduction,
stream fork, activation conversion or weight conversion. Compiler contraction
and scheduling changes require independent numerical qualification.

All kernels stay ordered on the single owned execution stream. The same
generation-owned `hyper_workspace` remains
`rows * (hidden_size * hc_count + hc_lowrank) * sizeof(float)`, or 42,240 bytes
for a canonical row. Low-rank and injection workgroups read normalized input
and write separate admitted outputs. Output mixing runs after the combined
launch. No global resource, allocation or host synchronization is added.
The conservative LDS bound remains 4,864 bytes: the maximum individual kernel
declaration rounded to a 256-byte boundary, rather than a sum of concurrent
kernel declarations.

Kernel IDs add `.fused-injection-v5`, the compiler identity is
`hip-gfx1151-hyper-mix-v5-fused-injection`, and launcher identities fold a
distinct revision constant. Selecting the flag does not change v4 identities.

The operator tests poison the whole admitted workspace and both output buffers
before eager dispatch and each of three captured graph replays, then compare
the results against immutable paired baselines using bit comparisons. A no-op
negative control demonstrates that poisoned output fails the comparison. Cases cover
FP32/BF16 activations, packed IQ4_XS/Q8_0 fixtures, signed dense-weight tails,
attention/MLP/final mixing, invalid selection, missing/wrong/short/unaligned
workspace and catalog resource/revision contracts. Static operator capture
does not qualify the full-model graph.

This is an experimental scheduling change. Workgroup independence permits
overlap but does not guarantee it, and fixture equality alone does not
establish full-model fidelity. Actual compiler resource/ISA receipts, native
tests, real-checkpoint operator/logit comparisons and complete-workload timing
are required before promotion. No speed gain is claimed by this document.

The corrected Core10.1 gfx1151 build passes all174 device-enabled tests on one
device. A real checkpoint trajectory generates512 tokens at context capacity
8192; all554 complete vocabulary logit arrays agree bitwise with the staged
reference. An additional eager shadow poisons outputs and the whole workspace
before comparing1940 complete output sets, with zero differences across
9,963,520 output bytes. This qualifies only the tested checkpoint, inputs and
eager trajectory. Second-device tests, complete-workload timing and broader
graph, multirow and distributed qualification remain pending.
