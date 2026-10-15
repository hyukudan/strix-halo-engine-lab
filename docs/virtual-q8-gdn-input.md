# Experimental virtual Q8 GDN input projection

`--virtual-q8-gdn-input` selects a one-wave candidate for Q8_0 BF16 input projections in GDN. It is disabled by default. Other admitted weight formats and the final FP32-to-BF16 output projection retain their original kernels. The application requires a meaningful aligned Q8 input projection before loading model residency.

Each physical lane owns eight original leaves, numbered `8*lane + local`. Each leaf consumes features in ascending steps of 256. Shuffle distances 16,8,4,2,1 reproduce the original reduction strides 128,64,32,16,8. Register additions 4,2,1 complete the original tree. The existing Q8 decoder and BF16 rounding epilogue remain in use. The output workspace still stores BF16-rounded values in FP32 slots.

The capability uses a distinct selected-kernel suffix, implementation hash and compiler identity. Default identity remains unchanged. The composite GDN resource limits remain 256 threads and 16640 LDS bytes because its other stages still require those bounds; the new projection itself launches 32 threads without LDS or new scratch allocations.

The golden default launcher hash is independently derived from the unchanged 37d00b7 FNV tag/hash fields for the canonical fixture; it was not copied from candidate output.

The host test `strix-virtual-q8-gdn-oracle-tests` executes an independent 256-leaf fused oracle and checks that eight deliberately incorrect implementations are observable after BF16 rounding. A two-term cancellation at features 0 and 256 distinguishes fused result -15.1875 from unfused -15.125. A width of 257 already admits two terms in a leaf.

HIP qualification fixtures use canonical 16 key heads, 48 value heads and 128 head width, with Q8 inputs at hidden widths 128,288,640 and 2560. They compare complete output/convolution/recurrent-state bits against the original kernel, inspect captured kernel functions and 32-versus-256 launch dimensions, and poison admitted arena/workspace between fresh trials. Supported IQ4 fallback dispatch is capture-only in those fixtures; relabeled blobs are never executed. A separate F32 fixture executes fallback with the capability enabled.

Run the host oracle through CTest. On supported gfx1151 hardware, run `strix-hip-gdn-virtual-q8-*-tests`, the original GDN fixture, launcher catalog and CLI tests. Hardware acceptance additionally requires actual candidate ISA verification, complete full-vocabulary logit comparisons against a corrected reference, paired full-model timing and the supported strict compiler checks. The initial claim is an experimental arithmetic design, not a measured speedup. Finite-normal oracle scope does not establish universal generated-NaN or subnormal behavior; the original rounding/status semantics are retained.

## Qualification scope

Dense Q8 fixtures qualify each of the four input-projection arrays bitwise against an independent fused 256-leaf BF16 oracle for both original and candidate kernels. They separately require exact complete arena, output, convolution and recurrent-state equality between the candidate and corrected original, including static graph replay and fresh poison. Isolated projection checks also preserve the input and require every arena word outside the four output slices to retain its canonical poison. They retain independent CPU convolution bits, the fused-leaf negative control, finite-state checks and the existing CPU output bound.

The original scalar-double/libm recurrent reference has an observed rounding difference from the unchanged GPU pipeline at dense-fixture magnitudes. Those dense matrix errors are printed explicitly as diagnostics. No larger matrix tolerance is substituted, and no device values are fed into an independent expected result. The canonical sparse recurrence/WAR fixture and the F32 fallback retain all existing 1e-7 independent CPU matrix checks. This candidate does not claim that dense GDN matches the scalar CPU recurrence at 1e-7. Hardware qualification of the latest fixtures and full model remains a separate required gate.
