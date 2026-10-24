# Experimental staged router logits

`--staged-moe-router-logits --parallel-moe-router` requests an experimental
router data-movement organization. It defaults off and numerical profiles do
not enable it. The CLI rejects MTP composition; speculative-verify compilation
also rejects a selected staged-router kernel. No direct MTP router support is added.

Dispatch uses full-row staging only for F32 router weights with hidden width
2560, 512 experts and an actual nonzero weight address aligned to 16 bytes.
Other types, shapes or addresses retain the original logits kernel. The
implementation identity records the request, including its fallback behavior.
The receipt reports this admission policy, not a count of observed launches.

The candidate uses 30,720 bytes of static LDS and 256 threads per workgroup.
It reuses the activation conversion and ascending fused FP32 accumulation
chain, followed by the existing selection kernel on the same stream and
workspace. It adds no model backing or runtime workspace. Resource admission
declares the larger LDS bound even for fallback dispatches.

Source availability does not establish native fidelity or performance. Admission
requires native comparison of every logits bit and selected route/weight bit,
then whole-model logits/state validation within the same numerical recipe and
complete-request performance measurements. Changed arithmetic is a failure of
this candidate's contract; no speedup is claimed.
