# Optional AMD approximate prefill

`DevicePrefill` retains the incumbent arithmetic unless explicitly selected. These experimental profiles failed full-model qualification and must not be promoted. Decode arithmetic is unchanged.

The registered gate is per prompt family and length: mean KL <= 0.002 and top-1 >= 98%. Local operator accuracy did not predict model fidelity. A conservative activation/residual revision also failed. Keep all switches off until a revised profile passes the complete gate.

- Build `-DSTRIX_DS41_PREFILL_FP16_LT=ON` to include the hipBLASLt dense backend. Select with `STRIX_DS41_PREFILL_FP16_DENSE=1` or `PrefillOptions::fp16_dense`.
- Select BF16 high/low attention independently with `STRIX_DS41_PREFILL_BF16_ATTN=1` or `PrefillOptions::bf16_attention`.

Dense projects original FP32 activations through FP16 operands with FP32 accumulation. A power-of-two scale per activation row protects FP16 range and is undone in FP32. Effective Q8 weights are reconstructed once per model-owned view, including grouped output projections and K-split parts. Staging and solutions remain allocated until the prefill is destroyed. This increases resident memory and has an initial conversion cost.

`STRIX_DS41_FP16_CACHE=DIR` stores one solution index per exact internal R/N/K, GPU architecture, runtime hipBLASLt version/revision and profile schema. External input/output strides are handled by compact staging. Every disk hit is reconstructed through the library and checked for problem support and the 32 MiB workspace limit. Missing or stale records use a heuristic; `STRIX_DS41_FP16_TUNE=1` explicitly measures up to 16 candidates on first use and saves the fastest. Ordinary replay does no tuning or source rebuild. Freeze the resulting directory for qualification and deployment; a different solution requires a new quality gate.

`STRIX_DS41_FP16_CACHE_READONLY=1` rejects a missing or unsupported record instead of changing the solution. The environment-derived prefix namespace requires this mode for FP16 dense and includes the directory content digest. BF16 attention also gets a distinct namespace. Flags off retain the previous namespace. Callers setting `PrefillOptions` directly own their corresponding external state/cache identities. Approximate states are not bitwise equivalent to incumbent states.

Attention splits FP32 queries and softmax probabilities into BF16 high and residual terms, using two WMMA operations for each QK/PV tile. The packed K/V formats fit BF16 in its normal range; extreme underflow remains lossy. RoPE, online softmax, sink, chunk combination, normalization and output A8 keep their FP32 implementations. The entry point is exclusive to prefill; decode and segmented batch attention retain their current kernels.

`PrefillOptions::quality_head_rows` may request up to 64 final teacher-forced positions. Values above one require cone pruning and row splitting disabled. The result's `teacher_logits` holds complete vocabulary rows, with the last row checked against the existing final-position head by `strix-deepseek41-prefill-quality`. The normal result and normal head dispatch remain unchanged.

Qualification compares every standard prompt family/length separately against the current path, with identical effective weights: mean KL(reference || candidate) <= 0.002 and top-1 agreement >= 98%, rejecting incomplete output, nonfinite logits and inconsistent ranks. Gate each switch independently. Operator error and theoretical throughput do not establish full-model fidelity or TTFT improvement. State continuation, prefix-cache behavior, other lengths and combinations need their own qualification before promotion.


The quality runner accepts `MODEL manifest.tsv outdir rank [budget_seconds]`. A manifest row is `name tokens ids_path [arm_mask]`, with bits 1=current, 2=dense-only, 4=attention-only. It stops between arms with a conservative reserve when its budget is near exhaustion, allowing only unfinished cells to resume in another bounded lease. Completed rows include a hash of all teacher logits for rank agreement. Each invocation checks an exact repeated run on its first current-path cell; the first revision's failed cell was reproduced with that control passing. These quality-run wall times are diagnostic, not a performance campaign.
