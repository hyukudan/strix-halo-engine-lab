# ERNIE 5.5 Flash Preview host engine binding

`strix-ernie-engine-plan CHECKPOINT_DIR [--list] [--cut N]` validates the sharded
safetensors inventory and manifest. Its TSV identifies first-row consumers, new
host candidate consumers, and pending tensors. `--list` includes all bound
tensors and their source ranges.
The command reads one 256-byte F32 attention sink from layer 0 after validating
the complete header set. The command exits nonzero on incomplete inventory,
unknown tensors, malformed metadata, or probe failure. Output and status can be
captured separately.

`strix-ernie-cpu CHECKPOINT_DIR PROMPTS.json NEW_TRACE_DIR --generate 32`
runs the target-only CPU reference. The prompt file contains an array of
objects with `messages` and optional `tools`. The runner renders ERNIE chat
with thinking disabled, preserves the causal history and selects greedily from
the scores after the checkpoint's learned linear and quadratic hinges. The trace
contains little-endian prompt and sampled IDs, raw and transformed FP32 logit
files, first-row top-20, per-layer stream digests/RMS and selected route IDs
and weights. Raw scores in the checkpoint host profile are widened from the
BF16 projection store. `--head-store-f32` exposes the FP32 accumulator for
comparison on the same states.

The host runner accepts `--threads` and `--cache-mib`. Its single-thread,
zero-cache configuration keeps the scalar dot order. For a schedule
comparison, `--replay PRIOR_TRACE_DIR` feeds the same continuation IDs and
reports full-row bit identity, KL and top-1 agreement. Capture diagonal moments
with `--capture MODEL_DESCRIPTION` on a free-running checkpoint trace. It
writes the per-layer files and a pack description with measured routes. The
two-bit and three-bit pilot plans replace the observed experts from a narrow
slice. Load an explicit sidecar with `--pack` and use `--replay` for paired
logit measurements. These options do not enter runtime tiers.

### Teacher-forced host diagnostics

```sh
strix-ernie-cpu CHECKPOINT_DIR TEXT.txt BASE --oracle --limit 64 --save-layers \
  --threads 12 --cache-mib 8192 --ordered-simd --table-mxfp4 --profile-host
strix-ernie-cpu CHECKPOINT_DIR TEXT.txt PROBE --oracle --limit 64 --resume BASE \
  --start-layer 7 --end-layer 8 --threads 12 --cache-mib 8192 \
  --ordered-simd --table-mxfp4
```

The oracle tokenizes ordinary UTF-8 text. Each row predicts the actual next
token from the causal prefix. `--bos` explicitly inserts the checkpoint's begin
token. `scores.tsv` contains IDs, negative log-likelihood in nats and top-1
hits. `run.json` includes the mean, perplexity and empirical unigram entropy for
the same target IDs. The oracle accepts up to 600 tokens. `rms.tsv`,
`stream-rms.tsv` and `layers.tsv` describe the residual before each indexed
layer and after the last layer. `--trace-sites` records the attention and FFN
residuals per stream in `site-rms.tsv`.
`--dump-logits` writes the FP32 scores before the selected MultiMax probe.

With `--save-layers`, the runner writes a little-endian F32 image at each
boundary. Use the same immutable checkpoint and text to resume that image.
The runner checks the saved token IDs and boundary size and rebuilds causal
history within every recomputed layer. Boundary 43 is the input to the output
head. Stopping earlier applies the final head as an intermediate diagnostic;
the resulting loss is a logit-lens score for that boundary. `--lens-stride N`
scores every Nth position at each boundary with the same final head and writes
`lens.tsv`. To score previously saved boundaries without rerunning the stack,
combine `--rescore-layers` with `--resume BASE` and the desired layer range.

The scalar reference and the opt-in ordered FMA path run the same column
chain for each output. The batch head streams each weight block once for
every position. The table MXFP4 decoder is also opt-in and widens the same
low-first E2M1 and E8M0 operands. `--profile-host` reports projection and
weight-access timings separately. Contract switches are experimental host
diagnostics and carry their settings into the receipt.
`--parallel-mla` distributes independent attention heads and folded head-mix
output rows over the same host workers. It admits the decoded KV-B matrix to
the bounded cache. `--variant hca-fp8-epilogue` selects block E4M3 stores at
the two HCA output projection inputs for explicit comparisons.

`--recovered-contract` selects the host candidate that applies the learned
expert gain before renormalization, keeps the weighted KV normalization
without a second key RMS, and reverses the VHA factor order used by the
initial candidate. It keeps the raw output head. The mode is available to
both the oracle and the chat runner. Set `STRIX_ERNIE_ORACLE_CHECKPOINT` when
configuring the build to register the slow next-token loss fixture.
For generation probes, `--bos` prepends the explicit BOS ID to the rendered
chat. `--min-new-tokens` delays EOS eligibility for the specified number of
generated tokens. Raw logit traces preserve every score for those probes.

The [pair plan](pair-plan.md) describes native weight placement, memory admission,
the shared transport seam and the first device run.

| Stage | Shared component | Model binding |
| --- | --- | --- |
| Weight access | Safetensors index, typed slices, block E4M3 and split MXFP4 decoders | Named tensors and scale companions, FP8 128×128 with F32 scales, MXFP4 E8M0 per 32 columns |
| Attention | Causal window and rotary, compressed-history softmax and carry, selected latent attention, FP4 index selection and folded head mix | HCA R128, MLA per-head KV-B planes, index head gate and output gate wiring |
| Residual and norm | HC host contract and BF16 RMS with FP32 collapse variance | Four streams, feature-major F32 HC function, three scales and 20 Sinkhorn iterations |
| FFN and experts | Linear, split-score routing, activation policies, ordered expert reference and latent expert composition | Dense first block, latent RMS after combine, top-10 floor renormalization with per-expert gain and an ungated shared expert |
| Logits | HC head collapse, RMS, linear reference and learned hinge epilogue | Independent head projection and two vectors of learned coefficients |
| Draft | Shared block and chain policies, BF16 tap projection and MTP join, Markov score correction and greedy verification | Bundled block geometry, taps, shared embedding and head. The target runner leaves draft execution disabled. |
| Session and HTTP | `session::Engine`, `KvStore`, `SessionRuntime`, shared tokenizer/API and ERNIE chat binding | The serve adapter configures the API from an admitted session runtime and installs the existing ERNIE protocol |

The first-row diagnostic exercises a single position in any target HCA block.
The caller supplies the activation arithmetic. Its synthetic fixture uses
elementwise multiplication and assumes the first router selects, the second
router supplies the renormalized mixture, and expert gain applies afterward.
The diagnostic also assumes a zero-value sink, residual `U(Vᵀ·values)`, the
shared HC coefficient order, and a BF16 rounding boundary after weighted RMS.
The full CPU reference sums both sigmoid planes before ranking, renormalizes
the selected scores and applies the learned gain. Its activation defaults to
soft-capped SwiGLU with a sigmoid on the raw gate. Hard-cap and plain SwiGLU
are available for operator probes. Native captures are still required for
checkpoint parity at the compression boundary and for activation quantization.

## Next tasks for a target-only first token

1. **Freeze the numerical contract.** Obtain independent fixtures for SiTU at
   both configured beta values, the two router planes and gain placement,
   E2M1 nibble order, attention RMS and gate/VHA ordering, HC coefficients and
   store rounding, and MultiMax. Include nonzero VHA and HC weights, FP8 tile
   tails and expert ties. Record outputs for layers 0, 1 and 7, including each
   route ID and mixture weight. Compare staged tensors and raw logits before
   comparing selected IDs.
2. **Validate the host attention wiring.** The synthetic cases exercise HCA
   positions 127–129 and an incomplete final group, plus selected latent
   attention, packed index keys, masking and tie order. Compare state and scores
   from the official checkpoint across the compression boundary and beyond
   2,048 visible positions to establish the HCA and FlashIndexer contracts.
3. **Build the native bindings and verify the shape boundary.**

   | Bind the existing HIP body when its arithmetic and stores match the fixtures | Extend the existing shared family |
   | --- | --- |
   | BF16 tall projection and head (`tall_family.hpp`, `bf16_lm_head.hpp`), embedding, RMS, `row_transforms.hpp` sigmoid, rotary pair | Add the 128×128 F32-scale FP8 prologue with partial tiles (MLA KV-A has 576 output rows), grouped O and fused epilogues as traits |
   | Dispatch/tiling and combine in `moe_core/grouped_core.hpp`, the Sinkhorn/collapse/expand bodies in `core/hc_core.hpp` | Add the split MXFP4 operand for 2048×2048 with E8M0/32 and consecutive low-first nibbles, two BF16 512×4096 routers plus top-10 reduction, SiTU epilogue and F32 feature-major HC access or a load-time transpose |
   | `attn_core/online_softmax.hpp`, scalar head ops, radix top-k in `attn_core/select_radix3.hpp` | HCA ratio 128 and FP8 compressed entries (existing DeepSeek wrappers bind 1/2 and FP4 compressed entries), R128 `group_pool.hpp` with bounded LDS, 64-head VHA/gates, `selected_exact_mla.hpp` for QK192+64/V256, 64×128 `index_score_wmma.hpp` and FlashIndexer gate/top-2048 |

   Start without native fusions that change arithmetic. Compile shared
   traits on the target GPU, inspect registers, spills and occupancy, and
   run no-model fixtures for each real shape. Existing HIP fixtures can
   establish the starting contracts with
   `timeout 120s ctest --test-dir build -R '^strix-hip-(hc-core|ds41-attention|ds41-native-fp8|qsa-index-score-wmma)-tests$' --output-on-failure`.
   Add the ERNIE shape cases before running the checkpoint, and recheck any
   existing engine that shares changed arithmetic.
4. **Admit one slot on each node.** Start at 4,096 context with a candidate
   cut at layer 22. Load the indexed local ranges directly into one packed
   resident image per rank. Embeddings belong to rank 0, the closing HC,
   norm, MultiMax and head to rank 1. Measure aligned weight residency,
   long-lived KV/carry, staging, allocator overhead, prefill scratch and the
   804,864-byte F32 logits row separately. Use measured usable GTT and
   headroom to admit the slot.

   | Rank | Layers | Native weight payload | Estimated KV at 4K | Estimated KV at 32K | Estimated KV at 1M |
   | --- | --- | ---: | ---: | ---: | ---: |
   | 0 | [0,22), 19 HCA + 3 MLA | 77,578,932,752 B | 9,715,200 B | 68,732,928 B | 2,159,646,720 B |
   | 1 | [22,43), 18 HCA + 3 MLA | 77,275,806,300 B | 9,630,720 B | 68,530,176 B | 2,155,253,760 B |

   The KV estimates use `528 × (min(context,128) + ceil(context/128))`
   bytes per HCA layer and `660 × context` per MLA layer. They exclude
   alignment, carry and workspace. Keep the 11,031,006,620-byte DSpark
   payload out of first-token residency.
5. **Bring up the pair and check the seam.** Declare the layers, placement and
   link costs through the shared layer cut and pipeline. Keep KV local to its
   owner. Match model, weight and tokenizer identities, context, split and
   numerical identity during startup. For four FP32 activation streams, the
   forward seam costs 65,536 bytes per row. Return the selected ID/control
   status to rank 0. Use the shared host USB4 channel for the baseline plan.
   On each rank measure latency and bandwidth at one row and representative
   prefill sizes with
   `timeout 120s build/strix-dist-host-exchange-bench --transport usb4auto --rank "$R" --sizes 65536,262144,1048576 --warmup 5 --iters 20`.
   Validate truncation, sequencing and peer failure in the channel tests
   before executing a model.
6. **Compare a generation run.** Use the checked chat fixture IDs, one slot,
   greedy, draft off and one output token. Compare the seam payload, both
   ranks' KV and route state, full final logits and the selected ID against
   the host oracle. Repeat across the first compression boundary and record
   decode and prefill independently. Extend context and add DSpark only after
   the target-only run passes. Treat the native result as BASE until it has
   bitwise identity for EXACT. SP2/SP3 packing needs separate calibration,
   a native consumer, and held-out gate and timing receipts through
   `bind_band_receipts` before any tier admission.
