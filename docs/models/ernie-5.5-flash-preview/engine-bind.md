# ERNIE 5.5 Flash Preview host engine binding

`strix-ernie-engine-plan CHECKPOINT_DIR [--list] [--cut N]` validates the sharded
safetensors inventory and manifest. Its TSV lists every indexed tensor without a
first-row consumer. `--list` includes the bound tensors and their source ranges.
The command reads one 256-byte F32 attention sink from layer 0 after validating
the complete header set. The command exits nonzero on incomplete inventory,
unknown tensors, malformed metadata, or probe failure. Output and status can be
captured separately.

| Stage | Shared component | Model binding |
| --- | --- | --- |
| Weight access | Safetensors index, typed slices, block E4M3 and split MXFP4 decoders | Named tensors and scale companions, FP8 128×128 with F32 scales, MXFP4 E8M0 per 32 columns |
| Attention | Causal GQA with sink, hybrid RoPE also bound by DeepSeek, head RMS, low-rank head mixing, grouped linear and sigmoid product | Position-zero HCA, Q LoRA, one KV head, VHA before grouped projection, sigmoid gate before final output projection |
| Residual and norm | HC host contract and BF16 RMS | Four streams, feature-major F32 HC function, three scales and 20 Sinkhorn iterations |
| FFN and experts | Linear, split-score routing, grouped expert reference, latent expert composition | Dense first block, latent RMS/down/up, top-10 with a separate ungated shared expert |
| Logits | HC head collapse, RMS and linear reference | Diagnostic head immediately after block 0, raw BF16-head scores |
| Session and HTTP | `session::Engine`, `KvStore`, `SessionRuntime`, shared tokenizer/API and ERNIE chat binding | The serve adapter configures the API from an admitted session runtime and installs the existing ERNIE protocol |

The host reference exercises a single position in any target HCA block. The
caller supplies the activation arithmetic. The synthetic fixture uses
elementwise multiplication and assumes the first router selects, the second
router supplies the renormalized mixture, and expert gain applies afterward.
The diagnostic also assumes a zero-value sink, residual `U(Vᵀ·values)`, the
shared HC coefficient order, and a BF16 rounding boundary after weighted RMS.
These assumptions need independent operator captures before they define
checkpoint numerics. The first-block scores test the data path. Generation
requires all 43 blocks and MultiMax.

## Next tasks for a target-only first token

1. **Freeze the numerical contract.** Obtain independent fixtures for SiTU at
   both configured beta values, the two router planes and gain placement,
   E2M1 nibble order, attention RMS and gate/VHA ordering, HC coefficients and
   store rounding, and MultiMax. Include nonzero VHA and HC weights, FP8 tile
   tails and expert ties. Record outputs for layers 0, 1 and 7, including each
   route ID and mixture weight. Compare staged tensors and raw logits before
   comparing selected IDs.
2. **Finish attention in the host reference.** Bind the shared gated/APE
   pooling and an incremental carry policy to 128×512 HCA groups. Freeze the
   merge order and separate RoPE domains with positions 127–129 and an
   incomplete final group. Bind the shared selected MLA core and determine
   how to split KV-B into K192 and V256 planes. Freeze FlashIndexer head
   selection, Hadamard/FP4 keys, causal top-2048 and tie order. Capture the six
   MLA layers and the HCA cache layout before using the CPU path as a
   first-token oracle.
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
