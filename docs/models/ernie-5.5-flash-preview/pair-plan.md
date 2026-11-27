# ERNIE target-only pair bring-up

`strix-ernie-pair-plan` reads the official checkpoint headers and the shared
topology declaration. It emits a whole-layer cut for two ranks, chosen by the
shared layer-chain planner (`pair::chain`, layer-owned state, priced
layer-owned decode) from ERNIE's per-layer cache bytes; there is no ERNIE cut
constant. ERNIE has no measured rates yet, so the planner is given unit costs:
every layer costs one unit, the final norm and head one more on the last rank,
both ranks read at one declared rate and a hop moves one row per millisecond
(the plan is a prefill flow shop over 16 windows of the prefill width). The
inputs are data only, so a rank-scoped checkpoint copy that omits the peer's
shards chooses the same cut. The rank memory admission then checks that cut.
With these declarations the planner chooses 22/21. Replace the unit costs with
measured per-layer costs and link rates when the device fixture provides them.
Rank 0
owns the embedding and layers [0,22). Rank 1 owns layers [22,43), the final
hyper-connection, final norm, MultiMax and the vocabulary projection. Each rank
keeps its own sequence state. The seam transfers four contiguous FP32 residual
streams in stream-major order, 65,536 bytes per row.

The host adapter validates the rank's actual byte budget and streams the
owned ranges into a single aligned image. The device loader supplies the
allocation and synchronous write method. The first-token adapter verifies
the peer's checkpoint/control SHA-256 digest, numerical identity, context,
seam geometry and prompt IDs through the shared host transport. It binds the
range and device-copy callbacks to the shared prefill pipe and relays the
selected ID from rank 1 through the same link. Fake devices exercise this
path with two prefill windows, check all four streams and verify peer rejection
on a different asset digest. Native forward callbacks remain to be implemented.

The pair hello uses protocol word `0x324E5245` ("ERN2", the FourCC style of
the other engines' pair hellos; version 1 was the plain integer 1, so a peer
built before this change is refused at the hello). The hello also carries the
SHA-256 of the running executable and the device architecture, so two
different executables or architectures are refused before any activation
moves. The host path reports the architecture "cpu"; a native rank
executable sets the GPU architecture name in `FirstTokenSpec::gpu_arch`.

## Local admission for one sequence

The plan adds alignment per tensor, a conservative HCA carry buffer for
projected values and pooling scores in FP32, three prefill-window seam
buffers, chunked load staging, and bytes supplied by the allocator probe and
the deployment reserve. KV uses the candidate compact layouts, 528 bytes per
window or compressed HCA entry and 660 bytes per sparse MLA position. The
HCA count includes one reserved entry for any incomplete compression group.
Recalculate admission if the device fixture chooses a different cache layout.

| Rank | Target tensors | Source bytes | Resident weights, 256 B alignment | KV at 4K | F32 carry |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 65,572 | 77,578,932,752 | 77,578,974,208 | 9,715,200 | 9,961,472 |
| 1 | 65,539 | 77,275,806,300 | 77,275,846,912 | 9,630,720 | 9,437,184 |

With a 4,096-token context, one row per prefill window, 16 MiB of load
staging, and illustrative budgets of 2 GiB scratch and 4 GiB reserve per
rank, rank 0 needs 84,058,075,648 bytes and rank 1 needs 83,755,144,448 bytes.
The second number includes 804,864 bytes for F32 logits. These are planning
inputs and calculated memory requirements. Obtain usable memory and scratch
from both live allocators before qualifying the device session. The plan
defers 11,031,006,620 bytes of draft-only weights.

## Kernel binding matrix

The unchanged-body column refers to shape and storage interfaces. Qualify
the numerical binding with input and output fixtures for each row before
using it to compare against a CPU receipt.

| Stage | Existing bodies for compatible operands | Geometry or storage binding to add |
| --- | --- | --- |
| Embedding, weighted RMS, LM head, sigmoid gate | HIP tall BF16 and head `src/core/tall_family.hpp:46`, `src/core/bf16_lm_head.hpp:24`, sigmoid product `src/core/row_transforms.hpp:102`. CUDA dense `src/cuda/core/dense/dense_mma.cu`, RMS `src/cuda/core/norms/norms.cu:22`, greedy reduction `src/cuda/core/head_argmax.cuh:112`. | Bind embeddings and head at 201,216 × 4,096. Verify store boundaries for sigmoid and norm. Add the MultiMax epilogue after fixing the checkpoint contract. |
| Rotary and head norm | HIP interleaved 64-channel transform of a 512-element head `src/hip_ds41_attention.hip:96`. CUDA generic rotary pair `src/cuda/core/norms/rope.cu:10`. | Two rotary tables and both directions. Add the head-RMS/store schedule with the HCA head width 512 and MLA QK width 192 + 64. |
| Dense FP8/F32 | Shared E4M3/F32 decoder `include/strix/weight/formats/fp8_block.hpp`, HIP tall schedule `src/core/tall_family.hpp:68`, CUDA schedule `src/cuda/core/dense/rows_mma_body.inc`. | Both need the 128 × 128 F32-scale prologue, including a 64-row tail in the 576-row MLA KV-A projection. |
| Residual mixing | HIP partial/reduction/collapse/expand `src/core/hc_core.hpp:458`. CUDA four-stream collapse/expand body `src/cuda/models/ds41/ds41_core_norm.cu:209`, `src/cuda/models/ds41/ds41_core_norm.cu:319`. | Match the F32 feature-major 16,384 × 24 layout, enabled terms and 20 Sinkhorn iterations. The CUDA FP16 partial at `src/cuda/models/ds41/ds41_core_norm.cu:120` requires an F32 operand trait in the shared family. |
| Routing and experts | HIP grouped dispatch/reduction `src/moe_core/grouped_core.hpp:29`. CUDA projection and selector templates `src/cuda/core/route_project.cuh:41`, `src/cuda/core/route_select.cuh`, grouped schedule `src/cuda/core/grouped_w4a4.cu:47`. | Bind two BF16 router planes, 512 experts / top-10, learned gain, separate MXFP4 low-first E2M1/E8M0 by 32, 2,048 × 2,048 expert maps and SiTU in the shared epilogue `src/moe_core/expert_activation.hpp:72`. Existing W4A4 formats require a distinct operand trait. |
| HCA | HIP streaming softmax `src/attn_core/online_softmax.hpp:25`, window addressing `src/hip_ds41_attention.hip:107`. CUDA flash schedule `src/cuda/core/attn_flash.cu`. | Add an R128 channel-tiled gated APE pool to `src/attn_core/group_pool.hpp:41` with bounded shared memory. Bind the two rotary domains, causal history, sink and output merge. Existing compressed-cache operands need a new trait. |
| MLA, indexer and head mixing | HIP latent score and value bodies `src/attn_core/selected_exact_mla.hpp:111`, `src/attn_core/selected_exact_mla.hpp:245`; grouped index scoring `src/attn_core/index_score_wmma.hpp:82` and radix selection `src/attn_core/select_radix3.hpp:120`. CUDA selected-block schedule `src/cuda/core/attn_selected_blocks.cuh`. | Bind MLA rank 512, RoPE 64, split K192/V256, the offset sink and 64-head indexer with Q/K width 128 and causal top-2048. Supply the Hadamard/FP4 keys, FlashIndexer head-gating policy, VHA rank 16 and gated attention as shared score, gather and epilogue traits. |

## First device run

Run from the checked-out engine. Set `SOURCE` to the complete official
checkpoint (the source volume), `MODEL` to a rank's local target directory,
`RECEIPTS` to a receipt directory and `R` to the rank. Each rank copies only
the shards that hold its own tensors: the rank-scoped plan reads the complete
source once and lists them, and the same plan on the rank copy refuses if any
tensor the rank owns is absent.

```sh
set -eu
mkdir -p "$RECEIPTS" "$MODEL/flashindexer"
build/ernie-pair/strix-ernie-pair-plan "$SOURCE" "$TOPOLOGY" 4096 1 256 16777216 \
  2147483648 2147483648 4294967296 4294967296 --ranges --rank "$R" > "$RECEIPTS/rank-$R-source-plan.tsv"
{
  printf '%s\n' config.json generation_config.json model.safetensors.index.json tokenizer.json tokenizer_config.json \
    flashindexer/flashindexer.safetensors
  awk -F '\t' '$3 ~ /^model-/ {print $3}' "$RECEIPTS/rank-$R-source-plan.tsv" | sort -u
} > "$RECEIPTS/rank-$R-assets.list"
(cd "$SOURCE" && xargs -d '\n' sha256sum < "$RECEIPTS/rank-$R-assets.list") > "$RECEIPTS/rank-$R-assets.sha256"
rsync -a --files-from="$RECEIPTS/rank-$R-assets.list" "$SOURCE/" "$MODEL/"
(cd "$MODEL" && sha256sum -c "$RECEIPTS/rank-$R-assets.sha256") > "$RECEIPTS/rank-$R-assets.log"
```

Rank 0 (embedding and layers [0,22)) reads 60 of the 128 primary shards,
77,766,645,728 bytes; rank 1 (layers [22,43), the closing hyper-connection,
final norm, MultiMax and the vocabulary projection) reads 61 shards,
79,672,826,672 bytes. One shard holds tensors of both ranks. The five JSON
control files (30,246,762 bytes) and the FlashIndexer sidecar (3,148,112
bytes) go to both ranks. The eight shards that hold only drafter tensors
(9,759,356,648 bytes) are copied to the head rank when the drafter binds.

Set `TOPOLOGY` to the shared `strix-topology-v1` TOML with ranks 0 and 1,
a USB4 link, and both nodes' usable memory. The illustrative scratch/reserve
figures below reproduce the memory table. Run the transport benchmark on
both ranks together, after the builds and fixture tests are complete.

```sh
cmake -S . -B build/ernie-pair -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DSTRIX_ENABLE_HIP_RUNTIME=ON
cmake --build build/ernie-pair --target strix-ernie-pair-plan strix-ernie-pair-plan-tests strix-dist-host-exchange-bench \
  strix-hip-hc-core-tests strix-hip-ds41-attention-tests strix-hip-ds41-native-fp8-tests strix-hip-qsa-index-score-wmma-tests
ctest --test-dir build/ernie-pair --no-tests=error --output-on-failure \
  -R '^(strix-ernie-pair-plan-tests|strix-hip-(hc-core|ds41-attention|ds41-native-fp8|qsa-index-score-wmma)-tests)$'

build/ernie-pair/strix-ernie-pair-plan "$MODEL" "$TOPOLOGY" 4096 1 256 16777216 \
  2147483648 2147483648 4294967296 4294967296 --ranges --rank "$R" > "$RECEIPTS/rank-$R-plan.tsv"
timeout 120s build/ernie-pair/strix-dist-host-exchange-bench --transport usb4auto --rank "$R" \
  --sizes 65536,262144,1048576 --warmup 5 --iters 20 > "$RECEIPTS/rank-$R-link.log"
```

The executable commands finish at the plan and link receipt. The next
step needs a native rank executable that uses the range loader and
`run_pair_first_token` hooks. First admit a single target-only slot and
load rank-owned ranges, using measured alignment, staging and scratch.
Feed the first oracle prompt's token IDs unchanged with greedy selection.
Save both ranks' ACT digest, local layer digests, the raw logits from the
final prompt row and the selected ID. Compare IDs, per-layer digests and
the first-token top-20 against the host oracle. A first native result is
BASE until bytewise equality establishes EXACT. The first HCA history
qualification must also exercise positions 127–129.

Use `prompt.u32`, `generated.u32`, and `hidden.tsv` in each rank receipt
directory. Put the first full FP32 vocabulary row in
`rank-1/raw-logits.f32`. Write the rank-local `seam.tsv` with columns
`position` and `sha256` for the outgoing or incoming FP32 streams. Once the
native callbacks emit these receipts, run the comparison against a completed
CPU prompt directory.

```sh
python -I tools/ernie_trace_compare.py "$ORACLE/prompt-0" "$RECEIPTS/rank-0" "$RECEIPTS/rank-1" --cut 22 > "$RECEIPTS/first-token-compare.json"
```

The receipt reports the reference identity. Agreement with a candidate raw-head
trace checks the same raw computation and stream layout.
