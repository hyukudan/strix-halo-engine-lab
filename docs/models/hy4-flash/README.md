# Hy4-flash

[Español](README.es.md)

Status: **experimental.** The model runs on HIP (Strix Halo, gfx1151) on one node or as a two-node layer split, through
the runner `strix-hy4-rank`. The tokenizer, the chat template, the serve options and the host reference (CPU oracle)
are bound on the host. Serving (`strix-serve --model hy4_flash`) admits the pack and resolves its options, then refuses
to create a session: the device binding is reached only through `strix-hy4-rank` so far.

## Pack

Q4_K_M GGUF in two parts, read together (pass both paths, in order):

| file | bytes |
|---|---|
| `Hy4-flash-Q4_K_M-00001-of-00002.gguf` | 104,986,517,504 |
| `Hy4-flash-Q4_K_M-00002-of-00002.gguf` | 63,360,616,128 |

Tensor bytes 156.78 GiB (1416 tensors). The Engram table stays on disk and is read row by row. The rest is resident:
about 91 GiB on one node, or about 45 GiB per node in the layer split.

## How the forward pass is bound

Support is a binding on shared code (AGENTS.md, "Adding a model"). The table maps every stage to the shared
component that runs it on HIP, and says what is specific to this model.

| stage | shared component | model-specific code |
|---|---|---|
| tokenizer, chat template, tool-call parser | `serve/bpe_tokenizer` (split rule `hy_v4`), `serve/hy4_chat` on `serve/chat_template_kit` | template dialect (special-token suffix, reasoning efforts) |
| weights, tensor names | GGUF reader + `weight/tensor_map` table, `core::HipRangeWeightLoader` per part and layer range | the naming map (`weight/adapters/hy4`), which tensors a layer range needs |
| embedding (Q8_0, four streams) | `launch_hip_embedding_rows` + `ops/glu` epilogue | none |
| Engram rows (layers 1, 11, 21, 31) | `engram::RowReader` (host pages) → `engram/hip` row prep + inject kernels | the hash spec (`src/hy4/hy4_engram_spec.cpp`), the gate/scale buffer layout |
| hyper-connection mixes, collapse, expand | `core/hc_core` (`include/strix/hc` contract) | none (Hy4 spec preset in the contract) |
| RMSNorm (7 per layer) | `ops/norm` kernel, `OutputForm::bf16_double` | norm traits (declared square root, FP32 weights) |
| dense projections (q_a, q_b, kv_a, out, gate, dense FFN, shared expert, latent down/up) | decode: `q8_rows` direct form (+ BF16 store epilogue); prefill: `core/f16_dense_tall` q8a8 rows GEMM; router: dense rows tile | none |
| RoPE (interleaved tail, theta per layer kind) | `ops/rope` launcher + native table | theta per layer kind |
| latent cache row [512 \| 64] | `attn_core/cache_rows` writer | per-layer cache plan (`hy4_hip_plan.hpp`) |
| absorbed MLA heads (K_b, V_b) | `q8_rows` grouped heads (one launch, decode); q8a8 rows (prefill) | none |
| group indexer (compressed pool, score, select) | `attn_core/group_pool`, `group_score_scan`, `group_select_kernel` (`include/strix/attn/group_indexer.hpp`) | the Hy4 indexer preset |
| sparse MLA with sinks (DSA list / SWA ring 512) | `attn_core::sparse::launch_flash_rows` with the `mla_contract` policies | DSA / SWA specs (contract presets) |
| output gate | `ops/glu` sigmoid gate | none |
| router (F32, top-16, scale) | dense rows (F32) + `moe_core` route wave kernel, `moe/route_policy` preset `hy4` | none |
| routed experts (gate/up Q4_K, down Q4_K/Q6_K, clamp) | decode: A8 probe + down kernels; prefill: grouped FP16 known-rows core | none |
| expert combine + shared add | `moe_core` combine + `ops/glu` epilogue | none |
| head (BF16, 120832 × 3072) and next token | `hip_lm_head_launcher`; device greedy row (`hip_generation_output`) | none |
| two-node split | `pair::layers` layer pipeline (windows cross as FP32 rows), `pair/link` transports | default cut = middle layer |
| tier word | shared resolver (`strix/engine_profile.hpp`) | the lever rows (`include/strix/profiles/hy4.hpp`) |

### Files under the model directory and why they cannot be shared

| file | what | why it is model-only |
|---|---|---|
| `src/hy4/hip/hy4_hip_forward.{hpp,hip}` | the stage order of the forward pass on the shared entries above | the composition of this model's forward (stage order, tensor names, traits); it defines no kernel, every device kernel is a shared template |
| `src/hy4/hip/hy4_hip_plan.hpp` | stage table, per-layer cache plan, prompt window rules, pair cut | Hy4 geometry and layer kinds (indexed / sliding / Engram) as data |
| `src/hy4/ref/hy4_host_ref.{hpp,cpp}` | the CPU oracle | the model's declared arithmetic, stage by stage; it calls the shared host references where they compute the same bits |
| `src/hy4/hy4_engram_spec.cpp` | Engram hash spec | the model's hash constants and table geometry |
| `src/hy4/serve/session_service.{hpp,cpp}` | family options, effective configuration, pack admission | the family adapter of the shared serve factory (the same shape as the other families) |
| `apps/hy4_rank_main.cpp`, `apps/hy4_ref_main.cpp` | the device runner and the oracle runner | command lines of this model |
| `include/strix/profiles/hy4.hpp` | lever rows of the tier word | data for the shared resolver |
| `models/hy4-flash.toml` | descriptive entry | settings |

## Arithmetic class

The device path is class **BASE**: a quantised-activation class (dense Linears and absorbed heads take Q8_0 weights
times activations quantised to 8 bits; the routed experts run on the A8 kernels; device reduction orders). It is not
bit-equal to the host reference. It is checked against the oracle by token ids (teacher-forced and greedy). The layer split changes no bit: the pair reproduces the one-node ids and last-position logits digest. The
prompt window width changes no bit either (1024- and 2048-row windows give the same digest), but a batched prompt is
not bit-equal to the position-by-position prompt: in one measured case the greedy ids part after 68 generated tokens.

The host reference (`src/hy4/ref/hy4_host_ref.hpp`, contract `hy4-host-ref-v1`) runs one decode step at a time over
the text model: weights decoded to their exact stored values and widened to FP32; activations rounded to BF16 where
the reference model rounds (Linear outputs, norm outputs, the four residual streams); FP32 for the router,
hyper-connection mixes, indexer pooling and scores, softmax and the head; the shared declared reduction order and
transcendental functions, no fused multiply-add, lower index on ties. `Config::tiny()` with `synthetic_weights()` runs
the whole structure on random weights.

Tokenizer: begin of sequence 120000, end 120025 (the stop token), padding 120002; added tokens 120000..120831 are
matched whole; the post-processor adds no token (the chat template writes them). The pack may carry its own copy of the
chat template; `hy4_chat_dialect()` reads its special-token suffix and reasoning efforts and refuses a template without
them.

## How to run

Build with `-DSTRIX_ENABLE_HIP_RUNTIME=ON -DSTRIX_FAMILY_HY4=ON -DSTRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE=ON`.

One node:

```
strix-hy4-rank PART1.gguf PART2.gguf --ids-file prompt.ids --gen 128 --context 4160 --top 0 [--prompt-rows 2048] [--profile fast|turbo]
```

`--profile turbo` (`hy4-turbo-v1`) runs the fast cells until a higher cell has a measured regime.

`--ids-file` holds token ids (comma or space separated). `--prompt-rows R` runs the prompt in batched windows of R rows.
Without it the prompt runs position by position under the default tier word. `--top 0` prints only the greedy id.

The speculative decode drafts with the pack's next-token block. The DFly2 block drafter is opt-in:
`--dfly-arm sel` runs it beside the next-token block under the shared per-regime selector (`--dfly-arm only`: alone),
its shard from `--dfly FILE` or beside the pack; the server takes `--dfly FILE|beside`. A shard beside the pack is never
bound by discovery alone. The selector stays opt-in until its policy fix (bar measured after warm-up, re-probe) lands
and an ABAB per regime shows it never slower: measured as the default it ran 44.90 vs 45.67 tok/s in the record regime
and 45.10 vs 48.55 tok/s (-7.1 %) in the agentic regime, with equal ids.

Two nodes (rank 0 holds the embedding and layers [0, K); rank 1 holds layers [K, 42) and the head; K defaults to 21):

```
node A: strix-hy4-rank PART1.gguf PART2.gguf --rank 0 --peer NODE_B --ids-file prompt.ids --gen 128 --context 4160 --prompt-rows 1024
node B: strix-hy4-rank PART1.gguf PART2.gguf --rank 1 --peer NODE_A --ids-file prompt.ids --gen 128 --context 4160 --prompt-rows 1024
```

`--transport usb4auto|usb4gpu|tcp` picks the link (default `usb4auto`). `--cut K` moves the split. The pair needs a
window width: `--prompt-rows R`, or a calibration `--window-costs ROWS:MS,...` (half-window milliseconds per width,
measured on the deployment) from which it picks the width with the shortest pipeline wall.

Exact tensor-parallel decode (both nodes hold the whole pack and prefill the prompt alone; every decode step is split):

```
node A: strix-hy4-rank PART1.gguf PART2.gguf --tp-rank 0 --tp-rates 217,217 --ids-file prompt.ids --gen 128 --context 4160 --prompt-rows 1024 --top 0
node B: strix-hy4-rank PART1.gguf PART2.gguf --tp-rank 1 --tp-rates 217,217 --ids-file prompt.ids --gen 128 --context 4160 --prompt-rows 1024 --top 0
```

Attention is split by heads, o_proj, the layer-0 FFN, the shared expert and the head by rows, the routed experts by
slot (8 + 8 of the 16); 128 all-gathers per token on the shared pair gather link (`--tp-transport`, default
`usb4auto`) join the shares. Every split output is a whole result of the same kernel and the joins are
concatenations, so the ids equal one node. At start-up both ranks measure the link and the shared split planner prints
`tp placement:`; it refuses the pair when the pair is predicted slower (`--tp-force` runs it anyway, `--tp-trip-us`
replaces the measured trip).

Before the first collective the two ranks exchange their placement decision (the form and a hash of the measured
receipt reference) and both refuse at start-up when they differ, for example when only one rank was given the
deployment receipt. The pair placement (layers or tensor) is a `strix-hy4-rank` feature: the HTTP server
(`strix-serve --model hy4_flash`) serves on one node and never resolves a pair placement.

## Measured

Class (see [Arithmetic class](#arithmetic-class)): the default-path line is BASE; the table rows use the FAST lever
`prompt_rows` and are unqualified. Q4_K_M pack; Strix Halo nodes with 128 GB each;
the pair is the layer split at cut 21 over USB4. Each row is one run of the runner per configuration (cold = first
prompt after load, warm = the same prompt again in the same process with `--repeat 2`).

Default path (prompt position by position, the qualified prompt path), one node: a 256-token prompt in 15.77 s
(16.2 tok/s, 61.6 ms per position), then 32 greedy ids at 16.1 tok/s.

Batched prompt (`--prompt-rows`, the `fast` row; NOT qualified: logit gate code PASS / prose FAIL, so it runs only when
set explicitly or with `--allow-pending` and the run is stamped `unqualified: prompt_rows`). The pair always runs the
batched prompt (it does not start without a window width), so every pair figure is in this class. tok/s of the prompt,
32 greedy ids after it. Synthetic prompt: a sequence of about 66 tokens repeated up to the prompt length, so the
Engram rows the prompt reads are almost all repeats and a real-text prompt is slower than these rows:

| prompt tokens | one node, 2048-row windows (cold / warm) | pair, cut 21 (cold / warm) |
|---|---|---|
| 4096 | 526.2 / 542.0 | 738.8 / 761.8 (1024-row windows) |
| 16384 (the 4096-token prompt four times) | not measured | 901.8 / 909.2 (2048-row windows) |
| 32768 (the 4096-token prompt eight times) | not measured | 955.3 / 957.8 (2048-row windows) |

At 4096 tokens the pair reproduced the one-node greedy ids and last-position logits digest.

The layer split does not speed up decode: for one token the two halves run one after the other, and the hop adds to
it. Decode reads about 9.7 GiB of weights per token, so one node is close to its memory-bandwidth floor.

Exact tensor-parallel decode (1024-token prompt, 128 greedy ids, device greedy pick, host-staged gather route), one
node | pair | one node in one sequence: 16.52 | 19.92 | 16.51 tok/s (step median 60.5 | 49.9 | 60.5 ms); the pair's
ids equal one node's on both ranks (and 300 of 300 in a longer run). Each rank reads about 56 % of the one-node
bytes per token; the rest of the step is the 128 gathers.

## Tests

| test | needs weights | checks |
|---|---|---|
| `strix-hy4-tokenizer-tests [PACK]` | no (optional first part, header only) | split rule; with the pack: vocabulary, special ids, whole-vocabulary texts |
| `strix-hy4-chat-tests` | no | model card example, efforts, preserved reasoning, tools and response order, dialect, parser |
| `strix-hy4-host-ref-tests [PACK]` | no (synthetic weights) | end-to-end synthetic run, determinism, window boundary, indexer fallback, router, Engram rows |
| `strix-hy4-hip-plan-tests [PACK]` | no | stage table, cache plan, prompt windows, pair cut and window picker |
| `strix-hy4-placement-tests` | no (byte table fixture) | layer placement over the pack's component bytes |
| `hy4-serve` | no | serve options, effective configuration, admission |

All of them run in the public host CI. `STRIX_HY4_GGUF` (CMake cache) passes the first pack file to the tests that
accept it; only its header is read.

## Open points

- The pack stores the Engram table as one byte per value with no scale; the reference model round-trips each row
  through FP8 with a per-row scale. The norm that follows removes the scale except through its epsilon.
- The next-token block of the pack (block 42) is not part of the forward pass here: no device drafter yet.
- The batched prompt windows (`--prompt-rows`) are not qualified: their logit gate against the position-by-position
  prompt passed on code and failed on prose. `--profile fast` leaves them pending (they run only when set explicitly
  or with `--allow-pending`), and every run that uses them prints `unqualified: prompt_rows`. The pair always runs
  the batched prompt. Decode is the same under every tier word.
