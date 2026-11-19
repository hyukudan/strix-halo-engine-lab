# Qwen3.8-Flash-Next on one RTX PRO: `strix-cuda-qwen-serve`

`strix-cuda-qwen-serve` serves Qwen3.8-Flash-Next (UD-Q4_K_XL) from one RTX PRO 6000 (96 GB, Windows WDDM)
with the CUDA engine: exact prefill and exact greedy decode with MTP verify windows. It speaks the same APIs
as the node server: OpenAI chat completions and completions, Anthropic messages, SSE streaming, tool calls,
`/tokenize`, `/health` and `/api/status`, with the GGUF BPE tokenizer and the Qwen chat template.

Output is exact greedy: every request gets the token ids it gets when it runs alone, at any slot count and
with every default below.

## Build

Windows, Visual Studio 2022 and CUDA 13.0, `sm_120`:

```powershell
cmake -S . -B build-rtx -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRIX_CUDA_QWEN_ONLY=ON `
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-rtx --target strix-cuda-qwen-serve
```

`STRIX_CUDA_QWEN_ONLY=ON` builds only the CUDA Qwen targets; Strix Halo builds do not compile these files.

## Run

```powershell
.\build-rtx\strix-cuda-qwen-serve.exe --model Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00003.gguf `
  --mtp-sidecar MTP.gguf --draft-vocab DRAFT_VOCAB --tables TABLES_DIR --rope-table ROPE_TABLE `
  --bind 127.0.0.1 --port 8090
```

`--model`, `--tables` and `--rope-table` are required. Split GGUF parts are found from the first part.

| Flag | Default | Meaning |
|---|---|---|
| `--bind`, `--port` | `127.0.0.1`, `8090` | Listen address. |
| `--api-key-file F` | none | Require the key in `F` as a bearer token. |
| `--model-id ID` | `qwen3.8-flash-next` | Model name reported by the API. |
| `--max-positions N` | 32768 | Context cap per session. A prompt longer than the context returns `context_length_exceeded`; a session never spills to system memory. |
| `--mtp-depth N` | 7 | MTP draft depth per verify window. |
| `--min-p P` | 0.7 | The MTP draft chain stops at a draft token below this probability. Output ids do not depend on it. |
| `--http-slots N` | 4 | Concurrent sessions (continuous batching). `1` serves one request at a time; `STRIX_QRTX_COHORT=0` does the same. |
| `--slot-margin-mib N` | 2048 | A new slot is admitted only while this much VRAM stays free after its session. |
| `--admit-grace-ms N` | 2 | Wait for a burst of requests before the next step (`0` = off). |
| `--max-queue N` | 64 | Requests waiting for a slot; beyond it the server answers busy. |
| `--prefill-chunk N\|auto` | 1024 | Prefill rows of a new request per chunk, run between the decode steps of the other sessions. `auto` fits the chunk to the decode step time. |
| `--cohort-depth fair\|max` | `fair` | How MTP depth is shared when several sessions decode in one step. |
| `--block-rows N` | flat 8192 | Prefill block rows (multiple of 16, at least 1024). Untyped: one repeating block of 8192 rows (lever `qwen_cuda_flat_blocks`, bitwise; `STRIX_QRTX_FLAT_BLOCKS=0` or `--blocks 1024,2048,4096,8192` restores the ramp). See [prefill scheduling](../../engine/serving/qwen-cuda-prefill.md#flat-blocks-on-the-serve-entry). |

Every response carries the `strix.rtx` receipt: emitted ids, prefill and decode timings and slot occupancy.

## Engine defaults

All of these are exact (same token ids as the reference paths) and ON by default. Each has an off switch
that selects the reference path:

| Default ON | Off switch |
|---|---|
| Decode-shaped kernels for windows of up to 8 rows (dense A8, HC mix, router, routed MoE, QSA attention) | `STRIX_QRTX_DECODE_KERNELS=0` |
| CUDA graphs per verify window | `STRIX_QRTX_GRAPHS=0` |
| Graph descriptor tables | `STRIX_QRTX_GRAPH_TABLE=0` |
| Multi-segment QSA cores between graph segments | `STRIX_QRTX_MULTISEG=0` |
| Shared expert on a side stream | `STRIX_QRTX_SIDE_SHARED=0` |
| MTP drafts batched across sessions | `STRIX_QRTX_DRAFT_BATCH=0` |

Opt-in and OFF by default:

| Opt-in | Effect |
|---|---|
| `STRIX_QRTX_WINDOW_ROWS=16` | Verify windows of 9 to 16 rows. |
| `STRIX_QRTX_HEAD1=1` | One-row LM head per window row. |
| `STRIX_QRTX_DRAFT_SYNC=1` | Host sync after each draft step. |
| `STRIX_QRTX_PREFILL_OVERLAP=1` | Prefill chunks on a second stream beside the decode steps. |
| `STRIX_QRTX_PREFILL_POLL=1` | With overlap, decode steps continue until the chunk completes. |
| `STRIX_QRTX_ALLOW_SYSMEM=1` | Allow an allocation to spill to system memory (refused by default). |
| `STRIX_QRTX_NVTX=1` | NVTX profiling ranges. |

## Prompt prefix cache

The server keeps exact checkpoints of the prompt state in blocks of 256 tokens. A request restores the longest
cached prefix of its token ids (at most n-2 tokens) and prefills only the rest. It matches token ids, not a
session id, so it also hits when an agent client resends the conversation without the previous turn's reasoning.
Checkpoints are taken at the last block boundary before the final prompt token and every 16,384 tokens, in pinned
host RAM, LRU-evicted; row segments are shared between checkpoints of the same prefix.

- **Default:** ON. **Off switch:** `--prefix-cache off` or `STRIX_QRTX_PREFIX_CACHE=0` (the previous code path).
- **Budgets:** `--prefix-cache-host-mb N` (default: the smaller of 8,192 MiB and 25 % of the available RAM at
  startup; pinned host memory is reserved in slabs of 32 segments as checkpoints are captured and stays reserved
  until shutdown) and `--prefix-cache-vram-mb N` (default 0). A capture that does not fit is skipped; a request
  never fails because of the cache.
- **Tuning:** `--prefix-cache-block T` (default 256), `--prefix-cache-stride S` (default 16384, 0 = only the last
  block boundary; a multiple of T).
- **Root mark (opt-in, not yet measured on device):** `--prefix-cache-root on` also checkpoints the last block boundary
  inside the request's system prompt + tools (the leading system messages and tools rendered alone, when their tokens
  begin the prompt), so a new conversation with the same system prompt and tools resumes there instead of at the
  nearest stride multiple (0 below 16,384 tokens). One extra checkpoint per distinct prefix; restored state stays
  bit-identical (a block-aligned checkpoint like the others). The positions come from the shared mark plan
  (`include/strix/session/mark_plan.hpp`).
- **Reporting:** `usage.prompt_tokens_details.cached_tokens` is the number of restored tokens. The `strix.rtx`
  receipt adds `prefix_cached`, `prefix_restore_ms` and `prefix_capture_ms`. `GET /v1/cache/stats` and the
  `prefix_cache` object of `/api/status` report lookups, hits, captures, bytes, evictions, capture time and the
  reserved host bytes.
- **Compatibility:** a checkpoint is used only by the same process and the same key: SHA-256 over the layout
  version, block size, model, MTP sidecar, draft vocabulary, rope and table files (path, size, mtime), context
  cap, block schedule, PLE placement and every engine environment variable that selects kernels. Nothing is
  written to disk.
- **Exactness:** restored state, 64 decoded ids and their logits are bit-identical to a fresh prefill
  (`qwen_prefix_cache_e2e`: checkpoint-continue, restore into another session, changed suffix, changed boundary
  token; sync and overlapped prefill). With MTP, a checkpoint at position P also records token P and is restored
  only when it matches. Over HTTP, 48-50K-token agent turns gave identical output with the cache on and off (8/8).
- **Cost to other slots:** a restore copies the saved rows on the decode stream, so the other sessions' decode
  pauses for the copy; the pause grows with the number of restored tokens.

## Tests

`qwen_decode_e2e` (decode gate, rollback arms, draft invariance), `qwen_cohort_e2e` and `qwen_cohort_plan_test`
(multi-session equality and window packing), `qwen_qsa_multi_test`, `qwen_quant_host_test` and
`qwen_quant_device_test` (GGUF decoders against the host reference), and the `qwen_decode_bench` timing bench. Prefix cache: `qwen_prefix_cache_host_test` (planner, identity,
store, slab pool, batched capture) and `qwen_prefix_cache_e2e`.

Profiles: `--profile exact|default|fast|quality` and the lever options (`--hc-mma`, `--mtp-mma`, `--qsa-mma`, `--moe-mma`,
`--prefill-moe-route-bf16`; env `STRIX_QRTX_HC_MMA` / `_MTP_MMA` / `_QSA_MMA` / `_MOE_MMA`) are parsed by one shared parser
(`include/strix/cuda/qwen_options.hpp`) and resolved by the shared Qwen profile resolver over the CUDA rows in
`include/strix/profiles/qwen_cuda.hpp`; `--print-effective-config` prints the result, with the serve-only values named and
explained. FAST quality gate kit: `scripts/bench/cuda/qwen_fast/README.md` (`qwen_tf_logits`, `gate.sh`); host test `qwen_cuda_profile_host_test`.

FAST prefill forms (`--profile fast`, each with its own off switch; the exact path is byte-identical when a form is off):
the HC projection/output and the MTP fused projection on FP16 tensor cores (quantised codes are exact in FP16, FP32
accumulation, per-32-K-block FP32 scale), QSA attention scores/softmax/values on BF16 tensor cores (the KV cache is
BF16), and the MoE prefill project/down GEMMs with one FMA scale chain per output on int8 tensor cores. Shared tile
code: `src/cuda/core/mma_tile.cuh`, `src/cuda/core/attention_tile.cuh`, `src/cuda/core/imma_tile.cuh`. Gate receipt:
`gate.sh` run `fast-20261116015045` (12 prompts at 2K/8K/32K, 128 teacher-forced rows each): pooled KL 0.0091,
top-1 0.986, dNLL +0.0026. Exact lever: the dense projection runs `project_tile2_kernel` (same FP32 chain per output,
bitwise) by default; `STRIX_QRTX_DENSE_TILE2=0` restores the first tiled kernel. Its weight loader is the remapped one
(`qwen_dense_t2_remap`, bitwise, default on: 16 lanes per weight row, stage-invariant addresses, one CTA per SM without
the 128-register cap; `STRIX_QRTX_DENSE_T2_REMAP=0` restores the first loader). Site microbench
(`qwen_dense_tests --bench-shape M,K,N`, 8192 rows, 2.6 GHz, equal output hashes): 2560 x 16384 3.148 -> 2.267 ms,
2560 x 13312 2.570 -> 1.837, 6144 x 2560 1.227 -> 0.838, 2560 x 1280 0.282 -> 0.207, 640 x 2560 0.162 -> 0.131 ms
(218 -> 303 TOPS on the largest site). Without any load the same kernel runs 1.529 ms and without the FP32 chain
3.040 ms: the stage is bound by the weight loader, not by the chain.

### Routed MoE rows in decode windows

Fully resident CUDA Qwen MoE engines route and execute only the real rows of a
9–15-row verification window, rather than also processing its padding to 16 rows.
This organization is on by default for CLI and serving. Prefill blocks and tiered
expert residency retain their existing padded execution. Dense models have no
routed-expert work to eliminate.

`STRIX_QRTX_MOE_REAL_ROWS=0` restores padded execution. The shared resolver reports
`engine.moe_window_rows` in effective configuration; the loaded engine receipt adds
requested/effective policy, fallback reason and window counts. A change requires a
new engine/process. The numerical identity and real-row arithmetic are unchanged.

Qualification includes bitwise native verification and partial-accept rollback for
T=2–16, plus poisoned-padding operator checks. Paired old/new and reversed-order
native window measurements (two observations per arm from opposite orders) on a
full-resident Qwen MoE Q4_K_XL profile, 1,024 prompt tokens, 144-token benchmark
generation setting, CUDA sm_120 and a 534 W configured GPU cap gave:

| Window rows | Padded median (ms) | Real-row median (ms) |
|---|---:|---:|
| 8 (control) | 17.465 | 17.445 |
| 9 | 49.815 | 45.105 |
| 12 | 51.030 | 47.705 |
| 16 (control) | 51.325 | 51.480 |

These are window timings, not whole-model generation throughput or a claim about
all request mixes. Tiered execution and numerical quality qualification are unchanged.

### Shared native expert pipeline

The shared native expert kernels use padded group storage and vector loads to
avoid bank conflicts. A compile-time decoder binding removes the per-group
runtime type switch, and each down-projection weight group is decoded once per
expert tile rather than once per eight-row chunk. Per-assignment arithmetic and
reduction order remain unchanged. There is no new numerical route or control.

On an RTX PRO 6000 at a configured 534 W, the same GSQ IQ3_S model files, exact
prefill, 8,076 actual prompt tokens, prefix caching off and one emitted token gave:

| Pipeline | First request (tokens/s) | Second request (tokens/s) |
|---|---:|---:|
| Initial native pipeline | 406.4 | 407.9 |
| Conflict-free group storage | 1,019.9 | 925.6 |
| Typed decoder and down-group reuse | 1,665.9 | 1,717.3 |

The final second-request time is 4.703 seconds versus the initial 19.800 seconds,
a 4.21× improvement in this served cell. Each version used one fresh process and
two fixed request-order observations; these are not all-workload figures. The
timing client saved output counts rather than token IDs. Bitwise evidence comes
from the separate native captures, not this full-model timing series.

The group-layout change retains the full 60-case capture. The typed decoder
retains all 20 changed native records plus two T=51 cases covering multiple
chunks, a tail and runtime-source fallback. Each comparison covers complete
record bytes; unchanged MMA controls are reused.

| Adapter | Binding of this shared correction |
|---|---|
| CUDA Qwen MoE | Binds the shared NativeScalar/CudaNativeFp32 expert source and CSR/rawdot kernels. |
| CUDA DeepSeek | Uses IQ2_XXS-to-A8/Q2_K expert arithmetic with different scale and clamp contracts; this FP32 body cannot be bound unchanged. |
| CUDA GLM and MiniMax | Use trellis projections and transform geometry rather than GGML native group decoding. |
| Dense CUDA Qwen | Uses the shared dense BF16/Q8/NVFP4 core, with no routed-expert CSR/rawdot body. |
| HIP-only adapters | Have no CUDA native-expert binding; CUDA intrinsics and types are not transplanted into HIP. |

The implementation lives in `src/cuda/core/moe_a8.cuh` and
`src/cuda/weight/gsq_sources.cuh`; the Qwen adapter only selects the shared source.

### Served 8,076-token cell on the integrated build

RTX PRO 6000 Blackwell at a 600 W power limit (read, not set), the same GSQ IQ3_S model files, 8,076 actual
chat-template prompt tokens, prefix cache off, one fresh server process per row, model files in the page cache:

| Path | Client | First request TTFT | Second request TTFT | Numerical check on the same build |
|---|---|---:|---:|---|
| Exact defaults | 64 greedy tokens, MTP sidecar loaded | 2,565.2 ms (3,148 tok/s) | 2,565.8 ms (3,148 tok/s) | greedy ids (8,076 x2 + 2K prompts) and 256 teacher-forced logit rows byte-equal to the previous main build |
| `--profile fast` (`qwen-cuda-fast-bundle-v1`, `hc_mma` v2) | 1 token, under a kernel trace | 1,144.5 ms (7,056 tok/s) | 1,144.5 ms (7,056 tok/s) | FAST gate PASS: 8K pooled KL 0.00198, top-1 0.9941; 32K KL 0.00621, top-1 0.9688; short code / structured KL 0.00030, top-1 0.9969 |

The previous main build took 19,984.6 / 19,735.0 ms for the exact row in the same run frame. Each row is one
process and two request observations, not an all-workload figure; the FAST row's time was taken under an Nsight
Systems kernel trace (warm kernel sum 1,173.4 ms on two streams). `--profile fast` stays opt-in and provisional.
