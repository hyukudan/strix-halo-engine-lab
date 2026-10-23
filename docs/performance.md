# Measured performance

Every speed number quoted in the [README](../README.md) is listed here with its date, configuration and
exactness check. All numbers are measured on our own machines; values are medians of 3 or more runs unless a row
says otherwise. Estimates and calculations are not listed as results.

**Re-measurement in progress (2026-10-23).** A defaults campaign is timing every new opt-in flag of the current
DeepSeek and Qwen builds against their production configuration. When it closes, the rows marked *(campaign)*
below are refreshed with the build it measured. Until then they show the last clean measurement.

Hardware names used below:

- **Node:** one AMD Strix Halo mini PC (Ryzen AI Max+ 395, Radeon 8060S `gfx1151`, 128 GB unified memory), Linux,
  ROCm.
- **Pair:** two nodes joined by two direct USB4 cables (one 40 Gb/s, one 20 Gb/s link) using the
  [USB4 driver series](https://github.com/hyukudan/strix-halo-usb4-driver).
- **Workstation:** a PC with an NVIDIA RTX PRO 6000 Blackwell (96 GB), reaching the pair over 1 GbE.

## Qwen3.8-Flash-Next

Model: unsloth UD-Q4_K_XL GGUF (111.33 GB) plus a locally distilled Q8_0 MTP draft sidecar (2.79 GB). Every row
uses MTP speculative decoding, greedy (temperature 0), thinking off, dense A8 numerical profile.

### One node

| What | Value | Date | Conditions | Exactness |
|---|---|---|---|---|
| Decode, code, 1K-token prompt | **59.9 tok/s** | 2026-10-22 | 256 tokens, MTP windows 2/4/8, production decode set | golden replay error 0 |
| Decode, prose, 1K-token prompt | **55.5 tok/s** | 2026-10-22 | same | same |
| Decode, prose, 8K-token prompt | 48.9 tok/s | 2026-10-22 | same | same |
| Decode, code, 32K-token prompt | 51.8 tok/s | 2026-10-22 | same | same |
| Prefill / time to first token, 1K | 916 tok/s / **1.13 s** | 2026-10-22 | fast prompt profile, 8192-row tiles | byte-identical to the serial route within the profile |
| Prefill / time to first token, 4K | 1,085 tok/s / 3.78 s | 2026-10-22 | same | same |
| Prefill / time to first token, 16K | 1,089 tok/s / 15.05 s | 2026-10-22 | same | same |
| Prefill / time to first token, 32K | 1,065 tok/s / 30.76 s | 2026-10-22 | same, `--unqualified-context` | same |

*(campaign)*: the code1k / prose8k prefill and decode rows are being re-timed against the new opt-in prefill
schedules; so far no schedule is faster than the default, so the default rows are expected to stand.

### Two replicas (one complete model per node, behind the router)

Setup: the [TensorMeld](https://github.com/hyukudan/tensormeld) router balancing active decodes; servers with
`--http-batch 8`, T 1024, a prefix-cache entry every 1024 tokens, per-request controller reset (D-057) and joint
windows (D-058); both servers restarted before each of 3 rounds, no warm-up; median of 3 rounds; 2026-10-22.
Token ids equal the single-node run in **570/570** responses. Evidence: [DECISIONS.md D-060](../DECISIONS.md#d-060-two-full-replicas-one-per-node-behind-the-router).

Two ways to count, both reported:

- **Sum of per-request decode:** completion tokens / decode time of each request, summed over the concurrent
  requests (what the hardware sustains while they all decode).
- **Wall aggregate:** completion tokens / wall time from release to the last response, including time to first
  token and queueing (what a client sees).

| Workload | Clients | Sum of per-request decode | Wall aggregate | TTFT p50 |
|---|---:|---:|---:|---:|
| Coding chat (~70-token prompt, 187-token answers) | 1 | **90.7** | 68.1 | 0.76 s |
| Coding chat | 2 | **181.5** | 136.2 | 0.76 s |
| Coding chat | 4 | **201.6** | 146.9 | 1.25 s |
| Coding chat | 8 | **255.0** | 166.7 | 2.37 s |
| Code, 1,024-token prompt, 256-token answers | 1 | 59.9 | 46.1 | 1.27 s |
| Code, 1,024-token prompt | 2 | 120.0 | 92.5 | 1.26 s |
| Code, 1,024-token prompt | 4 | 142.1 | 111.9 | 2.17 s |
| 8-category short set (36-303-token prompts, 82-200-token answers) | 1 | 58.2 | **42.8** | 0.89 s |
| 8-category short set | 2 | 116.6 | **85.8** | 0.85 s |
| 8-category short set | 4 | 142.3 | **102.9** | 1.32 s |
| 8-category short set | 6 | 159.4 | **110.6** | 1.91 s |
| 8-category short set | 8 | 177.5 | **120.4** | 2.55 s |

*(campaign)*: per-replica serving at 1 and 4 clients is part of the running campaign.

## DeepSeek V4.1 Flash

Model: DeepSeek V4.1 Flash (1017 checkpoint). Routed experts: the published 2-bit (Q2) GGUF. Dense weights: a
6.5-bit packed overlay with trellis-coded (EXL3/QTIP-style, [strix-trellis](https://github.com/hyukudan/strix-trellis))
shared-expert and head weights ("mix-B"). Production defaults: [ds41-production-defaults.md](ds41-production-defaults.md),
[DECISIONS.md D-061 to D-064](../DECISIONS.md#d-061-deepseek-v41-two-node-production-defaults).

Exactness on the pair: every gate run is bitwise. Greedy and DSpark windows equal the stored references on both
ranks, batch tokens equal the campaign reference, and the server's token ids equal the reference in 64/64 requests.

### The pair

| What | Value | Date | Conditions |
|---|---|---|---|
| Greedy decode, 1 stream *(campaign)* | **28.5 tok/s** | 2026-10-22 | production defaults with eager small sends (P17.3); bits identical with and without |
| DSpark speculative decode, 1 stream, 8 English / 8 Spanish / 8 code prompts × 256 tokens | **37.5 / 39.3 / 62.3 tok/s**, 46.4 over all 24 | 2026-10-22 | DSpark drafter on both ranks (P17.1) |
| Batched decode, 8 streams, aggregate *(campaign)* | **95.9 tok/s** | 2026-10-23 | production defaults, segmented batch steps, 8 prompts × 48 tokens, no drafting |
| Device prefill, 8K tokens *(campaign)* | **734 tok/s** | 2026-10-23 | expert-parallel prefill on the pair, final KV by owner (P17.5) |
| Device prefill, 1K / 4K / 16K tokens | 458 / 640 / 758 tok/s | 2026-10-22 | P17.1 build, before fused staging and final KV by owner |
| Server, 8-category short set, aggregate at 1 / 2 / 4 / 8 clients *(campaign)* | **38.4 / 52.7 / 59.3 / 66.8 tok/s** | 2026-10-22 | `strix-deepseek41-serve`, 8 slots, DSpark (drafting paused above 4 requests in flight), P17.4 |
| Server, 8 chats × 128 tokens at 1 / 2 / 4 / 8 clients | 41.4 / 54.0 / 59.7 / 67.9 tok/s | 2026-10-22 | same |
| Server, one client, prose / code chat, decode after the first token | 34.9 / **66.9 tok/s** (31.8 / 55.4 including time to first token; TTFT p50 0.50 s) | 2026-10-22 | P17.1 build |
| Server time to first token, 8K-token prompt, after 40 s idle *(campaign)* | **15.8 s** | 2026-10-22 | P17.5; 27.9 s while 4 other streams decode, longest decode stall 940 ms |

Measured effect of each production default (A/B inside the timing campaign, bitwise): segmented batch steps
N=8 +9.1 %; DSpark in the batch app +82 % at N=1, +53 % at N=2; DSpark in the server +62 / +52 / +19 % at
1 / 2 / 4 clients; eager small sends greedy +2.5 %; final KV by owner 8K prefill +1.8 % and idle 8K TTFT −9 %.

### The pair plus the RTX PRO 6000 (opt-in)

| What | Pair only | Pair + RTX PRO | Date | Conditions |
|---|---|---|---|---|
| DSpark decode, all 24 prompts | 46.4 tok/s | **61.8 tok/s (×1.33)** | 2026-10-22 | RTX runs layers [0, 22) with CUDA, the pair runs [22, 40) and the head; one hidden state crosses 1 GbE per step; P17.1 pair build |
| DSpark decode, English / Spanish / code | 37.5 / 39.3 / 62.3 | **50.3 / 54.2 / 81.1** | same | same |
| Greedy decode | 27.7 | 38.3 | same | pipeline fallback, one stream |

Exactness: without the trellis sites the split is bitwise equal to the pair-only run (61.7 tok/s, 24/24 prompts,
6,144/6,144 tokens). With them (the 61.8 row) the RTX layers sum the trellis shared expert in IEEE order: a
declared composed identity, checked against the pair (KL mean 1.7e-3, top-1 127/128), not bitwise. Evidence:
[ds41-pipeline-stage.md](ds41-pipeline-stage.md), [ds41-cuda-trellis-shexp.md](ds41-cuda-trellis-shexp.md),
[DECISIONS.md D-064](../DECISIONS.md#d-064-a-discrete-gpu-in-the-deepseek-v41-decode-rtx-split-opt-in).

**RTX PRO prompt reading, exact profile** (resident `strix-cuda-ds41-prefill-service`, 73.3 GB of experts pinned
in host RAM and the rest read from NVMe, warm, model load excluded, median of 3; the exported attention state's
SHA equals the exact reference; handoff measured to a local import endpoint), 2026-10-22:

| Prompt | Prefill time | Prefill tok/s | Time to first token including state export |
|---|---:|---:|---:|
| 2K | 3.39 s | 605 | **3.64 s** |
| 8K | 4.04 s | 2,029 | **4.39 s** |
| 32K | 11.55 s | 2,838 | **12.20 s** |

A faster CUDA prefill profile exists but fails the quality gate at long prompts; it is experimental and none of its
numbers are quoted as results.

**Hot-expert placement** (all dense weights and ~80 GB of the hottest experts on the RTX PRO, a 10 GiB expert
cache, the rest from `strix-deepseek41-expert-server` on the pair over 1 GbE): 40.1 tok/s greedy against 27.2
pair-only, 256/256 tokens equal; measured on an earlier pair build (2026-10-21). Evidence:
[ds41-hot-placement.md](ds41-hot-placement.md).

### Dense format quality

Trellis mix-B dense weights (all K4, head K5) against the 6.5-bit overlay, KL to the original-precision model over
103,908 tokens, computed in an emulation of the formats (not with the engine's kernels): mix-B 0.0446 at 2.27 GB per
token per node, 6.5-bit 0.0447 at 3.63 GB; paired ΔKL −0.0001.

## The USB4 link

Measured with the [driver series](https://github.com/hyukudan/strix-halo-usb4-driver) (four host streams over the
two cables, 2026-10-22): about **37 Gb/s per direction** of common-window throughput with both directions busy
(the sum of the four per-stream rates is 39.8 Gb/s per direction; the two are different measures). A 20 KB
exchange takes 18.4 µs p50 back to back (raw), 35.5 µs through the engine's host path.
