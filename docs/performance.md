# Measured performance

Every speed number quoted in the [README](../README.md) is listed here with its date, configuration and
exactness check. All numbers are measured on our own machines; values are medians of 3 or more runs unless a row
says otherwise. Estimates and calculations are not listed as results.

The README shows, per mode, the **best** measured result and a **typical** one. Rows measured after this release
on integration builds are collected in [After this release](#after-this-release-pre-release-builds) with their
build id; they are candidates for integration. Rows marked *(campaign)* below are superseded there where a newer
measurement exists.

Hardware names used below:

- **Node:** one AMD Strix Halo mini PC (Ryzen AI Max+ 395, Radeon 8060S `gfx1151`, 128 GB unified memory), Linux,
  ROCm.
- **Pair:** two nodes joined by two direct USB4 cables (one 40 Gb/s, one 20 Gb/s link) using the
  [USB4 driver series](https://github.com/hyukudan/strix-halo-usb4-driver).
- **Workstation:** a PC with an NVIDIA RTX PRO 6000 Blackwell (96 GB), reaching the pair over 1 GbE.

## Evening snapshot (2026-10-25)

These integration measurements supersede older rows below only for the named build and workload.
They do not qualify unmeasured combinations. Public documentation records project-owned results;
raw lab receipts and review correspondence remain outside Git.

| Engine / profile | Hardware and workload | Latest measured result | Verification / status |
|---|---|---|---|
| DeepSeek V4.1 Flash 1017, Q2 experts + 6.5-bit/trellis dense | Two Strix Halo nodes TP2 over USB4, short R1 benchmark; `69c6bdab` | Median **29.0 ms/token** (34.5 tok/s); **58.2 %** of calculated bandwidth ceiling | Three interleaved pairs; token/window/reject and batch reference gates pass |
| Same, batch N=8 | Same hardware, fixed decode benchmark | Median **72.20 ms/step** (110.8 tok/s aggregate); **64.4 %** of bandwidth ceiling | Three interleaved pairs; all batch rows match the reference |
| DeepSeek standalone RTX, Q2 GGUF (IQ2_XXS gate/up, Q2_K down), Q8_0 dense | RTX PRO 6000 (96 GB, Windows WDDM), exclusive VRAM + pinned-host cache, on-GPU keep-64 DSpark; `5a88c0d1` | Warm controller: code **77.8**, EN **83.7**, ES **91.8 tok/s**; **48.1 / 51.7 / 56.7 %** of the calculated 161.8 tok/s plain bandwidth ceiling | 7 variable-length chat prompts, 128 generated tokens, two warm repeats each; 70/70 arm/chat comparisons and canonical-state gates pass |
| Qwen3.8-Flash-Next, UD-Q4_K_XL, fast prefill | One Strix Halo node, 32,768-token fixed prompt, 8,192-row tiles, 32 generated tokens; `a85c60ca` | Mean TTFT **22.95 s**, **1,428 tok/s**, versus baseline 25.78 s; **45.9 %** of pooled/overlapped ceiling (3,112 tok/s), **48.7 %** of serial-phase ceiling (2,933 tok/s) | Interleaved baseline/candidate runs; equal final logits digests and tokens; code-1K/prose-8K gates pass; integration confirmation pending |
| MiniMax M3.1 Flash, EXL3 3.06 bpw, exchange-boundary candidate | Two Strix Halo nodes TP2 over USB4, matched replay, 1K / 8K | **44.40 / 48.32 ms/token** (22.52 / 20.70 tok/s); **65.7 / 60.4 %** of bandwidth ceiling | Exact replay; defaults integrated in `8cc30ac0`, combined confirmation pending |
| MiniMax, exact prefill core | Same model/hardware, fixed 1K / 8K benchmarks, chunk interleave; `3e1897a1` | TTFT **3.230 / 22.338 s** | State hashes and continuation tokens match previous core; separate-process measurements, supported by in-process 1K repeats |
| GLM-5.3 Flash, EXL3 4.05 bpw, default R1 | Two Strix Halo nodes TP2 over USB4, 31-token chat, 64 greedy tokens; `ad6ea409` | **47.46 ms/token = 21.07 tok/s**; **49 %** of calculated 23.2 ms/token bandwidth floor (inventory at ~1K, 246.7 GB/s per node) | Tokens equal prior default anchor; four tested prompts span 47.46–47.56 ms/token |
| GLM, versioned R1b summation order | Same hardware, chat and three code/prose/JSON prefixes, 64 tokens each | **45.41–45.59 ms/token**, one run per prompt | Separate identity; greedy chat/code diverge from R1; opt-in, wider validation pending |

RTX rates use pooled committed tokens / timed decode-window duration after warm-up, excluding load and
prefill. Whole-chat decode rates are **77.1 / 82.1 / 91.2 tok/s** (code / EN / ES). The 74.8 GiB expert arena
shares the GPU with target dense weights and draft. The warm controller uses a capped survival prior and
seeded, frozen costs with exploration disabled; it is selected as the speculative default. Its exactness is
relative to this RTX profile, separately from the pair's trellis identity. Kernel follow-up measurements are
not added to the end-to-end result.

The pair's shared-expert grouped launches remain enabled where admitted; server plan warm-up remains off
because its first-request benefit did not extend across serving workloads. R6 never-slower behavior and the
latest serving campaign remain unresolved.

Qwen's combined prefill candidate stays opt-in. Exact MTP draft attention is enabled in the integration
line; batched MTP drafting and router changes still need complete-workload qualification. GSQ is an opt-in
speed-over-quality profile, never the default; its partial code-decode comparison and measured quality cost
are in the [README GSQ table](../README.md#gsq-opt-in-speed-over-quality-profile).

MiniMax's eight-text quality gate passed on 2026-10-25: mean KL to the exact reference **0.1449** versus CPU
production arithmetic **0.1425**; top-1 agreement difference **−0.47 percentage points**, inside the
pre-registered **2-point** margin; next-token accuracy difference **+0.01 point**. Both independent TP2 runs
are byte-identical. Qualification is conditional on those eight texts. Combined prefill follow-ups await
end-to-end measurements; the older two-prompt failure below is historical.

GLM's R≥3 buffer fault is fixed; fused attention projections preserve R1. Kernel/engine prefill gates and
synthetic multi-row checks pass, but real-pair prefill and batch timings remain pending. R1b is a separate
versioned numerical identity and is not the default.

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
Token ids equal the single-node run in **570/570** responses. Evidence: [docs/decisions/decision-register.md D-060](decisions/decision-register.md#d-060-two-full-replicas-one-per-node-behind-the-router).

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
shared-expert and head weights ("mix-B"). Production defaults: [production-defaults.md](models/deepseek-v4.1/production-defaults.md),
[docs/decisions/decision-register.md D-061 to D-064](decisions/decision-register.md#d-061-deepseek-v41-two-node-production-defaults).

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
[pipeline-stages.md](models/deepseek-v4.1/pipeline-stages.md), [cuda-trellis-shexp.md](models/deepseek-v4.1/cuda-trellis-shexp.md),
[docs/decisions/decision-register.md D-064](decisions/decision-register.md#d-064-a-discrete-gpu-in-the-deepseek-v41-decode-rtx-split-opt-in).

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
[hot-placement.md](models/deepseek-v4.1/hot-placement.md).

### Dense format quality

Trellis mix-B dense weights (all K4, head K5) against the 6.5-bit overlay, KL to the original-precision model over
103,908 tokens, computed in an emulation of the formats (not with the engine's kernels): mix-B 0.0446 at 2.27 GB per
token per node, 6.5-bit 0.0447 at 3.63 GB; paired ΔKL −0.0001.

## After this release (pre-release builds)

Measured through 2026-10-25 on integration builds that are not in this repository yet (the hash is our build id).
Same hardware, greedy, every run exact against its reference before its time is counted.

### DeepSeek V4.1 Flash, pair

| What | Value | Date | Build | Conditions and exactness |
|---|---|---|---|---|
| Greedy decode, 1 stream | **28.8 ms/token = 34.7 tok/s** (production: 34.8 ms/token) | 2026-10-25 | pre-release `940a4b85` + trellis library release candidate | interleaved with the previous candidate, 3 clean runs; every run exact against its reference on both ranks |
| Verify window of 6 rows | 54.77 ms/window (production 63.30) | 2026-10-25 | pre-release `940a4b85` | same confirmation; verify windows including rejected rows are bit-equal to one-row decode on both ranks |
| Batched decode, 8 streams | **73.93 ms/step = 108.2 tok/s** aggregate (production 85.64 ms/step) | 2026-10-24 | pre-release `cab238d6` | no drafting, 8 prompts × 48 tokens; 3 interleaved runs per arm; tokens identical |
| Device prefill, 1K | TTFT **2.18 s** (469 tok/s; production 2.21 s) | 2026-10-24 | pre-release `cab238d6` | interleaved A/B ×3; logits identical on both ranks and between builds |
| Device prefill, 8K | TTFT **10.78 s** (760 tok/s) | 2026-10-24 | `f9332f3e` | production 11.21 s in the same session; logits identical on both ranks and between builds |
| Device prefill, 16K | TTFT **20.40 s** (803 tok/s; production 21.36 s) | 2026-10-24 | pre-release `f9332f3e` | A/B ×3 in the same session; logits identical |
| Device prefill, 32K | TTFT **39.55 s** (828 tok/s), 2 runs | 2026-10-23 | `43086d05` with the three prefill defaults of the next release | defaults campaign; base 41.37 s; logits identical |
| DSpark speculative decode, 1 stream | **49.97 tok/s** over all 24; code 70.57 / English 41.94 / Spanish 45.33 (production 45.11 over all 24; code 62.62 / English 38.68 / Spanish 40.34) | 2026-10-24 | pre-release `cab238d6` | 8 English / 8 Spanish / 8 code prompts × 256 tokens; 3 interleaved runs per arm; every speculative run equals plain greedy on 24/24 prompts |
| Server time to first token, 8K prompt while 4 other streams decode | **26.40 s** (production 28.18 s) | 2026-10-24 | pre-release `cab238d6` | 3 interleaved runs per arm; ids identical idle vs loaded |

The server aggregate rows at 1-8 clients will be refreshed with the same candidate.

### Qwen3.8-Flash-Next

| What | Value | Date | Build | Conditions and exactness |
|---|---|---|---|---|
| TTFT one node, 1K / 4K / 16K / 32K | 1.14 / **3.72 / 14.46 / 29.60 s** (production build, same session: 1.15 / 3.83 / 15.24 / 31.24 s) | 2026-10-24 | `0a646b90` with the pipelined dense and reworked MoE prefill schedules (defaults of the next release) | fast prompt profile, fresh process per run, A/B ×4 in ABBA/BAAB order, median of 4; prefix state and logits identical to the release build |
| Copy windows, long copy-edit: document fix / patch / C++ rewrite | **5.451 / 6.102 / 6.449 ms/token = 183.5 / 163.9 / 155.1 tok/s** (default MTP: 10.76 / 10.87 / 10.87 ms/token) | 2026-10-24 | `e1b115a2` | production fast preset, `--mtp-ngram-draft=wide --mtp-copy-wide-rows 32`, A/B ×3, tokens identical; wide-window gate (tokens, full-vocabulary logits of every row and state equal to serial) PASS |
| Copy windows, 7 long copy-edit tasks (token-weighted) | **7.987 ms/token = 125.2 tok/s** (default MTP 11.086 = 90.2 tok/s) | 2026-10-24 | `dd65dbd5` | 1,114-2,047-token answers; A/B ×3; 78/78 samples identical to the default. JSON with an edit in every record: neutral (+0.2 %) |
| Copy windows, 6 coding-agent sessions | 16.708 ms/token = 59.9 tok/s (default 16.714) | 2026-10-24 | `dd65dbd5` | neutral (−0.04 %), tokens identical |
| Next release candidate, one node: TTFT 1K / 4K / 16K / 32K | **1.12 / 3.73 / 14.64 / 30.17 s** (release build, same session: 1.15 / 3.86 / 15.42 / 31.79 s; −2.3 / −3.4 / −5.1 / −5.1 %, ranges disjoint) | 2026-10-24 | `0117a140` | integration of the prefill, decode-stall and attention work with its new defaults, no extra settings; fast prompt profile, fresh process per run, interleaved A/B, median of 4; prefix state and logits identical to the release reference with the new defaults and with each of them turned off |
| Next release candidate, one node: decode 1K code / 1K prose / 8K prose / 32K code | **60.0 / 55.7** / 48.1 / 47.9 tok/s (release, same session: 59.8 / 55.5 / 48.1 / 48.2); prompt reading 3.7-4.5 % faster | 2026-10-24 | `0117a140` | 256 tokens, median of 4; 1K ranges disjoint; 8K and 32K vary run to run in both builds (32K ranges 44.5-52.6 and 44.3-52.6 tok/s, same tokens per cycle); tokens, state and final logits identical to the release |
| Next release candidate, two replicas: coding chat wall throughput at concurrency 1 / 2 / 4 / 8 | 68.2 / 136.4 / **146.8 / 158.1 tok/s** (release, same session: 67.7 / 136.2 / 141.3 / 152.7); TTFT p50 at concurrency 4: 1.28 s (1.37 s) | 2026-10-24 | `0117a140` | `--http-batch 8`, router, fresh servers per run with one excluded warm-up request, 3 runs per build in ABBAAB order, median; 90/90 responses token-identical to the single-node reference. At concurrency 8 the per-request TTFT median is 2.51 s against 2.46 s: it falls on the step between the second and third prompt each node reads in turn, and one run per build was slowed by queueing; the best runs of the two builds are the same |
| RTX PRO prefill offload, TTFT 1K / 4K / 8K / 16K / 32K | **1.36 / 1.92 / 2.94 / 4.95 / 8.99 s** (same server alone: 1.80 / 5.20 / 10.46 / 21.43 / 44.02 s) | 2026-10-24 | `70b93af3` | exact CUDA prefill (A8 profile) reproduces the node's prefill bit for bit (61 state components); KV and recurrent state streamed to the node per layer over 1 GbE with a lossless codec on 2 connections; 64 generated tokens identical to the node-only run at every length; one run per length |

### MiniMax M3.1 Flash (bring-up, preliminary)

EXL3 trellis weights at 3.06 bits per weight, two nodes, tensor parallel over USB4. Not in this release.

| What | Value | Date | Build | Conditions and exactness |
|---|---|---|---|---|
| Greedy decode, 1K prompt | **48.44 ms/token = 20.64 tok/s**; chat prompts 20.66 tok/s | 2026-10-24 | pre-release `3ba49a3d` | all-defaults bench, 256 tokens ×3; ranks agree, token ids checked |
| Greedy decode, 8K prompt | **50.80 ms/token = 19.68 tok/s** | 2026-10-24 | pre-release `3ba49a3d` | all-defaults bench; ranks agree, token ids checked |
| Prefill / time to first token, 1K / 8K | **5.20 / 37.73 s** (217 tok/s) | 2026-10-24 | pre-release `3ba49a3d` | all-defaults bench; ranks agree, token ids checked |
| Batched decode, N = 1 / 2 / 4 / 8 | **20.75 / 30.52 / 42.09 / 57.59 tok/s** aggregate | 2026-10-24 | pre-release `3ba49a3d` | all-defaults bench; each row identical to running alone |
| DSpark decode, JSON table, 256 tokens | **52.03 tok/s** (×2.52) | 2026-10-24 | pre-release `3ba49a3d` | bundled K4 drafter; ids equal plain greedy on the same build (20.6 tok/s) |
| DSpark decode, Python code / prose, 256 tokens | Python **32.73 tok/s** (×1.59); English prose 22.68 tok/s (×1.10); on the wider set, DSpark is roughly at or below plain speed on prose (0.88-0.95×), and 1.8-2.2× on code and JSON | 2026-10-24; wider set 2026-10-25 | pre-release `3ba49a3d` | bundled K4 drafter; ids equal plain greedy in all three initial prompts and on the wider set of 16 prompts: 2 each in 6 languages plus code and JSON |
| Quality gate (teacher-forced, 7 pre-registered rules) | prompt B: 7/7 pass; prompt A: 6/7, top-1 agreement with the exact reference 71.7 % against 73.8 % required | 2026-10-24 | pair build | top-1 against the real text within 1 point of the CPU reference; **not declared passed**; a gate with more prompts is planned |

## The USB4 link

Measured with the [driver series](https://github.com/hyukudan/strix-halo-usb4-driver) (four host streams over the
two cables, 2026-10-22): about **37 Gb/s per direction** of common-window throughput with both directions busy
(the sum of the four per-stream rates is 39.8 Gb/s per direction; the two are different measures). A 20 KB
exchange takes 18.4 µs p50 back to back (raw), 35.5 µs through the engine's host path.
