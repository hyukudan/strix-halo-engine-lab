# Strix Halo engine lab

[English](README.md) · [Español](README.es.md)

This is a C++/HIP inference engine for AMD Strix Halo: Ryzen AI Max+ 395, Radeon 8060S (`gfx1151`), 128 GB of unified
memory, Linux and ROCm. It runs on one node, or on two nodes linked by USB4. An RTX PRO 6000 adds optional routes.
It runs Qwen3.8-Flash-Next, DeepSeek V4.1 Flash, MiniMax M3.1 Flash and GLM-5.3 Flash in production.
Opt-in builds include Qwen3.8-27B, Hy4-flash, Nemotron 3.5 Bolt E4B, MiniMax H3 (audio + video)
and MiniMax-H3-Image (text to image).
Profiles vary by engine. EXACT preserves the bits of its declared reference.
How we measure and what "exact" covers: [methodology](docs/methodology.md).

## Results

Measured on 2026-11-25 on `main` at commit `ca2de6ae8`, on AMD Strix Halo (Ryzen AI Max+ 395, Radeon 8060S, 128 GB per
node), one Release HIP build per node (GCC 15.2, HIP clang 24.0, ROCm 10.1, Linux 7.4 with the in-tree `amdgpu`). A pair
is two nodes over USB4 and runs the same binary on both. Power limits are the node defaults; the GPU clock and power were
sampled during every request: long prefills ran at 1,950–2,900 MHz and 120–150 W per node.
No lever was set by hand: `--profile exact|fast|turbo` decides, and every run printed its effective configuration.
Prompts are real English text (a set of short documents plus a project document). Token counts differ by tokenizer and
are shown per row. Each engine and tier was loaded once; "a / b" are two repeats on that resident server (or two CLI
processes). Each long request started below 70 °C. A lower second value comes from one node holding a lower GPU clock
after about two minutes of sustained load; the receipts show the clock per request.
Decode is one request (B1), greedy, in the engine's default decode mode, 128 tokens after a short chat prompt unless noted.
EXACT runs repeated the same ids; FAST and TURBO rows report what their profile prints (a qualification state or the
FAST identities it applies). See [methodology](docs/methodology.md) for measurement conditions.

| Model | Hardware | Route | Decode mode | Profile | Prompt (tokens) | Decode (tok/s) | Prefill (tok/s) | Main |
|---|---|---|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | pair | HTTP served | DSpark (default) | EXACT | 1,971 / 7,994 / 30,975; decode after 344 | 57.2 / 57.7 | 2K 782 / 739, 8K 1,101 / 1,111, 32K 1,273 / 1,245 | `ca2de6ae8` |
| DeepSeek V4.1 Flash | pair | HTTP served | DSpark (default) | FAST | same | 53.4 / 53.7 | 2K 969 / 922, 8K 1,425 / 1,588, 32K 1,689 / 1,708 | `ca2de6ae8` |
| DeepSeek V4.1 Flash | pair | HTTP served | DSpark (default) | TURBO (= FAST levers) | same | — | 8K 1,444 / 1,591, 32K 1,683 / 1,710 | `ca2de6ae8` |
| MiniMax M3.1 Flash | pair | HTTP served | plain | EXACT | 1,102 / 7,791 / 30,179; decode after 508 | 27.0 / 26.9 (after 1,102: 32.6 / 28.7) | 1K 417 / 451, 8K 667 / 587, 32K 590 / 565 | `ca2de6ae8` |
| MiniMax M3.1 Flash | pair | HTTP served | plain | FAST | same | 27.1 / 26.9 | 1K 415 / 457, 8K 709 / 660, 32K 640 / 611 | `ca2de6ae8` |
| MiniMax M3.1 Flash | pair | HTTP served | plain | TURBO | same | 26.7 / 26.6 | 1K 422 / 459, 8K 709 / 662, 32K 674 / 624 | `ca2de6ae8` |
| GLM-5.3 Flash | pair | HTTP served | default | EXACT | 2,037 / 8,108 / 31,258; decode after 362 | 40.2 / 42.8 | 2K 531, 8K 629 / 604, 32K 578 / 552 | `ca2de6ae8` |
| GLM-5.3 Flash | pair | HTTP served | default | FAST (resolves to the EXACT route) | same | 41.1 / 41.1 | 2K 530, 8K 628 / 596, 32K 568 / 545 | `ca2de6ae8` |
| GLM-5.3 Flash | pair | HTTP served | default | TURBO (FP16 WMMA prefill) | same | 41.3 / 41.3 | 2K 718, 8K 949 / 941, 32K 879 / 854 | `ca2de6ae8` |
| Qwen3.8-Flash-Next | 1 node | CLI | plain (no MTP drafter bundle on our nodes) | EXACT | 2,048 / 8,192 / 32,500 | 20.6–22.1 | 2K 1,435, 8K 1,667 / 1,840, 32K 1,669 / 1,677 | `ca2de6ae8` |
| Qwen3.8-Flash-Next | 1 node | CLI | plain | FAST | same | 21.6–21.9 | 2K 1,673, 8K 1,865 / 1,880, 32K 1,681 / 1,663 | `ca2de6ae8` |
| Qwen3.8-Flash-Next | 1 node | CLI | plain | TURBO (= FAST levers) | same | 21.5–21.8 | 2K 1,188 (one run), 8K 1,915, 32K 1,665 | `ca2de6ae8` |
| Qwen3.8-27B | 1 node | CLI | MTP (default) | EXACT | 2,048 / 8,192 | 2K 35.2 / 35.3, 8K 25.8 / 25.0 | 2K 187–199, 8K 115–117 | `ca2de6ae8` |
| Qwen3.8-27B | 1 node | CLI | plain (`--no-mtp`) | EXACT | 2,048 / 8,192 | 2K 14.05, 8K 13.22 | 2K 195, 8K 121 | `ca2de6ae8` |
| Qwen3.8-27B | 1 node | CLI | MTP (default) | FAST (TURBO = FAST) | 2,048 / 8,192 | 2K 35.2, 8K 25.0–25.1 | 2K 188–196, 8K 115–117 | `ca2de6ae8` |
| Qwen3.8-27B | pair (TP) | CLI | MTP (default) | EXACT / FAST | 2,048 | 45.4 / 45.5 | — | `ca2de6ae8` |
| Qwen3.8-27B | pair (TP) | CLI | plain | EXACT | 2,048 | 20.22 | — | `ca2de6ae8` |
| Hy4-flash | 1 node | CLI | plain | FAST (default, BASE reference; prompt rows 4096 by the tier) | 1,024 / 8,192 | 1K 17.5, 8K 14.25 | 1K 602, 8K 755 / 782 | `ca2de6ae8` |
| Hy4-flash | 1 node | CLI | plain | FAST, prompt rows 2048 set by hand | 2,048 / 8,192 / 32,500 | 16.9–17.0 | 2K 689–810, 8K 721–766, 32K 699 / 693 | `ca2de6ae8` |
| Hy4-flash | 1 node | CLI | plain | `exact` (BASE; prompt position by position) | 2,048 | 15.75 | 15.2 | `ca2de6ae8` |
| Hy4-flash | 1 node | CLI | plain | `exact` (BASE), prompt rows 2048 set by hand | 8,192 / 32,500 | 16.5–16.6 | 8K 687–723, 32K 661 (one run) | `ca2de6ae8` |
| Hy4-flash | 1 node | CLI | plain | TURBO (prompt rows 4096 by the tier) | 8,192 | 15.65 | 739 / 728 | `ca2de6ae8` |
| Hy4-flash | pair (layer split) | CLI | plain | FAST (the pair always runs a batched prompt) | 4,096 / 8,192 / 32,500 | 12.3–14.0 | 4K 1,265 / 1,216, 8K 1,327 / 1,387, 32K 1,328 (one run) | `ca2de6ae8` |
| Hy4-flash | pair (TP) | CLI | plain | FAST / `exact` | 1,024 | 22.64 / 22.42 | — | `ca2de6ae8` |
| Inkling 2 Mini | 1 node | HTTP served | plain | EXACT (FAST and TURBO resolve to the same route on HIP) | 2,280 / 9,116 / 35,170; decode after 396 | 23.6 / 23.6 (after 35,170: 20.9) | 2K 591 / 591, 8K 601 / 595, 32K 578 / 568 | `ca2de6ae8` |

**Lower than the previous table (2026-11-21, `d3a89ae16` / `f42edd05e`).** The previous rows were mostly CLI runs
with other prompt sets, so part of each change is the route and the prompt:
- DeepSeek V4.1 Flash EXACT prefill at 2K (791.9 before) and 8K (1,123.8 before).
- MiniMax M3.1 Flash EXACT prefill at 1K (472.3 before); the second 8K repeat (587) is below the previous 621.6.
- GLM-5.3 Flash EXACT prefill at 32K (595 before).
- Qwen3.8-27B prefill: the shipped default runs 64-row prompt windows, 2K 187–199 and 8K 115–117 against 644 / 606
  (EXACT) and 684 / 635 (FAST) before. Plain TP pair decode 20.22 (20.61 before).
- Qwen3.8-Flash-Next decode: 20.6–22.1 plain, against 51.0 with MTP before; the MTP drafter bundle is not installed
  beside the model on our nodes, so the default resolves to plain decode.
- Hy4-flash FAST decode with the tier's own 4096-row prompt: 14.25 at 8K (16.78 before).

**Not measured on `ca2de6ae8`:** Nemotron 3.5 Bolt E4B (its tokenizer and RoPE tables are not on our nodes); Qwen3.8-27B
at 32K (the shipped-default prefill did not finish inside our 470 s run limit); Qwen3.8-Flash-Next over HTTP
(`strix-serve --model qwen3.8-flash-next` needs an MTP calibration golden and the tiled MTP recipe to start); per-content
MTP and DSpark decode rows (code, prose, JSON, edit); Hy4-flash `exact` and TURBO pairs at 32K. TURBO rows marked
"= FAST levers" resolve to the FAST lever set on that engine.

### RTX PRO 6000 (CUDA)

Measured on 2026-11-21 on `main` at `f42edd05e`, on one machine with one RTX PRO 6000 and a 528 W power limit.

**Qwen 27B dense · CLI decode.** Warm rates in tok/s for 256 output tokens. Prompt lengths are unreported.
Within FAST, the output digest matched across all three modes for each content class.

| Profile | Content | Serial | MTP | MTP + wide |
|---|---|---:|---:|---:|
| FAST (`fast`, FAST4) | code | 85.90 | 283.3 | 292.2 |
| FAST (`fast`, FAST4) | prose | 87.15 | 162.3 | 172.8 |
| FAST (`fast`, FAST4) | reasoning | 86.52 | 244.0 | 271.1 |
| FAST (`fast`, FAST4) | JSON | 85.12 | 403.1 | 602.5 |
| FAST (`fast`, FAST4) | edit | 86.13 | 738.9 | 848.6 |
| EXACT (`exact`) | code | 84.46 | 280.9 | — |
| EXACT (`exact`) | prose | 87.01 | 161.4 | — |

**Qwen 27B dense · CLI prefill.** FAST4, admitted `large` route, 2,048-token chunks and two slots.
Rates and elapsed times are warm. Prompt class is unreported.

| Prompt (tokens) | Prefill (tok/s) | Time (s) |
|---|---:|---:|
| 2K (2,048) | 13,372 | 0.153 |
| 8K (8,192) | 13,642 | 0.600 |
| 32K (32,768) | 11,954 | 2.741 |
| 128K (130,326) | 7,910 | 16.48 |

**Qwen3.8-Flash-Next · HTTP decode.** Warm rates in tok/s with four server slots and 128 output tokens per request.
Prompt lengths are unreported. C1 is the rate per request with one client. C8 is the aggregate after the last
client receives its first signal, with eight concurrent clients. Slashes show both reported values.
The FAST prose output digest matched across plain, MTP and MTP + wide.

| Profile | Content | Mode | C1 (tok/s) | C8 (tok/s) |
|---|---|---|---:|---:|
| FAST | prose | plain | 96.5 / 96.4 | 506 / 504 |
| FAST | prose | MTP (sidecar) | 120.5 / 120.4 | 621 / 565 |
| FAST | prose | MTP + wide | 125.6 / 125.8 | 609 / 626 |
| FAST | edit | MTP + wide | 426.9 / 426.2 | 1,086 / 1,131 |
| FAST | JSON | MTP + wide | 294.9 / 294.2 | 921 / 1,055 |
| EXACT (`--profile default`) | prose | MTP + wide | 71.7 / 73.1 | 301.7 / 292.5 |

**Qwen3.8-Flash-Next · HTTP prefill.** FAST, warm. The decode mode and prompt class are unreported for these runs.

| Prompt (tokens) | TTFT (ms) | Prefill (tok/s) | Client TTFT (s) |
|---|---:|---:|---:|
| 2K (2,048) | 449 | 4,557 | 0.489 |
| 8K (8,192) | 1,359 | 6,030 | 1.392 |
| 32K (32,768) | 5,358 | 6,116 | 5.394 |

<a id="hardware"></a>

## Different hardware

The deployment planner budgets measured device memory for the requested KV context,
workspace and supported weight formats/topologies before loading. See
[deployment on different hardware](docs/deployment.md) for inspection, TOML examples,
orchestrator APIs and current integration limits, including pending INT4 runtimes.

## Quick start

`strix-serve --model NAME --profile fast` starts the configured server. Use
`strix run --model NAME --profile fast` for CLI generation. Replace `fast` with
`exact` or `turbo` to select another column in the model TOML. The default
is `fast`. FAST admits the native resolver's qualified levers for the active regime.
TURBO inherits that set. The optional E4M3 KV shadow is selected by TURBO on
CUDA Qwen3.8-27B. DeepSeek keeps its model-defined KV format in every tier.

For the short commands, put the local asset and launch arguments in
`launch/NAME.toml` in your working directory. Use one file per model on each node.
The paths in the file are relative to the working directory.

```toml
# launch/qwen3.8-flash-next.toml
[serve]
asset = "MODEL.gguf"

[run]
asset = "MODEL.gguf"
args = ["--prompt-ids", "prompt.ids", "--generate", "256"]
```

An optional `backend = "hip"` or `backend = "cuda"` selects a backend.
For a pair, put the rank and transport options in `args` on each node.
Alternatively, supply the asset and native arguments on the command line as shown below.
Add `--print-effective-config` before `--` to print the selected lever column, weight formats,
KV policy and drafter declaration, then exit. When an asset is configured or supplied,
the native resolver also prints its effective settings using the startup inputs it has.

| Model | Device | Serve | CLI |
|---|---|---|---|
| Qwen3.8 Flash Next | Strix, one node | `strix-serve --model qwen3.8-flash-next --profile fast --asset MODEL.gguf -- --prompt-ids calibration.ids --generate 64 --mtp-bundle MODEL.gguf.mtp-bundle.json` ([calibration prompt and bundle](docs/models/qwen3.8/README.md#serving-quick-start)) | `strix run qwen3.8-flash-next --profile fast --asset MODEL.gguf -- --prompt-ids prompt.ids --generate 256` |
| Qwen3.8 Flash Next | Strix pair | Pair CLI | `strix run qwen3.8-flash-next --profile fast --asset MODEL.gguf -- --prompt-ids prompt.ids --generate 256 --deploy deploy.toml --rank R --transport usb4auto` |
| Qwen3.8 Flash Next | RTX, Windows | `strix-serve --model qwen3.8-flash-next --profile fast --backend cuda --asset MODEL.gguf -- --tables TABLES --rope-table rope_cs.bin` | — |
| Qwen3.8-27B | Strix, one node | — | `strix run qwen3.8-27b --profile fast --asset MODEL.gguf -- --prompt-ids @prompt.ids --gen 256` |
| Qwen3.8-27B | Strix pair | — | `strix run qwen3.8-27b --profile fast --asset MODEL.gguf -- --prompt-ids @prompt.ids --gen 256 --tp-rank R --tp-transport usb4auto` |
| Qwen3.8-27B | RTX, Windows | — | `strix run qwen3.8-27b --profile fast --backend cuda --asset MODEL.gguf -- --prompt-ids IDS --n-gen 256` |
| DeepSeek V4.1 Flash | Strix pair | `strix-serve --model deepseek-v4.1-flash --profile fast --asset MODEL.gguf -- --rank R --transport usb4auto` | `strix run deepseek-v4.1-flash --profile fast --asset MODEL.gguf -- --rank R --transport usb4auto` |
| DeepSeek V4.1 Flash | RTX, Windows | Imported-state worker | `strix run deepseek-v4.1-flash --profile fast --backend cuda --asset MODEL.gguf -- --receipt RECEIPT.json --import-state STATE.bin --ref GREEDY.log` |
| MiniMax M3.1 Flash | Strix pair | `strix-serve --model minimax-m3.1-flash --profile fast --asset MODEL_DIR -- --rank R --transport usb4-auto` | `strix run minimax-m3.1-flash --profile fast --asset MODEL_DIR -- R usb4-auto OUTPUT_DIR` |
| GLM-5.3 Flash | Strix pair | `strix-serve --model glm-5.3-flash --profile fast --asset MODEL_DIR -- --world 2 --rank R --transport usb4-auto` | `strix run glm-5.3-flash --profile fast --asset MODEL_DIR -- --world 2 --rank R --transport usb4-auto --prompt PROMPT.u32 --decode 64` |
| Nemotron 3.5 Bolt E4B | Strix, one node | `strix-serve --model nemotron-3.5-bolt-e4b --profile fast --asset MODEL.gguf -- --rope-tables TABLES --tokenizer tokenizer.json` | `strix run nemotron-3.5-bolt-e4b --profile fast --asset MODEL.gguf -- --rope-tables TABLES --ids prompt.ids --max-tokens 256` |
| Hy4 Flash | Strix, one node | Host inspection | `strix run hy4-flash --profile fast --asset PART1.gguf -- PART2.gguf --ids-file prompt.ids --gen 256` |
| Hy4 Flash | Strix pair | Host inspection | `strix run hy4-flash --profile fast --asset PART1.gguf -- PART2.gguf --ids-file prompt.ids --gen 256 --rank R --transport usb4auto` |

Run pair commands on both nodes and substitute `0` and `1` for `R`. The Qwen Flash Next pair command
uses a deployment file declaring a pipeline. Dashes indicate unavailable launcher routes.
Hy4 CLI receipts use BASE as the device reference. Required build targets and sidecar assets
are listed in the engine guides. [Getting started](docs/getting-started/README.md)

### Native build and launch

One node. You need the ROCm toolchain, the model GGUF and a `prompt.ids` file from the tokenizer.

```bash
git clone https://github.com/hyukudan/strix-halo-engine-lab.git
cd strix-halo-engine-lab
export PATH=/opt/rocm/core-10.1/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/bin/amdclang++ \
  -DSTRIX_ENABLE_HIP_RUNTIME=ON \
  -DSTRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME=ON \
  -DSTRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE=ON \
  -DSTRIX_USE_OPENSSL_SHA256=ON -DBUILD_TESTING=ON
ninja -C build strix-hip-decode-token
./build/strix-hip-decode-token Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf $(cat prompt.ids) --generate 256
```

The run prints a JSON receipt with token IDs and timing. Add `--print-effective-config` to see the resolved
configuration before any device work. With the [drafter bundle](docs/engine/speculative/mtp-native-cli.md#default-mtp-drafter-bundle)
beside the model this command decodes with exact MTP speculation; without it, or with
`--no-mtp` / `STRIX_MTP=0`, it decodes the same tokens one at a time.

Pair (DeepSeek). Two nodes with 128 GB each and two USB4 cables. Run on both nodes, with `--rank 1` and the other
address on the second one:

```bash
scripts/pair/ds41-pair init --rank 0 --peer user@node-b --model-dir /path/to/models
scripts/pair/ds41-pair check && scripts/pair/ds41-pair driver && scripts/pair/ds41-pair build && scripts/pair/ds41-pair fetch
scripts/pair/ds41-pair run --both    # rank 0 only
scripts/pair/ds41-pair smoke         # rank 0 only
```

The script prints root steps and never runs them silently. Details: [pair setup](docs/pair-setup.md).
Every engine's build and launch command: [getting started](docs/getting-started/README.md).

<a id="models"></a>

## Model matrix

`fast` selects qualified options for the regime. `quality` uses a higher-precision pack where one exists.

| Model | Single / pair | Profiles | Memory | Weights |
|---|---|---|---|---|
| Qwen3.8-Flash-Next (UD-Q4_K_XL) | single, two replicas behind a router or opt-in pair prefill | MTP with drafter bundle; automatic prefill; `fast` | 111 GB weights on a 128 GB node | [UD-Q4_K_XL GGUF](https://huggingface.co/hyukudan/Qwen3.8-Flash-Next-GGUF-mirror) (mirror of `unsloth`, read as published); [distilled Q8_0 MTP draft head](https://huggingface.co/hyukudan/Qwen3.8-Flash-Next-MTP-distillG-Q8_0) |
| Qwen3.8-27B (IU4 GGUF) | single or TP pair, opt-in HIP build | EXACT default and opt-in FAST on one node | 128 GB per node | [GGUF contract](docs/models/qwen3.8-27b/README.md) |
| Hy4-flash (Q4_K_M GGUF) | single or USB4 pair, experimental HIP build | FAST default; BASE reference | 128 GB per node | [two-part GGUF](docs/models/hy4-flash/README.md#pack) |
| DeepSeek V4.1 Flash (2-bit experts, 6.5-bit trellis dense) | pair over USB4 | exact default; `fast` | 2 x 128 GB | [expert GGUF](https://huggingface.co/hyukudan/deepseek-v4.1-flash-gguf-mirror), [dense overlay](https://huggingface.co/hyukudan/DeepSeek-V4.1-Flash-1017-strix-dense-U6), [trellis packs](https://huggingface.co/hyukudan/DeepSeek-V4.1-Flash-1017-strix-trellis-EXL3) |
| MiniMax M3.1 Flash (EXL3 3.06 bpw) | pair TP2 | exact | 2 x 128 GB | [EXL3 3.06 bpw](https://huggingface.co/hyukudan/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw-mirror) (mirror of `Mia-AiLab`, with our EXL3 draft); [draft source](https://huggingface.co/hyukudan/MiniMax-M3.1-Flash-DFlash2-mirror) |
| GLM-5.3 Flash (EXL3 4.05 bpw) | pair TP2 | exact; opt-in WMMA prefill | 2 x 128 GB | [EXL3 4.05 bpw](https://huggingface.co/hyukudan/GLM-5.3-Flash-exl3-mirror) (mirror of `turboderp`) |
| Nemotron 3.5 Bolt E4B (Q4_K_M; EXL3 for `quality`) | single, opt-in build; opt-in pair prefill (`STRIX_G4_PAIR=1`) | exact default; `fast`; `quality` | 128 GB per node | official Q4_K_M GGUF (not mirrored by us); [EXL3 pack](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-EXL3); [MTP drafter](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-MTP-Drafter) |
| MiniMax H3 (audio + video) | single or pair, opt-in build | dense reference; FAST levers each with its own flag | 128 GB per node | not published by us; upstream [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/54472221d27c695fca9621b0c3ed52f06c4af707) ([files](docs/models/minimax-h3/README.md#install-and-run)) |
| MiniMax-H3-Image (int8 or W4A8) | single or pair (`strix-h3-image`), opt-in build | GPU VAE; `--parallel sequence` for one image, `request` for several | 128 GB per node | not published by us; upstream [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/0bb149f51e3156ed4b90dd45d5620346bfe1df8f) |

Our weights on Hugging Face: [huggingface.co/hyukudan](https://huggingface.co/hyukudan). Mirrors keep the upstream files; revisions, file names and licenses: [models and provenance](docs/reproduce.md#models-and-provenance).

<a id="discrete-gpu"></a>

Optional RTX PRO 6000 (96 GB) routes: DeepSeek standalone, DeepSeek RTX + pair, Qwen prefill offload, Qwen serve and
GLM on one card (draft). See [using a discrete GPU](docs/discrete-gpu.md).

Every switch, its default and when other hardware wants a different value: [toggles](docs/toggles.md).

## Image input

DeepSeek V4.1 on the two-node pair accepts images in OpenAI-style chat completions. Add an `image_url` part to a user
message. It is off by default: start the server with `--vision <vision GGUF>` on rank 0, together with
`--prefill-path device`.

```bash
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]}]}'
```

Limits: PNG and JPEG only; images over the row budget, video, and image requests with `session_id` are refused. Other
models do not take images yet. Details: [image input](docs/engine/vision.md).

## How it works

- One shared kernel core: GGUF and EXL3 readers, quantized matrix kernels, attention, the memory planner and the
  receipts. The core knows no model.
- Each model family is a module with its own contract, layer walk, plan and gates
  ([model families](docs/models/model-families.md)).
- The engine picks kernels by the number of rows in flight. One-row decode reads the smallest weights.
  Wide verify and prefill use the matrix-core and integer paths ([regime rule](docs/toggles.md#the-regime-rule)).
- Speculative decoding (MTP, DSpark) keeps the exact contract.
- The pair uses direct USB4 for tensor parallelism (TP2) or a layer pipeline depending on the engine ([pair](docs/pair.md)).
- Every default has a measured gain and an off switch ([optimizations](docs/optimizations.md)).
- Adding a model: [docs/engine/adding-a-model.md](docs/engine/adding-a-model.md).
- Map of the shared cores and which model uses each: [docs/engine/shared-cores.md](docs/engine/shared-cores.md).

## Documentation

| Topic | Docs |
|---|---|
| Start | [getting started](docs/getting-started/README.md) · [engine reference](docs/getting-started/engine-reference.md) · [reproduce](docs/reproduce.md) |
| Models | [Qwen Flash-Next](docs/models/qwen3.8/README.md) · [Qwen 27B](docs/models/qwen3.8-27b/README.md) · [Hy4-flash](docs/models/hy4-flash/README.md) · [DeepSeek](docs/models/deepseek-v4.1/README.md) · [MiniMax](docs/models/minimax-m3.1/README.md) · [GLM](docs/models/glm-5.3/README.md) · [Nemotron](docs/models/nemotron-3.5-bolt-e4b/README.md) · [MiniMax H3](docs/models/minimax-h3/README.md) · [families](docs/models/model-families.md) |
| Pair and serving | [pair](docs/pair.md) · [pair setup](docs/pair-setup.md) · [serving](docs/serving.md) · [discrete GPU](docs/discrete-gpu.md) |
| Switches | [toggles](docs/toggles.md) · [optimizations](docs/optimizations.md) ([JSON](docs/optimizations.json)) |
| Numbers | [performance](docs/performance.md) · [measured gains](docs/measured-gains.md) · [methodology](docs/methodology.md) |
| Internals | [architecture](docs/architecture/README.md) · [shared cores](docs/engine/shared-cores.md) · [decision register](docs/decisions/decision-register.md) · [implementation status](docs/reference/implementation-status.md) · [sources](docs/reference/sources-ledger.md) |
| Index | [full documentation index](docs/README.md) · [AGENTS.md](AGENTS.md) |

<a id="related-repositories"></a>

## Related repositories

- [strix-halo-usb4-driver](https://github.com/hyukudan/strix-halo-usb4-driver): the Linux patch series that turns
  the Strix Halo USB4 controllers into the direct link between the two nodes.
- [strix-trellis](https://github.com/hyukudan/strix-trellis): the trellis-quantization (EXL3-style) library
  used for DeepSeek's dense weights and MiniMax inference.
- [TensorMeld](https://github.com/hyukudan/tensormeld): placement and routing across nodes and GPUs; its router
  serves the Qwen replicas.
- [tierstream](https://github.com/hyukudan/tierstream): tiered expert streaming across VRAM, pinned host memory
  and NVMe used by the standalone DeepSeek RTX route.

<a id="license"></a>

## License

The engine source is licensed under the **[Apache License 2.0](LICENSE)**, except the Linux kernel
module under `lab/direct_gpu_stream/kernel/`, which is **GPL-2.0-only** as Linux kernel code must be (its license
texts are kept in that directory). Code adopted from other projects keeps its original license and notices. Model
weights are distributed under their own licenses. **Built with MiniMax M3.1-Flash**; its model license
is non-commercial by default. Preserve model and conversion notices and review the applicable terms. Third-party references used for formats and model semantics are
listed with their licenses in [docs/reference/sources-ledger.md](docs/reference/sources-ledger.md).
