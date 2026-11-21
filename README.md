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

Measured on 2026-11-21 on `main` with AMD Strix Halo (Ryzen AI Max+ 395, Radeon 8060S, 128 GB per node).
The measured commits were `d3a89ae16` and `f42edd05e`, shown per row. Pairs use USB4. Power limits and the
IOMMU setting per run were unreported in the measurement tables.
C1/C2/C4/C8 are streams per batch in the Nemotron decode benchmark, with aggregate rates.
Qwen HTTP rates are aggregate with one client and four slots configured. Clients completed their requests, and
the kit validator rejected the server exit code after intentional shutdown.
The Nemotron HTTP figure comes from the completed drafter-off arm with one client.
DSpark is speculative drafting. Hy4 FAST arms use the BASE device reference.
Full request cells show prompt tokens + output tokens. Prefill rates marked TTFT include time to first token.
See [methodology](docs/methodology.md) for measurement conditions.

| Model | Hardware | Route | Mode | Profile | Prompt (tokens) | Decode (tok/s) | Prefill (tok/s unless noted) | Full request | Main |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | 1 node | CLI | MTP | Default (auto FP16 prefill) | real text, 32K | 51.0 | 20.45 s | — | `d3a89ae16` |
| Qwen3.8-Flash-Next | 1 node | CLI | default mode unreported | Default | class unreported, 2K and 8K | — | — | 2K + 33: 2.71 s, 8K + 33: 5.95 s | `d3a89ae16` |
| Qwen3.8-Flash-Next | 1 node | CLI | plain | Default | real text, 32K | — | — | 32K + 128: 24.7 s | `d3a89ae16` |
| Qwen3.8-Flash-Next | 1 node | HTTP served | MTP + wide auto | Default | lengths unreported | prose 38.0–40.2, edit 153–155, JSON 85.7–87.6, code 60.8–61.2, reasoning 59.6–61.6 | — | — | `d3a89ae16` |
| Qwen3.8-27B | 1 node | CLI | plain | EXACT (default) | class unreported, 2K and 8K | 2K 13.90, 8K 13.10 | 2K 644, 8K 606 | — | `d3a89ae16` |
| Qwen3.8-27B | 1 node | CLI | plain | EXACT (default) | real text, 32K | 10.68 | 517 | — | `d3a89ae16` |
| Qwen3.8-27B | 1 node | CLI | plain | FAST (opt-in) | class unreported, 2K and 8K | — | 2K 684, 8K 635 | — | `d3a89ae16` |
| Qwen3.8-27B | 1 node | CLI | MTP | EXACT (default) | edit, 477 | 32.2 | — | — | `d3a89ae16` |
| Qwen3.8-27B | 1 node | CLI | MTP | EXACT (default) | lengths unreported | code 40.8, prose 25.9, reasoning 40.3, JSON 62.2 | — | — | `d3a89ae16` |
| Qwen3.8-27B | 1 node | CLI | plain | EXACT (default) | code, length unreported | 14.21–14.22 | — | — | `d3a89ae16` |
| Qwen3.8-27B | pair | CLI | plain | Default (TP) | code, length unreported | 20.61 | — | — | `d3a89ae16` |
| DeepSeek V4.1 Flash | pair | CLI | plain | EXACT | class unreported, 2K / 8K / 32K | — | 2K 791.9, 8K 1,123.8, 32K 1,224.1 (warm) | — | `d3a89ae16` |
| DeepSeek V4.1 Flash | pair | CLI | plain | EXACT (default) | 15–21 | code 26.85, prose 26.81 | — | — | `f42edd05e` |
| DeepSeek V4.1 Flash | pair | CLI | DSpark | EXACT (default) | 15–21 | code 41.70, prose 60.04, JSON 43.62, chat 39.98 | — | — | `f42edd05e` |
| MiniMax M3.1 Flash | pair | CLI | plain | EXACT | class unreported, 1K / 8K / 32K | 1K 24.20 | 1K 472.3, 8K 621.6 (warm), 32K 587.2 | — | `d3a89ae16` |
| MiniMax M3.1 Flash | pair | CLI | plain | EXACT (default) | 2,048 | code 23.28, prose 23.19, JSON 23.11, edit 23.07 | code, prose, JSON and edit 506–521 (TTFT) | — | `f42edd05e` |
| MiniMax M3.1 Flash | pair | CLI | DSpark auto | EXACT (default) | 2,048 | code 32.39, prose 23.83 | — | — | `f42edd05e` |
| GLM-5.3 Flash | pair | CLI | plain | EXACT (GEMV) | class unreported, 8K and 32K | — | 8K 617 (warm), 32K 595 | — | `d3a89ae16` |
| GLM-5.3 Flash | pair | CLI | plain | EXACT (R1b) | code, 2,048 | 25.73 | — | — | `f42edd05e` |
| GLM-5.3 Flash | pair | CLI | plain | EXACT (R1b) | prose, 1,884 | 25.94 | — | — | `f42edd05e` |
| GLM-5.3 Flash | pair | CLI | MTP + wide | EXACT (R1b) | code, 2,048 | 40.37 | — | — | `f42edd05e` |
| GLM-5.3 Flash | pair | CLI | MTP + wide | EXACT (R1b) | prose, 1,884 | 66.48 | — | — | `f42edd05e` |
| Hy4-flash | 1 node | CLI | plain | FAST (default), BASE reference | real text, 2K / 8K / 32K | 2K 16.75, 8K 16.78, 32K 16.69 | 2K 773.2, 8K 730.6 (warm), 32,568 tokens at 696.5 (cold) | — | `d3a89ae16` |
| Hy4-flash | 1 node | CLI | plain | `exact` (BASE) | real text, 8K | 16.39 | 679.3 (warm) | — | `d3a89ae16` |
| Hy4-flash | 1 node | CLI | MTP + wide | FAST (default), BASE reference | short prompts, lengths unreported | code 32.23, prose 22.93, reasoning 29.34 | — | — | `d3a89ae16` |
| Hy4-flash | pair | CLI | plain | FAST (default), BASE reference | real text, 4K / 8K / 32K | — | 4K 1,163.7, 8K 1,260.3 (warm), 32K 1,269.6 (cold) | — | `d3a89ae16` |
| Hy4-flash | pair | CLI | plain (TP) | FAST (default), BASE reference | class unreported, 1K | 22.3 | — | — | `d3a89ae16` |
| Hy4-flash | pair | CLI | MTP (TP) | FAST (default), BASE reference | lengths unreported | code 47.24, prose 28.12, reasoning 38.59 | — | — | `d3a89ae16` |
| Nemotron 3.5 Bolt E4B | 1 node | CLI | plain | EXACT (default) | benchmark, context 64 | C1 54.9, C2 103.2, C4 177.2, C8 243.1 | — | — | `d3a89ae16` |
| Nemotron 3.5 Bolt E4B | 1 node | CLI | plain | FAST (opt-in) | benchmark, context 64 | C1 53.2, C2 103.7, C4 196.0, C8 334.3 | — | — | `d3a89ae16` |
| Nemotron 3.5 Bolt E4B | 1 node | CLI | MTP | EXACT (default) | 18–23 | prose 208.3, code 194.7, JSON 193.9 | — | — | `d3a89ae16` |
| Nemotron 3.5 Bolt E4B | 1 node | CLI | MTP | FAST (opt-in) | 18–23 | prose 273.1, code 250.1, JSON 254.1 | — | — | `d3a89ae16` |
| Nemotron 3.5 Bolt E4B | 1 node | CLI | MTP | EXACT (default) | edit, 561 | 169.7 | — | — | `d3a89ae16` |
| Nemotron 3.5 Bolt E4B | 1 node | CLI | MTP | FAST (opt-in) | edit, 561 | 202.2 | — | — | `d3a89ae16` |
| Nemotron 3.5 Bolt E4B | 1 node | CLI | plain | FAST (opt-in) | repeated prose, 2K / 8K / 32K / 65,000 | — | 2K 5,050, 8K 4,909, 32K 4,249, 65,000: 3,610 | — | `d3a89ae16` |
| Nemotron 3.5 Bolt E4B | 1 node | HTTP served | plain | EXACT (default) | code, prose, JSON and edit, lengths unreported | 51.90 | — | — | `d3a89ae16` |

<a id="hardware"></a>

## Different hardware

The deployment planner budgets measured device memory for the requested KV context,
workspace and supported weight formats/topologies before loading. See
[deployment on different hardware](docs/deployment.md) for inspection, TOML examples,
orchestrator APIs and current integration limits, including pending INT4 runtimes.

## Quick start

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
