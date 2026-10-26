# DeepSeek V4.1 Flash on one RTX PRO 6000

## What it is

`strix-cuda-ds41-replica` runs the full DeepSeek V4.1 Flash target and a
DSpark drafter on one RTX PRO 6000 (96 GB, Windows WDDM). The standalone
implementation is [cuda_ds41_replica.cu](../../../apps/cuda_ds41_replica.cu),
with the worker protocol in
[cuda_ds41_replica_serve.hpp](../../../apps/cuda_ds41_replica_serve.hpp).

This profile uses the Q2 GGUF: routed gate/up **IQ2_XXS**, routed down
**Q2_K**, and dense **Q8_0**. Its exactness claims apply to this identity;
equality with the two-node trellis profile is not claimed.

Experts occupy an exclusive VRAM + pinned-host cache managed by
[tierstream](https://github.com/hyukudan/tierstream)'s device-gated decode
mode. Missing and predicted experts move over PCIe at the MoE sites. Dense
weights and the drafter share the target GPU; the drafter shares the target's
embedding and output weights. The built-in drafter pruning keeps 64 experts
per stage. See tierstream's
[library reproduction guide](https://github.com/hyukudan/tierstream/blob/main/docs/reproduce.md)
for the library side.

## Build on Windows

Use an x64 MSVC developer PowerShell with CMake (3.24 or newer), Ninja, and
a CUDA toolkit supporting architecture 120. Build tierstream's CUDA static
library first, using a compatible toolchain.

From the repository root, with a tierstream checkout at the relative
placeholder `TIERSTREAM`:

```powershell
cmake -S . -B build-rtx -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl `
  -DSTRIX_CUDA_DS41_ONLY=ON -DSTRIX_ENABLE_CUDA_DS41=ON `
  -DSTRIX_FAMILY_DEEPSEEK41=ON -DSTRIX_ENABLE_HIP_RUNTIME=OFF `
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRIX_TIERSTREAM_DIR=TIERSTREAM
cmake --build build-rtx --target strix-cuda-ds41-replica
```

[CMakeLists.txt](../../../CMakeLists.txt) defines these options. The CUDA-only
configuration enables the CUDA backend and requires the family enabled and
HIP disabled. `STRIX_TIERSTREAM_DIR` is a cache path to the library's source
tree: CMake imports `build-cuda/tierstream.lib` and its `include` directory,
then links the CUDA runtime/driver and Windows `avrt`/`winmm` libraries.
Leaving that variable empty omits the replica target. The executable links
`strix_ds41_cuda_tier`, `strix_ds41_cuda_prefill`, and `strix_ds41_cuda_dspark`.

## Run and persistent worker

This is an imported-state decode runner. Supply the admitted model, its
receipt binding SHA-256, size and modification time, an exported attention
state, the matching greedy reference, and tokenized chat sequences. These
assets are external. `CHATS.tsv` has a label, a tab, and comma-separated token
IDs per line; chat indices are zero-based. Each selected chat must contain
at least the requested prompt length. All filenames below are placeholders.

```powershell
.\build-rtx\strix-cuda-ds41-replica.exe MODEL.gguf `
  --receipt MODEL.receipt.json --import-state STATE.bin --ref GREEDY.log `
  --draft DRAFTER.gguf --draft-device 0 --arena-gib 80 `
  --phases exact0,pressure,exact8,wforce0,spec0,exact0,chats0,schats0,chats0 `
  --steps 64 --max-rows 6 --max-positions 4096 `
  --seqs-file CHATS.tsv --chats 0 --chat-prompt 96 --chat-steps 128 `
  --qual --json RESULTS.jsonl
```

The imported position plus the requested work and safety allowance must fit
`--max-positions`; increase it for a later imported state. Size the arena and
pinned-host budget for available memory; defaults are 80 GiB for the arena,
68 GiB maximum pinned allocation, and a 6 GiB available-host-memory margin.

To load once and accept repeated requests:

```powershell
.\build-rtx\strix-cuda-ds41-replica.exe MODEL.gguf `
  --receipt MODEL.receipt.json --import-state STATE.bin `
  --draft DRAFTER.gguf --draft-device 0 --arena-gib 80 `
  --phases chats0,schats0,chats0 `
  --seqs-file CHATS.tsv --chats 0 --chat-prompt 96 --chat-steps 128 `
  --max-rows 6 --max-positions 4096 --serve SPOOL --qual
```

Wait for `SPOOL/ready.json`. Write a request as `SPOOL/queue/REQUEST.tmp`,
then rename it to `REQUEST.req`. The file contains one argument per line,
for example:

```text
--phases
chats0,schats0,chats0
--reset-cache
open
--json
REQUEST.jsonl
```

The worker moves it through `run/` to `done/` and writes `done/REQUEST.json`
with `rc`, elapsed time and any error. Requests restore launch-time runtime
options before applying their overrides. `--reset-cache warm` is the default;
`open` restores the expert cache's initial placement for fresh-load comparisons.
Model, arena, drafter pruning and decode kernel choices require a new worker.
A request containing only `--quit` stops it. This is a file-queue worker.

## Defaults on and reference switches

The table follows the argument parser and default-flip block in the replica
source. These switches change scheduling, placement or draft proposals while
the target's greedy tokens remain the exactness reference.

| Default | Off/reference switch |
|---|---|
| Deferred expert write-backs | `--no-wb-defer` restores boundary write-backs |
| Built-in keep-64 drafter pruning | `--draft-keep 0` retains all draft experts |
| Warm controller, built-in survival prior (`builtin:64`) | `--controller-prior none` starts cold and also drops the implicit seeded costs |
| Priced controller costs, skipping the initial sweep | `--controller-costs none` removes seeded costs while retaining the prior |
| Router form 2 | `--router-pf0` restores form 0 |
| QA/KV split 1 | `--qakv-auto` restores automatic splitting |
| Decode bundle: router v2, single-launch output projection, shared/attention forks, expansion fusion, routed MoE variant, four-warp attention and clustered normalization | `--dsrtxk-off` disables the default bundle |
| Hyperconnection/shared-expert fusions | `--fuse-off` disables their defaults |
| Deferred step records | `--records-inline` writes records before the next step |
| 512 MiB host Engram record cache | `--engram-cache-mb 0` disables the cache |

The router and QA/KV reference switches override their corresponding defaults.
The bundle switches leave explicitly enabled options enabled. Controller
exploration and cost learning are off by default. Window expert prefetch is
also off by default; phases ending in `w` enable it, and omitting that suffix
disables it. No throughput gain is established for window prefetch.

## Exactness check

Use an off-switch run as the reference: repeat the same command and fixtures
with `--no-wb-defer --draft-keep 0 --controller-prior none
--controller-costs none --router-pf0 --qakv-auto --dsrtxk-off --fuse-off
--records-inline --engram-cache-mb 0`. Compare complete per-chat token vectors
and state digests against the default run, with the same capacity and model.
Drafter pruning and controller choices can change acceptance, never the
committed target-greedy sequence.

The example's phase order exercises plain greedy (`exact0`), forced
write-back pressure (`pressure`), predicted-expert prefetch (`exact8`), forced
verify windows (`wforce0`), speculation (`spec0`), and plain greedy again.
Each phase reimports the same state. Pressure must exercise demand banks,
resurrections and mid-token write-back collection. Forced windows vary row
and kept-prefix lengths, insert wrong tokens after the retained prefix, and
commit only that prefix. Speculative windows commit only verified greedy
tokens. Both compare retained rows with the plain greedy reference and must
finish at the same position and state digest. Exact phases require at least
64 compared steps and reference coverage for every position.

Read the JSON lines on stdout or in `--json` output:

- `gate`: `pass: true`, `compared == matched == steps`, and `digest_same: true`.
  Pressure also requires nonzero `bank`, `resurrected`, and `collected_mid`.
- `canonical`: `same: true`, with matching exported state digest and position.
- `chat_tokens`: `plain_found: true`, `pass: true`, and `same_prefix == compared`.
  With equal lengths, each plain/speculative chat vector has `chat_steps + 1`
  tokens. The first plain chat supplies the canonical vector; the example
  includes plain chats before and after speculation.
- `chat_hash`: compare `len` and `fnv` across the reference and default runs.
  `--qual` additionally emits chat token vectors and window-row records.
- `spec_summary` / `summary`: throughput and timing, separate from gate verdicts.
  A worker request must also finish with `rc: 0` in its completion JSON.

## Measurements and variance

The published [README](../../../README.md) and
[performance record](../../performance.md) report **82.3 / 87.7 / 94.4 tok/s**
for code / English / Spanish on the same load. Quote code throughput across
loads as **82–95 tok/s**. Load-to-load variation is up to **~10%**: compare
options within one loaded process, keeping fixtures, capacity and cache-reset
policy fixed. Prompt lengths and generated-token counts are unspecified for
these latest decode measurements.

The same published record reports full loading **108.7 → 61.8 s**, persistent
worker requests **8–18 s/request** without reloading, and fresh-load cache reset
**3.4 s**. Loading and persistent-worker work are marked **in development**;
these are workload-specific measurements, separate from decode throughput.
