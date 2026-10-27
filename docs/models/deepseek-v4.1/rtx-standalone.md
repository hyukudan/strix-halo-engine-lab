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
pinned-host budget for available memory; defaults are 74.8 GiB for a keep-64 drafter arena (76.5 for keep-32,
71.5 for a full drafter, 80 without a drafter),
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

## Optimizations (default ON) and off switches

These options change scheduling, placement or draft proposals. Committed
outputs retain target-greedy exactness. Gains below describe prior qualification;
this defaults change is checked on CPU and does not establish new GPU results.

| Item | What | Default | Measured gain | Exact | Off switch |
|---|---|---|---|---|---|
| Decode kernels | Router, output projection, forks, expansion fusion, routed MoE, attention and normalization | ON | All-hit 12.62 to 10.50 ms/token; full loop 5.8-10.8% | Bitwise | `--dsrtxk-off` |
| Fusions | Hyperconnection and shared expert | ON | Included in decode qualification | Bitwise at supported shapes | `--fuse-off` |
| Records | Move records off the critical path | Deferred | Qualified; isolated gain not listed | Same outputs | `--records-inline` |
| Router | Form 2 | ON | R1 17.3 to 15.4 us; 8-layer device time -0.5% | Bitwise rows 1-8 | `--router-pf0` |
| QA/KV | Split 1 | ON | 8-layer device time -1.54 to -1.94% | Bitwise | `--qakv-auto` |
| Q8 R1 | Compact single-row kernel, no padding | ON | 8-layer device time -0.80 / -0.53% | Bitwise | `--q8-r1-off`; `--q8-r1-pad` selects the padded form |
| Drafter | Keep most-used experts per stage | 64 | Qualified; isolated gain not listed | Target-greedy | `--draft-keep 0` |
| Controller | Built-in survival prior and seeded costs, no exploration or cost learning | Warm, chat scope | Qualified; isolated gain not listed | Target-greedy | `--controller-prior none` / `--controller-costs none`; `--controller-scope run` restores run scope |
| Write-backs | Deferred scheduling | ON | Qualified; isolated gain not listed | Bitwise | `--no-wb-defer` |
| Engram | RAM record cache | 512 MiB | Avoids repeat storage stalls | Same bytes | `--engram-cache-mb 0` |
| Cold load | Tierstream pipeline | ON | Qualified; isolated gain not listed | Bitwise | `DG_LOAD_SERIAL=1` |
| Graph warmup | Capture common graphs before worker readiness | `0,8` in serve mode | First 16 steps 56.05 to 33.99 ms/step | Bitwise | `--warmup-graphs none` |
| Residency | Reuse valid prior placement, save on worker exit | Auto | Step 0 332.6 to 156.0 ms; first 16 steps -20% | Bitwise | `--residency-seed none` / `--residency-save none` |
| Lean state | Decode-only attention state: an allocation ledger and a split at the maximum decode position instead of full prefill buffers | ON | 32K state 2,486 to 59 MiB (+264 expert slots, plain decode +3.0 %); 128K now loads (state 195 MiB; without it the replica does not fit) | Bitwise: prefill states, greedy logits, DSpark windows, chat tokens, worker requests | `--no-lean-state` |
| Arena sizing | Size the expert arena from free VRAM minus a headroom | Off (fixed arena by drafter) | More expert slots when VRAM is free | Same outputs | opt-in `--arena-gib auto [--arena-headroom-gib G]` |
| Eviction | Public tierstream scored victims after verify windows; plain steps keep LRU | Auto | DSpark misses/token −14 %, median +6.4 % tok/s (7 chats); plain decode neutral | Bitwise (only residency changes) | `--evict lru` |

`--baseline` restores all reference forms at that argument; later explicit
flags override their own settings. `STRIX_DS41_RTX_BASELINE=1` applies the
reference forms before all arguments. Baseline uses 96 host spares, run controller
scope, a 71.5 GiB arena with a drafter (80 without), full draft vocabulary,
and serial loading unless the caller already set `DG_LOAD_SERIAL`. It also
restores phase-inferred row capacity and 4 pressure spares.
`--dsrtxk`, `--fuse`, `--records-deferred`, `--router-pf`, `--qakv-s1`, and
`--q8-r1` can re-enable those features after baseline.

Normal defaults use 48 host spares, 8 pressure spares and 6 maximum rows.
An explicit `--arena-gib` wins over the drafter-dependent arena rule.
Vocabulary defaults to `auto:DIR` when `STRIX_DS41_DRAFT_VOCAB` names a directory,
otherwise when `ds41-draft-vocab` exists beside the drafter; absent a directory,
it uses `full` with a one-line notice. Explicit `--draft-vocab` wins.
The auto residency path is `MODEL.gguf.residency`, overridden by
`STRIX_DS41_RESIDENCY`. A missing seed quietly uses routes order; an invalid
auto seed reports a one-line notice and falls back. Explicit seed files retain
strict errors. Saves use a temporary file and rename. Unsupported Q8 R1 shapes
use the generic exact kernel with a one-time notice. Non-serve runs do not
warm up graphs or save residency unless explicitly requested.

`--print-config` prints one resolved JSON object and exits before any CUDA call,
fixture or model read. It can read small prior, pruning and residency files;
model and draft paths may be dummy paths. The JSON reports effective prior
counts and expert IDs so built-in and file-based configurations can be compared.
`strix-cuda-ds41-config-tests` is registered with CTest and requires no GPU or model.
Window expert prefetch remains opt-in through phases ending in `w`.

## Exactness check

Use an off-switch run as the reference: repeat the same command and fixtures
with `--baseline`. Compare complete per-chat token vectors
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
