# Engine reference

This page holds the precise technical material for the engine: the measured decode configuration, every
build option, the complete command line, the draft vocabularies, the environment switches and the repository
layout. The [README](../README.md) is the short, user-facing introduction; nothing here contradicts it, this
page only goes deeper.

- [Speculative decode configuration](#speculative-decode-configuration)
- [Decode by workload (earlier measurement)](#decode-by-workload-earlier-measurement)
- [Build options](#build-options)
- [Command line](#command-line)
- [Prompt prefill](#prompt-prefill)
- [Draft vocabulary](#draft-vocabulary)
- [Environment switches](#environment-switches)
- [Host-only tools](#host-only-tools)
- [Repository layout](#repository-layout)
- [Documentation map](#documentation-map)
- [Working rules](#working-rules)
- [Multi-node direction](#multi-node-direction)

## Speculative decode configuration

Decoding is speculative. The MTP head drafts a few tokens; the target model then verifies them in one pass over
a window of 2, 4 or 8 rows and commits the longest accepted prefix. Every multi-row kernel performs each row's
arithmetic exactly as the single-row kernel does, so the emitted tokens are bit-identical to serial greedy
decode (D-021).

The configuration behind the published decode numbers:

- **Adaptive verify windows `{2,4,8}`.** Each cycle the window size is 1 + the number of leading drafts with
  probability >= `--mtp-spec-min-p` (0.85), rounded up to an available window. An online cost model (expected
  tokens per ms for each window size, with acceptance calibrated per probability bin) can choose a smaller window
  or a serial step. Before each extra draft step the chain asks the same model whether a larger window could
  still pay off, and stops drafting when it cannot (D-022).
- **Longest-prefix acceptance with deferred GDN commit.** No state checkpoint is taken: the next window (or a
  flush before a serial step) applies the accepted recurrent rows (D-023).
- **Draft head over a vocabulary subset** (`--mtp-draft-vocab`), with Q4_0 copies of the draft projections.
  The draft only affects acceptance, never the output (D-026).
- **A8 numerical profiles** for the MoE (int8 activations, `dot4`), optionally also for the dense Q8_0
  projections (dense A8, D-032); routed-expert union per window (each expert read once for all rows, D-025);
  any-order launches for independent kernels (D-027); and the PLE table prefetched into the page cache after
  registration (D-030).

Per-cycle cost measured at `b0d1cd1`: R2 ~49 ms, R4 ~61 ms, R8 ~86 ms including the draft. The R8 cycle is
dominated by the routed-expert union, which is inherent to verifying eight exact rows. Decode is
memory-bandwidth bound: a 4-row window reads about 10.6 GB of weights.

Start-up to the first decode step takes about 11 s once the model file's SHA-256 is cached (the first start
hashes the 111 GB file, about 50 s).

## Decode by workload (earlier measurement)

Measured on one Bosgame M5 (Ryzen AI Max+ 395, 128 GB), UD-Q4_K_XL, A8 profile without dense A8, single stream,
256 generated tokens, at `b0d1cd1` (before D-032 and later decode work; the code figure has since risen to the
values in the README):

| Workload | tok/s | Tokens per cycle |
|---|---|---|
| Python code (raw completion prompt) | 48.5, stable across runs | 2.9 |
| English prose (500-word article, chat template, no thinking) | 37.2 | 1.9 |
| Spanish prose (same article) | 31-34 with an English-centred draft vocabulary, 37.4 with a Spanish pack | 1.7-1.9 |

<!-- BENCH: refresh this table per workload (code, English prose, Spanish prose, other languages) at the final campaign commit. -->

The generated tokens are identical in every run and identical to serial decode. The difference between
workloads is acceptance: code is far more predictable for the MTP head than prose. For languages other than
English, acceptance also depends on the draft vocabulary covering that language (see
[Draft vocabulary](#draft-vocabulary)).

## Build options

Requirements: Linux, CMake >= 3.24, Ninja, a C++20 compiler, ROCm (tested with ROCm core-10.1 under
`/opt/rocm/core-10.1`) and a `gfx1151` GPU for the HIP targets.

```sh
export PATH=/opt/rocm/core-10.1/bin:$PATH
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/bin/amdclang++ \
  -DSTRIX_ENABLE_HIP_RUNTIME=ON \
  -DSTRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME=ON \
  -DSTRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE=ON \
  -DSTRIX_USE_OPENSSL_SHA256=ON \
  -DBUILD_TESTING=ON
ninja -C build strix-hip-decode-token
```

| Option | Default | Purpose |
|---|---|---|
| `STRIX_ENABLE_HIP_RUNTIME` | OFF | Build the HIP runtime and the GPU executables, including `strix-hip-decode-token`. Without it only the host tools build. |
| `STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME` | OFF | Native Q4K/Q5K A8G32 MoE gate/up integration used by the A8 numerical profiles. |
| `STRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE` | OFF | Standalone Q4K A8G32 HIP probe and its tests. |
| `STRIX_USE_OPENSSL_SHA256` | OFF | Use OpenSSL for full-file SHA-256 verification. |
| `STRIX_WARNINGS_AS_ERRORS` | ON | Treat compiler warnings as errors. |
| `STRIX_EXPORT_OFFLINE_TUNING_PROVENANCE` | OFF | Export Ninja/FileAPI inputs for offline compile provenance (Ninja on POSIX only). |
| `STRIX_HC_REPACKED_ADMITTED_SCHEDULES` | empty | Externally admitted HC research schedules (subset of `31,32,33`). |

`ctest --test-dir build` runs the host and device tests. Many device tests need the GPU and model files.

## Command line

`strix-hip-decode-token` takes a GGUF model and the prompt as token IDs; it has no tokenizer. It writes a JSON
receipt to stdout that includes `generated_tokens`, cycle counts and timings. `--help` lists the general options;
the MTP options are documented in [mtp-native-cli.md](mtp-native-cli.md).

A speculative run has four groups of options.

**1. Target model and generation**

```
strix-hip-decode-token MODEL.gguf TOKEN_ID TOKEN_ID ... \
  --context 8192 --execution-mode eager --residency register --generate 256 \
  --optimization-profile auto \
  --numerical-profile hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1
```

The recommended profile adds dense A8: the dense Q8_0 projections (GDN, QSA, shared expert) also take int8
activations per group of 32. It is faster in prefill and decode at a small quality cost (D-032). To keep the dense
projections in FP32 (the exact profile), use `hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup-v1`.

The automatic optimization profile is admitted at context 8192 for the pinned model content. Other context sizes
need `--context N --unqualified-context`; they run the same profiles but are not covered by the automatic
profile's numerical and timing gates (see [decode-optimization-profile.md](decode-optimization-profile.md)).

`--prefill-numerical-profile NAME` runs the prompt prefill (tiled known-prefix prefill or the MTP prompt) under a
different profile from decode; today only the dense-A8 sibling of `--numerical-profile` is admitted. Decode and MTP
windows always use `--numerical-profile`, MTP goldens record both profiles, and MTP stays exact from the state the
prefill produced. See [prefill-numerical-profile.md](prefill-numerical-profile.md) (D-033).

Teacher-forced scoring (`--teacher-forced-score`, `--teacher-logits-file`) also runs over the tiled prefill with the
all-rows output policy, producing byte-identical logits files to the serial route within a profile, several times
faster. See [teacher-forced-tiled-prefill.md](teacher-forced-tiled-prefill.md) (D-034).

**2. Kernel organization required by the windows** (A8 MoE, wave8 Q8 GDN, deferred MoE status, HC chain)

```
  --q51-slot-parallel-down --batched-moe-activation --q4k-a8g32-gate-up --grouped-exact-q51-down \
  --deferred-moe-status --parallel-moe-router --wave-moe-router-topk \
  --virtual-q8-gdn-input --prefetch-q8-gdn-input --virtual-q8-gdn-output \
  --wave8-q8-gdn-input --wave8-q8-gdn-output --wave8-q8-qsa-output \
  --prefetch-q8-lm-head --wave8-q8-lm-head --virtual-q8-lm-head \
  --prefetch-q8-qsa-input --prefetch-q8-shared-expert \
  --parallel-hyper-mix --staged-hyper-projections --fused-staged-hyper-mix \
  --typed-fused-hyper-projections --prefetch-typed-hyper-rows \
  --prefetch-q8-hyper-output --lane-chain-hyper-output
```

**3. MTP sidecar, provenance and budgets.** The sidecar file and the target are pinned by SHA-256, and every
memory reservation is explicit. See [mtp-native-cli.md](mtp-native-cli.md) for the meaning of each value.

```
  --mtp --mtp-sidecar MTP.gguf --mtp-sidecar-logical-name NAME --mtp-sidecar-sha256 HEX --mtp-sidecar-bytes N \
  --mtp-target-sha256 HEX --mtp-target-content HEX \
  --mtp-source-repository REPO --mtp-source-revision REV --mtp-source-path PATH \
  --mtp-converter-revision REV --mtp-converter-sha256 HEX --mtp-norm-converter-sha256 HEX \
  --mtp-conversion effective-norm-embedding-first-v1 \
  --mtp-normalized-rounding bf16-rne --mtp-projection-rounding bf16-rne --mtp-sidecar-residency register \
  --mtp-header-bytes 33554432 --mtp-sidecar-retained-bytes 3221225472 --mtp-sidecar-setup-bytes 4294967296 \
  --mtp-golden prompt.golden --mtp-golden-sha256 HEX --mtp-golden-max-bytes 67108864 \
  --mtp-replay-host-bytes 67108864 --mtp-runtime-bytes 134217728 --mtp-target-capture-bytes 16777216 \
  --mtp-policy-id 1 --mtp-candidate-depth 7 --mtp-max-hidden-error 0 --mtp-max-logit-error 0
```

MTP replay needs a reference forward for the prompt. For benchmarking, generate one with the engine itself: run
once with `--mtp-golden-dump prompt.golden` instead of `--mtp-golden`/`--mtp-golden-sha256`, then pass the file
and the printed SHA-256. A self-generated reference is reported as not independent; it does not change the
output, because every emitted token is verified by the target. Size `--mtp-golden-max-bytes` at about 1.03 MB
per prompt row (64 MiB covers a 43-token prompt); see [mtp-native-cli.md](mtp-native-cli.md).

**4. Speculative decode**

```
  --mtp-target-window-rows 8 --mtp-target-window-rows-set 2,4,8 \
  --mtp-window-arena-bytes 2147483648 --mtp-window-capture-bytes 67108864 \
  --mtp-window-checkpoint-bytes 8589934592 --mtp-window-total-bytes 17179869184 \
  --mtp-window-history-intervals 8192 \
  --mtp-window-expert-union --mtp-window-expert-union-bytes 65536 \
  --mtp-window-prefix --mtp-window-hyper-mix --mtp-window-qsa-rows \
  --mtp-draft-chain --mtp-spec-min-p 0.85 --mtp-draft-q4 \
  --mtp-draft-vocab auto:data/draft-vocab --mtp-draft-context-vocab 4096
```

| Option | Meaning |
|---|---|
| `--mtp-target-window-rows-set 2,4,8` | Window sizes available to the verifier (ascending subset of 2/4/8). Requires `--mtp-window-prefix`; `--mtp-target-window-rows` must equal the largest entry. |
| `--mtp-window-prefix` | Longest-prefix acceptance with deferred GDN commit and per-row PLE snapshots (no checkpoint restore). |
| `--mtp-window-expert-union` | Routed experts read once per window for all rows. |
| `--mtp-window-hyper-mix`, `--mtp-window-qsa-rows` | Multi-row HC mixers and QSA attention inside the window. |
| `--mtp-draft-chain` | Chained MTP drafting on the device without per-draft host syncs. |
| `--mtp-spec-min-p P` | Stop drafting after the first draft whose probability is below P; sets the window size. |
| `--mtp-draft-q4` | Q4_0 copy of the draft head over the vocabulary subset (needs a subset). |
| `--mtp-draft-vocab FILE[,FILE...]` or `auto:DIR` | Draft vocabulary: little-endian `uint32` token IDs; several files are merged without duplicates. `auto:DIR` loads `DIR/core32k.bin` plus the `DIR/pack-<lang>16k.bin` that holds the most prompt tokens (see [Draft vocabulary](#draft-vocabulary)). |
| `--mtp-draft-context-vocab N` | Also admit up to N token IDs seen in the prompt and output into the draft vocabulary. |
| `--mtp-candidate-depth D` | Maximum drafts per cycle; at least `rows - 1` of the largest window. |

## Prompt prefill

Long prompts are read by the step-major tiled prefill:

```
--known-prefix-window --known-prefix-full-rows --known-prefix-prefill-rows T \
--known-prefix-workspace-budget-bytes N --known-prefix-host-metadata-budget-bytes N \
--known-prefix-uma-budget-bytes N --known-prefix-host-headroom-bytes N
```

T is 16 to 2048 rows per block. Every complete block runs each source step for all T rows before moving on;
the unpadded suffix runs as serial rows. GDN recurrence still updates rows in position order, so the result is
bit-identical to serial prefill within one numerical profile. On this route the full organization flags of
group 2 are stated explicitly. See [known-prefix-tiled-prefill.md](known-prefix-tiled-prefill.md),
[known-prefix-prefill.md](known-prefix-prefill.md) for the budgets, and
[prefill-numerical-profile.md](prefill-numerical-profile.md) for a prefill profile that differs from decode.

The prefill numbers in the README are interleaved request throughput (prompt tokens per second of wall time
for the request) at `dff1dbe`, 2026-10-18, same binary for both profiles; the commit message records the
kernel-level changes and the base it was measured against.

## Draft vocabulary

The draft head scores only a subset of the 248,320-token vocabulary, because reading the whole Q4_0 head costs
about 0.65 ms per draft step at 106k rows and grows linearly with the row count. A token outside the subset can
never be drafted, so the subset sets acceptance but never changes the output (the target verifies over the full
vocabulary).

The repository ships the vocabularies in `data/draft-vocab/`: a 32k language-independent core (code, English,
Markdown, numbers, chat control tokens) and 16k packs for `de`, `es`, `fr`, `it`, `pt`, `ru`, `ja` and `zh`.
`--mtp-draft-vocab auto:data/draft-vocab` picks the core plus the pack holding the most prompt tokens (at least 2
tokens and 5% of the prompt; otherwise the core alone) and prints the choice; explicit lists also work:

```
--mtp-draft-vocab data/draft-vocab/core32k.bin,data/draft-vocab/pack-es16k.bin --mtp-draft-context-vocab 4096
```

Measured against a 106k general subset: on a code prompt the core needs 86 verify cycles instead of 89 at the same
speed within noise; on a Spanish chat prompt `auto` selects core + `es16k` and decodes at 38.9 tok/s instead of
32.9 (+18%). Below 16k rows acceptance drops. `tools/build_draft_vocab.py` documents the file format and builds
lists from the GGUF tokenizer and text corpora; [data/draft-vocab/README.md](../data/draft-vocab/README.md)
describes how the shipped lists were built.

## Environment switches

Defaults are the measured configuration.

| Variable | Effect |
|---|---|
| `STRIX_HASH_CACHE=0` / `=DIR` | Disable, or relocate, the verified SHA-256 cache of model files (default `~/.cache/strix-engine/file-sha256`). A cached digest is reused only while the file keeps its device, inode, size, mtime and ctime, and it is still checked against the expected SHA-256. |
| `STRIX_MTP_DRAFT_WORTH=0` | Draft up to `--mtp-spec-min-p` only, without asking the window cost model before each extra step. |
| `STRIX_MTP_DRAFT_Q4_ALL=0` | Disable Q4_0 copies of the MTP draft projections. |
| `STRIX_ANYORDER=0` | Disable any-order launches of independent kernels. |
| `STRIX_PLE_WARM=0` | Skip prefetching the file-backed PLE table after registration. |
| `STRIX_MTP_GREEDY_ROWS=0` | Use per-row selectors instead of on-stream argmax of all window rows. |
| `STRIX_CYCLE_TIMING=1` | Print host phase timings per MTP cycle. |
| `STRIX_CYCLE_LOG=1` | Print each MTP cycle's window rows and wall time. |
| `STRIX_WINDOW_TIMING=1` | Print host-side phase totals of every known window at exit. |
| `STRIX_PREFILL_TIMING=1` | Print download, scoring and spool times per block of teacher-forced tiled prefill. |

Opt-in experiments that were measured and not adopted: `STRIX_WEIGHTS_COARSE_GRAINED=1`,
`STRIX_HC_ROWS_FUSED_NORM=1`, `STRIX_LM_HEAD_ROWS_WIDE=w16|direct`, `STRIX_MOE_GATEUP_CHUNK=5`.
Kernel-level fallbacks for the newest prefill kernels (`STRIX_HC_WIDE_PROJECTION_KERNEL=0`,
`STRIX_HC_WIDE_OUTPUT_KERNEL=0`, `STRIX_DENSE_A8_LDS=0|2`, `STRIX_GDN_FUSED_A8=0`, `STRIX_PREFILL_PLE_WIDE=0`)
select the previous bit-identical kernels; they exist for A/B checks.

## Host-only tools

These need no GPU: `strix-memory-plan` (UMA admission planning), `strix-inspect-gguf` (GGUF structure and
storage), `strix-model-manifest` (tensor roles, bytes, residency and node placement), `strix-platform-probe`
(read-only discovery of RAM, KFD GPU, ROCm and USB4 topology).

## Repository layout

| Path | Contents |
|---|---|
| `src/`, `include/strix/` | Engine: GGUF reader, model contract, HIP kernels and launchers, MTP runtime and window verifier |
| `apps/` | Executables: `strix-hip-decode-token` and host tools |
| `tests/` | Host and device tests (CTest), oracles and fixtures |
| `tools/` | Draft vocabulary builder, oracles, qualification and metadata scripts, README chart generator |
| `lab/` | Standalone experiments: kernels, quantization, UMA residency, USB4/direct GPU transport |
| `data/draft-vocab/` | Shipped MTP draft vocabularies |
| `docs/` | Architecture guide, this reference and implementation notes |
| `artifacts/` | Pinned official model configuration |

## Documentation map

- [docs/README.md](README.md): architecture guide (platform, memory, kernels, numerics, speculative decoding,
  multi-node USB4) and index of implementation notes.
- [docs/VERIFY_WINDOW_PLAN.md](VERIFY_WINDOW_PLAN.md): multi-row verify window design and status.
- [docs/mtp-native-cli.md](mtp-native-cli.md), [docs/mtp-native-runtime.md](mtp-native-runtime.md): MTP
  command line and runtime contract.
- [docs/decode-optimization-profile.md](decode-optimization-profile.md): automatic optimization profile and
  teacher-forced scoring.
- [docs/prefill-numerical-profile.md](prefill-numerical-profile.md),
  [docs/known-prefix-tiled-prefill.md](known-prefix-tiled-prefill.md),
  [docs/teacher-forced-tiled-prefill.md](teacher-forced-tiled-prefill.md): prompt prefill.
- [docs/qualification-ledger.md](qualification-ledger.md): what has been qualified, and how.
- [DECISIONS.md](../DECISIONS.md): accepted, pending and rejected decisions with their reasons.
- [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md): implemented components and how to reproduce them.
- [SPEC_DRAFT.md](../SPEC_DRAFT.md), [EVIDENCE.md](../EVIDENCE.md), [DESIGN_ACCEPTANCE.md](../DESIGN_ACCEPTANCE.md):
  specification, evidence register and acceptance criteria.
- [MEMORY_BUDGET.md](../MEMORY_BUDGET.md): UMA/GTT policy and admission guards.
- [GGUF_READER.md](../GGUF_READER.md): GGUF reader contract and defenses.
- [SOURCES_LEDGER.md](../SOURCES_LEDGER.md): canonical sources, licenses and provenance.

## Working rules

New observations go to `EVIDENCE.md` first. Design changes are reflected in `SPEC_DRAFT.md`, and the reason is
recorded in `DECISIONS.md`. A hypothesis does not become a requirement until a measurement or an explicit
decision supports it. Native `gfx1151` qualification is required before a HIP or direct-transport path becomes a
default.

## Multi-node direction

The distributed design treats every physical USB4 link as an independent resource with its own epoch, lease,
TX/RX pools, completion accounting and health state. Striping or splitting traffic across two links is allowed
only after simultaneous measurements show the links are independent. See
[architecture/16-direct-gpu-transport.md](architecture/16-direct-gpu-transport.md) and
[architecture/12-multi-node-and-usb4.md](architecture/12-multi-node-and-usb4.md).
