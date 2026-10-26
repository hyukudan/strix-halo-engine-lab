# Optional LookaheadCache proposal-head cache

The HIP resident drafter can use the external
[LookaheadCache](https://github.com/hyukudan/lookahead-cache) C ABI package,
version 0.4.2 or later, for its rank-256 proposal head. The library is installed
separately. Its source, models and benchmark data are not copied into this engine.
The library is AMD Strix Halo (HIP) first and modular; its CUDA backend exists
in the library, and CUDA engine integration is in progress (not part of this PR).

The build option `STRIX_ENABLE_LOOKAHEAD_CACHE` defaults to `OFF`.
`STRIX_ENABLE_LOOKAHEAD` remains accepted as a deprecated alias. An enabled build
still selects the existing head until `STRIX_LOOKAHEAD_CAPACITY` is set or the
caller explicitly invokes `DeviceDrafter::configure_lookahead`.

## What the cache stores

For predecessor token `p`, the head adds a fixed Markov bias row to the current
base logits, then selects the largest score, breaking ties by lower global token
ID. The optional library caches that bias row for an immutable vocabulary
binding. It combines the row with fresh base logits on every step; a cache hit
does not reuse an earlier selected token.

The adapter retains the split between core logits and extra-vocabulary logits,
the predecessor embeddings read by confidence, and the drafter's stream order.
Target verification, accepted-prefix handling and rollback stay in the existing
driver. `keep_logits` continues to expose the base logits expected by diagnostics.

## Build the external dependency and engine

Access to the dependency repository or a preinstalled HIP package must be
provided separately by its owner. The engine does not fetch the dependency over
the network during configuration or compilation.

Use an authorized, separate LookaheadCache checkout whose package version is at
least 0.4.2. The 0.4.2 source revision used for this integration is
[`9d060f7a59d3bf0bc6fe626cbd73309e85240036`](https://github.com/hyukudan/lookahead-cache/commit/9d060f7a59d3bf0bc6fe626cbd73309e85240036).
The commands below assume it already exists at `../lookahead-cache` and that
ROCm is installed at `/opt/rocm`; adjust the SDK prefix and HIP compiler when
using another location. For a reproducible qualification, record both
repositories' exact commit IDs and the compiler versions. If the package is
already installed, skip its build/install commands and set
`STRIX_LOOKAHEAD_PREFIX` to that installation prefix.

```sh
STRIX_LOOKAHEAD_PREFIX="$PWD/../lookahead-cache/local-install"

cmake -S ../lookahead-cache -B ../lookahead-cache/build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DSL_ENABLE_HIP=ON \
  -DSL_BUILD_EXAMPLES=OFF \
  -DBUILD_TESTING=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_INSTALL_PREFIX="$STRIX_LOOKAHEAD_PREFIX"
cmake --build ../lookahead-cache/build-hip \
  --target strix_lookahead_hip sl_hip_test --parallel 2
cmake --install ../lookahead-cache/build-hip

cmake -S . -B build-lookahead \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DSTRIX_ENABLE_HIP_RUNTIME=ON \
  -DSTRIX_ENABLE_LOOKAHEAD_CACHE=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_PREFIX_PATH="$STRIX_LOOKAHEAD_PREFIX;/opt/rocm"
cmake --build build-lookahead \
  --target strix-deepseek41-batch strix-hip-ds41-dspark-tests --parallel 2
```

CMake requires the installed `LookaheadCache::hip` target when the adapter is
enabled. A default build does not search for the package. To return to a build
without the adapter, configure with `-DSTRIX_ENABLE_LOOKAHEAD_CACHE=OFF`.

## Runtime modes and memory

| Configuration | Proposal head |
| --- | --- |
| `STRIX_LOOKAHEAD_CAPACITY` unset | Existing engine implementation |
| `STRIX_LOOKAHEAD_CAPACITY=0` | External library, without cached bias payload |
| `STRIX_LOOKAHEAD_CAPACITY=N`, `N > 0` | External library with `N` direct-mapped entries per sequence context |

The value must be an unsigned decimal integer. The default payload budget is
64 MiB **per context**, checked against the full vocabulary before allocation.
For the 129,280-token vocabulary, payload bytes are
`N * 129280 * sizeof(float)`; 16 entries use 7.89 MiB per context. Mapping and
workspace allocations are additional. The built-in budget permits up to 129
entries at the full vocabulary size.

`DeviceDrafter::footprint` includes the environment-requested cache, mapping and
workspace reservation when serving admission computes its memory requirements.
A caller that enables the adapter manually must reserve
`DeviceDrafter::lookahead_footprint(capacity, sequences, budget)` before calling
`configure_lookahead`. Actual active and parked payload bytes are exposed by
`lookahead_stats().payload_bytes`.

Each sequence context owns its binding and cache. Context switching preserves
that context's entries; reset invalidates them. Vocabulary replacement, including
reordering without a size change, rebuilds the binding. Vocabulary growth also
rebuilds it; adding only IDs already present retains it. Binding and reset calls
must occur outside graph capture on the owning device. This integration does not
make the complete drafter graph-capturable.

The first adapter path supports an unsharded proposal vocabulary (`world=1`).
The vocabulary-sharded `world=2` head uses the existing implementation and reports
fallbacks. This is distinct from distributing independent requests between two
drafter instances; this change makes no two-node performance claim. An unavailable
compiled adapter is also reported as a fallback. Allocation, enqueue or invalid
device-result failures stop the draft instead of silently changing paths midway
through its token chain.

## Correctness before timing

First run the library's native HIP contract test on the intended device:

```sh
ctest --test-dir ../lookahead-cache/build-hip \
  -R '^lookahead[.]hip_device_contract$' --output-on-failure
```

A skipped device test is not a correctness pass. Next supply the target and
drafter assets to the engine test:

```sh
STRIX_DS41_Q2_GGUF=/path/to/target.gguf \
STRIX_DS41_DSPARK_GGUF=/path/to/drafter.gguf \
STRIX_LOOKAHEAD_RECEIPT=1 \
  ./build-lookahead/strix-hip-ds41-dspark-tests
```

The integration fixture uses real checkpoint weights and synthetic hidden inputs.
It compares the existing head, the uncached library and the cached library for
tokens, confidence, predecessor embeddings, base logits and stage outputs. It
also exercises reduced vocabularies, same-size reordering, growth, context
switching, reset and context recreation. A successful integration fixture emits
`STRIX_LOOKAHEAD_MODEL_TEST` with `fixture=real_weights_synthetic_hidden`; both that
receipt and successful process exit are required. Skipping the weights or the
compiled adapter does not qualify the integration.

The target verifier can correct changed proposals, so an end-to-end token check
alone cannot establish proposal/confidence equivalence. After the native fixture,
the external library's qualification tool runs three arms: the existing head,
capacity zero, and the requested cached capacity.

```sh
python3 ../lookahead-cache/tools/qualify_engine.py \
  --executable ./build-lookahead/strix-deepseek41-batch \
  --native-test ./build-lookahead/strix-hip-ds41-dspark-tests \
  --model /path/to/target.gguf \
  --draft /path/to/drafter.gguf \
  --prompt-file /path/to/prompt-ids.csv \
  --concurrency 1,2,4 \
  --layers 40 --generate 128 --max-positions 4096 \
  --capacity 16 --repeats 3 \
  --output /path/to/new-evidence-directory
```

Prompt files contain comma-separated token IDs; repeat `--prompt-file` for
distinct requests. The first campaign uses single-node HIP, a fixed verification
window and full or numeric-core vocabularies. Eight concurrent requests fill all
eight verifier rows and leave no draft budget, so the qualification tool accepts
concurrency 1 through 7. At least three generated tokens are needed to exercise a
proposal after prefill. Fewer than 40 layers is a truncated diagnostic.

The tool checks native-model receipts, target token/state gates and adapter
activation before timing; fingerprints the supplied binaries and assets; and
rejects silent fallback or zero-work runs. Its decode ratios exclude model loading.
Process wall times and the recorded configuration should accompany any result.
The row/state FNV checks are digests, not collision-free bitwise proofs.

## Receipts and optional predecessor traces

`STRIX_LOOKAHEAD_RECEIPT=1` adds one `STRIX_LOOKAHEAD_RUN` JSON record per batch
case, outside the measured decode loop. It includes the effective mode, payload
bytes, step/hit/miss counters, binding rebuilds, fallbacks and errors, together
with workload and target-verification fields. Rebuild deltas include setup since
the case began, while decode time excludes that setup; do not interpret those
deltas as a decode-only breakdown.

`STRIX_LOOKAHEAD_TRACE=/path/to/new-file.jsonl` records each draft's predecessor
tokens, cache-hit flags, context, vocabulary epoch and capacity after the existing
result download. The file must not already exist. Records are flushed and checked;
tracing adds host I/O and must be disabled for timing. A separate file is required
per process/drafter. `qualify_engine.py --trace` enables separate qualification
traces and keeps tracing disabled during warmup and timing.

The implementation and compiler checks establish neither a device speedup nor a
complete-model speedup. Qualification requires real device execution and the
complete workload above, including vocabulary changes and cache misses.
