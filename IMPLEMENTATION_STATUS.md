# Estado de implementación

## Slice A0.1 — núcleo portable

Implementado:

- proyecto CMake C++20 sin dependencias externas;
- contrato de shapes Qwen3.8-Flash-Next fijado desde el config oficial;
- planificador de memoria con aritmética comprobada contra overflow;
- desglose KV base, MTP, indexer provisional, GDN FP32, rollback, modelo y cachés;
- decisión de admisión desde MemTotal, baseline y reserva explícita;
- CLI JSON `strix-memory-plan`, preparada para alimentar la futura WebUI;
- oráculo top-k candidato con empates deterministas y rechazo de NaN;
- pruebas CTest en Linux/WSL con warnings-as-errors.

No implementado o deliberadamente bloqueado:

- semántica QSA canónica, n-gram/PLE, hyper-connections y GDN: esperan contratos
  oficiales/oráculos antes de crear kernels;
- allocator/residency HIP y kernels `gfx1151`;
- MTP sampled;
- servidor y panel administrativo.

## Reproducción

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
ctest --test-dir build --output-on-failure
./build/strix-memory-plan --tokens 32768 --sessions 1 \
  --resident-gib 68 --baseline-gib 2 --reserve-gib 18
./build/strix-platform-probe
./build/strix-model-manifest MODEL.gguf
```

El indexer es una estimación provisional derivada del config. Se puede excluir
con `--no-indexer`. `--rollback-full` permite cuantificar por qué copiar el
estado GDN completo por borrador no escala; el camino objetivo será snapshot
mínimo más recompute si el contrato lo permite.

## Slice A0.2 — inspección GGUF

Implementado en `include/strix/gguf_reader.hpp` y `src/gguf_reader.cpp`: lector
GGUF v3 little-endian defensivo y CLI `strix-inspect-gguf`. Valida metadata,
arrays, directorio tensorial, geometría de bloques, tamaños, alineación, rangos
y solapamientos sin cargar los datos de pesos.
Las limitaciones y la revisión canónica se documentan en `GGUF_READER.md`.

Validación actual:

- build `RelWithDebInfo` con `-Werror`;
- fixtures válidos y corruptos en CTest;
- build Debug con AddressSanitizer y UndefinedBehaviorSanitizer;
- 100 % de las pruebas actuales superadas en ambos builds.


## Slice A0.4 — model manifest

Implemented:

- logical tensor manifest derived from validated GGUF metadata;
- conservative tensor-role classification;
- per-tensor physical bytes and offsets;
- layer extraction for common block naming;
- explicit residency policy;
- explicit placement with local node zero as the default;
- n-gram/sparse-table storage can be marked as a file-backed candidate without changing admission semantics for ordinary weights;
- JSON CLI `strix-model-manifest`;
- dedicated unit tests.

The manifest is intentionally conservative. Unknown tensors remain visible and resident by default. Complete architecture-specific tensor-name validation remains a later gate once the canonical development checkpoint is fixed.

## Slice A0.5 — platform capability probe

Implemented:

- dependency-free Linux host probe;
- physical RAM from `/proc/meminfo`;
- KFD GPU topology and `gfx_target_version`;
- derivation of `gfx1151` and compute-unit count when KFD exposes the required fields;
- ROCm version-file and `hipcc` presence checks;
- Thunderbolt/USB4 domains and host-router discovery from sysfs;
- USB4STREAM driver and `/dev/tbstream*` discovery;
- JSON CLI `strix-platform-probe`;
- fake-sysfs unit test for deterministic host-independent validation.

Not yet qualified on the target hardware:

- real port-to-router mapping;
- negotiated USB4 link rate;
- simultaneous dual-link independence;
- HIP runtime/device capabilities;
- GTT/TTM effective mapping limits;
- graph and library capability inventory.

Those require native Linux measurements on the Strix Halo hosts.


## Slice A0.5b — semantic and numerical oracle

Partially implemented in `tools/oracles/`:

- dependency-free Python reference code, separate from runtime inference;
- centred and grouped RMSNorm;
- partial rotary application;
- sequential Gated DeltaNet recurrent update in double precision;
- PLE/n-gram hashed row IDs with EOS-aware history;
- deterministic candidate top-k policy;
- tiny frozen fixtures;
- fresh-versus-resumed equivalence tests for recurrent and n-gram state;
- optional CTest integration when Python 3 is available.

Still open:

- canonical sampled RNG/replay semantics;
- MTP acceptance algorithm;
- complete QSA selection and tie fixtures;
- Gated Residual/Hyper-Connection checkpoint fixtures;
- tokenizer/chat-template golden tokens;
- first-layer/logit replay from the canonical development checkpoint.


## Slice A0.7 — transport core

Initial host-independent core implemented:

- explicit producer/transport/consumer buffer ownership state machine;
- abstract asynchronous transport-backend interface with prepare/submit/poll/cancel;
- distinct local-reuse, receive-complete and remote-transport-ack completion kinds; semantic commit remains above transport;
- transfer descriptors bound to epoch, model-plan, request, generation, execution boundary, shape, dtype and byte length;
- bounded transfer descriptors with protocol version and connection epoch;
- qualified-link profiles with latency, payload bandwidth and in-flight accounting;
- deterministic single-link cost estimation;
- flow scheduling that reacts to queued bytes;
- dual-link striping only when both links are explicitly qualified as independent and the estimated complete transfer beats the single-link path;
- unit tests for ownership, stale/oversized descriptors, single-link selection, queue-aware selection and striping;
- standalone TCP ping-pong and stream benchmark harness;
- direct USB4STREAM read/write benchmark harness;
- dual-link parallel-flow runner with configurable payload split and aggregate JSON metrics;
- read-only capture of configured stream index, busy-poll, ring-size, throttling and HopIDs;
- framed payload validation, optional checksums, warmup exclusion and JSON latency/throughput output;
- host-side CI workflow covering the core, oracles, UMA host lab and TCP smoke tests.

Still pending on the target hosts:

- real two-host TCP/network baseline measurements;
- real two-host direct USB4STREAM measurements;
- interrupt/event versus polling progress modes;
- mapped/registered transport-buffer tiers;
- real single-link and dual-link topology qualification.


## Slice A1.0 — kernel registry and basic gfx1151 kernels

Host-side registry implemented:

- explicit operator family and execution phase;
- exact logical-row, request-concurrency and geometry shape classes;
- independent Wave32/Wave64 identity;
- workgroup, LDS/register/scratch metadata fields;
- numerical contract with row-invariance and deterministic flags;
- graph-safety and qualification state;
- compiler identity and measured cost;
- deterministic selector;
- stable plan identity that deliberately excludes measured-time noise;
- qualified/production variants require a non-zero measured cost;
- CTest coverage for validation, selection, wave filtering and plan identity.

Standalone `lab/basic_kernels/` implemented for native ROCm:

- `gfx1151`-only build;
- native Wave32 candidate;
- explicit `-mwavefrontsize64` Wave64 candidate;
- device-side `warpSize` verification;
- FP32 centred RMSNorm;
- FP32 residual add;
- FP32 partial RoPE;
- deterministic argmax;
- deterministic top-k baseline up to k=16;
- solo-versus-batched row-invariance fixture;
- model-representative microbenchmarks for hidden size 2560, 24x256 partial RoPE, vocab 248320 and expert top-10 of 512;
- HIP-event timings and `hipFuncGetAttributes` resource reporting.

Still pending:

- native M5/Strix Halo build and execution;
- Wave32 versus Wave64 measurements;
- profiler-derived VGPR/LDS/spill confirmation;
- graph-capture qualification;
- promotion of any kernel from experimental to target-qualified;
- BF16/mixed-precision variants, which require separate numerical identities.


## Slice A1.1 — packed low-row projection

Host-side planning implemented:

- GGML tensor storage geometry is now a public contract shared by the loader and kernel planning;
- explicit low-row classes 1/2/4/8;
- exact block geometry and packed bytes per output row;
- exact tensor weight bytes;
- activation input/output byte accounting;
- serial repeated-weight traffic versus ideal reused-weight lower bound;
- overflow and quant-block divisibility checks;
- CTest coverage including Q4_K, Q8_0 and IQ4_XS geometry;\n- independent IQ4_XS 136-byte block/dequant contract with nonlinear codebook and 6-bit local-scale extraction;\n- host IQ4_XS fixture compiled with strict warnings and passing.

Standalone `lab/packed_projection/` implemented:

- Q8_0 packed reference kernel for `gfx1151`;\n- IQ4_XS compact packed candidate for `gfx1151`;
- packed block is consumed directly inside compute; no full dequantized weight tensor;
- compile-time C1/2/4/8 logical-row variants;
- each decoded packed weight is reused across active logical rows;
- native Wave32 and explicit Wave64 builds;
- device-side wave-size verification;
- CPU packed-format reference and true row-invariance test;
- model-shape benchmark at 2560x2560;
- HIP-event timings;
- physical packed-weight GiB/s and serial-equivalent weight-traffic metric;
- per-variant register/shared/local-memory attributes.

Still pending:

- native Strix Halo build/profile;
- canonical compact checkpoint format selection;
- native compile/profile of the IQ4_XS compact path and comparison against Q8_0;\n- canonical checkpoint-format decision after quality plus full memory/runtime qualification;
- BF16/mixed-precision activation variants;
- profiler spill/LDS/VALU/WMMA analysis;
- promotion into a target-qualified kernel plan.
