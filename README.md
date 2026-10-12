# Motor Strix Halo propio — espacio de investigación

Este repositorio desarrolla un motor propio con contratos matemáticos,
oráculos independientes y cualificación explícita sobre el hardware objetivo.

## Architecture documentation

The normative, source-independent technical decisions for the Strix Halo runtime live in [`docs/`](docs/README.md). Research notes and evidence remain separate from the architecture guide.

## Current implementation state

The project has moved beyond the initial host-only bootstrap.

Current experimental pieces include:

- defensive GGUF v3 parsing and model manifests;
- pinned Qwen3.8-Flash-Next geometry and semantic oracles;
- packed projection, GDN, MoE and QSA/PLE planning plus HIP harnesses;
- a compiled C1 serial execution DAG with versioned IO, physical scratch arena and persistent session-state layouts;
- experimental HIP launchers covering the serial DAG;
- shared opt-in integer projection launcher with owned allocations and epoch-bound completion;
- experimental device-position norm/RoPE/KV stage for graph replay;
- raw-token decode tooling that can materialize logits from a GGUF once native target qualification is available;
- greedy speculative target verification and snapshot/replay infrastructure;
- USB4 transport abstractions, TCP/direct-stream baselines and dual-link scheduling;
- bounded C1 host-copy tensor payload and distributed boundary adapter, tested with socketpairs;
- an experimental architecture for GPU-owned DMA-BUF USB4 transport with explicit commit/consumer-credit/RX-repost semantics.

Native `gfx1151` qualification remains mandatory before any HIP or direct-transport path is promoted.

### USB4 multi-node direction

The distributed design treats every physical USB4 connection as an independent resource.

For two links between the same pair of hosts, each link keeps its own epoch, lease, TX/RX pools, completion accounting and health state. The scheduler may later choose independent flows, directional specialization, role specialization or large-message striping, but only after simultaneous measurements prove the links are genuinely independent.

The normative synchronization and ownership contract is in [`docs/architecture/16-direct-gpu-transport.md`](docs/architecture/16-direct-gpu-transport.md). The broader placement/scheduler design remains in [`docs/architecture/12-multi-node-and-usb4.md`](docs/architecture/12-multi-node-and-usb4.md).

## Documentos

- [`RESEARCH.md`](RESEARCH.md): alcance de investigación y criterios de evidencia propios.
- [`SPEC_DRAFT.md`](SPEC_DRAFT.md): especificación técnica inicial del motor propio.
- [`EVIDENCE.md`](EVIDENCE.md): índice de afirmaciones, evidencia y nivel de confianza.
- [`DECISIONS.md`](DECISIONS.md): decisiones aceptadas, pendientes y descartadas.
- [`DESIGN_ACCEPTANCE.md`](DESIGN_ACCEPTANCE.md): acciones de aceptación del diseño y trazabilidad.
- [`SOURCES_LEDGER.md`](SOURCES_LEDGER.md): fuentes canónicas, licencias y bloqueos de procedencia.
- [`MEMORY_BUDGET.md`](MEMORY_BUDGET.md): política propuesta de UMA/GTT y guardas de admisión.
- [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md): código disponible, reproducción y bloqueos.
- [`GGUF_READER.md`](GGUF_READER.md): contrato, defensas y límites del lector GGUF propio.
- [`tools/`](tools): analizadores que conservan solo metadatos y recuentos agregados.

## Public documentation policy

Private comparative inputs remain outside Git. Public contracts and qualification
records describe this engine. Actual adopted components keep required notices.

## Regla de observación no derivativa

La observación de external runtime A se limita a comportamiento, metadatos de kernel,
mezcla agregada de instrucciones y medidas. No se copiará, reconstruirá ni
redistribuirá código fuente o kernels derivados. Las implementaciones nuevas se
escribirán desde contratos matemáticos públicos y se verificarán contra oráculos
independientes. En este alcance no se reutilizará código de external runtime B: se estudia
como referencia y el diseño presupone implementación propia. Cualquier cambio
futuro de esa política exigiría una decisión explícita y conservación de avisos.

## Método de actualización

Cada nueva observación se guarda primero en `EVIDENCE.md`. Si cambia el diseño,
se refleja en `SPEC_DRAFT.md` y se registra el motivo en `DECISIONS.md`. Las
hipótesis nunca se convierten en requisitos hasta que exista una medición o una
decisión explícita. Este orden permitirá mantener una spec trazable y
contrastar cada objeción contra evidencia concreta.

## Primer build offline

El slice A0.1 ya contiene un núcleo C++20 portable y el ejecutable
`strix-memory-plan`. Véase `IMPLEMENTATION_STATUS.md` para compilarlo en Linux o
WSL y obtener un desglose JSON consumible por la futura WebUI.


## Current host-side tools

- `strix-memory-plan`: conservative UMA admission planning.
- `strix-inspect-gguf`: defensive GGUF structure and storage inspection.
- `strix-model-manifest`: logical tensor roles, bytes, residency candidates and node placement.
- `strix-platform-probe`: read-only Linux discovery for RAM, KFD GPU identity, ROCm and USB4/USB4STREAM topology.

These tools remain useful independently of the experimental HIP execution layer and can be used to qualify a host before enabling GPU runtime or direct USB4 transport tiers.
