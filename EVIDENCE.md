# Registro de evidencias

Este archivo es la fuente de trazabilidad de la especificación. Los estados
permitidos son: **observado**, **publicado**, **inferido**, **hipótesis** y
**pendiente**.

| ID | Estado | Afirmación | Evidencia | Consecuencia para la spec |
| --- | --- | --- | --- | --- |
| E-001 | observado | La imagen external runtime A Flash 0.14.2 contiene tres objetos HIP `gfx1151`. | Extracción con `llvm-objdump --offloading`; hashes en `RESEARCH.md`. | Separar los subsistemas compilados es una opción a evaluar. |
| E-002 | observado | Los objetos contienen 1.057, 172 y 62 entradas de kernel. | `private artifact record`. | Usar generación declarativa de variantes y medir cuáles son necesarias. |
| E-003 | observado | Todas las entradas inspeccionadas declaran wavefront 32. | Metadatos AMDGPU de los tres objetos. | Probar wave32 como candidato, sin convertirlo en regla universal. |
| E-004 | observado | Algunas variantes usan hasta 256 VGPR, aproximadamente 64 KiB de LDS y spills. | JSON de metadatos agregado. | Evaluar tiempo completo; la ocupación aislada no decide el ganador. |
| E-005 | observado | La familia `q4moe` emplea WMMA F16/BF16, LDS, permutaciones y operaciones packed. | `private artifact record`. | Priorizar un prototipo MoE Q4 propio y comparar distintas estrategias de packing. |
| E-006 | observado | Las variantes DeltaNet combinan scan, WMMA/FMA, LDS y estado recurrente. | `private artifact record`. | Probar fusión de preparación, scan y escritura de estado. |
| E-007 | observado | Las variantes de atención conservan una parte importante del softmax en FP32. | `private artifact record`. | Fijar el orden de reducción y las fronteras de redondeo antes de optimizar. |
| E-008 | publicado | external runtime A integra `verify`, `inspect`, `ppl` y `niah`. | README/AGENTS de external runtime A Flash 0.14.2. | Incluir herramientas de calidad como comandos del producto. |
| E-009 | publicado | external runtime B implementa GGUF, HIP graphs, sesiones, snapshots, batching y MTP adaptativo. | Código y documentación del commit fijado en `RESEARCH.md`. | Adoptar contratos equivalentes mediante implementación propia. |
| E-010 | publicado | external runtime B exige controles de operador, replay y medición end-to-end. | `.agents/skills/optimize-kernel/SKILL.md` y documentación de calidad. | Hacer obligatorias tres puertas de calidad y benchmark emparejado. |
| E-011 | inferido | La especialización extrema puede explicar parte del rendimiento de external runtime A. | Número de variantes y ausencia de runtime general; aún sin perfil dinámico. | Diseñar un registry especializado, pero no fijar cuántas variantes necesita. |
| E-012 | hipótesis | Un formato interno lossless puede reducir el coste de arranque sin duplicar demasiada memoria/disco. | Todavía sin prototipo ni medida. | Mantenerlo opcional hasta medir carga, tamaño y tráfico. |
| E-013 | pendiente | Diferencia real external runtime A/external runtime B bajo condiciones idénticas. | Falta máquina Linux Strix Halo, pesos y harness emparejado. | No utilizar cifras públicas como comparación directa. |
| E-014 | pendiente | Mejor política wave32/wave64 por familia y forma. | Requiere compilar y medir ambos modos. | El registry debe admitir ambos durante investigación. |
| E-015 | publicado | HIP/Clang permite compilación offline para un target GFX mediante `--offload-arch`. | Documentación oficial AMD HIP/Clang. | Crear un laboratorio reproducible `gfx1151` antes de recibir el equipo. |
| E-016 | observado | external runtime A y external runtime B exponen API/health/métricas, pero no una WebUI administrativa completa. | Búsqueda en repositorios fijados. | Diseñar un plano administrativo propio; chat solo como playground opcional. |
| E-017 | observado | La imagen runtime de external runtime A no es un SDK completo: el probe HIP falla por ausencia de headers C++ estándar. | Compilación de `lab/cross_compile/probe.hip` dentro de la imagen OCI fijada. | No usar imágenes runtime ajenas como toolchain del proyecto. |
| E-018 | observado | `rocm/dev-ubuntu-24.04:7.2.4` compila offline el probe para `gfx1151`, y el HSACO extraído declara ese target, wave32 y cero spills. | Imagen por digest y hashes en `lab/cross_compile/README.md`. | El frontend/codegen básico queda validado; ejecución y rendimiento siguen pendientes. |
| E-019 | observado | La revisión de diseño registró bloqueos de memoria, residencia, semántica y backend, con alcance inicial reducido. | Acta reproducible en `DESIGN_ACCEPTANCE.md`. | No congelar la spec hasta resolver memoria, residencia UMA, fuentes canónicas y librerías `gfx1151`. |
| E-020 | publicado | Qwen publicó repositorio, informe técnico y checkpoint oficial de Flash-Next; describe 125B principales, 6B activos/token y 51B extra de tabla n-gram, con 36 capas GDN y 12 QSA. | `QwenLM/Qwen3.8-Flash-Next@69885871a64393807d988b27b1b5e380e8f28526`, arXiv:2608.30320 y `Qwen/Qwen3.8-Flash-Next@de4b8e4...`. | Usarlos como fuentes primarias de contratos; external runtime A/external runtime B/external platform study quedan como referencias secundarias. |
| E-021 | publicado | El checkpoint oficial usa Qwen Community License 1.0 y contempla condiciones adicionales para determinados usos comerciales MaaS/AI Work Assistant. | `Qwen/Qwen3.8-Flash-Next/LICENSE`. | Mantener abierta la decisión de licencia, distribución y caso de uso. |
| E-022 | publicado | En Strix Halo GTT es límite de mapeo GPUVM sobre RAM compartida, no VRAM adicional; AMD recomienda carve-out BIOS pequeño y elevar TTM/GTT. | Documentación oficial ROCm 7.2 de optimización Strix Halo. | Separar mapping limit de admission y reservar memoria efectiva del host. |
| E-023 | publicado | external runtime A propone para 128 GB `gttsize=126976` MiB y `pages_limit=32505856`; advierte GTT residual tras terminación sucia. | README actual de `private source record: external runtime A-flash-server`. | Perfil cercano a 124 GiB solo con guardas, shutdown limpio y recuperación. |
| E-024 | observado | `external platform study/strix-halo@f73872f` fija runtime y llama.cpp experimentales, publica medidas repetidas y conserva resultados negativos. | Copia local y `PLATFORM_REVIEW.md`. | Adoptar su disciplina experimental, no sus cifras como baseline propio. |
| E-025 | publicado | El mayor salto aislado publicado por external platform study fue un GDN tiled que mantiene recurrencia en registros y sustituye LDS por DPP preservando orden. | `journey.html`, paso 11: ~499,52→1181,77 t/s en su protocolo. | Primer prototipo GDN; exige oráculo independiente y hardware real. |
| E-026 | publicado | Su lector directo de PLE midió 1,69× frente a demand paging por mmap, mientras sparse QSA completo quedó aproximadamente break-even frente a denso hasta 140k. | `journey.html`, pasos 3–4 y ablaciones. | Investigar I/O PLE; usar denso como control de rendimiento/calidad sin asumir equivalencia semántica. |
| E-027 | publicado | PM4 retenido ayuda a grafos repetidos de decode, no a prefill con shapes que no se repiten; está experimental y validado en pocos targets. | Sitio `external platform study.github.io/strix-halo`, sección runtime. | Mantenerlo como experimento opt-in con fallback, fuera del camino crítico inicial. |
| E-028 | observado + inferido | El config oficial fijado declara 48 capas, 12 de atención completa, 2 cabezas KV y `head_dim=256`; de ello se infiere que un KV BF16 convencional requiere 24 KiB/token/sesión: 6 GiB a 262.144. | `artifacts/qwen-official-config-de4b8e4.json`, SHA-256 `889658F2508E8C61D409B02E70E0D78D8D4452EC65AAAFBE129805D213D2E74B`; fórmula en `MEMORY_BUDGET.md`. | C4 a contexto máximo serían 24 GiB solo de KV base; cualificar primero contextos menores. |
| E-029 | pendiente | El inventario offline `gfx1151` de rocBLAS/hipBLASLt no pudo repetirse en esta sesión porque el backend Linux de Docker Desktop no expuso su API, aunque el cliente y el proceso backend existen. | `docker info` contra `desktop-linux`, 2026-09-28. | P-011 sigue abierto; no inferir cobertura de librerías a partir de que el probe HIP compile. |
| E-030 | observado | El lector GGUF propio parsea v3 little-endian y rechaza límites, booleanos, tipos, bloques, alineación, rangos y solapamientos inválidos; ASan+UBSan no detectan fallos. | CTest en builds `build-wsl` y `build-sanitize`, código A0.2/A0.3 y geometrías fijadas de llama.cpp. | Ya se puede inspeccionar metadata/directorio y verificar el rango físico sin cargar pesos. |

| E-031 | publicado + observado | La revisión fijada de la implementación pública `qwen4_exp` define explícitamente RMSNorm centrado, RoPE parcial, la actualización recurrente Gated DeltaNet y el hashing PLE/n-gram; se han creado oráculos independientes y fixtures pequeños con replay fresh/resumed. | `huggingface/transformers@ebd5de00b43e94cae2da6d7aeb5a76ecdda51dae`; `tools/oracles/`. | A0.5b queda parcialmente implementado; sampling/MTP y semántica completa QSA siguen bloqueados. |


| E-032 | publicado | Los GGUF públicos actuales de Qwen3.8-Flash-Next sitúan aproximadamente Q8_0 en 188,26 GB, Q6_K en 168,11 GB, Q5_K_M en 134,67 GB, Q5_K_S en 128,10 GB, Q4_K_M en 119,60 GB, Q4_K_S en 113,14 GB, Q4_0 en 100,60 GB e IQ4_XS en 97,68 GB. | Catálogo público actual de GGUF Qwen3.8-Flash-Next, captura 2026-09-29. | Un perfil mononodo de 128 GiB no puede seleccionar cuantización solo por calidad nominal; debe dejar RAM física para reserva, runtime, KV, recurrencia y scratch. |
| E-033 | inferido | Con una reserva efectiva de 16–20 GiB y el resto del estado C1, un objetivo inicial de aproximadamente 100 GiB o menos de pesos residentes deja margen razonable en una máquina de 128 GiB; Q4_K_M no cumple ese margen y Q4_K_S queda muy próximo al límite antes de A0.6. | E-011, E-028, E-032 y `MEMORY_BUDGET.md`. | Fijar un envelope mononodo conservador; P-002 sigue pendiente de calidad y medidas reales. |
| E-034 | publicado | El config público actual de Qwen3.8-Flash-Next declara `hidden_size=2560`, 24 cabezas de atención, `head_dim=256`, `partial_rotary_factor=0.25`, vocabulario 248.320, 512 expertos y top-10 por token. | Config público de Qwen3.8-Flash-Next, captura 2026-09-29. | Usar estas geometrías como shapes representativas en los microbenchmarks A1.0/A1.1, sin convertirlas en resultados de rendimiento. |

| E-035 | publicado + observado | La revisión fijada de GGML define IQ4_XS como bloques de 256 elementos y 136 bytes: escala FP16 global, 8 escalas locales de 6 bits y 128 bytes de códigos de 4 bits contra una tabla no lineal de 16 valores. El core ya tiene un contrato de bytes y fixture independiente que cubre escalas positivas, negativas y cero. | `ggml-org/llama.cpp@8019dc563b1ecbae6b161a70c3a1359f1b206c1e`; `include/strix/iq4_xs.hpp`; `src/iq4_xs.cpp`; `tests/iq4_xs_tests.cpp`. | IQ4_XS puede pasar a kernel experimental A1.1 sin copiar una implementación backend y sin cerrar todavía P-002. |

| E-036 | publicado + observado | La revisión fijada de Transformers define assisted greedy verification como aceptación del prefijo candidato contiguo igual al `argmax` del target, seguido de un token bonus del target; tras EOS/longitud recompone `n_matches = valid_tokens - 1` y recorta `candidate_length - n_matches` posiciones de cache. La misma revisión define sampled speculative con ratio `p/q` y fallback `[p-q]+`, pero su replay exacto depende del RNG. | `huggingface/transformers@ebd5de00b43e94cae2da6d7aeb5a76ecdda51dae`, `generation/utils.py`, `generation/candidate_generator.py`; `include/strix/speculation.hpp`; `tests/speculation_tests.cpp`. | Cerrar aceptación MTP greedy; mantener sampled bloqueado hasta fijar stream/consumo RNG. |

| E-037 | publicado + observado | La PR abierta de contraste para qwen4exp MTP describe un head de una capa con combiner embedding+hidden, bloque de atención+MoE envuelto en hyper-connections y estado hidden recurrente entre pasos draft; la propia implementación contiene un TODO de QSA para el camino MTP y existen reportes posteriores de aislamiento multi-slot, por lo que no se adopta como oráculo semántico. El motor propio solo extrae de aquí familias estructurales para auditoría y mantiene la ejecución MTP bloqueada. | `ggml-org/llama.cpp#27836@1d8de7c1b0c7d2febf8f983174d8e6a711e2b1af`; `include/strix/mtp_draft_contract.hpp`; `include/strix/mtp_draft_tensor_audit.hpp`. | Mantener separación estructura/semántica; exigir contrato independiente antes de implementar el forward MTP. |

| E-038 | publicado + observado | Dos implementaciones independientes coinciden en la semántica del input fusion MTP con hyper-connections: el embedding normalizado se proyecta una vez y se comparte entre streams; el hidden ancho normalizado se interpreta como `hc_count` streams de `hidden_size`, se aplica la misma proyección `fc_hidden` a cada stream y se suma la proyección del embedding. Se añadió un oráculo CPU independiente para esta suboperación. Esto NO resuelve la atención/QSA ni el forward MTP completo. | `private semantic comparison@41cbe65de05b209f3263942389267f79bd749d6a:python/external semantic reference/srt/models/qwen4_exp_mtp.py`; `ggml-org/llama.cpp#27836@1d8de7c1b0c7d2febf8f983174d8e6a711e2b1af`; `include/strix/mtp_input_fusion.hpp`; `tests/mtp_input_fusion_tests.cpp`. | Marcar solo input-fusion como semántica resuelta; mantener bloqueada la construcción de ejecución MTP completa. |

| E-039 | publicado + observado | external semantic reference y la PR de contraste de llama.cpp coinciden en que el estado que alimenta el siguiente paso MTP es el residual ancho de hyper-connections después del bloque MTP y antes del mixer final de salida. Para el config fijado son 4 streams × 2560 = 10240 valores por fila/request. Se añadió un contrato host-side que rechaza usar el hidden ya colapsado como estado next-step. | `private semantic comparison@41cbe65de05b209f3263942389267f79bd749d6a`; `ggml-org/llama.cpp#27836@1d8de7c1b0c7d2febf8f983174d8e6a711e2b1af`; `include/strix/mtp_draft_state_contract.hpp`; `tests/mtp_draft_state_contract_tests.cpp`. | Marcar draft-state propagation como subcontrato resuelto; mantener QSA/output y forward completo bloqueados. |

| E-040 | observado + fixture propio | El oráculo QSA ya presente en `tools/oracles/qwen4_exp_oracle.py` fija para casos sin empate: promedio de keys por bloque comprimido, RMSNorm de la key pooled, RoPE aplicado a la key comprimida en la posición inicial del bloque, score como suma sobre heads de `max(0, q·k)` escalada por `1/sqrt(head_dim)`, top-k de bloques y conservación del tail parcial. El propio oráculo rechaza un empate que pueda afectar al cutoff. | `tools/oracles/qwen4_exp_oracle.py`; `include/strix/qsa_selection_plan.hpp`; `tests/qsa_selection_plan_tests.cpp`. | Marcar selección QSA sin empates como resuelta; mantener abierto el contrato de desempate y, por ello, el gate completo de atención MTP. |

| E-041 | publicado + observado | El forward canónico fijado de Transformers confirma el pipeline QSA: proyección Q/K del indexer, RMSNorm de Q, cache de raw K, promedio FP32 por grupos completos, RMSNorm de la key pooled, RoPE en la posición inicial del bloque, score `relu(q·k)` sumado sobre heads y escalado por `1/sqrt(head_dim)`, top-k de bloques y tail parcial sin comprimir. El desempate de `torch.topk` no constituye un contrato portable explícito. | `huggingface/transformers@ebd5de00b43e94cae2da6d7aeb5a76ecdda51dae:src/transformers/models/qwen4_exp/modeling_qwen4_exp.py`; `tools/oracles/qwen4_exp_oracle.py`. | Mantener resuelto el camino no-tie, mantener abierto el cutoff-tie y no habilitar todavía el forward MTP completo. |
| E-042 | publicado + observado | Dos implementaciones MTP de contraste usan un mixer final de hyper-connections antes de logits; el repo ya dispone de oráculo independiente para la fórmula gated-residual. La PR de llama.cpp además admite un `shared_head` MTP explícito y solo cae al output target si no está presente. El plan propio resuelve el binding únicamente cuando el manifest contiene un único `shared_head`. | `private semantic comparison@41cbe65de05b209f3263942389267f79bd749d6a`; `ggml-org/llama.cpp#27836@1d8de7c1b0c7d2febf8f983174d8e6a711e2b1af`; `include/strix/mtp_output_contract.hpp`. | Separar mixer resuelto de binding del head; no asumir fallback target sin binding explícito. |

| E-043 | publicado + decisión reproducible | El forward QSA canónico usa `torch.topk` y PyTorch no garantiza estabilidad de índices para elementos empatados. Por tanto upstream no define un orden portable en el cutoff. El motor fija score descendente y menor índice de bloque a igualdad de score, usando la misma regla para serial y MTP; el fixture `qsa_tie` cubre el caso. | PyTorch `torch.topk` docs; D-013; `qsa_select_tokens_deterministic`; `tools/oracles/fixtures/qwen4_exp_small.json`. | Cerrar el gate de desempate como contrato explícito del motor, sin atribuir esa regla a Qwen/PyTorch. |

| E-044 | publicado | Un transporte USB4/NHI cualificado sobre Strix Halo puede importar pools GPU exportados como DMA-BUF y transferir payloads sin pasar por el camino TCP/IP ni por una copia CPU read/write. La referencia pública mantiene TX/RX dedicados, framing propio y participación two-sided de ambos peers. | `private transport reference@1c7f1ae89298c5be2ac4f916b589c77a10c903d8`, `docs/architecture.md`, parches DMA-BUF de `kernel/linux-7.2.2/`. | Añadir un tier experimental de transporte GPU-owned/DMA-BUF por detrás de nuestra abstracción; no asumir disponibilidad por detectar USB4STREAM. |
| E-045 | publicado | La llegada DMA no basta para reutilizar RX: el camino cualificado espera consumo/validación GPU, contabilidad de completions y repost de frames. Un experimento separado de retirement diferido mostró que el coste de completar/repostar afecta al rendimiento de modelo, aunque la latencia aislada del transporte sea menor. | `private transport reference@1c7f1ae89298c5be2ac4f916b589c77a10c903d8`, `docs/architecture.md`, `docs/performance.md`. | Extender ownership a consumer-release -> repost-pending -> reusable y medir completed GPU operation, no solo DMA arrival. |
| E-046 | publicado | El protocolo GPU visible usa metadata terminal con epoch/sequence/payload y un commit publicado al final con semántica release/acquire; además trata la región terminal como frontera de ownership para no inferir visibilidad global solo por observar el último write de payload. | `private transport reference@1c7f1ae89298c5be2ac4f916b589c77a10c903d8`, `docs/architecture.md`. | Incorporar un commit record propio con epoch/sequence/stripe y exigir validación antes de publicar un tensor recibido. |
| E-047 | publicado | La preparación del stream directo es transaccional y depende del control USB4NET sano, reconciliación de ambos peers, lease exclusivo, RX primado y cleanup exacto; un endpoint preparado unilateralmente no equivale a un par listo. La reutilización de rings puede además fallar por estado MSI-X pendiente si el stack no limpia/rearma correctamente. | `private transport reference@1c7f1ae89298c5be2ac4f916b589c77a10c903d8`, `docs/transports.md`, patch de stale MSI-X. | Modelar lifecycle/lease por enlace físico y añadir gate explícito de stop/cleanup/re-arm/first-completion. |
| E-048 | publicado + inferido | Las medidas públicas muestran que el beneficio del camino directo depende del tamaño y de la operación completa: el caso de ~64 KiB BF16 obtiene una gran reducción de latencia aislada, mientras que un payload multi-MiB obtiene una mejora mucho menor; una compresión I8 ayuda en T128 pero empeora T1 y un all-reduce comprimido distinto fue rechazado por exactitud. | `private transport reference@1c7f1ae89298c5be2ac4f916b589c77a10c903d8`, `docs/performance.md`. | No generalizar “directo” o “comprimido” por ancho de banda; cualificar por shape/operación y coste por token. |

| E-049 | publicado | AMD especifica 2 puertos USB4 nativos de 40 Gbps en Ryzen AI Max+ 395/Strix Halo. La especificación de producto no demuestra independencia de routers/upstream ni 80 Gbps agregados sostenidos. | AMD Ryzen AI Max+ 395 product specification, consultada 2026-10-01. | Diseñar dos enlaces físicos como recursos separados y exigir prueba simultánea antes de habilitar striping/agregación. |

| E-050 | observado (host) | Experimental wave top-k: 6,864 independent insertion/wave bit cases plus slot/admission mutants; WSL GNU 13.3 Release HIP-OFF build and 152/152 CTest cases passed. | Project source `src/hip_moe_route_launcher.hip`, `tests/router_wave_topk_tests.cpp`, implementation document; `cmake --build /mnt/d/tmp/build-spec-v1-router-host -j 8`; `ctest --test-dir /mnt/d/tmp/build-spec-v1-router-host --output-on-failure --no-tests=error -j 8`. | Supports portable order/admission contracts only. Actual revised-source HIP compile, ISA/FP-mode and old-kernel preservation, device fixtures, 554 arrays/512 IDs and paired timing remain independent gates. OFF by default. |
| E-051 | observado | El modo `--deferred-moe-status` tiene identidad MoE distinta, conserva dominio/bits del primer error y finaliza dentro del límite de publicación de la sesión. Las pruebas host inyectan errores en la capa 17 y en el fence, comprueban historial host intacto, generaciones/posición/ID sin avanzar y sesión envenenada. | GNU 13.3, WSL, CMake Release, HIP OFF: 152/152 CTest; `tests/deferred_moe_status_tests.cpp`; `tests/compiled_execution_tests.cpp`; contrato en `docs/deferred-moe-status.md`. | No implica ejecución GPU, igualdad de arrays de modelo ni ganancia de rendimiento. Compilación HIP, ISA, fixtures nativos y cualificación de modelo/rendimiento pendientes. |

## Regla para nuevas entradas

Cada evidencia nueva debe incluir identidad del artefacto, comando o prueba,
condiciones del entorno y el requisito de la spec al que afecta. Si una medida
contradice una entrada anterior, ambas se conservan y se registra la resolución
en `DECISIONS.md`.


## E-052 (2026-10-15): Q8 indexed-attention input prefetch source increment

State: experimental source and host/device fixture implementation; native gates
pending. Artifact: `docs/implementation-2026-10-15-qsa-input-prefetch.md` and
`tests/qsa_input_prefetch_tests.cpp`, `tests/hip_qsa_input_prefetch_tests.hip`.
Commands: configure Release with `BUILD_TESTING=ON`; host target
`strix-qsa-input-prefetch-tests`, optional gfx1151 HIP target
`strix-hip-qsa-input-prefetch-tests`. The baseline project contains POSIX-only
sources, so a full Windows build is not a portable qualification environment.
Actual Linux HIP compiler/ISA/device/model/performance results are not claimed.

The follow-up source adds exact-message negative assertions, actual mixed catalog
and full-operator ON/OFF eager/graph fixtures, poisoned output/workspace/KV/raw
owners, five-candidate and four-candidate node assertions, and zero-node failure
checks. A host submission counter records only eager enqueue and graph capture;
it does not count graph replay. The host plan test explicitly covers Q8 q/k/v
with BF16 indexer inputs (36 eligible and 24 fallback projections at 12 steps).
These are fixture contracts, not claims that new device tests have executed.
## E-053

Experimental source and fixtures: `src/lane_chain_hyper_output.cpp`, `src/hip_hyper_mix_launcher.hip`, host `strix-lane-chain-hyper-output-tests` and HIP `strix-hip-lane-chain-hyper-output-tests` / `strix-hip-lane-chain-hyper-integration-tests`. Reproduce host checks with `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DSTRIX_ENABLE_HIP_RUNTIME=OFF -DSTRIX_WARNINGS_AS_ERRORS=ON`, build and `ctest --test-dir build --output-on-failure`. Host execution cannot qualify device arithmetic. HIP gfx1151 compiler/ISA/device, full-model exactness and performance gates remain required; see the lane-chain implementation document. No measured gain is claimed.
E-051 follow-up artifacts: `merge_deferred_moe_first_error` is now the same
host/device helper called by the actual publisher and the portable dispatcher
fixture. Host sequence zero -> A8 bits 5 -> legacy bits 1 -> zero must retain
`{2,5}`. Exact-message compiled admission and host-only launch-scope metadata
fixtures exercise their specific guards. Native scope/alias fixtures require
untouched poison and zero captured nodes where an owned scope exists. A bounded
host callback holds the fixture stream to test asynchronous launcher return;
this fixture has not been executed as part of the portable evidence above.
Actual owner receipts report completed tokens, without synthetic layer counters.


E-051 follow-up observed checks: GNU 13.3/WSL, Release HIP OFF, rebuilt targeted
`strix-deferred-moe-status-tests` and `strix-compiled-execution-tests`: 2/2 PASS.
The actual `hip_launch_context_tests.cpp` metadata fixture also passes as a host
executable linked to the real core with private HIP type/constant declarations;
it invokes no HIP API and provides no runtime qualification. Native fixture C++
API parsing succeeds both with and without experimental A8 capability enabled.
E-051 native/compiler/ISA/model/performance gates: **pendiente**.
## E-054 (2026-10-15): shared-expert Q8 projection loading source increment

Artifacts: `docs/implementation-2026-10-15-shared-expert-prefetch.md`,
`tests/shared_expert_prefetch_tests.cpp`,
`tests/hip_shared_expert_prefetch_tests.hip`, and full-operator/catalog fixtures.
Commands: build and run `strix-shared-expert-prefetch-tests`; optional native
`strix-hip-shared-expert-prefetch-tests`, existing shared launcher and catalog
targets. Host planning tests pass in a private external MSVC portability harness
excluding the baseline POSIX stream TU and disabling warnings-as-errors because
of pre-existing unreachable-code warnings. This is a limited host gate. C++ API
syntax checks are not HIP compilation. Actual linked legacy/new ISA, device tests,
complete model arrays/tokens and workload performance remain pending. Requirements:
exact gate/up raw FP32 and down scalar-gate/store behavior, explicit fallback and
identity admission, poisoned eager/graph fixtures and host dispatch counters.


E-054 follow-up source: catalog fixtures now provide a real statistics owner and
check the final native launcher configuration retains the same owner; the catalog
constructor forwards it explicitly. Tensor-name JSON escaping has portable
control/quote/backslash/null fixtures. New device-only payload cases compare
legacy, virtual and candidate outputs for distinct NaNs/infinities and half NaN/
infinite scales, without treating host FMA as a payload oracle. These new native
cases still require actual execution and linked-ISA qualification.

E-054 payload follow-up (pending native execution): a separate finite-scale,
nonzero-code fixture uses feature-dependent quiet NaNs whose distinct payloads
survive BF16 truncation. Host checks distinguish every reduction-stride pair;
finite output poison avoids legitimate NaN payload collisions. The mixed
nonfinite case also tests a distinct NaN scalar gate during graph replays.
