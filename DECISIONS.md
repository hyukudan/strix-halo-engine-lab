# Registro de decisiones

## Aceptadas

| ID | Decisión | Motivo | Evidencia | Revisión |
| --- | --- | --- | --- | --- |
| D-001 | Construir un motor propio; external runtime A y external runtime B son referencias. | Evita depender de una base cerrada o convertir external runtime B en destino de cambios. | Alcance confirmado por el usuario. | Reabrir solo por decisión explícita. |
| D-002 | Mantener observación no derivativa respecto a external runtime A. | Obtener ideas generales sin copiar, reconstruir ni redistribuir implementaciones. | Licencia external runtime A y alcance confirmado. | Auditar artefactos y procedencia antes de publicar código. |
| D-003 | Empezar por Linux nativo, ROCm y `gfx1151`. | Es el entorno real de los dos motores y permite optimización específica. | E-001, documentación de ambos proyectos. | Revisar después de una primera versión estable. |
| D-004 | Usar GGUF como entrada inicial. | Formato público, pesos disponibles y menor dependencia de formatos privados. | E-009. | Medir después un caché interno lossless. |
| D-005 | Calidad y reproducibilidad son puertas, no tareas posteriores. | MTP, recurrencia y cuantización pueden producir ganancias engañosas. | E-008, E-010. | Permanente. |
| D-006 | La UI prioritaria será un plano administrativo remoto. | Operación, observabilidad y benchmarks aportan más valor que duplicar clientes de chat existentes. | E-016 y alcance confirmado por el usuario. | Chat queda como playground opcional integrado. |
| D-007 | Avanzar con compilación offline antes del hardware. | Permite validar toolchain, objetos y contratos sin fingir resultados dinámicos. | E-015. | Revalidar todo en el equipo real. |
| D-008 | El primer slice será AR + MTP, C1/C2/C4 y un contexto cualificado menor que 262K. | Reduce simultáneamente riesgo numérico, memoria y scheduling. | E-019, O-11. | Ampliar solo al superar corrección, memoria y estabilidad. |
| D-009 | Separar motor, gateway/chat y administración como dominios de privilegio. | Una UI cómoda no debe ganar control directo del proceso GPU ni shell. | E-016, O-14, O-15. | Permanente salvo modelo de amenazas revisado. |
| D-010 | Tratar repositorio, paper y checkpoint oficiales de Qwen como fuente primaria de arquitectura. | Permite especificar contratos sin inferirlos de motores ajenos. | E-020, E-021. | Reabrir si Qwen publica revisión incompatible; fijar siempre SHA. |
| D-011 | Exponer GTT casi máximo, pero admitir cargas con una reserva efectiva independiente de 16–20 GiB. | GTT amplía el mapping; no crea RAM ni protege al host de OOM. | E-022, E-023, `MEMORY_BUDGET.md`. | Ajustar reserva solo con telemetría del hardware real. |
| D-012 | Mantener inicialmente el perfil mononodo de 128 GiB dentro de un envelope aproximado de 100 GiB de pesos residentes, hasta que A0.6 justifique otro límite. | Preserva la reserva efectiva y deja capacidad para KV, recurrencia, scratch, runtime y recuperación; evita convertir el GTT máximo en presupuesto de carga. | E-011, E-028, E-032, E-033. | Reabrir tras medidas UMA reales y calidad de las cuantizaciones candidatas. |

| D-013 | Fijar el desempate QSA por menor índice de bloque cuando los scores sean idénticos. | El forward canónico usa `torch.topk`, cuyos índices empatados no tienen estabilidad garantizada; el motor necesita replay reproducible y la misma regla debe usarse en serial y MTP. | E-041, E-043 y fixture `qsa_tie`. | Reabrir solo si Qwen publica un desempate canónico explícito incompatible. |

| D-014 | Modelar cada USB4 físico como transporte autónomo con epoch, lease, TX/RX pools y contabilidad propios; el modo dual-link es una política por encima de dos enlaces cualificados. | Evita asumir suma de ancho de banda, aísla fallos y permite flow distribution/directional split/striping solo con evidencia de independencia. | E-044 a E-048; docs 12 y 16. | Reabrir solo si el hardware demuestra un recurso agregado con ownership/sincronización comunes y beneficio end-to-end. |

| D-015 | Keep wave router selection OFF by default, with unchanged softmax/slot-order FP expressions, immutable preflight manifest admission and launcher-only K32; retain default/ranked alternatives. | Pointer identity to a validated owned deep copy avoids hot-path tensor scans and unsafe caches on mutable declared IDs. Raw GGUF rejects groups/bias before shape reduction; registration cannot inspect metadata absent from its API. | E-050; `docs/implementation-2026-10-15-wave-router-topk.md`; host oracle/admission checks. | Reopen for expressible grouped routing/bias, ISA changes, full-array mismatch or failure to improve measured latency. Target and model gates remain pending. |

## Pendientes

| ID | Decisión pendiente | Datos necesarios |
| --- | --- | --- |
| P-001 | Nombre y licencia del proyecto. | Preferencias del propietario y política de contribuciones. |
| P-002 | Checkpoint/quantización canónica de desarrollo. | Tamaño, calidad, MTP disponible y licencia de pesos. |
| P-003 | mmap registrado frente a copia anónima/huge pages. | Medidas de carga, decode, memoria y presión de página. |
| P-004 | Política exacta de continuous batching. | Resultados C1/C2/C4/C6/C8 y coste de admisión. |
| P-005 | Wave32/wave64 por familia. | Experimentos emparejados por forma y end-to-end. |
| P-006 | Caché interna de pesos. | Prototipo lossless, tamaño y tiempo de arranque. |
| P-007 | Stack del plano administrativo. | Prototipo, modelo de amenazas y preferencias de despliegue. |
| P-008 (P0) | Checkpoint canónico y presupuesto completo dentro de 128 GB UMA. | Tamaño real por tensor más SO, KV, recurrencia, rollback, scratch, librerías y margen. |
| P-009 (P0) | Modelo de residencia/registro de memoria en `gfx1151`. | Documentación y prueba con kernel Linux/ROCm exactos; confirmar XNACK/HMM. |
| P-010 (P0) | Ledger de fuentes canónicas de la arquitectura. | Repo, commit, licencia y contrato de QSA, DeltaNet, MoE, MTP, tokenizer y template. |
| P-011 (P0) | Cobertura `gfx1151` de las librerías ROCm elegidas. | Inventario offline de code objects de rocBLAS/hipBLASLt y fallbacks. |
| P-012 (P0) | Política compatible con Qwen Community License 1.0. | Caso de uso, distribución, servicio comercial y revisión legal si aplica. |
| P-013 | Valores finales de GTT, reserva y recuperación tras dirty kill. | RAM física, BIOS, kernel/ROCm final y pruebas de presión/reinicio. |
| P-014 (P0) | Semántica canónica QSA: selección top-k, desempate, indexer y relación con atención densa. | Paper/código oficial, oráculo CPU y pruebas KL/NIAH; no asumir que denso es equivalente. |
| P-015 | Backend directo GPU-owned DMA-BUF y política dual-link preferida. | Kernel/ROCm exactos en M5, soporte export/import DMA-BUF, consumer-credit/repost, re-arm fiable, medidas 1×USB4/2×USB4 y efecto real sobre decode/prefill. |
| H-001 | GDN tiled/DPP como primer candidato de optimización tras perfil. | Contrato/oráculo GDN y perfil propio de tráfico/tiempo. |
| H-002 | Lector directo PLE antes que mmap demand-paged. | Contrato n-gram/PLE, latencia p99, page cache y A/B en hardware. |

## Descartadas por ahora

| ID | Alternativa | Motivo |
| --- | --- | --- |
| X-001 | Modificar external runtime B como producto final. | No coincide con la meta de una solución propia. |
| X-002 | Copiar o reconstruir kernels de external runtime A. | Fuera del alcance clean-room y contrario al objetivo de código propio. |
| X-003 | Diseñar desde el inicio para múltiples GPU/modelos. | Añadiría abstracción antes de validar el fast path `gfx1151`. |

## Formato de revisión

Toda decisión nueva indicará: alternativas, evidencia, riesgo, consecuencia en
la spec y condición de reapertura. Las acciones de aceptación se enlazarán aquí y
en `DESIGN_ACCEPTANCE.md`.


## D-017 (2026-10-15): explicit indexed-attention Q8 input prefetch experiment

Keep the prefetch schedule off by default and in the original HIP translation
unit. Manifest eligibility is per matrix; operator bindings explicitly retain
legacy IDs for wholly ineligible layers. Receipts distinguish operator-step and
projection counts. Do not infer target qualification from an arithmetic argument
or a host test. Reopen adoption after linked legacy/candidate ISA, device fixtures,
full-array and token fidelity, and paired complete-workload measurements. The first
increment rejects speculative verify because the actual indexed launcher is C1.

Alternatives: retain only the original T256 schedule, or choose a distinct opt-in
loading schedule with the same leaves and reduction. The latter permits isolated
qualification without replacing the existing path. Risk: source equivalence does
not establish compiler behavior or complete-workload benefit. Spec consequence:
require explicit per-tensor eligibility, topology/dispatch receipts and all native
gates before adoption. Evidence: E-052. Reopening conditions are the gates above.
## D-HC-LANE-CHAIN-20261015

Preserve each original HC gate chain, coefficient rounding, sigmoid and stream order while assigning independent chains to lanes. Keep normalization/projection/injection kernels untouched. Stage original Q8 bytes without repacking, use explicit metadata-selected base fallback, and bound dynamic LDS to64KiB. Rank320 bank permutation is documented narrowly; other admitted ranks are not presumed conflict free. Reject speculative selection for this serial decode slice. Alternatives that split the reduction or alter rounding are outside this exact optimization. Reopen after actual ISA/resource checks and bitwise complete-model evidence, or reject the candidate if any arithmetic mismatch appears.
