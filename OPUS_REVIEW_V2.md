# Segunda revisión externa con Claude Opus 5.5

Fecha: 2026-09-28. Modelo solicitado: `claude-opus-5-5`, esfuerzo `high`,
permisos de solo lectura. Documentos revisados: spec v0.4, memoria, pwilkin,
evidencias, decisiones, ledger, config oficial y laboratorio cross-compile.

## Veredicto

No congelar todavía. Se consideran bien resueltos: aritmética KV base, separación
GTT/admisión, no-faults GPU, PM4 solo experimental de decode, fuentes Qwen
primarias, separación de privilegios y honestidad de A0. Persisten los P0 de
presupuesto, residency, contratos completos, librerías `gfx1151` y licencia.

## Objeciones incorporadas

| ID | Severidad | Objeción | Resolución |
| --- | --- | --- | --- |
| O2-01 | P0 | Denso no puede llamarse “exacto” hasta saber si QSA top-k es semántica del forward entrenado. | P-014; oráculo sparse pendiente. Denso será baseline de rendimiento/calidad. |
| O2-02 | Alta | Faltaban KV de MTP, indexer, estado GDN FP32 y rollback. | Fórmulas provisionales añadidas a `MEMORY_BUDGET.md`; medir/revisar contrato. |
| O2-03 | Alta | “Reserva 16–20 GiB” no definía la fórmula de admisión. | Tope desde MemTotal/baseline y contabilidad interna; definir MemAvailable corregida. |
| O2-04 | Alta | Faltan escenarios PLE residente/directa y límites de page/BF16 cache. | Convertidos en requisito del planner de memoria. |
| O2-05 | P0 | Residency sigue sin matriz experimental concreta. | Añadida prueba `hipMalloc`/host register/mmap/managed para Fase A. |
| O2-06 | Alta | D-012 convertía ideas de pwilkin en decisión antes de perfilar. | Movida a hipótesis pendientes; el contrato prima sobre mecanismo DPP/ring. |
| O2-07 | Alta | Hay que fijar contratos/oráculos antes de optimizar GDN/PLE/QSA. | Orden de fases corregido. |
| O2-08 | P0 | Ledger omitía n-gram/PLE, hyper-connections, output gate y mRoPE. | Filas añadidas; contratos siguen abiertos. |
| O2-09 | Alta | MTP sampled carece aún de algoritmo canónico/RNG. | Primer slice MTP solo greedy; sampled se aplaza. |
| O2-10/11 | Alta | Un cambio remoto de boot puede impedir el propio rollback; IOMMU no debe tocarse desde web. | Solo GTT/TTM allowlisted, arranque one-shot y health-confirm; IOMMU solo consola/SSH. |
| O2-12 | Media-alta | Chat y admin en el mismo origen crean riesgo XSS por salida del modelo. | Playground en origen/cookies separados del panel admin. |
| O2-13/15 | Media | Hay que leer parámetros efectivos y distinguir GTT residual de uso legítimo. | Telemetría y criterio por procesos con `/dev/kfd`/render abiertos. |
| O2-14 | Media | El inventario de librerías `gfx1151` puede hacerse ya. | P-011 sigue siendo siguiente trabajo offline prioritario. |
| O2-16 | Baja | Evidencia externa de una sola fuente no debe sostener una decisión aceptada. | Estado rebajado a hipótesis hasta reproducción propia. |

El texto completo de la respuesta quedó capturado en la salida de la sesión CLI;
este acta conserva todas las objeciones accionables sin tratar la opinión de un
segundo modelo como evidencia experimental.
