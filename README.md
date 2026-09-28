# Motor Strix Halo propio — espacio de investigación

Este directorio separa nuestro trabajo de los repositorios de referencia. No
contiene código derivado de Halogen ni modifica GUFO.

## Documentos

- [`RESEARCH.md`](RESEARCH.md): hechos observados, comparación y límites de licencia.
- [`SPEC_DRAFT.md`](SPEC_DRAFT.md): especificación técnica inicial del motor propio.
- [`EVIDENCE.md`](EVIDENCE.md): índice de afirmaciones, evidencia y nivel de confianza.
- [`DECISIONS.md`](DECISIONS.md): decisiones aceptadas, pendientes y descartadas.
- [`OPUS_REVIEW.md`](OPUS_REVIEW.md): revisión ejecutada con Opus 5.5 y objeciones trazadas.
- [`OPUS_REVIEW_V2.md`](OPUS_REVIEW_V2.md): segunda revisión adversarial tras Qwen/GTT/pwilkin.
- [`SOURCES_LEDGER.md`](SOURCES_LEDGER.md): fuentes canónicas, licencias y bloqueos de procedencia.
- [`PWILKIN_REVIEW.md`](PWILKIN_REVIEW.md): ideas, resultados negativos y protocolo extraídos de pwilkin.
- [`MEMORY_BUDGET.md`](MEMORY_BUDGET.md): política propuesta de UMA/GTT y guardas de admisión.
- [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md): código disponible, reproducción y bloqueos.
- [`tools/`](tools): analizadores que conservan solo metadatos y recuentos agregados.

## Referencias locales

- `../halogen-server`: despliegue de Halogen Qwen3.8-27B.
- `../halogen-flash-server`: despliegue de Halogen Qwen3.8-Flash-Next.
- `../gufo`: motor GUFO, consultado como referencia MIT.
- `../pwilkin-strix-halo`: instalador/sitio MIT de pwilkin, fijado solo como referencia.
- `../halogen-analysis`: binario y objetos GPU extraídos de la imagen oficial.

## Regla de observación no derivativa

La observación de Halogen se limita a comportamiento, metadatos de kernel,
mezcla agregada de instrucciones y medidas. No se copiará, reconstruirá ni
redistribuirá código fuente o kernels derivados. Las implementaciones nuevas se
escribirán desde contratos matemáticos públicos y se verificarán contra oráculos
independientes. En este alcance no se reutilizará código de GUFO: se estudia
como referencia y el diseño presupone implementación propia. Cualquier cambio
futuro de esa política exigiría una decisión explícita y conservación de avisos.

## Método de actualización

Cada nueva observación se guarda primero en `EVIDENCE.md`. Si cambia el diseño,
se refleja en `SPEC_DRAFT.md` y se registra el motivo en `DECISIONS.md`. Las
hipótesis nunca se convierten en requisitos hasta que exista una medición o una
decisión explícita. Este orden permitirá entregar a Opus una spec trazable y
contrastar cada objeción contra evidencia concreta.

## Primer build offline

El slice A0.1 ya contiene un núcleo C++20 portable y el ejecutable
`strix-memory-plan`. Véase `IMPLEMENTATION_STATUS.md` para compilarlo en Linux o
WSL y obtener un desglose JSON consumible por la futura WebUI.
