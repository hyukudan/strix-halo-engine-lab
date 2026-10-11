# Especificación técnica v0.4 — motor propio para Strix Halo

Estado: borrador de investigación revisado para aceptación técnica y ampliado con fuentes
oficiales Qwen, política GTT y revisión external platform study. Persisten puertas P0 y la
medición en hardware. Nombre del proyecto aún no decidido.

Trazabilidad: los datos que sustentan esta especificación están indexados en
[`EVIDENCE.md`](EVIDENCE.md); los cambios de dirección se registran en
[`DECISIONS.md`](DECISIONS.md). Una afirmación marcada como hipótesis no debe
interpretarse como requisito cerrado.

## 1. Objetivo

Construir un motor de inferencia local, reproducible y auditable para AMD Strix
Halo `gfx1151`, optimizado primero para Qwen3.8-Flash-Next. Debe combinar alto
prefill, decode competitivo, concurrencia real y calidad verificable. No será un
fork de external runtime B ni una reconstrucción de external runtime A.

### Metas iniciales

- Linux x86-64 nativo, ROCm/HIP, `gfx1151`.
- C++20 + HIP; Python solo para herramientas y oráculos.
- GGUF como formato de entrada; caché interna opcional, lossless y versionada.
- Primer slice: AR greedy/sampled + MTP solo greedy, Chat Completions en gateway
  separado y contrato de replay explícito. MTP sampled espera contrato RNG/replay canónico; el algoritmo probabilístico de aceptación ya está fijado.
- C1 como fast path y C2/C4 como primeras rutas concurrentes.
- Contexto inicial cualificado según el presupuesto UMA demostrado; 262.144,
  Responses API, prompt lookup y C6/C8 quedan como ampliaciones posteriores.

### No objetivos de la primera versión

- portabilidad a otras GPU;
- múltiples familias de modelo;
- visión antes de que texto/MTP esté cualificado;
- aproximaciones numéricas sin una puerta de calidad explícita;
- compatibilidad con cada cuantización existente.

## 2. Principios

1. **Código propio y procedencia auditable.** Contratos desde papers, fuentes
   públicas canónicas y oráculos independientes. external runtime A solo aporta
   observaciones agregadas anonimizadas; no se copian cuerpos, firmas ni
   estructuras de kernels.
2. **Especialización finita y justificada.** Cada variante debe corresponder a
   formas observadas y superar un control end-to-end.
3. **Corrección antes de velocidad.** Ningún cambio entra por renovar goldens o
   relajar tolerancias.
4. **Reproducibilidad.** Compilador, ROCm, kernel Linux, BIOS/IOMMU, potencia,
   commit, pesos, prompt y caché forman parte del resultado.
5. **Una implementación ganadora.** Experimentos perdedores se documentan y se
   eliminan; no se acumulan flags permanentes.
6. **Fuente primaria antes que referencia.** QSA, GDN, Gated Residual, MoE y
   n-gram se especifican desde Qwen oficial; external runtime A, external runtime B y external platform study sirven para
   formular y contrastar hipótesis, nunca como único oráculo.

## 3. Arquitectura

```text
Red/VPN
  -> TLS reverse proxy
    -> inference gateway (Chat API + playground opcional)
      -> UDS/loopback -> motor
        -> admission + scheduler
          -> session state (KV, recurrence, RNG, snapshots, rollback)
            -> model executor -> kernel registry -> HIP/hipBLASLt/HIP graphs
    -> admin API (credencial más fuerte, sin shell)
      -> operaciones tipadas -> supervisor/model manager/benchmark runner

Motor y supervisor -> métricas/logs de solo lectura -> panel administrativo
```

### 3.1 Loader y pesos

- lector GGUF propio y mínimo;
- validación estricta de arquitectura, dimensiones, dtype y hash;
- vista mmap read-only de los shards para carga CPU; antes de lanzar GPU, todo
  rango usado por el hot path debe estar registrado y residente conforme al
  contrato confirmado de la plataforma;
- planner lossless que decide por tensor: lectura directa, repack temporal o
  caché interna persistente;
- subsistema PLE/n-gram con lectura directa asíncrona candidata (`pread` o
  equivalente), pool acotado, coalescing, caché de filas y telemetría; `mmap`
  demand-paged será un control, no la opción asumida;
- caché identificada por hash de pesos + versión de layout + compilador;
- ninguna requantización implícita;
- comando `inspect` que publique bytes, bpw, familias, layouts y hashes.

### 3.2 Memoria UMA

- allocator con categorías visibles: pesos mapeados, pesos copiados, KV,
  recurrencia, rollback, scratch, librerías y caché;
- prefault CPU paralelo y acotado para rangos activos, fuera del kernel GPU;
- pruebas A/B de mmap registrado frente a copia anónima con páginas grandes;
- pool de sesiones dimensionado en posiciones, no solo en slots;
- snapshots solo de filas válidas;
- backpressure antes de comprometer memoria que no cabe; el tope parte de
  `MemTotal` medido menos baseline sin motor y reserva, y la contabilidad interna
  manda sobre una `MemAvailable` que puede sobreestimar páginas fijadas;
- métricas de GTT, MemAvailable corregida y page faults durante preparación;
- presupuesto previo a cargar: pesos y mappings, tablas, KV, recurrencia,
  rollback, scratch, librerías, SO y margen de recuperación. Un checkpoint que
  nominalmente quepa no se admite si incumple el margen.
- configurar GTT como límite de mapeo cercano al máximo útil (objetivo inicial
  124 GiB en una máquina nominal de 128 GB), calculado desde la RAM real;
- mantener separada una reserva efectiva inicial de 16–20 GiB en el admission
  controller: ampliar GTT no autoriza a consumir esa RAM;
- BIOS con carve-out mínimo verificado; `amd_iommu=off` y otros parámetros
  experimentales nunca son defaults y requieren perfil dedicado, aviso y
  rollback;
- shutdown limpio y watchdog de GTT residual: ante terminación sucia o uso
  incoherente se bloquea la recarga y se escala a reinicio controlado.

### 3.3 Ejecutor

- caminos separados para prefill, AR decode, verificación MTP y catch-up;
- grafo de operadores estático por modelo, sin runtime general en el hot path;
- HIP graph por forma estable de decode;
- invariante: ningún replay observa un input reutilizado; ring versionado por
  slot/generación es un mecanismo candidato, no el contrato;
- PM4 retenido solo como experimento opt-in para decode repetitivo, negociado
  por capabilities y con fallback al runtime estándar; no condiciona prefill;
- modo eager para warm-up, tuning y formas ragged;
- solapamiento únicamente con streams y ownership demostrablemente seguros;
- cada request conserva KV, estado recurrente, RNG y rollback independientes.

### 3.4 Kernel registry

Cada entrada declarará:

- operador, forma/rango y fase;
- dtype/layout de entrada y salida;
- wave32/wave64, workgroup, VGPR, SGPR, LDS y scratch;
- orden de reducción y fronteras de redondeo que debe conservar;
- flags de compilador, contracción FMA, modo de denormales y versión ABI;
- evidencia de corrección;
- coste medido por batch, profundidad y estado de caché;
- identidad del toolchain que produjo el código.

El selector lee un plan generado fuera de línea. Si la identidad no coincide,
usa una ruta conservadora y solicita retuning; nunca cronometra algoritmos en
producción de forma no determinista.

### 3.5 Familias de kernel prioritarias

1. RMSNorm/residual/RoPE y conversiones.
2. Proyecciones Q4/Q5/Q8 para C1 y batches 2–8.
3. MoE: routing, agrupación de expertos, gate/up fusionado y down.
4. Gated DeltaNet: fijar primero FP32/estado/orden/rollback; tiled, estado en
   registros y DPP/LDS son candidatos posteriores al perfil propio.
5. QSA: oráculo sparse canónico pendiente de confirmar desde Qwen; denso será
   baseline de rendimiento y calidad, no se llamará exacto ni equivalente.
   Indexer, top-k con empates, metadata y atención se evalúan como pipeline.
6. Sampling/argmax y verificación MTP.

Para cada familia se probarán wave32 y wave64 donde tengan sentido; el dato de
external runtime A no se convertirá en una regla universal.

## 4. Scheduling y especulación

- cola FIFO con límites explícitos de memoria y tokens;
- C1 no paga estructuras de batching que no usa;
- C2/C4 agrupa filas compatibles en el primer slice, manteniendo estado privado;
- batching del transformer completo, incluyendo predictor y verificación;
- MTP de 1–7 borradores con control basado en coste de ciclo completo;
- control temporal solo en greedy; sampled usa política fija para replay;
- rollback exacto para cada prefijo aceptado;
- prompt lookup se incorporará después del primer slice y se medirá separado;
- cancelación antes de tocar estado compartido y snapshots consistentes.

## 5. Calidad

Tres niveles obligatorios:

1. **Oráculo de operador:** fórmula CPU FP64/FP32 independiente, incluidos tails
   1/8/9/32/33 y valores pequeños no saturados.
2. **Replay de modelo:** primer layer/logit divergente con historial de tokens
   idéntico; AR, MTP, sampling, EOS, continuación y restauración.
3. **Calidad de tarea:** PPL/KL/top-1, NIAH y corpus de agentes/código.

Una optimización debe preservar el contrato declarado. Si el contrato permite
tolerancia, la tolerancia se fija antes de medir rendimiento.

El contrato distingue replay C1 y concurrente. “Determinista” nunca será una
etiqueta global: cada modo declara si exige identidad bitwise, identidad de
tokens o tolerancia numérica, incluyendo el orden de reducción.

## 6. Benchmark

Matriz mínima:

- prefill: 2K, 8K, 32K, 128K;
- decode: profundidad 0/4K/16K/32K/64K/128K;
- tg128, C1/C2/C4 inicialmente; C6/C8 después de cualificar C4;
- AR, MTP en texto mixto y repetitivo, prompt lookup;
- arranque frío/caliente, memoria pico y tiempo a readiness;
- baseline, candidato y control no afectado en el mismo entorno.
- mínimo tres repeticiones, dispersión publicada, lock exclusivo de GPU y
  guardia de memoria; un cambio conceptual por experimento siempre que sea
  posible.

Guardar latencias individuales, throughput agregado, aceptación, tokens
completados, launches, GPU busy, page faults y bytes de memoria. Un gain de
microbenchmark no cuenta hasta sobrevivir al request completo sin profiler.

## 7. Fases

### Fase A0 — laboratorio completamente offline

- imagen de desarrollo pública fijada por digest y segundo toolchain de control;
- compilación `gfx1151`, separación del HSACO y auditoría de ELF/metadata/ISA;
- inventario de code objects `gfx1151` en rocBLAS, hipBLASLt y dependencias;
- doble compilación con hashes y diff normalizado;
- gates CI de VGPR, SGPR, LDS, spills y tamaño de código;
- oráculos CPU y modelos/tensores diminutos sin dependencia del hardware final.

### Fase A — laboratorio en Strix Halo

- inventario de hardware/software;
- confirmación de ejecución de los objetos producidos en A0;
- medidor de ancho de banda y techo WMMA;
- harness común para external runtime A, external runtime B y nuestro binario;
- corpus y pesos fijados por hash.

### Fase B — referencia correcta

- GGUF, tokenizer y CPU/oráculo mínimo;
- ejecución AR correcta aunque sea lenta;
- pruebas de estado, snapshot y sampling.

### Fase C — texto C1

- kernels nativos de proyección, MoE, DeltaNet y QSA;
- contratos/oráculos CPU de GDN, QSA sparse, n-gram/PLE, hyper-connections y routing;
- tras AR correcto y perfil propio: prototipos GDN tiled y lector directo PLE;
- ablation QSA denso frente a sparse completo, incluida calidad; no se acepta
  integración parcial;
- prefill y decode separados;
- HIP graphs y tuning offline.

### Fase D — MTP y concurrencia inicial

- predictor, aceptación, rollback y catch-up;
- batching C2/C4 y admission por memoria;
- expansión posterior a C6/C8 y prompt cache solo tras cualificar C4.

### Fase E — servidor y producto

- APIs, health/metrics, inspect/verify/ppl/niah;
- empaquetado reproducible y documentación;
- hardening de cancelación, fallos de driver y presión de memoria.

El servidor incluye watchdog, drenaje de requests, política explícita ante
cuelgue/reset del driver y estado degradado visible. Cada benchmark registra
potencia/TDP, clocks y temperatura junto al entorno software.

## 8. Puertas de decisión antes de escribir el motor completo

1. Elegir checkpoint público y cuantización canónica tras demostrar un
   presupuesto completo dentro de 128 GB UMA.
2. Medir external runtime A y external runtime B con el mismo prompt, contexto, potencia y criterio de
   tokens completados.
3. Confirmar Linux/ROCm/kernel de la máquina objetivo.
4. Decidir si el caché interno de pesos compensa tiempo de arranque y espacio.
5. Fijar licencia y política de reutilización MIT.
6. Resolver los P0 de `DESIGN_ACCEPTANCE.md` y someter la spec consolidada a una
   segunda revisión.
7. Fijar repo, commit y licencia canónicos para QSA, DeltaNet, routing MoE,
   MTP, tokenizer y chat template.
8. Demostrar que las librerías ROCm elegidas contienen rutas `gfx1151` o definir
   fallbacks propios.
9. Aprobar el caso de uso y distribución frente a Qwen Community License 1.0.
10. Confirmar la semántica QSA y su oráculo; no sustituirla por denso por una
    observación de rendimiento externa.

## 8.1 Trabajo permitido antes de recibir el hardware

- compilar HIP offline para `gfx1151` con toolchain fijado;
- inspeccionar ISA, VGPR/SGPR/LDS, spills y tamaño de código;
- implementar cargador GGUF, hashes y planner de layouts;
- crear oráculos CPU y tests de operadores/estado;
- simular admission, batching, cancelación, snapshots y rollback;
- definir API, métricas, formatos de artefacto y plano administrativo;
- construir microkernels candidatos sin declarar ganadores.

Queda prohibido convertir una compilación correcta o un recuento estático en
una afirmación de rendimiento. Selección de kernels, tuning, UMA, page faults,
potencia, temperatura y estabilidad requieren el Strix Halo real.

## 8.2 Plano administrativo remoto

El producto tendrá tres dominios separados: motor local sin exposición de red,
gateway de inferencia/chat y plano administrativo. Sus funciones prioritarias
serán salud/versiones, configuración
validada, ciclo de vida de servicios, logs, memoria/GTT, temperatura/potencia,
modelos, caché/sesiones y ejecución/comparación de benchmarks. Un playground de
chat podrá vivir como pestaña opcional dentro del panel, únicamente para smoke
tests y reproducción de peticiones.

El puerto interno del motor nunca se publica: se usa UDS o loopback restringido.
El gateway usa API keys u OIDC, cuotas y auditoría. El panel usa un agente local
con operaciones tipadas permitidas explícitamente, WebAuthn o mTLS, TLS,
auditoría, CSP y protección CSRF. La entrada remota se limita preferentemente a
VPN. Las acciones destructivas o reinicios muestran el objetivo y requieren
confirmación. La pestaña de chat no hereda privilegios administrativos y la UI
no ejecuta shell arbitrario.

La sección GTT mostrará límite configurado frente a consumo y reserva efectiva,
carve-out detectado, `MemAvailable` y estado residual. Podrá preparar/aplicar un
perfil allowlisted de GTT/TTM mediante arranque one-shot, preflight,
reautenticación y confirmación. Un health check confirma driver/GTT/agente; si
falla, el siguiente arranque vuelve a la entrada anterior. `amd_iommu=off` y
parámetros experimentales quedan fuera de la WebUI y requieren consola o SSH.
El playground se sirve desde origen y cookies distintos al panel administrativo,
con salida del modelo estrictamente sanitizada.

## 9. Historial de la especificación

| Versión | Fecha | Cambio |
| --- | --- | --- |
| 0.4 | 2026-09-28 | Fuentes Qwen oficiales, política UMA/GTT y aprendizajes de external platform study: GDN tiled, PLE directo, UMA ring y controles negativos de QSA/PM4. |
| 0.3 | 2026-09-28 | Incorpora revisión de diseño: P0, alcance reducido, contratos numéricos, A0 y separación de privilegios. |
| 0.2 | 2026-09-28 | Cross-compilación previa al hardware y plano administrativo remoto. |
| 0.1 | 2026-09-28 | Alcance inicial, arquitectura, calidad, benchmark y fases. |
