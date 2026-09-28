# Investigación inicial: Halogen, GUFO, pwilkin y motor propio

Fecha de captura: 2026-09-28. Este documento distingue hechos observados de
hipótesis que todavía necesitan medición en una máquina Strix Halo con Linux
nativo.

## Material fijado

| Referencia | Identidad |
| --- | --- |
| Halogen Flash repo | `cf0d9cc5a3d585c0eb26edd421ff21a01be5e91c` (`0.14.2`) |
| Halogen clásico repo | `5a0f952c9e07794a0de0a617a377bae6a63ca56a` (`0.1.4`) |
| GUFO repo | `6cd5fa8de360cdb8f848efe2bd241cbe22cc81b0` |
| pwilkin/strix-halo | `f73872fe20dfdef460653f18e2fe63e5366ae958` |
| pwilkin/llama.cpp `strix-halo` | `b0f31f5876ef3856b55f5bb88072cc96e5effafe` |
| pwilkin/rocm-systems `ilintar-experiments` | `7dda3ac6cfe6bbe0b7f08c23a67cfa118d8641a1` |
| Qwen Flash-Next oficial | repo/paper y checkpoint `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| Imagen Halogen Flash | `ghcr.io/peonist-ai/halogen-flash-server:0.14.2` |
| Digest OCI | `sha256:f3f99aa48f3a051f18da9ee24b333ca108fe745773fd036a365fe1875871d0be` |
| Binario `flash_serve` | 15,023,528 bytes; SHA-256 `e8f74704f8148d9512cd33cc6abb46de21daac5385cb686285fc5ecbbf9cf70d` |

No se descargaron pesos de modelo. Halogen Flash indica aproximadamente 118
GiB para su checkpoint; esa descarga debe hacerse solo cuando exista una
máquina Linux/gfx1151 destinada a medir.

## Qué se pudo observar de Halogen

La licencia de Halogen permite expresamente la ingeniería inversa y reconoce
que el binario contiene código de dispositivo `gfx1151` legible con un
desensamblador. También pide no redistribuir kernels o fuente derivados como
obra propia. Nuestro alcance es más conservador: extraer patrones y métricas,
no reconstruir implementaciones.

El ELF x86-64 contiene una sección `.hip_fatbin` de 13,052,576 bytes y tres
objetos HIP para `gfx1151`:

| Objeto | Tamaño | Kernels | Lectura probable por nombres/metadatos |
| --- | ---: | ---: | --- |
| 0 | 7,640,960 B | 1,057 | cuantización, proyecciones, MoE, visión y utilidades |
| 1 | 3,418,224 B | 172 | Gated DeltaNet/scan recurrente |
| 2 | 1,976,992 B | 62 | atención QSA/Flash y selección |

Total: 1.291 entradas de kernel. En los metadatos de esta build, todas declaran
wavefront 32. No debe inferirse que wave64 sea siempre peor: GUFO conserva
rutas wave64 seleccionadas cuando la medida completa las justifica.

### Patrones estructurales útiles

1. **Especialización por forma y fase.** Hay muchas instancias template para
   tamaños, número de filas, profundidad y variantes de precisión. La idea
   aprovechable es compilar un conjunto finito basado en el histograma real de
   formas, no copiar esas instancias.
2. **Separación por subsistema.** Quant/MoE, DeltaNet y atención viven en
   objetos distintos. Esto facilita medir, versionar y reemplazar familias sin
   recompilar o invalidar todo el motor.
3. **MoE Q4 dedicado.** La selección `q4moe` contiene ocho kernels principales
   y dos helpers. En conjunto combinan WMMA BF16/F16, permutaciones/packed
   arithmetic, LDS y accesos globales. La mezcla agregada suma 101.490
   instrucciones estáticas; es una descripción del binario, no tiempo
   ejecutado.
4. **DeltaNet fusionado.** Dieciocho variantes `dnc_scan3x` combinan WMMA F16,
   acumulación FMA, carga LDS intensiva y estado recurrente. La hipótesis a
   probar es que fusionar preparación, scan y escritura de estado ahorra más
   que maximizar ocupación de cada microkernel aislado.
5. **Atención muy especializada.** Veintiséis variantes `attn_fa2` contienen
   rutas distintas para bloques/selección. El ISA agregado muestra softmax FP32
   (`exp`, `ldexp`, FMA) y bastante control de flujo. La precisión de reducción
   debe ser contrato explícito antes de optimizar.
6. **Presupuesto de recursos agresivo.** Algunas variantes llegan a 256 VGPR,
   ~64 KiB de LDS y spills. Esto demuestra que la ocupación máxima no es el
   objetivo por sí sola; hace falta medir tiempo completo y tráfico de memoria.
7. **Plan de GEMM fuera de línea.** La imagen incluye `flash-tune.plan` y una
   selección reproducible por forma. Nuestro motor necesita un manifiesto de
   tuning versionado por GPU, compilador, modelo y precisión.
8. **Herramientas de checkpoint integradas.** `verify`, `inspect`, `ppl` y
   `niah` convierten formato, calidad y regresión en capacidades del producto,
   no scripts de laboratorio desconectados.

Los recuentos de ISA son agregados estáticos. No indican frecuencia de
ejecución ni coste dinámico; deben cruzarse con trazas y tiempos en hardware.

## Qué aporta GUFO como referencia abierta

GUFO publica código MIT C++20/HIP, CMake/Nix, cargador GGUF, servidor,
planificador, sesiones, snapshots, pruebas de operadores y rutas específicas de
Qwen3.8-Flash-Next. Sus kernels cuantizados adaptados de llama.cpp conservan el
aviso MIT correspondiente; no todo el árbol tiene la misma procedencia.

Las ideas arquitectónicas más valiosas son:

- runtime HIP nativo sin PyTorch;
- ejecutor propio y HIP graphs para decode;
- estado privado por sesión para KV, recurrencia, rollback y RNG;
- batching de cuerpo completo, no solo de la cabeza de vocabulario;
- MTP adaptativo con verificación exacta y rollback de 1–8 tokens;
- separación entre rutas prefill, autoregresiva y especulativa;
- snapshots dimensionados por estado válido, no por capacidad máxima;
- pruebas analíticas, operador independiente y replay de sesión;
- registro explícito de experimentos retenidos y rechazados;
- mediciones C1/C2/C4/C6/C8, profundidad y calidad junto al rendimiento.

Sus números publicados no son directamente comparables con Halogen: cambian
checkpoint/precisión, predictor, versión ROCm, corpus, configuración y fecha.
Hasta ejecutar un harness idéntico, solo sirven para fijar objetivos y detectar
zonas prometedoras.

## Qué aporta pwilkin como laboratorio abierto

Su trabajo es especialmente útil porque publica la secuencia de experimentos y
las ablaciones, no solo el resultado final. La señal más fuerte es un Gated
DeltaNet tiled: estado recurrente en registros a través de varios tokens y
reducción DPP en lugar de cinco viajes `ds_bpermute`, preservando el orden. En
su protocolo fue el mayor salto de toda la serie. También documenta un lector
directo `pread` para la PLE/n-gram dispersa, ring de inputs para evitar aliasing
en UMA, dequant BF16 cacheado para WMMA y fusiones que reducen pasadas de
memoria.

Igual de valiosos son los negativos: QSA disperso completo quedó alrededor del
empate contra atención densa incluso a 140k; el scorer WMMA del indexer fue
neutral/peor; y retained PM4 no intervino en prefill porque no se repetían las
formas de grafo. Por tanto, GDN tiled y el lector PLE pasan arriba en la cola;
QSA sparse debe compararse end-to-end con un control denso, incluida calidad;
el denso no se presume semánticamente equivalente. PM4 retenido será un
experimento posterior de decode. Detalle y límites en `PWILKIN_REVIEW.md`.

El repositorio de sitio/instalador es MIT, pero las ramas de llama.cpp, ROCm,
modelos y material tercero mantienen licencias propias. No hemos ejecutado el
instalador ni importaremos sus implementaciones: reimplementaremos ideas desde
contratos oficiales de Qwen y oráculos independientes.

## Síntesis para nuestro motor

Conservar:

- de Halogen: especialización extrema pero medida, subsistemas compilados por
  separado, tuning reproducible, fast path de un solo modelo/GPU y herramientas
  de calidad integradas;
- de GUFO: arquitectura verificable, GGUF, sesiones correctas, batching,
  snapshots, HIP graphs, disciplina experimental y trazabilidad.
- de pwilkin: GDN tiled como primer candidato, lector directo para la tabla PLE,
  ring de inputs UMA, ablaciones negativas y benchmark con repeticiones.

Mejorar:

- generar variantes desde una tabla declarativa de formas, evitando miles de
  instancias sin evidencia de uso;
- mantener un kernel registry con coste medido y restricciones numéricas;
- separar contrato matemático, selección de kernel y scheduling;
- hacer del modo C1 una ruta de primera clase sin sacrificar C2–C8;
- diseñar el allocator para UMA desde el principio: mmap/registro, prefault,
  páginas grandes opcionales y límites visibles;
- almacenar todos los resultados crudos y la identidad completa del entorno;
- negar una optimización si solo gana en microbenchmark o altera el contrato de
  calidad.

## Preguntas que requieren hardware

1. ¿Qué parte de la diferencia de decode es ancho de banda, aceptación MTP,
   formato de peso o scheduling?
2. ¿Qué familias Halogen con spills aparecen realmente en los hot paths?
3. ¿Wave32 domina por arquitectura o por las formas particulares de esta build?
4. ¿Cuánto cuesta registrar pesos mmap frente a una copia anónima/huge-page?
5. ¿Qué especializaciones sobreviven al end-to-end en C1 y C8?
6. ¿Puede un único formato interno lossless cubrir Q4_K_XL/IQ4_XS sin duplicar
   decenas de GiB?
