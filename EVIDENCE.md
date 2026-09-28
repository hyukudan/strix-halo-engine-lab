# Registro de evidencias

Este archivo es la fuente de trazabilidad de la especificación. Los estados
permitidos son: **observado**, **publicado**, **inferido**, **hipótesis** y
**pendiente**.

| ID | Estado | Afirmación | Evidencia | Consecuencia para la spec |
| --- | --- | --- | --- | --- |
| E-001 | observado | La imagen Halogen Flash 0.14.2 contiene tres objetos HIP `gfx1151`. | Extracción con `llvm-objdump --offloading`; hashes en `RESEARCH.md`. | Separar los subsistemas compilados es una opción a evaluar. |
| E-002 | observado | Los objetos contienen 1.057, 172 y 62 entradas de kernel. | `artifacts/halogen-metadata-{0,1,2}.json`. | Usar generación declarativa de variantes y medir cuáles son necesarias. |
| E-003 | observado | Todas las entradas inspeccionadas declaran wavefront 32. | Metadatos AMDGPU de los tres objetos. | Probar wave32 como candidato, sin convertirlo en regla universal. |
| E-004 | observado | Algunas variantes usan hasta 256 VGPR, aproximadamente 64 KiB de LDS y spills. | JSON de metadatos agregado. | Evaluar tiempo completo; la ocupación aislada no decide el ganador. |
| E-005 | observado | La familia `q4moe` emplea WMMA F16/BF16, LDS, permutaciones y operaciones packed. | `artifacts/halogen-isa-q4moe.json`. | Priorizar un prototipo MoE Q4 propio y comparar distintas estrategias de packing. |
| E-006 | observado | Las variantes DeltaNet combinan scan, WMMA/FMA, LDS y estado recurrente. | `artifacts/halogen-isa-deltanet.json`. | Probar fusión de preparación, scan y escritura de estado. |
| E-007 | observado | Las variantes de atención conservan una parte importante del softmax en FP32. | `artifacts/halogen-isa-attention.json`. | Fijar el orden de reducción y las fronteras de redondeo antes de optimizar. |
| E-008 | publicado | Halogen integra `verify`, `inspect`, `ppl` y `niah`. | README/AGENTS de Halogen Flash 0.14.2. | Incluir herramientas de calidad como comandos del producto. |
| E-009 | publicado | GUFO implementa GGUF, HIP graphs, sesiones, snapshots, batching y MTP adaptativo. | Código y documentación del commit fijado en `RESEARCH.md`. | Adoptar contratos equivalentes mediante implementación propia. |
| E-010 | publicado | GUFO exige controles de operador, replay y medición end-to-end. | `.agents/skills/optimize-kernel/SKILL.md` y documentación de calidad. | Hacer obligatorias tres puertas de calidad y benchmark emparejado. |
| E-011 | inferido | La especialización extrema puede explicar parte del rendimiento de Halogen. | Número de variantes y ausencia de runtime general; aún sin perfil dinámico. | Diseñar un registry especializado, pero no fijar cuántas variantes necesita. |
| E-012 | hipótesis | Un formato interno lossless puede reducir el coste de arranque sin duplicar demasiada memoria/disco. | Todavía sin prototipo ni medida. | Mantenerlo opcional hasta medir carga, tamaño y tráfico. |
| E-013 | pendiente | Diferencia real Halogen/GUFO bajo condiciones idénticas. | Falta máquina Linux Strix Halo, pesos y harness emparejado. | No utilizar cifras públicas como comparación directa. |
| E-014 | pendiente | Mejor política wave32/wave64 por familia y forma. | Requiere compilar y medir ambos modos. | El registry debe admitir ambos durante investigación. |
| E-015 | publicado | HIP/Clang permite compilación offline para un target GFX mediante `--offload-arch`. | Documentación oficial AMD HIP/Clang. | Crear un laboratorio reproducible `gfx1151` antes de recibir el equipo. |
| E-016 | observado | Halogen y GUFO exponen API/health/métricas, pero no una WebUI administrativa completa. | Búsqueda en repositorios fijados. | Diseñar un plano administrativo propio; chat solo como playground opcional. |
| E-017 | observado | La imagen runtime de Halogen no es un SDK completo: el probe HIP falla por ausencia de headers C++ estándar. | Compilación de `lab/cross_compile/probe.hip` dentro de la imagen OCI fijada. | No usar imágenes runtime ajenas como toolchain del proyecto. |
| E-018 | observado | `rocm/dev-ubuntu-24.04:7.2.4` compila offline el probe para `gfx1151`, y el HSACO extraído declara ese target, wave32 y cero spills. | Imagen por digest y hashes en `lab/cross_compile/README.md`. | El frontend/codegen básico queda validado; ejecución y rendimiento siguen pendientes. |
| E-019 | observado | Opus 5.5 identificó cuatro bloqueos P0 y recomendó reducir el primer slice. | Acta reproducible en `OPUS_REVIEW.md`. | No congelar la spec hasta resolver memoria, residencia UMA, fuentes canónicas y librerías `gfx1151`. |
| E-020 | publicado | Qwen publicó repositorio, informe técnico y checkpoint oficial de Flash-Next; describe 125B principales, 6B activos/token y 51B extra de tabla n-gram, con 36 capas GDN y 12 QSA. | `QwenLM/Qwen3.8-Flash-Next@69885871a64393807d988b27b1b5e380e8f28526`, arXiv:2608.30320 y `Qwen/Qwen3.8-Flash-Next@de4b8e4...`. | Usarlos como fuentes primarias de contratos; Halogen/GUFO/pwilkin quedan como referencias secundarias. |
| E-021 | publicado | El checkpoint oficial usa Qwen Community License 1.0 y contempla condiciones adicionales para determinados usos comerciales MaaS/AI Work Assistant. | `Qwen/Qwen3.8-Flash-Next/LICENSE`. | Mantener abierta la decisión de licencia, distribución y caso de uso. |
| E-022 | publicado | En Strix Halo GTT es límite de mapeo GPUVM sobre RAM compartida, no VRAM adicional; AMD recomienda carve-out BIOS pequeño y elevar TTM/GTT. | Documentación oficial ROCm 7.2 de optimización Strix Halo. | Separar mapping limit de admission y reservar memoria efectiva del host. |
| E-023 | publicado | Halogen propone para 128 GB `gttsize=126976` MiB y `pages_limit=32505856`; advierte GTT residual tras terminación sucia. | README actual de `peonist-ai/halogen-flash-server`. | Perfil cercano a 124 GiB solo con guardas, shutdown limpio y recuperación. |
| E-024 | observado | `pwilkin/strix-halo@f73872f` fija runtime y llama.cpp experimentales, publica medidas repetidas y conserva resultados negativos. | Copia local y `PWILKIN_REVIEW.md`. | Adoptar su disciplina experimental, no sus cifras como baseline propio. |
| E-025 | publicado | El mayor salto aislado publicado por pwilkin fue un GDN tiled que mantiene recurrencia en registros y sustituye LDS por DPP preservando orden. | `journey.html`, paso 11: ~499,52→1181,77 t/s en su protocolo. | Primer prototipo GDN; exige oráculo independiente y hardware real. |
| E-026 | publicado | Su lector directo de PLE midió 1,69× frente a demand paging por mmap, mientras sparse QSA completo quedó aproximadamente break-even frente a denso hasta 140k. | `journey.html`, pasos 3–4 y ablaciones. | Investigar I/O PLE; usar denso como control de rendimiento/calidad sin asumir equivalencia semántica. |
| E-027 | publicado | PM4 retenido ayuda a grafos repetidos de decode, no a prefill con shapes que no se repiten; está experimental y validado en pocos targets. | Sitio `pwilkin.github.io/strix-halo`, sección runtime. | Mantenerlo como experimento opt-in con fallback, fuera del camino crítico inicial. |
| E-028 | observado + inferido | El config oficial fijado declara 48 capas, 12 de atención completa, 2 cabezas KV y `head_dim=256`; de ello se infiere que un KV BF16 convencional requiere 24 KiB/token/sesión: 6 GiB a 262.144. | `artifacts/qwen-official-config-de4b8e4.json`, SHA-256 `889658F2508E8C61D409B02E70E0D78D8D4452EC65AAAFBE129805D213D2E74B`; fórmula en `MEMORY_BUDGET.md`. | C4 a contexto máximo serían 24 GiB solo de KV base; cualificar primero contextos menores. |
| E-029 | pendiente | El inventario offline `gfx1151` de rocBLAS/hipBLASLt no pudo repetirse en esta sesión porque el backend Linux de Docker Desktop no expuso su API, aunque el cliente y el proceso backend existen. | `docker info` contra `desktop-linux`, 2026-09-28. | P-011 sigue abierto; no inferir cobertura de librerías a partir de que el probe HIP compile. |

## Regla para nuevas entradas

Cada evidencia nueva debe incluir identidad del artefacto, comando o prueba,
condiciones del entorno y el requisito de la spec al que afecta. Si una medida
contradice una entrada anterior, ambas se conservan y se registra la resolución
en `DECISIONS.md`.
