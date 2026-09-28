# Revisión externa con Claude Opus 5.5

Fecha: 2026-09-28  
Modelo solicitado y ejecutado: `claude-opus-5-5`, esfuerzo `high`  
Claude Code: `2.1.283`  
Modo: no interactivo, solo lectura (`Read`, `Glob`, `Grep`), sin persistencia.

La revisión comenzó sobre la instantánea v0.1. Mientras se ejecutaba se añadió
la compilación offline y el plano administrativo en v0.2; por tanto, la
objeción de que faltaba una fase previa al hardware quedó parcialmente resuelta
antes de recibir el informe. El resto se conserva sin suavizarlo.

## Resultado ejecutivo

Opus considera buena la trazabilidad y la separación entre hechos, inferencias
e hipótesis, pero no recomienda congelar todavía la especificación. Hay cuatro
bloqueos P0: memoria real del checkpoint, modelo correcto de residencia UMA,
fuentes públicas canónicas de la arquitectura y disponibilidad efectiva de
código `gfx1151` en las librerías ROCm elegidas.

También recomienda reducir el primer hito a AR + MTP, C1/C2/C4 y contexto menor,
dejando Responses API, prompt lookup, C6/C8 y 262K para después de validar el
motor básico.

## Acta de objeciones

| ID | Objeción de Opus | Respuesta/decisión | Cambio o prueba requerida |
| --- | --- | --- | --- |
| O-01 | Un checkpoint anunciado como ~118 GiB deja margen peligroso en 128 GB UMA. | Aceptada como P0. Tamaño de fichero y working set se separan: referencias publican ~68 GiB residentes más tabla dispersa, pero falta reproducción. | `MEMORY_BUDGET.md` ya calcula 6 GiB de KV BF16 C1 a 262K (24 GiB C4); completar recurrencia, MTP, rollback, scratch, librerías, SO y margen antes de elegir checkpoint. |
| O-02 | El texto sobre page faults podía sugerir demanda de páginas durante kernels GPU; `gfx11` probablemente no ofrece XNACK utilizable para ese diseño. | Aceptada. El hot path exigirá rangos registrados y residentes; los faults se medirán en preparación/prefault CPU. | Verificar XNACK y semántica HMM/registro con la versión exacta de kernel/ROCm. |
| O-03 | “Determinista” no define C1 frente a C8, ni bitwise frente a tolerancia. | Aceptada. | Crear contratos separados: replay bitwise cuando sea exigible y tolerancias predeclaradas para kernels paralelos. |
| O-04 | Orden de reducción WMMA, contracción FMA y denormales forman parte de la identidad numérica. | Aceptada. | Añadir flags de compilador, modo de denormales, `ffp-contract` y estrategia de reducción al manifiesto del kernel. |
| O-05 | Faltan repositorio, commit y licencia canónicos para QSA, Gated DeltaNet, routing MoE, MTP, tokenizer y chat template. | Parcialmente resuelta: repo, paper y checkpoint oficiales Qwen están fijados; la licencia exige decisión de uso/distribución. | Extraer contratos/oráculos y fijar aún tokenizer, template y aceptación MTP en `SOURCES_LEDGER.md`. |
| O-06 | Que el compilador acepte `gfx1151` no prueba que rocBLAS/hipBLASLt y demás librerías contengan objetos compatibles. | Aceptada como P0. | Auditar code objects de cada librería del toolchain fijado y construir fallbacks propios/hipBLAS cuando falte cobertura. |
| O-07 | El clasificador de familias y varios grupos ISA tenían errores; `v_dual` no es packed dot y faltaba `ds_bpermute`. | Corregida. | Herramientas actualizadas y artefactos agregados regenerables. |
| O-08 | “Diez variantes q4moe” mezclaba ocho kernels principales y dos helpers. | Aceptada. | Corregir la descripción sin inferir función dinámica por nombres. |
| O-09 | Los JSON con firmas/templates completas no eran compatibles con la política clean-room declarada. | Corregida y política aclarada. | Artefactos anonimizados; solo conteos agregados, sin nombres completos ni cuerpos ISA. Auditoría antes de publicar. |
| O-10 | Falta una fase A0 completamente offline y reproducible. | Parcialmente resuelta en v0.2. | Fijar imagen por digest, dos toolchains, hashes, oráculos CPU y puertas CI de recursos. |
| O-11 | El alcance v1 es demasiado ancho. | Aceptada. | Primer slice: AR + MTP, C1/C2/C4 y contexto cualificado menor; expansión solo tras puertas. |
| O-12 | Falta una baseline externa común como llama.cpp ROCm/Vulkan. | Aceptada. | Añadirla al harness sin convertirla en dependencia del motor. |
| O-13 | Faltan política de cuelgue/reset del driver y potencia/TDP en la identidad. | Aceptada. | Definir watchdog, drenaje, recuperación y registro de clocks/potencia/temperatura. |
| O-14 | Falta autenticación en la API de inferencia, no solo en la WebUI. | Aceptada. | Gateway con API keys/OIDC, cuotas y auditoría; motor solo por loopback/UDS. |
| O-15 | Chat, gateway API y administración no deberían compartir el mismo nivel de privilegio. | Aceptada. | Chat será pestaña opcional sin privilegios administrativos; administración con WebAuthn/mTLS y operaciones tipadas. |
| O-16 | Conviene comprobar reproducibilidad del HSACO y límites de VGPR/LDS/tamaño en CI. | Aceptada. | Doble compilación, hashes y diff de metadata/ISA normalizada; gates de recursos. |
| O-17 | No habrá afirmaciones de rendimiento válidas sin el hardware. | Confirmada. | La fase offline solo valida compilabilidad, formatos, recursos estáticos y corrección contra oráculos disponibles. |

## Cierre

La revisión no se considera cerrada hasta resolver O-01, O-02, O-05 y O-06
con evidencia reproducible. Las demás objeciones pasan a requisitos de la spec
o a tareas de laboratorio. Una segunda revisión se hará sobre la spec
consolidada y el ledger de fuentes, no sobre recuerdos de esta conversación.
