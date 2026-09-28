# Ledger de fuentes y procedencia

Estado: P0 parcialmente resuelto. Una referencia usada por Halogen, GUFO o pwilkin no se convierte
automáticamente en fuente canónica de nuestro motor.

| Componente | Fuente candidata fijada | Licencia verificada | Contrato independiente | Estado |
| --- | --- | --- | --- | --- |
| Arquitectura Flash-Next | `QwenLM/Qwen3.8-Flash-Next@69885871a64393807d988b27b1b5e380e8f28526`, paper `arXiv:2608.30320` y checkpoint oficial `Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540` | Qwen Community License 1.0; revisar caso de uso/distribución | Contratos QSA/GDN/Gated Residual/n-gram por extraer | **Fuente primaria fijada** |
| Configuración y pesos Flash-Next Q4 | `unsloth/Qwen3.8-Flash-Next-GGUF@38bb39ee97821de2c9009abb7e93950eec396e66` | Derivada; pendiente mapear condiciones del modelo oficial | Hash/tensor map frente al checkpoint oficial | Candidato; no aprobado |
| Predictor MTP Q8_0 | mismo snapshot, `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | Pendiente | Pendiente | Candidato; no aprobado |
| Implementación AR de contraste | `llama.cpp@b11069` según documentación GUFO | Pendiente de resolver a SHA completo | CPU/GPU de contraste | Referencia secundaria |
| Implementación MTP de contraste | `llama.cpp@6fcaa16f` según documentación GUFO | Pendiente de resolver a SHA completo | Replay de tokens/aceptación | Referencia secundaria |
| QSA/indexer/top-k | Paper y checkpoint/config oficiales Qwen fijados arriba | Sí, sujeta a política de uso | Oráculo y shapes exactos pendientes | Parcial; ya no depende de Halogen/GUFO |
| Gated DeltaNet/scan | Paper y checkpoint/config oficiales Qwen fijados arriba | Sí, sujeta a política de uso | Ecuaciones, estado y orden numérico pendientes | Parcial; ya no depende de Halogen/GUFO |
| Routing y expertos MoE | Paper y checkpoint/config oficiales Qwen fijados arriba | Sí, sujeta a política de uso | Tie-breaking, orden y layout pendientes | Parcial |
| Algoritmo de aceptación MTP | No identificada | No | No | **P0 bloqueado** |
| Tokenizer | Debe extraerse del snapshot canónico y fijarse por hash | Pendiente | IDs exactos en fixtures | Abierto |
| Chat template/reasoning controls | Debe extraerse del snapshot canónico y fijarse por hash | Pendiente | prompts y tokens golden | Abierto |
| N-gram/PLE | Config oficial fijado (`ngram_size=3`, base 20M, 128 particiones, PLE capa 2) y paper/código Qwen | Qwen Community License 1.0 | hashing, filas, conv y timing de lookup pendientes | **P0 abierto** |
| Hyper-connections/output gate | Config oficial (`hc_count=4`, `hc_lowrank=320`, sigmoid) y paper/código Qwen | Qwen Community License 1.0 | mezcla residual, scratch y orden numérico pendientes | **P0 abierto** |
| mRoPE parcial/interleaved | Config oficial fijado (`partial_rotary_factor=0.25`) y código Qwen | Qwen Community License 1.0 | posiciones multimodales y oráculo texto pendientes | Abierto |
| Ideas de optimización Strix Halo | `pwilkin/strix-halo@f73872fe20dfdef460653f18e2fe63e5366ae958`; ramas fijadas en `PWILKIN_REVIEW.md` | Sitio/instalador MIT; dependencias separadas | Solo hipótesis y protocolo, sin importar implementación | Referencia secundaria |
| Estructura GGUF v3 | `ggml-org/ggml@353b63b439f27ab2cc19dac97ab1681ba6d2d084`, `docs/gguf.md`; geometrías en `ggml-org/llama.cpp@6c7a87f7e5e5cd75b8a641c3471f2dee84a6ed17` | MIT de ambos proyectos; especificación/catálogo públicos | Fixtures propios de parseo, límites, bloques y rangos | Fuente primaria fijada para A0.2/A0.3 |

## Regla de aprobación

Cada fila necesita URL/repo, SHA o revisión inmutable, licencia aplicable,
hashes de artefactos y un contrato verificable sin depender de Halogen. El
código de GUFO puede orientar pruebas y arquitectura, pero no será el oráculo
único de una implementación propia.
