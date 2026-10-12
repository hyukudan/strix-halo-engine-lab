# Ledger de fuentes y procedencia

Estado: P0 parcialmente resuelto. Una referencia usada por external runtime A, external runtime B o external platform study no se convierte
automáticamente en fuente canónica de nuestro motor.

| Componente | Fuente candidata fijada | Licencia verificada | Contrato independiente | Estado |
| --- | --- | --- | --- | --- |
| Arquitectura Flash-Next | `QwenLM/Qwen3.8-Flash-Next@69885871a64393807d988b27b1b5e380e8f28526`, paper `arXiv:2608.30320` y checkpoint oficial `Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540` | Qwen Community License 1.0; revisar caso de uso/distribución | Contratos QSA/GDN/Gated Residual/n-gram por extraer | **Fuente primaria fijada** |
| Configuración y pesos Flash-Next Q4 | `unsloth/Qwen3.8-Flash-Next-GGUF@38bb39ee97821de2c9009abb7e93950eec396e66` | Derivada; pendiente mapear condiciones del modelo oficial | Hash/tensor map frente al checkpoint oficial | Candidato; no aprobado |
| Predictor MTP Q8_0 | mismo snapshot, `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | Pendiente | Pendiente | Candidato; no aprobado |
| Implementación MTP de contraste | `ggml-org/llama.cpp#27836@1d8de7c1b0c7d2febf8f983174d8e6a711e2b1af` | MIT del proyecto; PR abierta/no canónica | Contraste estructural y del input combiner; atención contiene TODO QSA, por lo que NO es oráculo del forward completo | **Secundaria; input fusion corroborado, forward completo bloqueado** |
| Routing y expertos MoE | Paper y checkpoint/config oficiales Qwen fijados arriba | Sí, sujeta a política de uso | Tie-breaking, orden y layout pendientes | Parcial |
| Algoritmo de aceptación MTP | `huggingface/transformers@ebd5de00b43e94cae2da6d7aeb5a76ecdda51dae`, `src/transformers/generation/utils.py` y `candidate_generator.py` | Apache-2.0 | Greedy acceptance/crop fixture propio; sampled mantiene RNG pendiente | **Greedy fijado; sampled RNG abierto** |
| Tokenizer | Debe extraerse del snapshot canónico y fijarse por hash | Pendiente | IDs exactos en fixtures | Abierto |
| Chat template/reasoning controls | Debe extraerse del snapshot canónico y fijarse por hash | Pendiente | prompts y tokens golden | Abierto |
| N-gram/PLE | Config oficial fijado (`ngram_size=3`, base 20M, 128 particiones, PLE capa 2) y paper/código Qwen | Qwen Community License 1.0 | hashing, filas, conv y timing de lookup pendientes | **P0 abierto** |
| Hyper-connections/output gate | Config oficial (`hc_count=4`, `hc_lowrank=320`, sigmoid) y paper/código Qwen | Qwen Community License 1.0 | mezcla residual, scratch y orden numérico pendientes | **P0 abierto** |
| mRoPE parcial/interleaved | Config oficial fijado (`partial_rotary_factor=0.25`) y código Qwen | Qwen Community License 1.0 | posiciones multimodales y oráculo texto pendientes | Abierto |
| Transformers `qwen4_exp` semantics | `huggingface/transformers@ebd5de00b43e94cae2da6d7aeb5a76ecdda51dae`, `src/transformers/models/qwen4_exp/modeling_qwen4_exp.py` | Apache-2.0 | Independent tiny fixtures for RMSNorm, partial RoPE, recurrent GDN and PLE n-gram hashing | **Pinned for A0.5b; greedy MTP acceptance fixed, sampled RNG/QSA ties remain open** |
| IQ4_XS storage/dequant contract | `ggml-org/llama.cpp@8019dc563b1ecbae6b161a70c3a1359f1b206c1e`, `ggml/src/ggml-common.h` and `ggml/src/ggml-quants.c` | MIT | Independent byte-layout parser, FP16 conversion and full 256-value dequant fixture | **Pinned for A1.1 compact-format experiments; not a canonical checkpoint decision** |
| Estructura GGUF v3 | `ggml-org/ggml@353b63b439f27ab2cc19dac97ab1681ba6d2d084`, `docs/gguf.md`; geometrías en `ggml-org/llama.cpp@6c7a87f7e5e5cd75b8a641c3471f2dee84a6ed17` | MIT de ambos proyectos; especificación/catálogo públicos | Fixtures propios de parseo, límites, bloques y rangos | Fuente primaria fijada para A0.2/A0.3 |
| Experimental USB4 DMA-BUF extension | [Linux USB4STREAM source](https://github.com/torvalds/linux/blob/72d3fcf802c45d00b300f25b848a93c3a2bd7c7e/drivers/thunderbolt/stream.c), pinned commit `72d3fcf802c45d00b300f25b848a93c3a2bd7c7e` | GPL-2.0; upstream Intel notices preserved in the integration patch. Project kernel code GPL-2.0-only; public UAPI GPL-2.0 WITH Linux-syscall-note. Exact upstream license/exception texts retained in `lab/direct_gpu_stream/kernel/` | Project-owned experimental ABI, attachment/SG/controller-drain contracts and shared property tests | Enabled real kernel driver object compiled with strict warnings; no module load, DMA execution or GPU/cache qualification |

## Regla de aprobación

Cada fila necesita URL/repo, SHA o revisión inmutable, licencia aplicable,
hashes de artefactos y un contrato verificable sin depender de external runtime A. El
código de external runtime B puede orientar pruebas y arquitectura, pero no será el oráculo
único de una implementación propia.
