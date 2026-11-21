# Strix Halo engine lab

[English](README.md) · [Español](README.es.md)

Es un motor de inferencia en C++/HIP para AMD Strix Halo: Ryzen AI Max+ 395, Radeon 8060S (`gfx1151`), 128 GB de memoria
unificada, Linux y ROCm. Funciona en un nodo, o en dos nodos unidos por USB4. Una RTX PRO 6000 añade rutas opcionales.
Ejecuta Qwen3.8-Flash-Next, DeepSeek V4.1 Flash, MiniMax M3.1 Flash y GLM-5.3 Flash en producción.
También incluye compilaciones opcionales de Qwen3.8-27B, Hy4-flash, Nemotron 3.5 Bolt E4B,
MiniMax H3 (audio + vídeo) y MiniMax-H3-Image (texto a imagen).
Los perfiles dependen del motor. EXACT conserva los bits de su referencia declarada.
Cómo medimos y qué cubre "exacto": [metodología](docs/methodology.es.md).

## Resultados

Mediciones del 2026-11-21 sobre `main`, en AMD Strix Halo (Ryzen AI Max+ 395, Radeon 8060S, 128 GB por nodo).
Las revisiones medidas fueron `d3a89ae16` y `f42edd05e`, indicadas por fila. Los pares usan USB4. El límite de
potencia y el ajuste IOMMU de cada medición quedaron sin registrar en las tablas de medición.
C1/C2/C4/C8 indican flujos por lote en las pruebas de decode de Nemotron, con tasas agregadas.
Las tasas HTTP de Qwen son agregadas con un cliente y cuatro plazas configuradas. Los clientes completaron
sus peticiones y el verificador del kit rechazó el código de salida del servidor tras el cierre intencionado.
La cifra HTTP de Nemotron corresponde al brazo completo con el borrador desactivado y un cliente.
DSpark es un borrador especulativo. Los brazos FAST de Hy4 usan su referencia de dispositivo BASE.
Las celdas de petición completa muestran tokens del prompt + tokens de salida. Las tasas de prefill marcadas
con TTFT incluyen el tiempo hasta el primer token. Consulte las condiciones en [metodología](docs/methodology.es.md).

| Modelo | Hardware | Ruta | Modo | Perfil | Prompt (tokens) | Decode (tok/s) | Prefill (tok/s salvo indicación) | Petición completa | Main |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | 1 nodo | CLI | MTP | Por defecto (prefill FP16 automático) | texto real, 32K | 51.0 | 20.45 s | — | `d3a89ae16` |
| Qwen3.8-Flash-Next | 1 nodo | CLI | modo por defecto sin registrar | Por defecto | clase sin registrar, 2K y 8K | — | — | 2K + 33: 2.71 s, 8K + 33: 5.95 s | `d3a89ae16` |
| Qwen3.8-Flash-Next | 1 nodo | CLI | plano | Por defecto | texto real, 32K | — | — | 32K + 128: 24.7 s | `d3a89ae16` |
| Qwen3.8-Flash-Next | 1 nodo | HTTP servido | MTP + wide automático | Por defecto | longitudes sin registrar | prosa 38.0–40.2, edición 153–155, JSON 85.7–87.6, código 60.8–61.2, razonamiento 59.6–61.6 | — | — | `d3a89ae16` |
| Qwen3.8-27B | 1 nodo | CLI | plano | EXACT (por defecto) | clase sin registrar, 2K y 8K | 2K 13.90, 8K 13.10 | 2K 644, 8K 606 | — | `d3a89ae16` |
| Qwen3.8-27B | 1 nodo | CLI | plano | EXACT (por defecto) | texto real, 32K | 10.68 | 517 | — | `d3a89ae16` |
| Qwen3.8-27B | 1 nodo | CLI | plano | FAST (opcional) | clase sin registrar, 2K y 8K | — | 2K 684, 8K 635 | — | `d3a89ae16` |
| Qwen3.8-27B | 1 nodo | CLI | MTP | EXACT (por defecto) | edición, 477 | 32.2 | — | — | `d3a89ae16` |
| Qwen3.8-27B | 1 nodo | CLI | MTP | EXACT (por defecto) | longitudes sin registrar | código 40.8, prosa 25.9, razonamiento 40.3, JSON 62.2 | — | — | `d3a89ae16` |
| Qwen3.8-27B | 1 nodo | CLI | plano | EXACT (por defecto) | código, longitud sin registrar | 14.21–14.22 | — | — | `d3a89ae16` |
| Qwen3.8-27B | par | CLI | plano | Por defecto (TP) | código, longitud sin registrar | 20.61 | — | — | `d3a89ae16` |
| DeepSeek V4.1 Flash | par | CLI | plano | EXACT | clase sin registrar, 2K / 8K / 32K | — | 2K 791.9, 8K 1,123.8, 32K 1,224.1 (en caliente) | — | `d3a89ae16` |
| DeepSeek V4.1 Flash | par | CLI | plano | EXACT (por defecto) | 15–21 | código 26.85, prosa 26.81 | — | — | `f42edd05e` |
| DeepSeek V4.1 Flash | par | CLI | DSpark | EXACT (por defecto) | 15–21 | código 41.70, prosa 60.04, JSON 43.62, chat 39.98 | — | — | `f42edd05e` |
| MiniMax M3.1 Flash | par | CLI | plano | EXACT | clase sin registrar, 1K / 8K / 32K | 1K 24.20 | 1K 472.3, 8K 621.6 (en caliente), 32K 587.2 | — | `d3a89ae16` |
| MiniMax M3.1 Flash | par | CLI | plano | EXACT (por defecto) | 2,048 | código 23.28, prosa 23.19, JSON 23.11, edición 23.07 | código, prosa, JSON y edición 506–521 (TTFT) | — | `f42edd05e` |
| MiniMax M3.1 Flash | par | CLI | DSpark auto | EXACT (por defecto) | 2,048 | código 32.39, prosa 23.83 | — | — | `f42edd05e` |
| GLM-5.3 Flash | par | CLI | plano | EXACT (GEMV) | clase sin registrar, 8K y 32K | — | 8K 617 (en caliente), 32K 595 | — | `d3a89ae16` |
| GLM-5.3 Flash | par | CLI | plano | EXACT (R1b) | código, 2,048 | 25.73 | — | — | `f42edd05e` |
| GLM-5.3 Flash | par | CLI | plano | EXACT (R1b) | prosa, 1,884 | 25.94 | — | — | `f42edd05e` |
| GLM-5.3 Flash | par | CLI | MTP + wide | EXACT (R1b) | código, 2,048 | 40.37 | — | — | `f42edd05e` |
| GLM-5.3 Flash | par | CLI | MTP + wide | EXACT (R1b) | prosa, 1,884 | 66.48 | — | — | `f42edd05e` |
| Hy4-flash | 1 nodo | CLI | plano | FAST (por defecto), ref. BASE | texto real, 2K / 8K / 32K | 2K 16.75, 8K 16.78, 32K 16.69 | 2K 773.2, 8K 730.6 (en caliente), 32,568 tokens a 696.5 (en frío) | — | `d3a89ae16` |
| Hy4-flash | 1 nodo | CLI | plano | `exact` (BASE) | texto real, 8K | 16.39 | 679.3 (en caliente) | — | `d3a89ae16` |
| Hy4-flash | 1 nodo | CLI | MTP + wide | FAST (por defecto), ref. BASE | prompts cortos, longitudes sin registrar | código 32.23, prosa 22.93, razonamiento 29.34 | — | — | `d3a89ae16` |
| Hy4-flash | par | CLI | plano | FAST (por defecto), ref. BASE | texto real, 4K / 8K / 32K | — | 4K 1,163.7, 8K 1,260.3 (en caliente), 32K 1,269.6 (en frío) | — | `d3a89ae16` |
| Hy4-flash | par | CLI | plano (TP) | FAST (por defecto), ref. BASE | clase sin registrar, 1K | 22.3 | — | — | `d3a89ae16` |
| Hy4-flash | par | CLI | MTP (TP) | FAST (por defecto), ref. BASE | longitudes sin registrar | código 47.24, prosa 28.12, razonamiento 38.59 | — | — | `d3a89ae16` |
| Nemotron 3.5 Bolt E4B | 1 nodo | CLI | plano | EXACT (por defecto) | prueba, contexto 64 | C1 54.9, C2 103.2, C4 177.2, C8 243.1 | — | — | `d3a89ae16` |
| Nemotron 3.5 Bolt E4B | 1 nodo | CLI | plano | FAST (opcional) | prueba, contexto 64 | C1 53.2, C2 103.7, C4 196.0, C8 334.3 | — | — | `d3a89ae16` |
| Nemotron 3.5 Bolt E4B | 1 nodo | CLI | MTP | EXACT (por defecto) | 18–23 | prosa 208.3, código 194.7, JSON 193.9 | — | — | `d3a89ae16` |
| Nemotron 3.5 Bolt E4B | 1 nodo | CLI | MTP | FAST (opcional) | 18–23 | prosa 273.1, código 250.1, JSON 254.1 | — | — | `d3a89ae16` |
| Nemotron 3.5 Bolt E4B | 1 nodo | CLI | MTP | EXACT (por defecto) | edición, 561 | 169.7 | — | — | `d3a89ae16` |
| Nemotron 3.5 Bolt E4B | 1 nodo | CLI | MTP | FAST (opcional) | edición, 561 | 202.2 | — | — | `d3a89ae16` |
| Nemotron 3.5 Bolt E4B | 1 nodo | CLI | plano | FAST (opcional) | prosa repetida, 2K / 8K / 32K / 65,000 | — | 2K 5,050, 8K 4,909, 32K 4,249, 65,000: 3,610 | — | `d3a89ae16` |
| Nemotron 3.5 Bolt E4B | 1 nodo | HTTP servido | plano | EXACT (por defecto) | código, prosa, JSON y edición, longitudes sin registrar | 51.90 | — | — | `d3a89ae16` |

### RTX PRO 6000 (CUDA)

Mediciones del 2026-11-21 sobre `main` en `f42edd05e`, en una máquina con una RTX PRO 6000 y un límite de potencia de 528 W.

**Qwen 27B denso · decode por CLI.** Tasas en caliente en tok/s para 256 tokens de salida. Las longitudes de los prompts
quedaron sin registrar. En FAST, el digest de salida coincidió entre los tres modos para cada clase de contenido.

| Perfil | Contenido | Serial | MTP | MTP + wide |
|---|---|---:|---:|---:|
| FAST (`fast`, FAST4) | código | 85.90 | 283.3 | 292.2 |
| FAST (`fast`, FAST4) | prosa | 87.15 | 162.3 | 172.8 |
| FAST (`fast`, FAST4) | razonamiento | 86.52 | 244.0 | 271.1 |
| FAST (`fast`, FAST4) | JSON | 85.12 | 403.1 | 602.5 |
| FAST (`fast`, FAST4) | edición | 86.13 | 738.9 | 848.6 |
| EXACT (`exact`) | código | 84.46 | 280.9 | — |
| EXACT (`exact`) | prosa | 87.01 | 161.4 | — |

**Qwen 27B denso · prefill por CLI.** FAST4, ruta `large` admitida, bloques de 2,048 tokens y dos plazas.
Tasas y tiempos en caliente. La clase del prompt quedó sin registrar.

| Prompt (tokens) | Prefill (tok/s) | Tiempo (s) |
|---|---:|---:|
| 2K (2,048) | 13,372 | 0.153 |
| 8K (8,192) | 13,642 | 0.600 |
| 32K (32,768) | 11,954 | 2.741 |
| 128K (130,326) | 7,910 | 16.48 |

**Qwen3.8-Flash-Next · decode por HTTP.** Tasas en caliente en tok/s con cuatro plazas en el servidor y 128 tokens
de salida por petición. Las longitudes de los prompts quedaron sin registrar. C1 es la tasa por petición con un cliente.
C8 es la tasa agregada desde que el último cliente recibe su primera señal, con ocho clientes concurrentes. Las barras
muestran los dos valores registrados. El digest de salida para prosa en FAST coincidió entre plano, MTP y MTP + wide.

| Perfil | Contenido | Modo | C1 (tok/s) | C8 (tok/s) |
|---|---|---|---:|---:|
| FAST | prosa | plano | 96.5 / 96.4 | 506 / 504 |
| FAST | prosa | MTP (sidecar) | 120.5 / 120.4 | 621 / 565 |
| FAST | prosa | MTP + wide | 125.6 / 125.8 | 609 / 626 |
| FAST | edición | MTP + wide | 426.9 / 426.2 | 1,086 / 1,131 |
| FAST | JSON | MTP + wide | 294.9 / 294.2 | 921 / 1,055 |
| EXACT (`--profile default`) | prosa | MTP + wide | 71.7 / 73.1 | 301.7 / 292.5 |

**Qwen3.8-Flash-Next · prefill por HTTP.** FAST, en caliente. El modo de decode y la clase del prompt quedaron sin registrar en estas pruebas.

| Prompt (tokens) | TTFT (ms) | Prefill (tok/s) | TTFT del cliente (s) |
|---|---:|---:|---:|
| 2K (2,048) | 449 | 4,557 | 0.489 |
| 8K (8,192) | 1,359 | 6,030 | 1.392 |
| 32K (32,768) | 5,358 | 6,116 | 5.394 |

<a id="hardware"></a>

## Otro hardware

El planificador reserva la memoria medida del dispositivo para KV del contexto
solicitado, workspace y formatos/topologías compatibles antes de cargar. Consulte
[despliegue en otro hardware](docs/deployment.es.md) para inspección, ejemplos TOML,
API de orquestación y límites actuales, incluidos los runtimes INT4 pendientes.

## Inicio rápido

Un nodo. Necesita el toolchain de ROCm, el GGUF del modelo y un fichero `prompt.ids` generado con el tokenizador.

```bash
git clone https://github.com/hyukudan/strix-halo-engine-lab.git
cd strix-halo-engine-lab
export PATH=/opt/rocm/core-10.1/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/bin/amdclang++ \
  -DSTRIX_ENABLE_HIP_RUNTIME=ON \
  -DSTRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME=ON \
  -DSTRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE=ON \
  -DSTRIX_USE_OPENSSL_SHA256=ON -DBUILD_TESTING=ON
ninja -C build strix-hip-decode-token
./build/strix-hip-decode-token Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf $(cat prompt.ids) --generate 256
```

La ejecución imprime un recibo JSON con los IDs de token y los tiempos. Añada `--print-effective-config` para ver la
configuración resuelta antes de cualquier trabajo en el dispositivo.
Con el [paquete del borrador](docs/engine/speculative/mtp-native-cli.md#default-mtp-drafter-bundle) junto al modelo,
el comando usa MTP. Sin el paquete, o con `--no-mtp` / `STRIX_MTP=0`, genera los mismos tokens uno a uno.

Par (DeepSeek). Dos nodos de 128 GB y dos cables USB4. Ejecute en ambos nodos, con `--rank 1` y la otra dirección en
el segundo:

```bash
scripts/pair/ds41-pair init --rank 0 --peer user@node-b --model-dir /path/to/models
scripts/pair/ds41-pair check && scripts/pair/ds41-pair driver && scripts/pair/ds41-pair build && scripts/pair/ds41-pair fetch
scripts/pair/ds41-pair run --both    # solo rank 0
scripts/pair/ds41-pair smoke         # solo rank 0
```

El script imprime los pasos de root y nunca los ejecuta en silencio. Detalles: [configuración del par](docs/pair-setup.es.md).
La compilación y el lanzamiento de cada motor: [primeros pasos](docs/getting-started/README.es.md).

<a id="models"></a>

## Matriz de modelos

`fast` selecciona las opciones cualificadas para el régimen. `quality` usa un pack de mayor precisión donde existe.

| Modelo | Uno / par | Perfiles | Memoria | Pesos |
|---|---|---|---|---|
| Qwen3.8-Flash-Next (UD-Q4_K_XL) | uno, dos réplicas detrás de un router o prefill en par opcional | MTP con el paquete del borrador; prefill automático; `fast` | 111 GB de pesos en un nodo de 128 GB | [GGUF UD-Q4_K_XL](https://huggingface.co/hyukudan/Qwen3.8-Flash-Next-GGUF-mirror) (réplica de `unsloth`, leído tal cual se publica); [cabeza de draft MTP Q8_0 destilada](https://huggingface.co/hyukudan/Qwen3.8-Flash-Next-MTP-distillG-Q8_0) |
| Qwen3.8-27B (IU4 GGUF) | uno o par TP, compilación HIP opcional | EXACT por defecto y FAST opcional en un nodo | 128 GB por nodo | [contrato GGUF](docs/models/qwen3.8-27b/README.md) |
| Hy4-flash (Q4_K_M GGUF) | uno o par USB4, compilación HIP experimental | FAST por defecto; referencia BASE | 128 GB por nodo | [GGUF en dos partes](docs/models/hy4-flash/README.es.md#pack) |
| DeepSeek V4.1 Flash (expertos de 2 bits, denso trellis de 6.5 bits) | par por USB4 | exacto por defecto; `fast` | 2 x 128 GB | [GGUF de expertos](https://huggingface.co/hyukudan/deepseek-v4.1-flash-gguf-mirror), [capa densa](https://huggingface.co/hyukudan/DeepSeek-V4.1-Flash-1017-strix-dense-U6), [packs trellis](https://huggingface.co/hyukudan/DeepSeek-V4.1-Flash-1017-strix-trellis-EXL3) |
| MiniMax M3.1 Flash (EXL3 3.06 bpw) | par TP2 | exacto | 2 x 128 GB | [EXL3 3.06 bpw](https://huggingface.co/hyukudan/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw-mirror) (réplica de `Mia-AiLab`, con nuestro draft EXL3); [origen del draft](https://huggingface.co/hyukudan/MiniMax-M3.1-Flash-DFlash2-mirror) |
| GLM-5.3 Flash (EXL3 4.05 bpw) | par TP2 | exacto; prefill WMMA opcional | 2 x 128 GB | [EXL3 4.05 bpw](https://huggingface.co/hyukudan/GLM-5.3-Flash-exl3-mirror) (réplica de `turboderp`) |
| Nemotron 3.5 Bolt E4B (Q4_K_M; EXL3 para `quality`) | uno, compilación opcional; prefill en par opcional (`STRIX_G4_PAIR=1`) | exacto por defecto; `fast`; `quality` | 128 GB por nodo | GGUF Q4_K_M oficial (no lo replicamos); [pack EXL3](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-EXL3); [drafter MTP](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-MTP-Drafter) |
| MiniMax H3 (audio + vídeo) | uno o par, compilación opcional | referencia densa; palancas FAST, cada una con su flag | 128 GB por nodo | no publicado por nosotros; origen [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/54472221d27c695fca9621b0c3ed52f06c4af707) ([ficheros](docs/models/minimax-h3/README.es.md#instalación-y-ejecución)) |
| MiniMax-H3-Image (int8 o W4A8) | uno o par (`strix-h3-image`), compilación opcional | VAE en GPU; `--parallel sequence` para una imagen, `request` para varias | 128 GB por nodo | no publicado por nosotros; origen [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/0bb149f51e3156ed4b90dd45d5620346bfe1df8f) |

Nuestros pesos en Hugging Face: [huggingface.co/hyukudan](https://huggingface.co/hyukudan). Las réplicas conservan los ficheros de origen; revisiones, nombres de fichero y licencias: [modelos y procedencia](docs/reproduce.md#models-and-provenance).

<a id="discrete-gpu"></a>

Rutas opcionales con RTX PRO 6000 (96 GB): DeepSeek independiente, DeepSeek RTX + par, descarga del prefill de Qwen,
servicio de Qwen y GLM en una tarjeta (borrador). Vea [uso de una GPU discreta](docs/discrete-gpu.es.md).

Cada interruptor, su valor por defecto y cuándo otro hardware quiere otro valor: [toggles](docs/toggles.es.md).

## Entrada de imágenes

DeepSeek V4.1 en el par de dos nodos acepta imágenes en chat completions al estilo OpenAI. Añada una parte `image_url`
a un mensaje de usuario. Está desactivada por defecto: arranque el servidor con `--vision <GGUF de visión>` en el rank 0,
junto con `--prefill-path device`.

```bash
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]}]}'
```

Límites: solo PNG y JPEG; se rechazan las imágenes que superan el presupuesto de filas, el vídeo y las peticiones con
imagen y `session_id`. Los demás modelos aún no admiten imágenes. Detalles: [entrada de imágenes](docs/engine/vision.es.md).

## Cómo funciona

- Un núcleo de kernels compartido: lectores GGUF y EXL3, kernels de matrices cuantizadas, atención, el planificador de
  memoria y los recibos. El núcleo no conoce ningún modelo.
- Cada familia de modelos es un módulo con su propio contrato, recorrido de capas, plan y compuertas
  ([familias de modelos](docs/models/model-families.md)).
- El motor elige kernels según el número de filas en vuelo. El decode de una fila lee los pesos más pequeños.
  La verificación ancha y el prefill usan las rutas de matrix cores y de enteros ([regla de régimen](docs/toggles.es.md#la-regla-de-régimen)).
- La decodificación especulativa (MTP, DSpark) mantiene el contrato exacto.
- El par usa USB4 directo para el paralelismo de tensores (TP2) o el reparto por etapas según el motor ([par](docs/pair.es.md)).
- Cada valor por defecto tiene una ganancia medida y un interruptor de apagado ([optimizaciones](docs/optimizations.md)).
- Añadir un modelo: [docs/engine/adding-a-model.es.md](docs/engine/adding-a-model.es.md).
- Mapa de los núcleos compartidos y qué modelo usa cada uno: [docs/engine/shared-cores.es.md](docs/engine/shared-cores.es.md).

## Documentación

| Tema | Documentos |
|---|---|
| Inicio | [primeros pasos](docs/getting-started/README.es.md) · [referencia del motor](docs/getting-started/engine-reference.md) · [reproducir](docs/reproduce.md) |
| Modelos | [Qwen Flash-Next](docs/models/qwen3.8/README.es.md) · [Qwen 27B](docs/models/qwen3.8-27b/README.md) · [Hy4-flash](docs/models/hy4-flash/README.es.md) · [DeepSeek](docs/models/deepseek-v4.1/README.es.md) · [MiniMax](docs/models/minimax-m3.1/README.es.md) · [GLM](docs/models/glm-5.3/README.es.md) · [Nemotron](docs/models/nemotron-3.5-bolt-e4b/README.es.md) · [MiniMax H3](docs/models/minimax-h3/README.es.md) · [familias](docs/models/model-families.md) |
| Par y servicio | [par](docs/pair.es.md) · [configuración del par](docs/pair-setup.es.md) · [servicio](docs/serving.es.md) · [GPU discreta](docs/discrete-gpu.es.md) |
| Interruptores | [toggles](docs/toggles.es.md) · [optimizaciones](docs/optimizations.md) ([JSON](docs/optimizations.json)) |
| Números | [rendimiento](docs/performance.md) · [ganancias medidas](docs/measured-gains.es.md) · [metodología](docs/methodology.es.md) |
| Internos | [arquitectura](docs/architecture/README.md) · [núcleos compartidos](docs/engine/shared-cores.es.md) · [registro de decisiones](docs/decisions/decision-register.md) · [estado de implementación](docs/reference/implementation-status.md) · [fuentes](docs/reference/sources-ledger.md) |
| Índice | [índice completo de la documentación](docs/README.md) · [AGENTS.md](AGENTS.md) |

<a id="related-repositories"></a>

## Repositorios relacionados

- [strix-halo-usb4-driver](https://github.com/hyukudan/strix-halo-usb4-driver): la serie de parches de Linux que convierte
  los controladores USB4 de Strix Halo en el enlace directo entre los dos nodos.
- [strix-trellis](https://github.com/hyukudan/strix-trellis): la biblioteca de cuantización trellis (estilo EXL3)
  usada para los pesos densos de DeepSeek y la inferencia de MiniMax.
- [TensorMeld](https://github.com/hyukudan/tensormeld): colocación y enrutamiento entre nodos y GPU; su router
  sirve las réplicas de Qwen.
- [tierstream](https://github.com/hyukudan/tierstream): streaming de expertos por niveles entre VRAM, memoria de host fijada
  y NVMe, usado por la ruta de DeepSeek en RTX independiente.

<a id="license"></a>

## Licencia

El código fuente del motor se distribuye bajo la **[Apache License 2.0](LICENSE)**, excepto el módulo del kernel de Linux
en `lab/direct_gpu_stream/kernel/`, que es **GPL-2.0-only** como debe serlo el código del kernel de Linux (sus textos
de licencia se conservan en ese directorio). El código adoptado de otros proyectos conserva su licencia y avisos originales. Los pesos
de los modelos se distribuyen bajo sus propias licencias. **Built with MiniMax M3.1-Flash**; la licencia de su modelo
es no comercial por defecto. Conserve los avisos del modelo y de la conversión y revise los términos aplicables. Las referencias de terceros usadas para los formatos y la semántica de los modelos
se enumeran con sus licencias en [docs/reference/sources-ledger.md](docs/reference/sources-ledger.md).
