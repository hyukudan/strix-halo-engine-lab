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

Mediciones del 2026-11-25 sobre `main` en la revisión `ca2de6ae8`, en AMD Strix Halo (Ryzen AI Max+ 395, Radeon 8060S,
128 GB por nodo), con un build HIP Release por nodo (GCC 15.2, HIP clang 24.0, ROCm 10.1, Linux 7.4 con el `amdgpu` del
kernel). Un par son dos nodos unidos por USB4 que ejecutan el mismo binario. Los límites de potencia son los de fábrica del
nodo; el reloj y la potencia de la GPU se muestrearon en cada petición: los prefills largos fueron a 1,950–2,900 MHz y
120–150 W por nodo.
No se fijó ninguna palanca a mano: decide `--profile exact|fast|turbo`, y cada ejecución imprimió su configuración efectiva.
Los prompts son texto real en inglés (un conjunto de documentos cortos más un documento del proyecto). El número de tokens
depende del tokenizador y se indica por fila. Cada motor y tier se cargó una vez; "a / b" son dos repeticiones sobre ese
servidor residente (o dos procesos CLI). Cada petición larga empezó por debajo de 70 °C. Un segundo valor más bajo se debe a
que un nodo mantiene un reloj de GPU menor tras unos dos minutos de carga sostenida; los recibos dan el reloj por petición.
El decode es una petición (B1), voraz, en el modo de decode por defecto del motor, 128 tokens tras un prompt de chat corto
salvo que se indique otra cosa. Las ejecuciones EXACT repitieron los mismos ids; las filas FAST y TURBO indican lo que imprime
su perfil (estado de calificación o las identidades FAST que aplica). Consulte las condiciones en
[metodología](docs/methodology.es.md).

| Modelo | Hardware | Ruta | Modo de decode | Perfil | Prompt (tokens) | Decode (tok/s) | Prefill (tok/s) | Main |
|---|---|---|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | par | HTTP servido | DSpark (por defecto) | EXACT | 1,971 / 7,994 / 30,975; decode tras 344 | 57.2 / 57.7 | 2K 782 / 739, 8K 1,101 / 1,111, 32K 1,273 / 1,245 | `ca2de6ae8` |
| DeepSeek V4.1 Flash | par | HTTP servido | DSpark (por defecto) | FAST | igual | 53.4 / 53.7 | 2K 969 / 922, 8K 1,425 / 1,588, 32K 1,689 / 1,708 | `ca2de6ae8` |
| DeepSeek V4.1 Flash | par | HTTP servido | DSpark (por defecto) | TURBO (= palancas FAST) | igual | — | 8K 1,444 / 1,591, 32K 1,683 / 1,710 | `ca2de6ae8` |
| MiniMax M3.1 Flash | par | HTTP servido | plain | EXACT | 1,102 / 7,791 / 30,179; decode tras 508 | 27.0 / 26.9 (tras 1,102: 32.6 / 28.7) | 1K 417 / 451, 8K 667 / 587, 32K 590 / 565 | `ca2de6ae8` |
| MiniMax M3.1 Flash | par | HTTP servido | plain | FAST | igual | 27.1 / 26.9 | 1K 415 / 457, 8K 709 / 660, 32K 640 / 611 | `ca2de6ae8` |
| MiniMax M3.1 Flash | par | HTTP servido | plain | TURBO | igual | 26.7 / 26.6 | 1K 422 / 459, 8K 709 / 662, 32K 674 / 624 | `ca2de6ae8` |
| GLM-5.3 Flash | par | HTTP servido | por defecto | EXACT | 2,037 / 8,108 / 31,258; decode tras 362 | 40.2 / 42.8 | 2K 531, 8K 629 / 604, 32K 578 / 552 | `ca2de6ae8` |
| GLM-5.3 Flash | par | HTTP servido | por defecto | FAST (resuelve a la ruta EXACT) | igual | 41.1 / 41.1 | 2K 530, 8K 628 / 596, 32K 568 / 545 | `ca2de6ae8` |
| GLM-5.3 Flash | par | HTTP servido | por defecto | TURBO (prefill WMMA FP16) | igual | 41.3 / 41.3 | 2K 718, 8K 949 / 941, 32K 879 / 854 | `ca2de6ae8` |
| Qwen3.8-Flash-Next | 1 nodo | CLI | plain (sin bundle del drafter MTP en nuestros nodos) | EXACT | 2,048 / 8,192 / 32,500 | 20.6–22.1 | 2K 1,435, 8K 1,667 / 1,840, 32K 1,669 / 1,677 | `ca2de6ae8` |
| Qwen3.8-Flash-Next | 1 nodo | CLI | plain | FAST | igual | 21.6–21.9 | 2K 1,673, 8K 1,865 / 1,880, 32K 1,681 / 1,663 | `ca2de6ae8` |
| Qwen3.8-Flash-Next | 1 nodo | CLI | plain | TURBO (= palancas FAST) | igual | 21.5–21.8 | 2K 1,188 (una ejecución), 8K 1,915, 32K 1,665 | `ca2de6ae8` |
| Qwen3.8-27B | 1 nodo | CLI | MTP (por defecto) | EXACT | 2,048 / 8,192 | 2K 35.2 / 35.3, 8K 25.8 / 25.0 | 2K 187–199, 8K 115–117 | `ca2de6ae8` |
| Qwen3.8-27B | 1 nodo | CLI | plain (`--no-mtp`) | EXACT | 2,048 / 8,192 | 2K 14.05, 8K 13.22 | 2K 195, 8K 121 | `ca2de6ae8` |
| Qwen3.8-27B | 1 nodo | CLI | MTP (por defecto) | FAST (TURBO = FAST) | 2,048 / 8,192 | 2K 35.2, 8K 25.0–25.1 | 2K 188–196, 8K 115–117 | `ca2de6ae8` |
| Qwen3.8-27B | par (TP) | CLI | MTP (por defecto) | EXACT / FAST | 2,048 | 45.4 / 45.5 | — | `ca2de6ae8` |
| Qwen3.8-27B | par (TP) | CLI | plain | EXACT | 2,048 | 20.22 | — | `ca2de6ae8` |
| Hy4-flash | 1 nodo | CLI | plain | FAST (por defecto, referencia BASE; el tier elige filas de prompt 4096) | 1,024 / 8,192 | 1K 17.5, 8K 14.25 | 1K 602, 8K 755 / 782 | `ca2de6ae8` |
| Hy4-flash | 1 nodo | CLI | plain | FAST, filas de prompt 2048 fijadas a mano | 2,048 / 8,192 / 32,500 | 16.9–17.0 | 2K 689–810, 8K 721–766, 32K 699 / 693 | `ca2de6ae8` |
| Hy4-flash | 1 nodo | CLI | plain | `exact` (BASE; prompt posición a posición) | 2,048 | 15.75 | 15.2 | `ca2de6ae8` |
| Hy4-flash | 1 nodo | CLI | plain | `exact` (BASE), filas de prompt 2048 fijadas a mano | 8,192 / 32,500 | 16.5–16.6 | 8K 687–723, 32K 661 (una ejecución) | `ca2de6ae8` |
| Hy4-flash | 1 nodo | CLI | plain | TURBO (el tier elige filas de prompt 4096) | 8,192 | 15.65 | 739 / 728 | `ca2de6ae8` |
| Hy4-flash | par (reparto por capas) | CLI | plain | FAST (el par siempre usa prompt por lotes) | 4,096 / 8,192 / 32,500 | 12.3–14.0 | 4K 1,265 / 1,216, 8K 1,327 / 1,387, 32K 1,328 (una ejecución) | `ca2de6ae8` |
| Hy4-flash | par (TP) | CLI | plain | FAST / `exact` | 1,024 | 22.64 / 22.42 | — | `ca2de6ae8` |
| Inkling 2 Mini | 1 nodo | HTTP servido | plain | EXACT (FAST y TURBO resuelven a la misma ruta en HIP) | 2,280 / 9,116 / 35,170; decode tras 396 | 23.6 / 23.6 (tras 35,170: 20.9) | 2K 591 / 591, 8K 601 / 595, 32K 578 / 568 | `ca2de6ae8` |

**Más bajo que la tabla anterior (2026-11-21, `d3a89ae16` / `f42edd05e`).** Las filas anteriores eran sobre todo
ejecuciones CLI con otros prompts, así que parte de cada cambio se debe a la ruta y al prompt:
- Prefill EXACT de DeepSeek V4.1 Flash a 2K (antes 791.9) y a 8K (antes 1,123.8).
- Prefill EXACT de MiniMax M3.1 Flash a 1K (antes 472.3); la segunda repetición a 8K (587) queda por debajo de 621.6.
- Prefill EXACT de GLM-5.3 Flash a 32K (antes 595).
- Prefill de Qwen3.8-27B: la configuración por defecto usa ventanas de prompt de 64 filas, 2K 187–199 y 8K 115–117
  frente a 644 / 606 (EXACT) y 684 / 635 (FAST) antes. Decode plain en par TP 20.22 (antes 20.61).
- Decode de Qwen3.8-Flash-Next: 20.6–22.1 en plain, frente a 51.0 con MTP antes; el bundle del drafter MTP no está
  instalado junto al modelo en nuestros nodos, así que el valor por defecto resuelve a decode plain.
- Decode FAST de Hy4-flash con el prompt de 4096 filas que elige el tier: 14.25 a 8K (antes 16.78).

**Sin medir en `ca2de6ae8`:** Nemotron 3.5 Bolt E4B (su tokenizador y sus tablas RoPE no están en nuestros nodos);
Qwen3.8-27B a 32K (el prefill por defecto no terminó dentro de nuestro límite de 470 s por ejecución); Qwen3.8-Flash-Next
por HTTP (`strix-serve --model qwen3.8-flash-next` necesita un golden de calibración MTP y la receta MTP en mosaico para
arrancar); las filas de decode MTP y DSpark por contenido (código, prosa, JSON, edición); los pares `exact` y TURBO de
Hy4-flash a 32K. Las filas TURBO marcadas "= palancas FAST" resuelven al conjunto de palancas FAST de ese motor.

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
