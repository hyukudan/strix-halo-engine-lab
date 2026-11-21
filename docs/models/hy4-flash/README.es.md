# Hy4-flash

[English](README.md)

Estado: **experimental.** El modelo corre en HIP (Strix Halo, gfx1151) en un nodo o repartido por capas entre dos
nodos, mediante el ejecutor `strix-hy4-rank`. El tokenizador, la plantilla de chat, las opciones de servicio y la
referencia de host (oráculo en CPU) están enlazados en el host. El servicio (`strix-serve --model hy4_flash`) admite el
pack y resuelve sus opciones, y después rehúsa crear la sesión: de momento el enlace de dispositivo solo se alcanza con
`strix-hy4-rank`.

## Pack

GGUF Q4_K_M en dos partes que se leen juntas (se pasan las dos rutas, en orden):

| fichero | bytes |
|---|---|
| `Hy4-flash-Q4_K_M-00001-of-00002.gguf` | 104,986,517,504 |
| `Hy4-flash-Q4_K_M-00002-of-00002.gguf` | 63,360,616,128 |

156.78 GiB de tensores (1416 tensores). La tabla Engram se queda en disco y se lee fila a fila. El resto es residente:
unos 91 GiB en un nodo, o unos 45 GiB por nodo en el reparto por capas.

## Cómo se enlaza la pasada hacia delante

El soporte es un enlace sobre código compartido (AGENTS.md, "Adding a model"). La tabla asigna cada etapa al
componente compartido que la ejecuta en HIP e indica qué es específico de este modelo.

| etapa | componente compartido | código específico del modelo |
|---|---|---|
| tokenizador, plantilla de chat, parser de llamadas a herramientas | `serve/bpe_tokenizer` (regla `hy_v4`), `serve/hy4_chat` sobre `serve/chat_template_kit` | dialecto de la plantilla (sufijo de tokens especiales, esfuerzos de razonamiento) |
| pesos, nombres de tensores | lector GGUF + tabla `weight/tensor_map`, `core::HipRangeWeightLoader` por parte y rango de capas | el mapa de nombres (`weight/adapters/hy4`), qué tensores necesita un rango de capas |
| embedding (Q8_0, cuatro flujos) | `launch_hip_embedding_rows` + epílogo `ops/glu` | ninguno |
| filas Engram (capas 1, 11, 21, 31) | `engram::RowReader` (páginas de host) → kernels `engram/hip` de preparación e inyección | la especificación del hash (`src/hy4/hy4_engram_spec.cpp`), la disposición del buffer de compuerta/escala |
| mezclas, colapso y expansión de hiperconexión | `core/hc_core` (contrato `include/strix/hc`) | ninguno (preset Hy4 en el contrato) |
| RMSNorm (7 por capa) | kernel `ops/norm`, `OutputForm::bf16_double` | rasgos de la norma (raíz cuadrada declarada, pesos FP32) |
| proyecciones densas (q_a, q_b, kv_a, out, gate, FFN denso, experto compartido, latente down/up) | decode: forma directa `q8_rows` (+ epílogo de almacenamiento BF16); prefill: GEMM de filas q8a8 `core/f16_dense_tall`; router: tile de filas densas | ninguno |
| RoPE (cola intercalada, theta por tipo de capa) | lanzador `ops/rope` + tabla nativa | theta por tipo de capa |
| fila de caché latente [512 \| 64] | escritor `attn_core/cache_rows` | plan de caché por capa (`hy4_hip_plan.hpp`) |
| cabezas MLA absorbidas (K_b, V_b) | cabezas agrupadas `q8_rows` (un lanzamiento, decode); filas q8a8 (prefill) | ninguno |
| indexador por grupos (pool comprimido, puntuación, selección) | `attn_core/group_pool`, `group_score_scan`, `group_select_kernel` (`include/strix/attn/group_indexer.hpp`) | el preset del indexador Hy4 |
| MLA dispersa con sumideros (lista DSA / anillo SWA 512) | `attn_core::sparse::launch_flash_rows` con las políticas de `mla_contract` | especificaciones DSA / SWA (presets del contrato) |
| compuerta de salida | compuerta sigmoide `ops/glu` | ninguno |
| router (F32, top-16, escala) | filas densas (F32) + kernel de ruta `moe_core`, preset `hy4` de `moe/route_policy` | ninguno |
| expertos enrutados (gate/up Q4_K, down Q4_K/Q6_K, recorte) | decode: kernels A8 de sonda y down; prefill: núcleo FP16 agrupado de filas conocidas | ninguno |
| combinación de expertos + suma del compartido | combinación `moe_core` + epílogo `ops/glu` | ninguno |
| cabeza (BF16, 120832 × 3072) y siguiente token | `hip_lm_head_launcher`; fila voraz en dispositivo (`hip_generation_output`) | ninguno |
| reparto en dos nodos | canalización por capas `pair::layers` (las ventanas cruzan como filas FP32), transportes `pair/link` | corte por defecto = capa central |
| palabra de nivel | resolutor compartido (`strix/engine_profile.hpp`) | las filas de palancas (`include/strix/profiles/hy4.hpp`) |

### Ficheros bajo el directorio del modelo y por qué no se pueden compartir

| fichero | qué | por qué es solo del modelo |
|---|---|---|
| `src/hy4/hip/hy4_hip_forward.{hpp,hip}` | el orden de etapas de la pasada sobre las entradas compartidas de arriba | la composición de la pasada de este modelo (orden de etapas, nombres de tensores, rasgos); no define ningún kernel, todos los kernels de dispositivo son plantillas compartidas |
| `src/hy4/hip/hy4_hip_plan.hpp` | tabla de etapas, plan de caché por capa, reglas de ventana del prompt, corte del par | geometría y tipos de capa de Hy4 (indexada / deslizante / Engram) como datos |
| `src/hy4/ref/hy4_host_ref.{hpp,cpp}` | el oráculo en CPU | la aritmética declarada del modelo, etapa por etapa; llama a las referencias de host compartidas donde calculan los mismos bits |
| `src/hy4/hy4_engram_spec.cpp` | especificación del hash Engram | constantes del hash y geometría de la tabla de este modelo |
| `src/hy4/serve/session_service.{hpp,cpp}` | opciones de la familia, configuración efectiva, admisión del pack | el adaptador de familia de la fábrica de servicio compartida (la misma forma que las demás familias) |
| `apps/hy4_rank_main.cpp`, `apps/hy4_ref_main.cpp` | el ejecutor de dispositivo y el del oráculo | líneas de órdenes de este modelo |
| `include/strix/profiles/hy4.hpp` | filas de palancas de la palabra de nivel | datos para el resolutor compartido |
| `models/hy4-flash.toml` | entrada descriptiva | ajustes |

## Clase aritmética

La ruta de dispositivo es de clase **BASE**: una clase con activaciones cuantizadas (las Linear densas y las cabezas
absorbidas multiplican pesos Q8_0 por activaciones cuantizadas a 8 bits; los expertos enrutados corren en los kernels
A8; órdenes de reducción del dispositivo). No es idéntica bit a bit a la referencia de host. Se contrasta con el oráculo
por ids de tokens (forzados por el maestro y voraces). El reparto por capas no cambia ningún
bit: el par reproduce los ids de un nodo y el digest de los logits de la última posición. El ancho de ventana del prompt
tampoco cambia ningún bit (ventanas de 1024 y 2048 filas dan el mismo resumen), pero un prompt por lotes no es
idéntico bit a bit al prompt posición a posición: en un caso medido los ids voraces se separan tras 68 tokens generados.

La referencia de host (`src/hy4/ref/hy4_host_ref.hpp`, contrato `hy4-host-ref-v1`) ejecuta un paso de decode cada vez
sobre el modelo de texto: pesos decodificados a sus valores almacenados exactos y ensanchados a FP32; activaciones
redondeadas a BF16 donde el modelo de referencia redondea (salidas de Linear, salidas de norma, los cuatro flujos
residuales); FP32 para el router, las mezclas de hiperconexión, el pooling y las puntuaciones del indexador, el softmax
y la cabeza; el orden de reducción declarado compartido y las funciones trascendentes compartidas, sin multiplicación-suma
fusionada, índice menor en empates. `Config::tiny()` con `synthetic_weights()` recorre toda la estructura con pesos
aleatorios.

Tokenizador: inicio de secuencia 120000, fin 120025 (el token de parada), relleno 120002; los tokens añadidos
120000..120831 se reconocen enteros; el postprocesador no añade ningún token (los escribe la plantilla de chat). El pack
puede traer su propia copia de la plantilla de chat; `hy4_chat_dialect()` lee de ella el sufijo de tokens especiales y
los esfuerzos de razonamiento, y rechaza una plantilla que no los tenga.

## Cómo ejecutarlo

Compilar con `-DSTRIX_ENABLE_HIP_RUNTIME=ON -DSTRIX_FAMILY_HY4=ON -DSTRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE=ON`.

Un nodo:

```
strix-hy4-rank PART1.gguf PART2.gguf --ids-file prompt.ids --gen 128 --context 4160 --top 0 [--prompt-rows 2048] [--profile fast]
```

`--ids-file` contiene ids de tokens (separados por comas o espacios). `--prompt-rows R` ejecuta el prompt en ventanas
de R filas. Sin esa opción, con la palabra de nivel por defecto, el prompt va posición a posición. `--top 0` imprime
solo el id voraz.

Dos nodos (el rango 0 tiene el embedding y las capas [0, K); el rango 1, las capas [K, 42) y la cabeza; K vale 21 por
defecto):

```
nodo A: strix-hy4-rank PART1.gguf PART2.gguf --rank 0 --peer NODO_B --ids-file prompt.ids --gen 128 --context 4160 --prompt-rows 1024
nodo B: strix-hy4-rank PART1.gguf PART2.gguf --rank 1 --peer NODO_A --ids-file prompt.ids --gen 128 --context 4160 --prompt-rows 1024
```

`--transport usb4auto|usb4gpu|tcp` elige el enlace (por defecto `usb4auto`). `--cut K` mueve el corte. El par necesita
un ancho de ventana: `--prompt-rows R`, o una calibración `--window-costs FILAS:MS,...` (milisegundos de media ventana
por ancho, medidos en el despliegue) con la que elige el ancho de menor tiempo de canalización.

Decodificación exacta en paralelo de tensores (los dos nodos tienen el pack entero y prellenan el prompt cada uno; cada
paso de decodificación se reparte):

```
nodo A: strix-hy4-rank PART1.gguf PART2.gguf --tp-rank 0 --tp-rates 217,217 --ids-file prompt.ids --gen 128 --context 4160 --prompt-rows 1024 --top 0
nodo B: strix-hy4-rank PART1.gguf PART2.gguf --tp-rank 1 --tp-rates 217,217 --ids-file prompt.ids --gen 128 --context 4160 --prompt-rows 1024 --top 0
```

La atención se reparte por cabezas; o_proj, la FFN de la capa 0, el experto compartido y la cabeza, por filas; los
expertos enrutados, por ranura (8 + 8 de las 16). 128 all-gathers por token sobre el enlace de gather compartido del
par (`--tp-transport`, por defecto `usb4auto`) unen las partes. Cada salida repartida es un resultado entero del mismo
kernel y las uniones son concatenaciones, así que los ids son los de un nodo. Al arrancar, los dos rangos miden el
enlace y el planificador compartido imprime `tp placement:`; rechaza el par cuando lo predice más lento (`--tp-force`
lo ejecuta igualmente, `--tp-trip-us` sustituye el viaje medido).

## Medidas

Clase (ver [Clase aritmética](#clase-aritmética)): la línea de la ruta por defecto es BASE; las filas de la tabla usan la
palanca FAST `prompt_rows` y no están cualificadas; paquete Q4_K_M; nodos Strix Halo de 128 GB
cada uno; el par es el reparto por capas con corte 21 sobre USB4. Cada fila es una ejecución del runner por
configuración (frío = primer prompt tras la carga, caliente = el mismo prompt otra vez en el mismo proceso con
`--repeat 2`).

Camino por defecto (prompt posición a posición, el camino de prompt cualificado), un nodo: un prompt de 256 tokens en
15,77 s (16,2 tok/s, 61,6 ms por posición), después 32 ids voraces a 16,1 tok/s.

Prompt por lotes (`--prompt-rows`, la fila `fast`; NO cualificado: compuerta de logits código PASA / prosa FALLA, así
que solo corre si se pide de forma explícita o con `--allow-pending`, y la ejecución queda marcada
`unqualified: prompt_rows`). El par siempre ejecuta el prompt por lotes (no arranca sin un ancho de ventana), así que
todas las cifras del par son de esta clase. tok/s del prompt, 32 ids voraces tras él. Prompt sintético: una secuencia de unos 66 tokens repetida hasta la
longitud del prompt, así que las filas Engram que lee el prompt son casi todas repetidas y un prompt de texto real es más
lento que estas filas:

| tokens del prompt | un nodo, ventanas de 2048 filas (frío / caliente) | par, corte 21 (frío / caliente) |
|---|---|---|
| 4096 | 526,2 / 542,0 | 738,8 / 761,8 (ventanas de 1024 filas) |
| 16384 (el prompt de 4096 tokens cuatro veces) | sin medir | 901,8 / 909,2 (ventanas de 2048 filas) |
| 32768 (el prompt de 4096 tokens ocho veces) | sin medir | 955,3 / 957,8 (ventanas de 2048 filas) |

Con 4096 tokens el par reprodujo los ids voraces y el resumen de logits de la última posición de un nodo.

La división por capas no acelera la decodificación: para un token las dos mitades se ejecutan una tras otra y el salto
se suma. La decodificación lee unos 9,7 GiB de pesos por token, así que un nodo está cerca de su límite de ancho de banda.

Decodificación exacta en paralelo de tensores (prompt de 1024 tokens, 128 ids voraces, elección voraz en el
dispositivo, ruta de gather por el host), un nodo | par | un nodo en una secuencia: 16,52 | 19,92 | 16,51 tok/s
(mediana por paso 60,5 | 49,9 | 60,5 ms); los ids del par son los de un nodo en los dos rangos (y 300 de 300 en una
ejecución más larga). Cada rango lee cerca del 56 % de los bytes por token de un nodo; el resto del paso son los 128
gathers.

## Pruebas

| prueba | necesita pesos | comprueba |
|---|---|---|
| `strix-hy4-tokenizer-tests [PACK]` | no (primera parte opcional, solo la cabecera) | regla de división; con el pack: vocabulario, ids especiales, textos de todo el vocabulario |
| `strix-hy4-chat-tests` | no | ejemplo de la ficha del modelo, esfuerzos, razonamiento conservado, herramientas y orden de respuestas, dialecto, parser |
| `strix-hy4-host-ref-tests [PACK]` | no (pesos sintéticos) | recorrido sintético completo, determinismo, borde de ventana, respaldo del indexador, router, filas Engram |
| `strix-hy4-hip-plan-tests [PACK]` | no | tabla de etapas, plan de caché, ventanas del prompt, corte del par y selector de ventana |
| `strix-hy4-placement-tests` | no (fixture de bytes por componente) | colocación por capas sobre los bytes de componentes del pack |
| `hy4-serve` | no | opciones de servicio, configuración efectiva, admisión |

Todas corren en la CI pública de host. `STRIX_HY4_GGUF` (caché de CMake) pasa la primera parte del pack a las pruebas
que la aceptan; solo se lee su cabecera.

## Puntos abiertos

- El pack guarda la tabla Engram como un byte por valor sin escala; el modelo de referencia pasa cada fila por FP8 con
  una escala por fila. La norma siguiente elimina la escala salvo a través de su épsilon.
- El bloque de siguiente token del pack (bloque 42) no forma parte de esta pasada: todavía no hay borrador en
  dispositivo.
- Las ventanas del prompt por lotes (`--prompt-rows`) no están cualificadas: su compuerta de logits frente al prompt
  posición a posición pasó en código y falló en prosa. `--profile fast` las deja pendientes (solo corren si se piden
  de forma explícita o con `--allow-pending`), y cada ejecución que las usa imprime `unqualified: prompt_rows`. El par
  siempre ejecuta el prompt por lotes. El decode es el mismo con cualquier palabra de nivel.
