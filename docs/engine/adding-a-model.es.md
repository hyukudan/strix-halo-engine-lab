# Añadir un modelo

Versión en inglés: [adding-a-model.md](adding-a-model.md).

Registrar metadatos descriptivos de un contrato compilado ya soportado requiere un archivo de descripción y una fila
del registro. Estos archivos no añaden soporte de inferencia para otro tamaño o contrato de checkpoint. Cambiar el
contrato sigue requiriendo adaptar el motor de la familia y superar su cualificación, y una arquitectura nueva necesita
además su propio módulo de familia; las partes comunes a las que se conecta están al final.

## Contrato soportado: dos ficheros

### 1. La descripción, `models/<nombre>.toml`

Parte del fichero de la misma familia (`models/qwen3.8-flash-next.toml`, `models/deepseek-v4.1-flash.toml`,
`models/glm-5.3-flash.toml`, `models/minimax-m3.1-flash.toml`, `models/nemotron-3.5-bolt-e4b.toml`). Herramientas:

- Qwen: `strix-model-describe MODELO.toml [FICHERO.gguf]` carga una descripción y, con un GGUF, comprueba que
  coinciden.
- DeepSeek: `strix-deepseek41-model-describe --config-json CONFIG.json|--gguf MODELO.gguf --output MODELO.toml` escribe
  una descripción a partir de una configuración o de un GGUF.
- GLM: `src/glm/load/glm_model_description.hpp` (`validate_model_directory`); MiniMax:
  `include/strix/minimax/model_description.hpp`; Nemotron: `src/gemma4/model/description.hpp`.

Campos comunes a todos los ficheros: `schema = "strix-model/v1"` y un `name` único. Las demás tablas (`[source]`,
`[dims]`, atención, MoE, formatos) las lee el adaptador de la familia, que es el único código que las interpreta. El
fichero es descriptivo: no elige kernel, calendario ni interruptor y no imprime ninguna identidad. Qué se comprueba, y
contra qué, cambia según la familia; véase la lista de alcance más abajo.

`[multi_device]` opcional (lo lee el resolver de perfiles común, `include/strix/profile/multi_device.hpp`): las formas
de colocación que el modelo admite en varios dispositivos (`replica`, `layers`, `tensor`, `speculation`, con sus datos
de corte, filas, cable y drafter) y `regimes` ordenados que eligen forma por fase, filas del prompt, peticiones listas y
tier. Un modelo sin la sección resuelve `single` con cualquier número de dispositivos. El cable `reference` conserva
cada unión exacta; un cable más estrecho nunca se resuelve en EXACT. `strix run|serve --device-count N --placement FORMA
--print-effective-config` imprime la decisión y quién la ejecuta (`executed_by`). Inspeccionar y ejecutar son cosas
distintas: un lanzamiento `run` / `serve` ejecuta un solo motor, así que solo se admite cuando la decisión es `single`;
`replica` solo la ejecuta `strix-router`, que tiene los endpoints (cada worker se arranca con `--device-count 1` y
después el router con las direcciones de los workers); cualquier otra forma se rechaza con su motivo hasta que un
motor la consuma. Un régimen solo puede nombrar formas declaradas, y su `wire` solo formas que lo llevan (`layers`,
`tensor`, `speculation`). Los interruptores
locales de los motores (`STRIX_*_PF_PIPE_CUT`, ...) siguen actuando dentro del motor; la config efectiva lista los que
están puestos en `multi_device.legacy_env`. Ese inventario son datos del propio modelo: cada modelo declara los interruptores que sus motores leen bajo `[multi_device.legacy_env]` (`STRIX_NOMBRE = "forms.layers.cut"`, la clave prevista para sustituirlo) y solo se listan los del modelo elegido. Una sección con solo `legacy_env` no declara política de colocación. El `wire` de un régimen que difiere del de la forma lo sustituye en el informe (`wire`, `settings.wire`); el de la forma se muestra al lado como `declared_wire`, y un wire que cambia bits es solo informe (`wire_status`) hasta que el resolvedor compartido resuelva una palanca cualificada (`lever` se lee, no se resuelve).

El lector (`src/model_description.cpp`) acepta un subconjunto de TOML: claves simples y con puntos, tablas, tablas en
línea, arrays, cadenas de una línea, enteros y decimales, booleanos. Cualquier otra cosa es un error con línea y
columna.

### 2. La fila del registro, `models/registry.toml`

Añade una tabla `[models.<id>]` (`schema = "strix-registry/v1"` se declara una vez al principio del fichero):

| clave | contenido |
|---|---|
| `name` | igual al `name` de la descripción |
| `description` | ruta relativa al repositorio de la descripción |
| `family` | familia de motor (`qwen`, `deepseek41`, `glm5`, `minimax`, `gemma4`) |
| `cmake` | opción de CMake que compila la familia (vacío: se compila siempre) |
| `profile_tag`, `allow_pending_env` | etiqueta de la tabla de perfil y su variable allow-pending (vacías si el motor no tiene) |
| `defaults` | ficheros fuente con las tablas de interruptores y valores por defecto del motor (array, no vacío) |
| `apps` | puntos de entrada (array, no vacío) |
| `adapter` | cabecera del adaptador de la familia que valida la descripción (array) |

El registro son datos. Ningún motor lo lee y no añade soporte de inferencia. `src/model_family_registry.cpp` es otro registro,
en tiempo de ejecución (las arquitecturas GGUF que admite una familia); `models/registry.toml` solo lista lo que existe.

### Alcance de la comprobación de la descripción por familia

La descripción se comprueba contra lo que cada motor lee de verdad; nada es genérico.

- **Qwen:** la admisión de decode compara los metadatos GGUF presentes con el contrato compilado
  (`STRIX_QWEN_MODEL_DESCRIPTION_CHECK=0` la desactiva). Un fichero de descripción solo se carga cuando
  `STRIX_QWEN_MODEL_DESCRIPTION` lo nombra.
- **DeepSeek:** solo inspección y exportación (`strix-deepseek41-model-describe`); sin enganche en inferencia.
- **GLM:** la carga del modelo valida un `glm-5.3-flash.toml` desplegado y, si existe `config.json`, las claves que
  contiene; un directorio sin `config.json` sigue cargando. `STRIX_GLM_MODEL_DESCRIPTION_CHECK=0` omite la
  comprobación.
- **MiniMax:** la carga valida un `minimax-m3.1-flash.toml` desplegado y las aserciones de roles y dimensiones sobre
  las cabeceras safetensors (las comprobaciones estructurales se mantienen siempre).
  `STRIX_MM_MODEL_DESCRIPTION_CHECK=0` omite el sidecar y las aserciones.
- **Nemotron:** solo inspección y exportación; sin enganche en inferencia.

Los adaptadores comparan con nombres, dimensiones y plantillas compilados. Cambiar el nombre o una dimensión en un TOML
y añadir una fila del registro no hace que se ejecute otro tamaño.

### Comandos

Desde la raíz del repositorio, solo CPU (sin GPU ni ficheros de modelo):

```sh
cmake -S . -B build
cmake --build build --target strix-model-registry-tests strix-model-description-tests
ctest --test-dir build -R "registry|description"
```

`strix-model-registry-tests` falla cuando una fila apunta a un fichero que no existe, cuando su `name` difiere del de la
descripción, cuando dos filas comparten nombre o cuando un `models/*.toml` no tiene fila. Comprueba nombres, existencia
de archivos y cobertura; no demuestra soporte ni ejecuta los adaptadores de la familia sobre una fila nueva. Los tests
de familia son
`strix-deepseek41-model-description-tests`, `strix-minimax-model-description-tests` (`-DSTRIX_FAMILY_MINIMAX=ON`),
`strix-glm5-model-description-tests` (`-DSTRIX_FAMILY_GLM5=ON`) y `strix-gemma4-description-tests`
(`-DSTRIX_FAMILY_GEMMA4=ON`).

La puerta de host aislada solo necesita un compilador C++20:

```sh
cmake -S tests/engine_profile_host -B host-gate
cmake --build host-gate
ctest --test-dir host-gate
```

Compila los tests de perfil, valores por defecto, recibo e identidad de ruta y la comprobación de la tabla de
interruptores.

Si añades o cambias una fila de interruptor en la tabla de valores por defecto de un motor, regenera la tabla y
súbela:

```sh
host-gate/strix-optimization-table > docs/engine/optimization-table.md
```

(los generadores multi-configuración dejan el ejecutable en una subcarpeta `Release/`).
`strix-optimization-table-current` falla mientras el fichero versionado difiera del generado.

Si el cambio toca `src/minimax/`, `src/dspark_common/`, `apps/minimax_*` o `CMakeLists.txt`, el recibo de fuentes
revisado `tests/minimax_route_identity_sources.inc` deja de coincidir. No actualices los hashes para silenciar la
comprobación: audita los selectores de ruta cambiados o nuevos, actualiza la tabla de permitidos de
`tests/minimax_route_identity_tests.cpp` si hace falta, recalcula el hash FNV-1a de cada fichero listado (ignorando los
bytes CR, como hace el test) y ejecuta `strix-minimax-route-identity-tests --audit-sources`.

### Antes de dar el modelo por soportado

Las comprobaciones de host no prueban un resultado numérico. En el dispositivo:

1. Ejecuta las puertas de exactitud integradas del motor para la configuración por defecto y compara los ids de token
   completos y los digests de logits/estado con una referencia establecida de forma independiente (`AGENTS.md`).
   `tools/strix-gate` compara salidas de comandos y digests entre dos compilaciones; por sí solo no es una puerta del
   motor.
2. Revisa las líneas impresas de configuración, ruta y perfil y el recibo de ejecución ([receipts.md](receipts.md),
   esquema [receipt-v1.schema.json](receipt-v1.schema.json)): los campos de identidad y los digests deben estar
   rellenos, no vacíos.
3. En una familia con comprobación en la carga (Qwen, GLM, MiniMax), confirma que un fichero que no coincide a propósito
   se rechaza y que el interruptor OFF de la comprobación restablece la admisión anterior; la configuración, la ruta y
   el recibo impresos deben ser iguales con y sin ella.
4. Registra cada interruptor que se publique con su medición y su recibo en `docs/measured-gains.md`.

## Arquitectura nueva

Sigue la [guía de integración de motores](../adding-an-engine.md) para el adaptador y el registro de la familia.
Mantén en la familia los nombres y diseños de tensores y los enlaces de cada etapa. Extiende el mecanismo compartido
mediante un trait si necesitas una variante. Las implementaciones compartidas incluyen operadores y kernels
(`src/ops`, `src/core`, `src/cuda/core`), intercambio entre nodos (`src/xport`, `include/strix/pair`) y borradores
especulativos (`src/spec`, `include/strix/spec`).

Cuando un adaptador utilice constantes generadas, ejecuta `strix-model-gen` después de cambiar su TOML y
`strix-model-constants-tests` para comprobar que la cabecera coincide con el origen. Los enlaces comunes de grafos
y ejecutores están en `src/graph` y `src/exec`. El censo `tools/env_switch_census.py` comprueba las lecturas
directas de variables de entorno y los valores predeterminados contradictorios.
