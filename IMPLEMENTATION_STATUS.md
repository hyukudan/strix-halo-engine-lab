# Estado de implementación

## Slice A0.1 — núcleo portable

Implementado:

- proyecto CMake C++20 sin dependencias externas;
- contrato de shapes Qwen3.8-Flash-Next fijado desde el config oficial;
- planificador de memoria con aritmética comprobada contra overflow;
- desglose KV base, MTP, indexer provisional, GDN FP32, rollback, modelo y cachés;
- decisión de admisión desde MemTotal, baseline y reserva explícita;
- CLI JSON `strix-memory-plan`, preparada para alimentar la futura WebUI;
- oráculo top-k candidato con empates deterministas y rechazo de NaN;
- pruebas CTest en Linux/WSL con warnings-as-errors.

No implementado o deliberadamente bloqueado:

- semántica QSA canónica, n-gram/PLE, hyper-connections y GDN: esperan contratos
  oficiales/oráculos antes de crear kernels;
- parser GGUF y mapa de tensores;
- allocator/residency HIP y kernels `gfx1151`;
- MTP sampled;
- servidor y panel administrativo.

## Reproducción

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
ctest --test-dir build --output-on-failure
./build/strix-memory-plan --tokens 32768 --sessions 1 \
  --resident-gib 68 --baseline-gib 2 --reserve-gib 18
```

El indexer es una estimación provisional derivada del config. Se puede excluir
con `--no-indexer`. `--rollback-full` permite cuantificar por qué copiar el
estado GDN completo por borrador no escala; el camino objetivo será snapshot
mínimo más recompute si el contrato lo permite.

## Slice A0.2 — inspección GGUF

Implementado en `include/strix/gguf_reader.hpp` y `src/gguf_reader.cpp`: lector
GGUF v3 little-endian defensivo y CLI `strix-inspect-gguf`. Valida metadata,
arrays, directorio tensorial, alineación y offsets sin tocar los datos de pesos.
Las limitaciones y la revisión canónica se documentan en `GGUF_READER.md`.

Validación actual:

- build `RelWithDebInfo` con `-Werror`;
- fixtures válidos y corruptos en CTest;
- build Debug con AddressSanitizer y UndefinedBehaviorSanitizer;
- 100 % de las pruebas actuales superadas en ambos builds.
