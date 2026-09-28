# Lector GGUF A0.2

Fuente estructural fijada: `ggml-org/ggml@353b63b439f27ab2cc19dac97ab1681ba6d2d084`,
`docs/gguf.md` (GGUF v3).

Geometría `(elementos por bloque, bytes por bloque)` fijada desde
`ggml-org/llama.cpp@6c7a87f7e5e5cd75b8a641c3471f2dee84a6ed17`,
`gguf-py/gguf/constants.py::GGML_QUANT_SIZES`.

## Alcance implementado

- magic y versión v3;
- enteros y floats little-endian sin depender del endianess del host;
- tipos escalares, strings y arrays anidados de metadata;
- límites previos a reservar para metadata, tensors, strings, arrays y profundidad;
- claves jerárquicas ASCII, claves y nombres de tensor únicos;
- booleanos restringidos a 0/1;
- `general.alignment`, padding cero, dimensiones y offsets relativos/absolutos;
- rechazo de tipos GGML retirados o desconocidos en la revisión fijada;
- cálculo overflow-safe de bytes/rango para los tipos 0–39 vigentes, incluida
  divisibilidad de la primera dimensión por el bloque cuantizado;
- rechazo de rangos que salen del fichero o se solapan;
- resumen de arrays sin conservar vocabularios/listas gigantes en memoria;
- CLI JSON `strix-inspect-gguf`.

## Límites deliberados

- Solo little-endian. GGUF v3 permite big-endian, pero la especificación fijada
  reconoce que el fichero no permite autodetectarlo de forma fiable.
- Los nuevos tipos posteriores a la revisión fijada se rechazan hasta actualizar
  explícitamente la tabla y sus fixtures.
- No se aceptan versiones futuras por defecto. Actualizar la revisión de la spec
  debe ser un cambio explícito con fixtures nuevos.
- La metadata no se interpreta aún como contrato Qwen; el loader genérico y la
  validación de arquitectura son capas separadas.

## Seguridad

El lector trata todo GGUF como entrada no confiable. Los contadores se validan
antes de `reserve`, las sumas de offsets detectan overflow y los arrays llevan
un presupuesto agregado. Los tests se ejecutan también con ASan+UBSan.
