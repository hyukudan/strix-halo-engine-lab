# GGUF reader A0.2

Pinned structural source: `ggml-org/ggml@353b63b439f27ab2cc19dac97ab1681ba6d2d084`,
`docs/gguf.md` (GGUF v3).

`(elements per block, bytes per block)` geometry pinned from
`ggml-org/llama.cpp@6c7a87f7e5e5cd75b8a641c3471f2dee84a6ed17`,
`gguf-py/gguf/constants.py::GGML_QUANT_SIZES`.

## Implemented scope

- magic and version v3;
- little-endian integers and floats without depending on host endianness;
- scalar types, strings and nested metadata arrays;
- limits checked before reserving for metadata, tensors, strings, arrays and depth;
- hierarchical ASCII keys, unique keys and tensor names;
- booleans restricted to 0/1;
- `general.alignment`, zero padding, dimensions and relative/absolute offsets;
- rejection of GGML types that are retired or unknown at the pinned revision;
- overflow-safe byte/range computation for the current types 0–39, including
  divisibility of the first dimension by the quantization block;
- rejection of ranges that extend past the file or overlap;
- array summaries without keeping huge vocabularies/lists in memory; top-level numeric
  arrays up to `Limits::max_retained_array_elements` keep their values when a model
  contract asks for them (default 0: summaries only);
- `strix-inspect-gguf` JSON CLI.

## Deliberate limits

- Little-endian only. GGUF v3 allows big-endian, but the pinned specification
  acknowledges that the file does not allow reliable autodetection.
- New types added after the pinned revision are rejected until the table and
  its fixtures are explicitly updated.
- Future versions are not accepted by default. Updating the spec revision must
  be an explicit change with new fixtures.
- Metadata is not yet interpreted as a Qwen contract; the generic loader and
  architecture validation are separate layers.

## Security

The reader treats every GGUF file as untrusted input. Counts are validated
before `reserve`, offset sums detect overflow and arrays share an aggregate
budget. The tests also run under ASan+UBSan.
