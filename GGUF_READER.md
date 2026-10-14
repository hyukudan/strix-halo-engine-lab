# GGUF reader A0.2

Pinned structural source: `ggml-org/ggml@353b63b439f27ab2cc19dac97ab1681ba6d2d084`,
`docs/gguf.md` (GGUF v3).

The `(elements per block, bytes per block)` geometry is pinned to
`ggml-org/llama.cpp@6c7a87f7e5e5cd75b8a641c3471f2dee84a6ed17`,
`gguf-py/gguf/constants.py::GGML_QUANT_SIZES`.

## Implemented scope

- magic and version v3;
- little-endian integers and floats independent of host endianness;
- scalar types, strings and nested metadata arrays;
- limits checked before allocating metadata, tensors, strings, arrays and nesting depth;
- hierarchical ASCII keys, unique keys and tensor names;
- booleans restricted to 0/1;
- `general.alignment`, zero padding, dimensions and relative/absolute offsets;
- rejection of GGML types retired or unknown in the pinned revision;
- overflow-safe byte/range calculation for current types 0–39, including
  divisibility of the first dimension by the quantization block;
- rejection of ranges outside the file or overlapping other ranges;
- array summaries without retaining huge vocabularies/lists in memory;
- JSON CLI `strix-inspect-gguf`.

## Deliberate limits

- Little-endian only. GGUF v3 permits big-endian, but the pinned specification
  acknowledges that the file does not support reliable automatic detection.
- New types introduced after the pinned revision are rejected until the table
  and its fixtures are explicitly updated.
- Future versions are not accepted by default. Updating the specification
  revision requires an explicit change with new fixtures.
- Metadata is not yet interpreted as a Qwen contract; the generic loader and
  architecture validation are separate layers.

## Build and inspect

From the repository root, using CMake, Ninja and a C++20 compiler:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
ctest --test-dir build --output-on-failure
./build/strix-inspect-gguf /path/to/MODEL.gguf
```

Inspection checks the directory and physical ranges without loading model
weights for inference. See [implementation history](IMPLEMENTATION_STATUS.md)
and [qualification status](docs/qualification-ledger.md) for the distinct
loader, architecture and execution gates.

## Security

The reader treats every GGUF as untrusted input. Counts are validated before
`reserve`, offset sums detect overflow and arrays have an aggregate budget.
Tests also run under ASan+UBSan.
