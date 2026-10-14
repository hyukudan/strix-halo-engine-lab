# Offline gfx1151 compilation lab

Goal: demonstrate AMDGPU object generation for Strix Halo without the GPU
present, inspect metadata/ISA and pin the toolchain.

`probe.hip` is not a benchmark. It checks only the pipeline:

1. HIP frontend;
2. `--offload-arch=gfx1151` target;
3. AMDGPU object emission;
4. target, symbol and resource inspection;
5. aggregate disassembly through the tools in `../../tools`.

Without target hardware this does not validate execution, driver correctness,
performance, dynamic occupancy, UMA memory or stability. Successful compilation
establishes neither speed nor correct execution.

## Provisionally verified toolchain

The runtime image from external runtime A lacks the necessary C++ headers and
is not a complete development environment. The successful test uses the public
`rocm/dev-ubuntu-24.04:7.2.4` image, pinned by digest:

`sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054`

Command executed inside the container, from `lab/cross_compile`:

```sh
mkdir -p out
hipcc -std=c++20 -O3 --offload-arch=gfx1151 --cuda-device-only \
  -c probe.hip -o out/probe.gfx1151.o
```

The bundle contains target `hipv4-amdgcn-amd-amdhsa--gfx1151`. After device-object
extraction, the ELF declares AMDGPU-HSA ABI v4 and the `gfx1151` flag;
`saxpy_probe` declares wave32, 14 SGPR, 6 VGPR, 0 LDS and 0 spills.

## Verified outputs

- `out/probe.gfx1151.o`, SHA-256
  `3CE334B698C6BCA3D824637ECC7E712350308B47C3AC672CF71824A6072FC562`;
- `out/probe.gfx1151.hsaco`, 4,976 bytes, SHA-256
  `70C2BCDBAD899D22786ED065F8F4F546F676D7924F1D9E5F166426AA3CC19E2F`;
- ELF header with AMDGPU target;
- metadata declaring `amdgcn-amd-amdhsa--gfx1151`;
- resource and ISA summary.

Generated binaries in `out/` are hash-pinned artifacts, not performance results.
Bitwise reproducibility still requires a second clean build and normalized
comparison.

See [implementation history](../../IMPLEMENTATION_STATUS.md) and the
[qualification ledger](../../docs/qualification-ledger.md) for the distinction
between offline compilation, native execution and full-workload qualification.

## Further reading

See the [lab index](../README.md) for prerequisites and related experiments,
the [qualification ledger](../../docs/qualification-ledger.md) for current gates,
and [independent oracles](../../tools/oracles/README.md) for semantic checks.
