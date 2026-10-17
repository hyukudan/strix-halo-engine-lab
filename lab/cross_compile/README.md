# Offline gfx1151 compilation lab

Goal: demonstrate that we can generate AMDGPU objects for Strix Halo without
the GPU present, inspect their metadata/ISA and pin the toolchain.

`probe.hip` is not a benchmark. It only checks the pipeline:

1. HIP frontend;
2. `--offload-arch=gfx1151` target;
3. AMDGPU object emission;
4. reading of target, symbols and resources;
5. aggregate disassembly using the tools in `../../tools`.

Without hardware it is impossible to validate execution, driver correctness,
performance, dynamic occupancy, UMA memory and stability. An object that
compiles proves neither that it is fast nor even that it executes correctly.

## Provisional verified toolchain

This lab needs a full ROCm development image that includes the C++ headers; a
runtime-only image is not a complete development environment. The verified
build uses the public image `rocm/dev-ubuntu-24.04:7.2.4`, pinned by digest:

`sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054`

Command run inside the container:

```sh
hipcc -std=c++20 -O3 --offload-arch=gfx1151 --cuda-device-only \
  -c probe.hip -o out/probe.gfx1151.o
```

The bundle contains a `hipv4-amdgcn-amd-amdhsa--gfx1151` target. After
extracting the device object, the ELF declares AMDGPU-HSA ABI v4, flag
`gfx1151`; the `saxpy_probe` kernel declares wave32, 14 SGPRs, 6 VGPRs, 0 LDS
and 0 spills.

## Verified outputs

- `out/probe.gfx1151.o`, SHA-256
  `3CE334B698C6BCA3D824637ECC7E712350308B47C3AC672CF71824A6072FC562`;
- `out/probe.gfx1151.hsaco`, 4,976 bytes, SHA-256
  `70C2BCDBAD899D22786ED065F8F4F546F676D7924F1D9E5F166426AA3CC19E2F`;
- ELF header with an AMDGPU target;
- metadata declaring `amdgcn-amd-amdhsa--gfx1151`;
- resource and ISA summary.

The binaries generated in `out/` are hash-pinned artifacts and are not
considered performance results. Bit-for-bit reproducibility still requires a
second clean build and a normalized comparison.
