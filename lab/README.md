# Experiment labs

These labs isolate compiler, operator, memory and transport gates before
promotion into the engine. Run each lab's commands from the repository root
unless its README explicitly gives another working directory. A passing
fixture, emitted instruction or transport microbenchmark qualifies only its
declared scope. Current promotion status is recorded in the
[qualification ledger](../docs/qualification-ledger.md).

## Choose a lab

| Goal | Entry point | Execution requirements |
| --- | --- | --- |
| Offline AMDGPU object generation and inspection | [Cross compilation](cross_compile/README.md) | Pinned HIP development toolchain; no GPU needed for compilation |
| Basic operator correctness and wave variants | [Basic kernels](basic_kernels/README.md) | HIP compiler and native `gfx1151` execution |
| Packed low-row projections | [Packed projection](packed_projection/README.md) | HIP compiler and native `gfx1151` execution |
| Scalar low-precision contracts and instruction probes | [Low precision](low_precision/README.md) | Portable C++20; HIP toolchain only for offline probes |
| Experimental integer projections | [Integer projection](integer_projection/README.md) | Portable reference; native HIP requirements described per path |
| Device routing and expert scheduling | [MoE routing](moe_routing/README.md), [FP32 experts](moe_experts/README.md) | HIP development environment and native execution |
| Packed expert formats | [IQ4_XS experts](moe_iq4_xs/README.md), [Q8_0 experts](moe_q8_0/README.md) | HIP development environment and native execution |
| Recurrent-state arithmetic and scheduling | [GDN](recurrent_gdn/README.md) | HIP development environment and native execution |
| Bounded lookup and direct-read behavior | [Sparse lookup](sparse_lookup/README.md) | Host-side Linux I/O; local fixture files |
| UMA placement and registration | [UMA residency](uma_residency/README.md) | Linux host tests; optional HIP tests on the target |
| Network-stack and copied-stream controls | [Transport](transport/README.md) | Linux hosts; peer connectivity for paired tests |
| Direct GPU-owned transport and kernel ABI | [Commissioning](direct_gpu_stream/COMMISSIONING.md), [kernel integration](direct_gpu_stream/kernel/README.md) | Exact kernel/toolchain and explicit native commissioning gates |
| Paired model recording and numerical evaluation | [Native numerical campaign](native_campaign/numerical/README.md) | Private model/prompts and the campaign's documented runtime gates |

## Build prerequisites

CMake labs require CMake 3.24 or newer, Ninja when using `-G Ninja`, and a
C++20 compiler. Host-only labs do not require the final GPU. HIP labs additionally
need the HIP development compiler, headers and libraries; a runtime-only image
is insufficient. Confirm the actual compiler and install layout rather than
assuming `/opt/rocm`. The [basic-kernel guide](basic_kernels/README.md) shows
explicit CMake compiler selection for a distribution layout.

Use separate build directories for different compilers, wave modes or runtime
configurations. The top-level portable build and standalone labs have different
options; consult the individual README before passing a flag. Successful
configuration or compilation does not demonstrate GPU execution or model quality.

## Record and interpret results

Keep benchmark records, models, private prompts, host inventories and generated
compiler/result artifacts outside Git. Bind each private result to the exact
source revision, compiler/runtime, device, workload, shape, numerical plan and
repetition conditions. Preserve failed tests and negative outcomes. Use the
[publication policy](../AGENTS.md) and [source ledger](../SOURCES_LEDGER.md) to
retain accurate provenance without publishing private comparative inputs.

For independent mathematical checks, begin with
[the semantic oracles](../tools/oracles/README.md). For portable build commands
and implementation history, see [implementation status](../IMPLEMENTATION_STATUS.md).
