# GPU visibility probes — compatibility route and current multi-slot route

> **Superseded** for current installs by [USB4 multi-slot GPU queue integration](usb4-gpu-queue-integration.md). This page documents the older compatibility probes, which remain in the tree.

The original PR #27 probe is retained in `include/strix/hip_gpu_visibility_probe.hpp`,
`src/hip_gpu_visibility_probe.hip` and `lab/gpu_visibility`. The integrated C1
pre-copy variant from PR #28 remains in `lab/direct_gpu_stream`.

**For current USB4 multi-slot installation and testing, start with
[usb4-gpu-queue-integration.md](usb4-gpu-queue-integration.md).**
Its `strix-gpu-queue-visibility` executable defaults to eight operations per
direction and reuses the original GPU verifier in a real QUEUE/PREPARE_SLOT
transport. The older C1 executables' one-operation limit is not the limit of
that new route. All are now source-integrated, not separate ZIP prerequisites.

## Shared probe contract

`HipGpuVisibilityProbe` generates and compares every payload byte with actual
GPU kernels when built/run natively. Patterns bind connection epoch, sequence,
seed and byte position. Each verification returns a bounded 64-byte GPU result
with a fresh nonce, coverage, mismatch position/count and a consumer transform.
The CPU does not decide GPU visibility by downloading the entire payload.

A negative control changes a byte only in independent execution storage, requires
that the verifier detects it, restores it and verifies again. Imported DMA-owned
RX memory is not overwritten for that control. Retained owners, stream/device
identity checks, bounded event waits and conservative cleanup remain mandatory.
A controller fence is not remote acknowledgment or GPU cache visibility proof.

The original standalone `strix-gpu-visibility` C1 runner verifies raw RX after
its acquire copy. The PR #28 C1 observation adapter and the current multi-slot
runner verify raw RX before copying, then verify the actual consumer destination
separately. Do not combine these scopes into a claim about a hypothetical
copy-free path. The production model dispatcher is not automatically switched
or qualified by a passing commissioning report.

## Original probe build and tests

The standalone C1 target can still be built explicitly for comparison:

```sh
cmake -S lab/gpu_visibility -B build/gpu-visibility \
  -DSTRIX_ENABLE_HIP_RUNTIME=ON -DBUILD_TESTING=OFF \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build/gpu-visibility --target strix-gpu-visibility -j2
python3 -m unittest discover -s tests -p 'test_gpu_visibility*.py' -v
```

Portable probe tests use sequential HIP doubles and sanitizers; no physical
cache behavior, GPU instructions or USB4 DMA are simulated by those doubles.
Native HIP compilation, real two-host correctness and recovery campaigns remain
execution gates. Every successful experimental receipt still declares
`production_qualified=false`. An evidence parser validates supplied records,
not their authenticity or permission to enable the production backend.
