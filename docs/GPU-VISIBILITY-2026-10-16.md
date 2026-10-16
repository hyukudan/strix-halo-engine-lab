# GPU visibility: native commissioning and remaining cluster gates

## Status

This change adds a **native two-host commissioning executable**, GPU producer and
GPU byte-verification kernels, negative controls and a paired-report validator.
It integrates with the existing `LinuxDirectGpuStreamCommissioning`,
`HipDirectGpuStreamGpu`, retained DMA-BUF exporter and bounded control channel.
It does not bypass the existing production qualification boundary and does not
change the model decoder's active backend. Every report says
`production_qualified: false`, including successful experimental reports.

Driver counterpart: patch 0007 in the USB4 driver recovery PR rejects failed
external fences and retains pending dependencies across blocking/nonblocking
retries and reservation pruning. Driver capability, successful local DMA,
remote receipt, GPU visibility, and completion of GPU consumption are different
facts. None may be substituted for another.

## Exact path exercised

`lab/gpu_visibility/visibility_main.hip` constructs the actual retained HIP
exporter, pool allocator, GPU helper, Linux stream provider and control session.
The producer kernel fills the execution source with a pattern keyed by epoch,
sequence, seed and byte position. The existing helper packs that source into
its imported TX window and publishes the terminal on its owned stream, using
system-fence HIP events. The receiver's existing acquire path copies imported
RX into its separately owned execution buffer and unpacks it into the registered
consumer destination.

The existing commissioning callback then executes two GPU checks before it can
return consumer credit or repost RX: an ordinary-load check of the imported RX
payload and an ordinary-load check of the unpacked execution destination. Every
byte is checked, and an integer consumer transform is computed. The CPU reads
only a fixed **64-byte result per verification**, not a payload copy. Separate
32-byte terminal/control metadata and descriptor/event processing are unchanged.
Negative controls use additional small result copies; 64 bytes is not the total
metadata of the whole session.

The raw RX check occurs **after the existing acquisition copy**. A passing run
would support that exact current engine path, not a hypothetical direct
external-DMA-to-kernel path that omits the acquisition copy. GPU-to-GPU copies
are still present. This is not a claim of copy-free GPU RDMA.

For every receive, the test flips one byte of the separately owned execution
destination using a GPU kernel, requires the GPU verifier to detect exactly that
byte, restores it, and verifies the complete payload again. It never writes an
imported/posted RX window during this control. This is a test-only destination,
not a buffer concurrently consumed by the model. A mismatch or incomplete GPU
result blocks credit/repost rather than silently using a CPU fallback.

The result contains epoch, sequence, a monotonically allocated invocation nonce,
checked-byte count, mismatch count, first mismatch, and two diagnostic digests.
A digest match does not replace the per-byte mismatch count. A stale result nonce
or incomplete coverage is an error. Epoch is included in the new byte pattern so
restarting sequence numbering is not treated as continuation of an old session.
The shape fingerprint includes `gpu-byte-check-v2`; do not mix this executable
with the older CPU-readback commissioning peer.

## Lifetime, synchronization and bounds

The probe retains the execution owner and the helper's stream lifetime until its
work actually completes. It validates resident/nonmanaged device allocation
extents, active device, non-overlap, bounded total bytes and noncaptured stream
before launching. Events use `hipEventDisableTiming`, **not**
`hipEventDisableSystemFence`. The wait has a bounded elapsed-time budget and
cooperative pause. A timeout, failed record or failed query is not quiescence.

If an event cannot prove completion, cleanup queries the original retained
stream. If that cannot prove quiescence, storage/owners remain retained. A native
free with an uncertain outcome is sticky: a later close must neither retry a
possibly reused handle nor discard the retained owner. There is no automatic
GPU reset, process-kill workaround, IOMMU change, or controller reset here.

The underlying native helper remains **one operation per direction**. The new
runner supports sequential reuse or paired TX+RX duplex, at most 128 transfers
and 128 MiB aggregate payload. Paired duplex is not eight operations per
direction. A previous raw-buffer read is recorded during reuse; this does not
assert that every cache line was still resident when DMA subsequently occurred.
The optional producer/consumer delays are explicit **host-side delays** before
producer launch or consumer verification, not claims of a stalled GPU kernel.

## Native build and use

Use a complete checkout and an actual ROCm HIP toolchain for gfx1151:

```sh
cmake -S lab/gpu_visibility -B build/gpu-visibility \
  -DSTRIX_ENABLE_HIP_RUNTIME=ON -DBUILD_TESTING=OFF \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build/gpu-visibility --target strix-gpu-visibility -j2
```

This separate lab target does not alter the root default build or enable a new
production path. It links the same `strix_hip_runtime` as the existing native
commissioning tool. The root engine and all its dependencies must be present.

The CLI preserves the identity-oriented commissioning arguments:
`--experimental-commissioning`, `--campaign-id`, `--run-id`, `--dma-device`,
`--local-endpoint`, `--peer-endpoint`, `--gpu-uuid`, `--gpu-runtime-version`,
`--gpu-driver-version`, `--kernel-release`, `--driver-build-id`, `--physical-link`,
`--epoch`, `--request-id`, `--payload-bytes`, `--control-port`, `--role`
(`sender`/`receiver`), and `--control-mode` (`listen`/`connect`). GPU device defaults
to zero; ring frames default to 32 and must match the configured stream. Use the
actual observed UUIDs/builds, not invented identity strings. Both peers must use
the same campaign/run, epoch, request, seed, payload and schedule, with opposite
roles and exchanged endpoint identities. Control reachability follows the
existing bounded commissioning channel; this tool does not configure networking.

`--check-config` validates arguments without opening a GPU, stream or socket and
emits an **unexecuted** configuration record. It is not an evidence report.
Use `--iterations 4` for reuse, add `--duplex-pairs` for paired duplex, and use
`--producer-delay-ms` / `--consumer-delay-ms` (0..1000) to exercise withholding
credit. The overall timeout must exceed the delays. Run the two actual hosts,
retaining their respective JSON stdout reports. No tests or hardware operations
are scheduled automatically by committing/building this code.

```sh
python3 tools/qualification/gpu_visibility_evidence.py \
  --pair host-a.json host-b.json
```

The reader rejects duplicate JSON keys, nonfinite values, booleans as integers,
oversized/nonregular files, missing byte coverage, wrong epochs, repeated result
nonces, mismatched peers/directions, failed cleanup and claimed production
qualification. It compares raw/copied GPU diagnostics and identifies covered
reuse/duplex/delay cases. The reported payload rate includes verification,
negative controls, control messages and deliberate delays. It is a commissioning
rate, not peak link bandwidth.

Reports are **operator-supplied and unattested**. Merely setting `native_gpu` in
JSON is not authenticity evidence; synthetically consistent fixtures can only
test the reader. Successful structural validation never activates a backend or
authorizes production. Review the real binary/source/driver provenance and
physical campaign independently.

## Validation performed for this change

22 local tests passed with GCC and Clang. Ten execute the complete native probe
function bodies through a sequential HIP API double, exercising full-byte and
consumer-result checking, cross-epoch mismatch, single-byte corruption,
nonoverlapping bounds, foreign device, stream capture, stale result, failed event
record, timeout and ambiguous cleanup. The ambiguous-cleanup case deliberately
exits with retained state; it is not a successful reclamation/leak-freedom test.
Other normal cleanup paths use ASan leak detection. C/C++ tests also use UBSan.

The remaining tests cover paired-report rejection rules and compile/execute the
bounded CLI parser, including duplicate options and inconsistent schedules.
These tests do **not** execute GPU instructions, external DMA, real HIP events,
cache invalidation, the native two-host main, or the whole engine test suite.
The native executable has **not been compiled or run** here: no ROCm HIP
compiler/device was available in this execution environment. Its actual build
and physical commissioning remain explicit acceptance gates.

```sh
python3 -m unittest discover -s tests -p 'test_gpu_visibility*.py' -v
CXX=clang++ python3 -m unittest discover -s tests -p 'test_gpu_visibility*.py' -v
```

## Remaining work, without stale completion claims

1. Compile/reconstruct the full patched pinned kernel and compatible modules;
   do not mix old `tb_nhi` consumers with the recovery header layout. Build this
   native tool with the actual engine/ROCm toolchain and run both physical hosts.
2. Establish the exporter/importer cache contract for the precise HIP allocation,
   `hipMemGetHandleForAddressRange(..., hipMemRangeHandleTypeDmaBufFd, 0)` export,
   NHI mapping and GPU-consumer path. The new tests can detect bad observations;
   they are not a universal cache-acquire implementation. Do not reinterpret a
   Linux sync_file FD as an opaque HIP external-semaphore handle.
3. Integrate more than one operation per direction into the native engine only
   with per-slot terminal geometry, disjoint execution destinations, retained
   owners and consumer-credit demultiplexing. Driver support for eight slots
   alone is not that implementation. The current C1 path remains unchanged.
4. Exercise unplug, suspend/resume, reset rejection with sibling streams,
   reconnect/new epochs and actual GPU work. PR #2's reset is deliberately
   exclusive/administrative, not automatic transparent multi-consumer recovery.
5. Unprovable disconnected controller quarantines remain retained. Reclamation
   requires an actual hardware quiescence/revocation proof, not elapsed time.
6. Preserve the property-notification revert until a captured discovery trace
   establishes a minimal corrective change. Diagnose lane training separately;
   none of these changes claims the problematic link now negotiates 40 Gb/s.

## Primary contracts

- Linux DMA-BUF/fences and the status/error distinction:
  https://docs.kernel.org/driver-api/dma-buf.html
- AMD HIP runtime memory and event API documentation:
  https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api.html
- AMD ROCr export/import and scope guarantees:
  https://rocm.docs.amd.com/projects/ROCR-Runtime/en/latest/api-reference/api.html

A successful export or local controller fence is not, by itself, evidence of
coherence through every other importing device. This document records the
project-owned experiment and its limits, not external benchmark comparisons.
