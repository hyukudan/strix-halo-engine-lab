# Experimental DMA-BUF USB4STREAM extension

This directory supplies actual kernel source and a project-owned UAPI for a
separate, explicitly selected DMA-BUF mode of the pinned Linux stream driver.
It is not a stock Linux ioctl interface. The existing copied read/write mode
remains the default. No driver has been installed or loaded for this work.

The exact upstream source is Linux commit
`72d3fcf802c45d00b300f25b848a93c3a2bd7c7e`:
[stream.c](https://github.com/torvalds/linux/blob/72d3fcf802c45d00b300f25b848a93c3a2bd7c7e/drivers/thunderbolt/stream.c).
Its SHA256 is
`94598012d67f3225af3b572fc7f1ee5471daa00e1c0a4752cac908b1b0d7251d`.
`stream.patch` preserves the original GPL-2.0 notice, Intel Corporation 2026
copyright and authors Alan Borzeszkowski and Mika Westerberg. The extension and
shared helper/test source are GPL-2.0-only. The syscall header is
GPL-2.0 WITH Linux-syscall-note. Exact upstream license/exception text is included
in `GPL-2.0.txt` and `Linux-syscall-note.txt`; this does not change the license of
unrelated project components.

## Source preparation and compile check

Keep the complete source, build directory and receipts outside Git. The pinned
[source archive](https://codeload.github.com/torvalds/linux/tar.gz/72d3fcf802c45d00b300f25b848a93c3a2bd7c7e)
has SHA256
`f265c0eeb342d2f697c0ede885dcca5292e89bd6100ea3265fed409c589e537d`.
Verify it before extracting. Use a fresh extraction for each application of the
patches; the preparation script checks the exact stock stream/Kconfig inputs.

```sh
python3 lab/direct_gpu_stream/kernel/apply_pinned_kernel.py --source "$source"
make -C "$source" O="$build" allnoconfig
"$source/scripts/config" --file "$build/.config" \
  -e 64BIT -e PCI -e MODULES -e CONFIGFS_FS -e USB4 -e USB4_CONFIGFS \
  -m USB4_STREAM -e USB4_GPU_STREAM
make -C "$source" O="$build" olddefconfig
grep '^CONFIG_USB4_GPU_STREAM=y$' "$build/.config"
make -C "$source" O="$build" -j4 W=1 KCFLAGS=-Werror drivers/thunderbolt/stream.o
nm "$build/drivers/thunderbolt/stream.o" | grep 'sgu_\|tbstream_gpu'
```

`source` and `build` are explicit private absolute directories; the build is out
of source. The host build needs ordinary kernel tools including flex, bison,
libelf headers/library and a supported compiler. A local build used GCC 13.3 and
private libelf headers/library extracted from the already cached ROCm 7.2.4
development image; no host packages were installed. `HOSTCFLAGS`/`HOSTLDFLAGS`
pointed at those private dependencies. Checking the enabled configuration and
actual extension symbols prevents a disabled-feature object being counted as
extension compilation. This command only compiles an object; it neither builds
a tested bootable kernel nor installs or loads a module.

The shared helper property test runs without a kernel or GPU:

```sh
cc -std=c11 -O2 -DNDEBUG -Wall -Wextra -Werror -Iinclude \
  -Ilab/direct_gpu_stream/kernel lab/direct_gpu_stream/kernel/logic_tests.c \
  -o "$private_output/logic-tests"
"$private_output/logic-tests"
```

Assertions remain active under `NDEBUG`. These tests cover randomized mapped-SG
address resolution, split frames, range/endpoint overflow, zero DMA address,
partial final payload geometry, every endpoint/epoch/import fence field,
bounded outstanding callbacks, partial completion, cancellation acknowledgement,
confirmed stop and failed path teardown. The functions are consumed by the
actual driver; the test does not supply fake kernel headers or assert physical
DMA completion. Separate host tests pin the 32/64-bit UAPI layout and ioctl sizes.

## Supported first version

`CONFIG_USB4_GPU_STREAM` defaults off and selects `DMA_SHARED_BUFFER`. An endpoint
must explicitly select its ConfigFS `gpu_direct` attribute before opening.
One open-file description owns the endpoint; duplicating that description shares
the same serialized owner. Changing mode, hop IDs, ring size, interrupt behavior
or throttling is forbidden while open or while a failed context is retained.
Copy read/write calls are rejected in DMA mode. Interrupt mode and the pinned
software connection manager are supported; firmware-managed ICM and busy polling
are rejected. They have no inferred compatibility from sharing a device name.

Open allocates disabled rings and identity metadata only. IMPORT attaches and
maps separate TX/RX DMA-BUF objects into the actual NHI DMA device. Full FD
extents, file device/inode, allocation generations, window offsets and window
lengths are exact inputs. The two directions cannot share a buffer or reservation
object. Imports are bounded to 128 MiB total, at most 65536 mapped SG entries,
and one `ring_frames * 4096` window per direction (32 through 4096 frames).
Each full RX capacity must fit a single mapped DMA segment. Addresses come only
from `sg_dma_address`, are checked against the controller DMA mask, and must fit
`dma_addr_t`. SWIOTLB bounce mappings are rejected. No CPU payload mapping or
copy is used by the extension.

The final physical frame of each window is dedicated to protocol terminal
metadata. The kernel treats it as bytes; it does not validate model or GPU commit
records. Payload starts are frame aligned; the last payload frame may be partial.
The first version is C1 per direction, contiguous and has no window wrap. A
maximum payload leaves the dedicated terminal frame available. Payload/terminal
wire sizes are checked independently from the full 4096-byte RX capacities.

## UAPI and ownership

The shared header is `include/strix/linux_direct_gpu_stream_uapi.h`. Every mutation
and event carries exact ABI version/cookie, per-open instance UUID, local/peer
UUID, route/hop IDs, peer-agreed epoch, pool binding and kernel import generation.
QUERY exposes raw identity, state and limits; it contains no GPU or hardware
qualification fields. Fresh open changes the instance UUID. New epochs must
increase for the lifetime of the ConfigFS endpoint; same-open/same-epoch import
reopen retains directional sequence history. Handles and directional sequences
are consumed monotonically when preparing an operation, including failed or
canceled attempts. Module/device recreation still requires a fresh peer-agreed
epoch and new endpoint qualification at the control layer.

The callable sequence is IMPORT, ARM, PREPARE/SUBMIT RX, peer-ready control,
then PREPARE/SUBMIT TX on the sending side. ARM may start empty rings and is not
reciprocal readiness. The caller must post the exact RX frame plan before telling
its peer to send. PREPARE does not enqueue DMA. SUBMIT adds a directional DMA
reservation fence and enqueues actual ring frames. Independent TX/RX fence
contexts preserve ordered timelines without assuming order across the rings.

EVENT is available after every frame of an operation has completed or canceled.
It reports exact expected/actual sizes, slot offsets, capacities, CRC/overrun,
protocol/length errors and controller completion time in batches of at most 64
frames. The caller must consume all batches before retiring TX or reposting RX.
A failed userspace event copy does not consume a batch. Local TX completion is
not remote transport acknowledgement or consumer completion.

RX storage remains held after DMA completion and event consumption. Following
actual GPU consumer release, RX_REPOST physically enqueues the same slots and
returns a controller repost acknowledgement. RETIRE closes the old logical
operation but retains standby DMA and its import. The next RX PREPARE stops and
drains that RX ring before changing geometry, then rearms it; the peer must wait
for the next explicit readiness message. An early standby arrival fails the
endpoint instead of silently assigning old data to a new operation.

CANCEL acknowledges a request and makes the endpoint fail closed. It does not
decrement pending callbacks or release an import. STOP calls `tb_ring_stop` on
both directions without holding the callback spinlock and waits for all callback
invocations. It also requires confirmed path disable. A failed path-disable
return retains the old path/ring identity. Only confirmed stop, removed paths
and zero pending callbacks permit RELEASE/unmap/detach. No GPU quiescence is
claimed by the STOP result; the userspace provider must independently drain GPU
events and retain export/allocation owners before release.

Partial enqueue failure retains already submitted frames and their callbacks.
A lost IMPORT or action reply retains the exact import/generation and fails the
endpoint. Close or disconnect stops before dropping endpoint paths; no implicit
rebind or suspend/resume rearm is allowed. If quiescence/path teardown cannot be
proved, the driver retains its sdev, stream, service, XDomain/domain, DMA-device,
module, rings, DMA-BUF references and callback sources. This quarantine can block
device removal or module unload and is not normal recovery or an automatic retry
path. A new opener or ConfigFS edit cannot overwrite a retained context.

## Evidence limits

The pinned actual driver object compiled with the feature enabled and strict
kernel warnings. Shared helper tests and UAPI layout tests passed on the host.
There was no module load, GPU/controller transfer, firmware change, network
change or physical reset test. Exporter/controller compatibility, external-DMA
cache visibility, GPU release/acquire behavior, reciprocal peer protocol,
completion/rearm fidelity and complete-workload performance remain target
qualification gates. The userspace provider must use actual GPU events and
separate transport ACK/consumer-credit control; no provided hash or QUERY value
can turn those missing observations into native readiness.
