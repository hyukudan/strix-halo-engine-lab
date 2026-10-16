# Native GPU visibility test integration

This extends the existing `strix-direct-gpu-commission` executable, its real
Linux driver provider, HIP export/import owner and ACK/consumer-credit protocol.
It is not a standalone replacement transport or a production qualification
provider. Select the new profile explicitly with `--gpu-visibility`.

## What the integrated path does

Before IMPORT/ARM, each process fills its still-unimported RX allocation with an
old pattern, reads it from a GPU kernel, deliberately corrupts one byte, checks
that the GPU verifier detects exactly that byte, restores it and reads it again.
This is a positive/negative verifier self-test and an initial GPU pre-read, not
proof that a particular cache physically retained the data. The existing native
pool allocator's memory policy remains unchanged. It is never legal to repeat
these writes after RX_REPOST has handed ownership back to DMA.

The sender produces the normal per-sequence pattern on the helper-owned HIP
stream. In this profile the helper's release/acquire/consumer events explicitly
request `hipEventReleaseToSystem` in addition to disabled timing. The normal
producer-release and control handshakes still precede controller submission.
This event policy is part of the test profile, not a generic external-DMA cache
invalidation implementation or a claim about arbitrary allocation types.

After every exact RX controller frame completes, an observation adapter invokes
normal GPU loads on the **original DMA destination, before the existing acquire
copy**. It compares every payload byte with the expected sequence/seed pattern.
Only then does the ordinary provider acquire/copy the payload, check its terminal
record and send the transport ACK. After unpacking, another GPU kernel verifies
the complete actual consumer destination. Its own system-scope completion event
must finish before the existing facade can publish consumed credit or RX_REPOST.
A mismatch, failed API call, late result or missing byte blocks that progress.

Each verifier downloads only its **64-byte result**, never the payload. The
existing 32-byte terminal metadata transfer is unchanged. The GPU result echoes
epoch, sequence, stage and a fresh invocation nonce, and records checked bytes,
full mismatch count and first mismatch. Zeroed, partial, stale, duplicated and
wrong-stage results cannot pass. `observed_payload_sha256` deliberately remains
null: an expected CPU-generated hash is not an observed GPU payload hash.

All diagnostic allocations are included in the retained execution owner; all
kernels/copies/events use the existing owned stream. Unknown completion retains
resources. Deliberate producer/consumer delays are bounded host-side stress
controls; they do not authorize buffer reuse or simulate GPU completion.

## Scope

This integration makes the **first real two-host visibility test runnable** on
the engine's existing transport path. It supports one operation per direction,
sequential reuse, alternating directions and one TX+RX paired duplex. It does
**not** implement an eight-outstanding-operations engine scheduler, run a model,
qualify model fidelity, or enable the production backend. The driver supporting
eight slots is not evidence that this executable uses eight. `--queue-depth 8`
is rejected rather than silently ignored.

Repeated transfers keep one import and ARM, with a changing pattern sequence.
The GPU reads from each generation precede the next reuse. The new profile
rejects changed payload frame counts per direction because legacy RX standby
requires unchanged geometry. Tail lengths within the same frame count are
allowed. Use a new process/import with a strictly newer shared epoch to test a
different geometry; do not overwrite standby memory to prepare the next case.

The profile and stress delays enter the existing descriptor fingerprint. Both
peers must select the same profile/delays and exactly matching transfer schedule.

## Build (no Actions)

Use the same engine branch/commit and driver ABI on both hosts. The driver needs
the recovery/dependency series through patch 0007, deliberately built, installed
and booted by the operator. Its internal NHI changes require compatible rebuilt
modules; do not use the old lab kernel snapshot as a substitute for that series.
Retain a known-good boot entry and keep the IOMMU enabled.

On a host with the ROCm development toolchain already configured:

```sh
cmake -S . -B build-usb4 -G Ninja \
  -DSTRIX_ENABLE_HIP_RUNTIME=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-usb4 --target strix-direct-gpu-commission strix-direct-gpu-observe-driver -j2
build-usb4/strix-direct-gpu-commission --help
```

The native target is already wired into the root CMake build. The new headers
are included by its existing `.hip` source; no generated patch/build step or
workflow change is required. A full HIP build was not possible in the editing
environment. The native compiler/link step above is still required on the hosts.

## Prepare each host from actual observations

Maintain exclusive ConfigFS ownership: configure the intended endpoint's
`gpu_direct` mode and interrupt mode, and match ring/HopID geometry. Do not open
an arbitrary stock/copied-mode stream to discover whether it supports this ABI.
See `COMMISSIONING.md` for those prerequisites and the two observation tools.
Keep every observation, configuration and run receipt outside Git.

On **each host**, with local values for the known configured device and the GNU
build ID of the deliberately installed module:

```sh
mkdir -p "$HOME/usb4-evidence"
build-usb4/strix-direct-gpu-commission --observe-gpu 0 > "$HOME/usb4-evidence/gpu.json"
build-usb4/strix-direct-gpu-observe-driver --configured-gpu-direct \
  --observe-driver /dev/tbstreamX \
  --kernel-release "$(uname -r)" --driver-build-id ACTUAL_MODULE_GNU_BUILD_ID \
  > "$HOME/usb4-evidence/driver.json"
```

`/dev/tbstreamX` and `ACTUAL_MODULE_GNU_BUILD_ID` are placeholders, not inferred
values. The observer checks the actual loaded identity. Its per-open device
instance must not be reused by the later process; prepare copies only stable
endpoint/kernel/GPU expectations, and the native runner binds its own new FD.

Choose a **new increasing epoch shared by both hosts** and a common campaign,
run ID, request ID, seed and physical-link label. On host A:

```sh
python3 lab/direct_gpu_stream/visibility_campaign.py prepare \
  --gpu-observation "$HOME/usb4-evidence/gpu.json" \
  --driver-observation "$HOME/usb4-evidence/driver.json" \
  --output "$HOME/usb4-evidence/host.json" \
  --campaign-id usb4-vis --run-id reuse-01 --role sender \
  --physical-link link-0 --epoch NEW_SHARED_EPOCH --request-id 1 \
  --control-port 39001 --payload-bytes 65537 --iterations 32
```

On host B use its own observation files, `--role receiver`, the same shared
fields, and its forwarded local port. `NEW_SHARED_EPOCH` is a placeholder that
must be replaced by an actual integer greater than previous sessions. The helper
rejects missing, retained, active or incompatible observations. It never changes
driver configuration, opens sockets or executes GPU work while preparing JSON.

On each host, validate without GPU/DMA/socket work:

```sh
python3 lab/direct_gpu_stream/visibility_campaign.py plan \
  --binary build-usb4/strix-direct-gpu-commission \
  --config "$HOME/usb4-evidence/host.json"
```

## Run and compare

The control adapter accepts **loopback only**. Establish authenticated forwarding
on B to A before the run, for example with an operator-managed SSH session:

```sh
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:39001:127.0.0.1:39001 USER@HOST_A
```

No script here creates that session or assumes its credentials. Do not expose
the control port over unauthenticated LAN forwarding. Start A (sender/listener),
then B (receiver/connector), in separate terminals:

```sh
python3 lab/direct_gpu_stream/visibility_campaign.py run \
  --binary build-usb4/strix-direct-gpu-commission \
  --config "$HOME/usb4-evidence/host.json" \
  --output "$HOME/usb4-evidence/result.json" --configured-gpu-direct
```

Output files use exclusive creation and are never overwritten. A native failure
receipt is kept; a killed process is not a DMA/GPU quiescence proof. After both
processes finish, copy the two receipts to one machine under distinct filenames:

```sh
python3 lab/direct_gpu_stream/visibility_campaign.py compare host-a-result.json host-b-result.json
```

The comparison rejects partial sessions, retained cleanup, wrong peer/epoch,
missing raw/consumer GPU results, duplicate nonces, incomplete byte counts and
missing ACK/consumer credit. It reports record agreement for this run, never
creates a production qualification record. Keep both original receipts and the
native build identities; JSON agreement is not a substitute for the native run.

After a passing one-way reuse test, prepare **new files and a newer epoch** for
`--alternate-directions`, then `--duplex-pairs`. Use an even iteration count for
duplex. Repeat with `--producer-delay-ms 50 --consumer-delay-ms 50` on both hosts.
For each new session choose lengths including 1, 255, 256, 257, 4095, 4096, 4097
and larger multi-frame values. Keep frame count fixed inside each reused receive
direction. On failure stop the campaign, preserve evidence and establish safe
controller/GPU cleanup before restarting; do not loop automatic resets.

## Local validation and remaining hardware work

`python3 lab/direct_gpu_stream/test_visibility.py` runs portable tests. The
actual new kernel function bodies, observer adapter and result rules execute
against CPU API doubles with ASan/UBSan. The complete modified native translation
unit is type/syntax checked against explicit interface declarations; its real
parser and pre-import self-test execute with those doubles. This is **not** a
HIP device compilation, cache simulation, pinned-kernel build or hardware run.

The first delivery passed 16 tests with GCC and Clang, including payload tails,
grid-stride coverage, one-byte corruption, stale results, omitted verifier work,
API errors, unknown-completion retention, hook ordering, parser rejection and
strict receipt comparison. Run them again after any integration change.

Still requiring actual evidence: GPU-visible correctness of the exact exported
allocation/importer on both hosts, native compiled-kernel behavior, USB4 bandwidth,
unplug/suspend/reconnect, and any different memory/export policy. Eight-slot
engine pipelining and production model integration remain separate from this
commissioning executable. No IOMMU disable, cache-bypass load, unconditional NHI
reset or synthetic successful qualification is part of this change.
