# Two-rank communicator

Status: first layer of the distributed runtime. `include/strix/dist/communicator.hpp`, `src/dist_communicator.cpp`
(library `strix_dist`), the probe `strix-dist-tcp-probe` and the two-rank fixtures in
`tests/hip_ds41_formats_tests.hip`.

## Pieces

- **`Transport`**: moves whole messages between the two ranks, in order. Backends:
  - `memory_pair()`: both ranks in one process (the emulated two-node setup: two host threads, one GPU, one
    stream per rank);
  - `tcp_listen` / `tcp_connect`: one TCP connection with `TCP_NODELAY`, bounded waits, for bring-up and
    debugging between hosts; a persistent sender thread does the send half of an exchange;
  - `usb4_host(devices, weights)`: the host path of USB4 streams. The message is split over the listed streams
    (in the same order on both ranks) at 64-byte boundaries, in proportion to integer weights (equal by default),
    so the trailer travels in the last part; each stream has a persistent thread that writes its part and then
    reads the peer's part;
  - `usb4_host_auto(options, &topology)`: the same transport over every live stream, found and paired at startup
    (see *Automatic USB4 pairing*). This is the entry point for runs; the explicit list is for bring-up.
  Each stream has a persistent writer and a persistent reader, so a stream's part is written and read at the
  same time: messages of any size go through (the first version wrote a whole part before reading, and two
  ranks doing that deadlocked once a part exceeded what the two rings buffer, about 8 MB: a 42 MB message on
  four streams). Messages shorter than `min_part_bytes` per stream use fewer streams; the striping order takes
  the first stream of every cable, then the second, so a short message still spreads over the cables.
- **Failures**: every transfer waits in `poll()` slices with a progress deadline (`io_timeout`, 120 s by
  default, `$STRIX_DIST_USB4_TIMEOUT_MS`) and an abort flag. The first failing stream (ENODEV after an unplug,
  ETIMEDOUT from the driver, the peer's CLOSE, the deadline) is reported as the cause, the other transfers stop,
  every stream is closed (the peer reads CLOSE on its live streams and fails at once instead of waiting) and the
  transport refuses later calls. `strix-dist-transport-tests` checks this over Unix socket pairs: 42 MB
  exchanges, a peer that dies mid-exchange (reported in ~50 ms), a silent peer (the deadline).
  `DirectTransport` declares the device-buffer path (registered windows, submit, repost, completion by sequence).
- **`Transport::exchange(out, in)`**: both ranks send and receive at once; backends use persistent threads, never a
  thread per call.
- **Message format**: the payload, then a 64-byte trailer (`magic, epoch, plan, sequence, request, bytes, rank,
  tag, layer, crc32`). The trailer is the last thing on the wire, so its sequence number marks the payload before it
  as complete; the receiver checks every field and the CRC-32 of the payload. Sizes are never sent: both ranks
  derive them from shared data (rows, routes, plan).
- **`Communicator::exchange(stream, local, remote, bytes, tag)`**: copies the device buffer to pinned memory after
  its producer, waits for the copy on the CPU, sends it with its trailer while receiving the peer's message,
  checks the peer's trailer and CRC, and enqueues the copy of the payload into the device buffer `remote`. Both
  ranks call the same collectives in the same order.
- **`start(stream, local, bytes, tag)` / `finish(stream, remote)`**: the same exchange in two halves. `start`
  enqueues the copy out and returns; the communicator's progress thread waits for the copy on the CPU, then
  sends and receives. Work enqueued between the two calls runs on the GPU while the message travels. `finish`
  waits for the progress thread, checks the peer's message and enqueues the copy in. `exchange` is `start`
  followed by `finish`.

## Synchronization contract

Bytes of a device buffer leave only after the work producing them is complete as seen by the CPU, and a receive
buffer is reused only after the work reading it is complete as seen by the CPU. No backend may rely on implicit
device fences: work submitted through HIP user queues publishes none to other DMA engines. The host-staged
backends satisfy this with the stream synchronization at the start of `exchange` and the synchronous copy back;
a device-buffer backend must check producer and consumer completion on the CPU before `submit` and `repost`.

## Automatic USB4 pairing

Cabling is not assumed: any number of cables, straight or crossed, 20 or 40 Gb/s links, and stream device
numbers that change between boots. `usb4_host_auto` builds the pairing at startup:

1. **Candidates.** The stream entries of the layout file (`$STRIX_DIST_USB4_LAYOUT`, default
   `/run/usb4stream-layout.json`): keys `<service>/<stream>` with a `device`. A stream is opened only if the
   service is present under `/sys/bus/thunderbolt/devices` (a peer is attached on that port on this boot; a stream configured on a domain
   whose controller is down is never opened). Link rate and lanes (both directions) and the peer's identity
   come from the XDomain's sysfs entries. sysfs decides: the layout is written at one moment and can be stale both
   ways (streams of a link that went down still listed as openable, or live streams listed as unattached by a
   layout written during boot). `Usb4AutoOptions::exclude` / `$STRIX_DIST_USB4_EXCLUDE` (device files or keys)
   keeps streams out explicitly. Streams are opened non-blocking, so a stream whose link is down fails at once
   instead of waiting for it (the explicit-list `usb4_host` opens the same way, then switches to blocking I/O).
   The plan reads after the handshake wait at most 20 s (poll), so a peer or link lost during pairing fails it
   instead of blocking it.
2. **Handshake on every opened stream.** Messages are 64-byte units: hello (protocol version, rank, process
   nonce, the sender's stream index, device number and open generation, host name, CRC-32C) and acknowledgement
   (the same, plus the peer's nonce, stream index and generation, which proves both directions). With the stream
   module in use, frames written before the peer has opened its end can hold back later frames of that
   direction, so rank 0 never writes first:
   - rank 1 sends a hello every 250 ms; without an answer within 1.5 s of opening it closes the stream, waits
     0.5 s and opens it again (fresh rings, next generation);
   - rank 0 answers each new hello (nonce and generation) with its hello and an acknowledgement in one write,
     and reopens at once when it reads the peer's CLOSE (its own CLOSE then finds rank 1's end closed);
   - rank 1 acknowledges rank 0's hello.
   Every unit written is consumed by the other side inside the handshake, so the streams are clean afterwards.
   Units that are not valid handshake units (left over from an earlier session) are skipped and counted. After
   the first stream pairs, the others have a grace period (5 s by default).
3. **Rejection with reasons.** Streams that do not complete are closed and listed in the receipt (no answer,
   one direction only, reopen count, last error). Streams answered by a different peer process, or by a process
   with the same rank, fail the pairing: two jobs or two rank-0 processes are on the link.
4. **Order and plan.** Accepted streams are ordered by rank 0's stream index. Rank 0 completes a stream only on
   rank 1's acknowledgement, so its set is a subset of rank 1's; rank 0 sends its set on its first stream,
   rank 1 finds it, keeps exactly that set (closing any extra stream) and echoes it.
5. **Per-cable capacity.** The streams of one cable (one XDomain) are measured together with a short exchange
   in both directions (16 MiB by default, a 64-byte barrier on the cable before each of five repetitions, the
   median after a warm-up, contents checked). The measured
   rate in Mb/s, shared equally among the cable's streams, becomes the integer striping weight. Rank 0's
   measurement decides and travels in a second plan. With `probe_bytes = 0` the weights come from the link rate.
6. **Receipt.** `Usb4Topology::receipt()` lists, per stream, the local key and device, domain, port, cable label,
   link rate, the peer's device and index, the cable's measured rate and the weight, and every candidate not
   used with its reason. The two-node fixtures print it at startup.

Fewer live cables or streams simply give a smaller plan. TCP is never chosen automatically; it stays an explicit
debugging transport.

What the stream service could expose to make this simpler (none of it is required):
- the peer's stream identity per stream (host UUID, service and stream name, device number) once the rings are
  up, which would make the hello round confirmatory only;
- a per-stream state (rings started, peer open, peer closed) readable without opening the device;
- the XDomain link state and lanes in the layout file, refreshed on hot-plug, with a generation counter;
- an rx polling mode (`rx_napi`/`spin_us`) as a per-open option, so a job can choose it without module
  parameters.

## Expert parallelism of the deepseek41 MoE layer

Each rank runs the router and the shared expert on the same inputs (replicated, identical bits), computes the
routed experts it owns (ids below or above the split, `restrict_routes`), forms the sum of its half
(`enqueue_half_sum`), exchanges it, and forms `(shared + low) + high` (`enqueue_canonical_sum`). This is the
single-node combine operation by operation, so both ranks produce the single-node output bitwise. The raw slot
vectors can be exchanged instead (`TagKind::raw_slots`) when the experts are not split by id.

Overlap: a rank enqueues the router, its routed experts and its half sum, starts the exchange, enqueues the shared
expert (replicated, independent of the exchange) and then finishes the exchange and forms the canonical sum. The
shared expert's GPU time hides the transfer. The order of operations inside every sum is unchanged, so the bits are
the same with and without overlap.

## Measurements, four streams on two cables (kernel with receive polling)

- Round trip of a 20 KB message: 32 us p50 / 58 us p99 striped over 4 streams, 57 / 68 us on one stream; 120 KB:
  77 / 82 us. Default: every stream for every message (`min_part_bytes` 0).
- 21 MB exchange: 4.7 ms; 42 MB: 9.3 ms (about 4.7 GB/s per direction). Each cable carries 17-19 Gb/s per
  direction whatever its link rate (20 or 40 Gb/s): the host path is the limit.
- A rank killed during 42 MB exchanges: the other rank fails about 2 s later with the cause ("the peer closed the
  stream"), and nothing hangs.
- Two-node decode of the deepseek41 runner (40 layers, tensor-parallel attention, 81 exchanges per token): with the
  shared expert overlapped with the half-sum exchange, 17.70 tok/s (56.5 ms/token), against 16.94 tok/s
  (59.0 ms/token) in the serial order. Same tokens on both ranks and in every variant.

## Measurements (cluster1 <-> cluster2, one USB4 cable, two streams, host path, interrupt-driven module)

- Expert parallelism by halves over USB4 between the two nodes: the output of layer 20 equals the single-node
  output bitwise on both ranks (1 and 6 rows, 20-30 inputs each), and the two ranks' outputs agree.
- Back-to-back exchange of one layer's half sums, staging included: 71-151 us (1 row, 20 KB) and 142-198 us
  (6 rows, 120 KB) per exchange; the link alone (`strix-dist-tcp-probe usb4`): 42-51 us and 133-143 us round trip.
- After the receiver has been idle for 1-3 ms (the compute between two layers' exchanges), the first message is
  late: round trip p50 75-360 us at 20 KB and p99 1-3 ms, with blocking or polled reads alike. Clock-aligned
  timelines of the layer runs show single receives delayed by 1-2 ms. This is in the receive interrupt path of the
  loaded stream module (no receive polling); the per-layer exchange needs a polled receive path before expert
  parallelism pays.
- `strix-dist-tcp-probe` over the 1 GbE LAN between the two nodes: round trip 56 us at 64 B, 410 us at 20 KB,
  0.118 GB/s one way. TCP over this link is for bring-up only.
