# Strix Halo Transport Lab

This lab establishes the network-stack baseline that future direct USB4
transport modes must beat.

It is intentionally separate from the inference runtime. The benchmark measures
transport properties with controlled payload sizes before model execution adds
GPU, scheduler and state costs.

## Build

```bash
cmake -S lab/transport -B build/transport-lab -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/transport-lab
```

## Server

On the receiving host:

```bash
./build/transport-lab/strix-transport-tcp server \
  --bind 0.0.0.0 --port 19090 --max-bytes 64MiB
```

The server accepts one benchmark connection, validates framing and serves until
the client sends a stop record.

## Client

Round-trip payload benchmark:

```bash
./build/transport-lab/strix-transport-tcp client \
  --host PEER --port 19090 --mode pingpong \
  --bytes 64KiB --warmup 20 --iterations 200
```

One-way bulk plus small acknowledgement:

```bash
./build/transport-lab/strix-transport-tcp client \
  --host PEER --port 19090 --mode stream \
  --bytes 16MiB --warmup 5 --iterations 50
```

Add `--verify` for correctness runs. Verification hashes every payload and is
therefore intentionally separate from performance qualification.

## Modes

### pingpong

The client sends a payload and the server returns the same payload.

Use this for:

- queue-depth-one round-trip latency;
- small-message comparison;
- symmetric payload paths.

### stream

The client sends the payload and the server returns only a framed
acknowledgement.

Use this for:

- bulk client-to-server payload throughput;
- large tensor-sized messages.

The measured transaction still includes acknowledgement latency; it is not a
clock-synchronized one-way latency measurement.

## Output

The client emits JSON containing:

- mode;
- payload bytes;
- warmup and measured iteration counts;
- total measured seconds;
- effective payload GiB/s;
- p50/p95/p99/max transaction latency;
- whether payload verification was enabled.

The server emits a final JSON summary with payload and message counts.

## Qualification matrix

Run both modes for sizes derived from real model boundaries. Until those are
known, useful provisional sizes include:

- 4 KiB;
- 64 KiB;
- 256 KiB;
- 1 MiB;
- 4 MiB;
- 16 MiB;
- 32 MiB.

For each size record:

1. Ethernet/network baseline;
2. IP over a single USB4 link, if available;
3. later, direct USB4 streaming through the same logical benchmark shape.

The direct transport does not qualify because it wins a synthetic bandwidth
test. It must improve the corresponding model boundary and end-to-end token
cost.

## Dual-link work

This first executable is deliberately single connection/link.

The core scheduler already models independent physical links. The next lab step
will run two qualified endpoints concurrently and compare:

- same-direction dual-link flow distribution;
- opposite-direction dedicated links;
- large-message striping.

Do not use operating-system bonding as the reference for engine-level
multi-link scheduling: the engine needs per-link latency, queue and failure
visibility.


## Direct USB4STREAM baseline

On a Linux kernel with direct USB4 streaming support, configure a named stream
through configfs on both hosts. Once the stream is active, its `index`
attribute identifies the matching `/dev/tbstreamX` character device.

Example receiver:

```bash
./build/transport-lab/strix-transport-tbstream server \
  --device /dev/tbstream0 \
  --config /sys/kernel/config/thunderbolt/stream/0-1.0/data \
  --max-bytes 64MiB --verify
```

Example sender:

```bash
./build/transport-lab/strix-transport-tbstream client \
  --device /dev/tbstream0 \
  --config /sys/kernel/config/thunderbolt/stream/0-3.0/data \
  --mode stream --bytes 16MiB --warmup 5 --iterations 50 --verify
```

The direct backend deliberately uses only normal `read(2)` and `write(2)`.
It does not assume mmap, DMA-BUF, ioctls, GPU-direct DMA or message boundaries
provided by the character device.

The optional `--config` path is read-only. Benchmark JSON records:

- stream index;
- busy-poll mode;
- ring size;
- interrupt throttling;
- input HopID;
- output HopID.

Those settings are part of the benchmark identity.

### Progress modes

Run separate qualifications with interrupt-driven completion and
`busy_poll=1`.

Busy polling may reduce latency but consumes CPU and changes polling semantics,
so it is a distinct transport mode rather than a hidden tuning flag.

### Ring and interrupt tuning

The direct-stream ring size and interrupt-throttling value are tunable kernel
attributes. Tune them offline as a matrix and record them with each result.
Never infer the best decode setting from bulk throughput alone.

### Config responsibility

The benchmark does not create or modify configfs streams.

Stream creation, HopID allocation and tuning require system-level coordination
between both hosts and remain an explicit deployment step. This keeps the
benchmark read-only with respect to transport configuration.


## Dual-link parallel-flow runner

Before treating two physical USB4 links as one striped tensor transport, measure
whether they can sustain useful traffic concurrently.

On the receiving host:

```bash
python3 lab/transport/dual_tbstream.py server \
  --left-device /dev/tbstream0 \
  --right-device /dev/tbstream1 \
  --left-config /sys/kernel/config/thunderbolt/stream/0-1.0/data0 \
  --right-config /sys/kernel/config/thunderbolt/stream/1-1.0/data1 \
  --verify
```

On the sending host:

```bash
python3 lab/transport/dual_tbstream.py client \
  --left-device /dev/tbstream0 \
  --right-device /dev/tbstream1 \
  --left-config /sys/kernel/config/thunderbolt/stream/0-3.0/data0 \
  --right-config /sys/kernel/config/thunderbolt/stream/1-3.0/data1 \
  --bytes 33554432 --left-share 0.5 \
  --mode stream --warmup 5 --iterations 50 --verify
```

This runner launches one benchmark process per physical stream concurrently and
reports both lane results plus aggregate useful payload throughput.

It is intentionally called **parallel flows**, not single-tensor striping:

- it tests whether the two physical paths interfere;
- it avoids hiding one weak link behind a reassembly policy;
- it can vary the left/right payload share;
- it provides the evidence needed before enabling the core scheduler's striped
  plan for a workload class.

A later single-tensor striping harness must add reassembly cost to the measured
critical path.

## GPU-owned direct-DMA qualification plan

The copy/read-write USB4STREAM harness is the baseline for a later GPU-owned transport tier.

The direct-DMA experiment must keep separate TX and RX pools per physical link and must not reuse the inference scratch arena or persistent model/session state as transport storage.

Before a direct-DMA result can be compared with the copy baseline, record:

- physical link identity;
- connection epoch;
- direct-stream ABI/capability identity;
- TX/RX pool bytes and allocation generations;
- exclusive lease state;
- ring/frame count;
- progress mode;
- interrupt throttling;
- RX posted/in-flight/repost counts;
- local TX reuse completion;
- RX DMA completion;
- remote GPU consumer-complete credit;
- completed useful GPU-operation latency.

The first direct-DMA operation should be a fixed-shape activation transfer or exact two-peer reduction with an ordinary portable fallback.

### Synchronization gate

A received payload is not reusable storage merely because DMA completed.

The experiment must prove:

1. producer GPU writes payload;
2. producer publishes a terminal commit record with release semantics;
3. external DMA transfers payload plus terminal record;
4. consumer observes the commit with acquire semantics;
5. consumer validates epoch/sequence/length/stripe metadata;
6. consumer GPU finishes using the RX slot;
7. transport completion accounting retires the message;
8. RX frames are reposted;
9. only then is the RX slot reusable.

### Dual-link gate

For two physical USB4 links, create one independent direct endpoint per link.

Each endpoint owns its own:

- TX/RX pools;
- epoch;
- lease;
- frame counters;
- completion/repost queue;
- performance profile.

Qualify in this order:

1. link A alone;
2. link B alone;
3. two independent same-direction flows;
4. two independent opposite-direction flows;
5. latency-critical traffic on one link plus bulk traffic on the other;
6. large logical message striped across both links;
7. failure of one link during independent flows;
8. failure of one link during a striped message;
9. clean degradation to one link;
10. cleanup/re-arm/requalification of the failed link.

Single-message striping is not valid until every stripe commits and the total stripe byte count exactly matches the logical transfer descriptor. A partial tensor must never be published.
