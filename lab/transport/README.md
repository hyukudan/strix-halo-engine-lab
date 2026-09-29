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
