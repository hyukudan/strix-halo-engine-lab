# State hand-off transport: measured link use and the shared policy

The prompt-state hand-off (prefill host -> serving ranks) is shared code: `include/strix/handoff.hpp` /
`src/handoff.cpp` (framed TCP, 1..64 streams, bp-v1 byte-plane Huffman codec, per-piece XXH64, manifest fold) and
`include/strix/handoff_policy.hpp` (the cost model and the one resolver of the hand-off options). Every number below
was measured on 2026-11-15 with `strix-xfer-probe` (no GPU) or `scripts/bench/xfer/copy_probe.cpp`.

## What the link gives today (1 GbE, prefill host -> node)

| Pieces | 1 stream | 2 streams | 4 streams |
|---|---:|---:|---:|
| 64 KiB | 115.6 MB/s (92.5 % of line) | 117.0 | 113.2 |
| 1 MiB | 117.1 (93.7 %) | 116.5 | 115.5 |
| 16 MiB | 118.0 (94.4 %) | 112.6 | 114.7 |

TCP/IPv4 at MTU 1500 carries at most 1448/1538 = 94.1 % of the line as payload: one stream at 16 MiB pieces already
takes all of it, and more streams do not add anything on this link. The fixed cost of a piece is about 12 us.

## Codec on real state bytes (a Qwen state dump, one thread / 8 threads)

| Region | bp-v1 ratio | bp-v1 encode | zstd-1 | zstd-1 on byte planes |
|---|---:|---:|---:|---:|
| KV rows (BF16) | 1.468 | 2.16 / 5.27 GB/s | 1.265 (0.94 GB/s) | 1.506 (0.63 GB/s) |
| GDN recurrent matrix (FP32) | 1.171 | 2.22 / 4.75 GB/s | 1.068 | 1.181 (0.80 GB/s) |

lz4 has no entropy stage, so it cannot beat zstd-1 on these bytes, and zstd-1 alone loses to bp-v1. zstd-1 on byte
planes is 2.6 % / 0.8 % smaller than bp-v1 at about a third of its speed. That would save about 0.15 s on a 32K Qwen
hand-off at 1 GbE, but there is no zstd on the Windows prefill host, so it is not built. bp-v1 decodes at 11-23 GB/s.

## Staging copies

| Device | D2H pinned | D2H pageable | H2D pinned | H2D pageable | busy kernel beside the copy |
|---|---:|---:|---:|---:|---|
| RTX PRO 6000 (PCIe 5.0) | 35.6-49.0 GB/s | 9.8-13.7 | 33-38 | 13-23 | 0.243 -> 0.243 s |
| Strix Halo node (unified) | 68-90 | 62-83 | 61-81 | 43-82 | 0.184 -> 0.184 s |

An async copy on its own stream overlaps compute fully on both. Pinned staging matters on the discrete card (3.5x
D2H), but even a pageable copy is 100x faster than 1 GbE. On the node, pageable and pinned cost the same.

## Hand-off seconds at 32K (1 GbE, shared transport, real Qwen bytes, pieces released at the producer's times)

| Case | Wire | Tail after the last ready piece | Link busy |
|---|---:|---:|---:|
| Qwen 32K, prefill 7.59 s, bp-v1, per layer and block | 767.7 MB of 1,097.9 | 0.708-0.711 s | 74.7 % |
| Qwen 32K, identity | 1,097.9 MB | 1.757 s | 94.8 % |
| DS41 32K, whole state after the prefill (today's v1 shape) | 32.2 MB | 0.284 s (bp-v1 0.212) | - |
| DS41 32K, streamed per layer | 32.2 MB | 0.012-0.017 s | 13 % |

For Qwen the tail is the last block's rows plus the GDN recurrent state, which is final only when its layer finishes
the last block. Those 97 MB on the wire are the floor at 1 GbE.

## Library changes (bitwise: the imported bytes are the same)

- `ReceiverOptions` path, `recv_all` with `MSG_WAITALL`: the receiver used to wake per arriving segment burst and
  burned about one core at 118 MB/s. With the change it uses 0.041-0.056 s of CPU per 134 MB session instead of
  1.130-1.136 s (-96 %, 3 ABAB pairs), with the same throughput. On the Qwen 32K session the receiver CPU fell from
  7.24 s to 0.92 s. Rollback: `STRIX_HANDOFF_RECV_WAITALL=0`.
- `SenderOptions::encode_batch` (default 16): bp-v1 encodes up to 16 queued pieces side by side on the pool.
  Before, a piece of one codec block (<= 1 MiB) encoded on one thread. 1 MiB pieces went from 1.30 to 2.07-2.13 GB/s
  and 256 KiB pieces from 1.24 to 2.02-2.11 GB/s on loopback (8 threads, 3 ABAB pairs). The frames are the same bytes,
  and the 1 GbE Qwen tail is unchanged (0.708 s).

## Shared resolver (`strix/handoff_policy.hpp`)

`resolve(family_defaults, request)`: CLI > `STRIX_HANDOFF_*` > family defaults. Each read is recorded for
`--print-effective-config`, and each value gets a line `handoff <key> <value> (default|env|cli|auto: ...)`.

| Switch | Values |
|---|---|
| `STRIX_HANDOFF_LINK` / `--handoff-link` | `1gbe`, `2.5gbe`, `10gbe`, `usb4`, `oculink`, `loopback`, or goodput in MB/s |
| `STRIX_HANDOFF_CODEC` | `identity`, `bp-v1`, `auto` (needs a declared link) |
| `STRIX_HANDOFF_STREAMS` | `1..64`, `auto` |
| `STRIX_HANDOFF_ENCODE_THREADS` | `0..256`, `auto` |
| `STRIX_HANDOFF_ENCODE_BATCH` | `1..256`, `auto` |

`auto` runs `predict()` over the request's pieces for every candidate. It picks the cheapest choice within 1 % of the
earliest end: identity before bp-v1, then fewer streams, then fewer threads. The defaults stay what each family ships.
The Qwen prefill service resolves `--handoff-codec/-connections/-encode-threads/-link` through it (connections
`auto` is refused: an HTTP chunk POST waits for its reply).

Cost model: wire(piece) = per-piece cost + wire bytes / min(link goodput, streams x per-stream cap); bp-v1 encode =
plain / min(threads x 2.16, 4.8) GB/s; pieces pass encoder then wire in order; 10 ms session finish. Against the 1 GbE
receipts: Qwen bp-v1 0.713 vs 0.708 s, identity 1.820 vs 1.757 s, DS41 after 0.284 vs 0.284 s.

## Predicted tails for the other links (`strix-xfer-probe plan`)

| Link | Qwen 32K (prefill 7.59 s) | DS41 32K after / streamed | MiniMax 5.03 GB pure transfer |
|---|---:|---:|---:|
| 1 GbE (measured constants) | 0.72 s, bp-v1 | 0.28 / 0.02 s | 29.1 s bp-v1 (42.7 identity) |
| 2.5 GbE | 0.06-0.08 s | 0.12 / 0.01 s | 11.7 s |
| 10 GbE | 0.02-0.03 s | 0.04 / 0.01 s | 2.9 s |
| USB4 host path (4.4 GB/s) | 0.02 s | 0.02 / 0.01 s | 1.05 s |
| OCuLink (PCIe 4.0 x4, nominal) | 0.01 s | 0.02 / 0.01 s | 0.78 s identity |

From 2.5 GbE up, a per-layer Qwen hand-off hides behind its prefill. From USB4 up, bp-v1 at 8 threads is slower than
the link, and `auto` picks identity.

Kit: `scripts/bench/xfer/` (README there).
