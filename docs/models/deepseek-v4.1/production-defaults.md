# DeepSeek-V4.1 two-node: production defaults (P17.2)

From the P17.1 timing campaign (3 interleaved rounds per arm, every arm's tokens checked against its identity's
reference). Code defaults change only what was measured as a gain and is exact; the production identity needs the
deployment's environment (pack paths), listed below.

## Code defaults (unset = on)

| setting | default | off | effect (P17.1 campaign) |
|---|---|---|---|
| `STRIX_DS41_BATCH_SEGMENTS` | on (off by itself with `STRIX_DS41_BATCH_ROWS16=1`) | `=0` | batch N=8 +9 % |
| `STRIX_DS41_TP_DOWN` | on (the expert-TP down, width 1152 only) | `=0` | R6 window -3 %, batch N=8 +1.6 % |
| `STRIX_DS41_SERVE_SCHED` | `interleave` | `=off` | 8K prompt under load: decode stall <= 1 s instead of ~13 s (its own TTFT about doubles) |
| serve `--dspark-ranks` | 2 (a drafter on each rank) | `--dspark-ranks 1` | |
| serve `--dspark-max-inflight` | 4: plain steps while more requests are in flight | `0`: always draft | keeps C1..C4 (+19..+76 %) without the C8 TTFT cost |
| batch app `--draft-ranks` | 2 | `--draft-ranks 1` | |
| `STRIX_DIST_USB4_EAGER` | 65536 (P17.3) | `=0` | R1 decode +2.8 %, R6 / batch equal |
| `STRIX_DS41_LATE_HC_LAYERS` | `all` (P17.5): the attention site's hc mixes run during the o_b exchange (not with `STRIX_DS41_BATCH_PREFETCH`) | `=none` | R1 -0.2 ms, R6 -0.35 ms, batch (no prefetch) N=2/4/8 -1.5/-0.7/-0.6 ms/step; bitwise |
| `STRIX_DS41_PREFILL_FINAL_KV` | `all` (P17.5): row-split prefill owners send the final ring / compressed entries (+ ratio-2 carry) instead of FP32 K/V and compressor inputs | `=none` | 8K prefill +1.8 % (714.9 -> 729.4 tok/s); prefill --reference bitwise, decode continuation identical |
| `STRIX_DIST_STAGING` | `fused` (P17.4): the TP o_b product writes the pinned send buffer, the add reads the pinned receive buffer (host-path routes) | `=copy` | R1 -0.2..-0.4 ms/token, batch N=1..2 -0.25 ms/step, R6 / N=8 equal; bitwise in layouts 1 and 2bq |
| `STRIX_DS41_EXPERT_TP`, `EXPERT_SLOTS`, `ROUTES_FAST`, `F16_PAIR` | on (P16) | `=0` | |

Unchanged (off unless set): `STRIX_DS41_PREPARED_LOAD`, the tpserve prefill
(`STRIX_DS41_PREFILL_EXPERT_TP`, slower than the EP prefill), `STRIX_DS41_BATCH_ROWS16`, serve `--cache-profile`
(opt-in for agent loops: prompts shorter than a 256-token block are never cached), `STRIX_DS41_COMPRESS_FAST`,
`STRIX_DS41_EXPERT_ROWS`.

## Transport and USB4 layout (P17.4)

- Every app takes `--transport usb4gpu` (prefill: `usb4-gpu`); the serve manifest kind is `usb4_gpu_auto`.
  - With gpu_direct streams (layout 2bq: data0 host, data1 gpu_direct + gpu_poll=1, latency_qos_us -1) they build the device exchange. Route A is `65536:host,1048576:gpu:2,inf:mix:2:35`, with prepost.
  - Without gpu_direct streams (layout 1) they print "gpu-direct unavailable ... host path only" and run the host path. The bits are the same in both layouts.
- Serve quiesces both ranks when it goes idle after work, so no posted GPU receive outlives the driver's 30 s DMA deadline.
- 2bq vs layout 1, the same binary and kernel (P17.4):
  - faster: R6 window +2.9 %, prefill 8K +1.6 %, serve 8K TTFT -10 %, serve C2..C8 +4..7 %;
  - slower: R1 -1 % (2 host streams instead of 4);
  - equal: batch.
- 2b at latency_qos_us 0 is slower everywhere: do not use it.
- `STRIX_DS41_OB_SPLIT=K` (o_b exchange in K column chunks overlapping the products) is exact but slower on the host path (R6 +1.2 / +3 / +5 ms at K=2 / 4 / 8). It stays opt-in.

## Production identity (environment of both ranks)

```
STRIX_DS41_OVERLAY=<requant>/ov/c-u6.gguf STRIX_DS41_CB4=1
STRIX_DS41_TRELLIS=<packs>/ours-K4 STRIX_DS41_TRELLIS_MIX=head=<packs>/ours-K5
STRIX_DS41_TRELLIS_SITES=shexp,head STRIX_DS41_TRELLIS_HEADOUT=1
```

Trellis mix-B (shexp + head sites, K4 dense / K5 head) is the production identity after its quality gate
(full-set KL equal to the 6.5-bit format). Either trellis library (CMake `STRIX_DS41_TRELLIS_API` v12 or splitk04)
gives the same bits. Serve: `--prefill-path device --dspark <DSpark.gguf> --dspark-vocab auto:<vocab>`.
