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

## Additional qualified two-node code defaults (unset = on, `=0` = off)

Qualified on 2026-10-23 with three interleaved ABA rounds, every run bitwise against its reference, and the
full-workload gate; confirmed in a combined measurement.

| setting | default | qualification effect |
|---|---|---|
| `STRIX_DIST_CRC_PARALLEL`, `STRIX_DIST_HOST_CRC_OVERLAP` | on | hp: R6 -1.37 %, N8 -1.77 %, TTFT 2K/8K/32K -1.15/-1.24/-0.89 % |
| `STRIX_DIST_HOST_CHUNK_BYTES` | 1048576 (`=0`: one piece) | part of hp |
| `STRIX_DS41_ENGRAM_PAYLOAD_CACHE` | on (`_MIB` caps the budget) | TTFT -0.66/-1.17/-1.63 %, decode equal |
| `STRIX_DS41_ENGRAM_TP` | on for every single-row DeviceForward / BatchForward step (decode or prompt row; `=0` off, `=1` also up to 8 rows) | the two ranks split the Engram projection's output rows (same per-row arithmetic) and exchange the halves: R1 -0.61 ms/step (-2.1 %), same tokens/logits/state; at R6 not significant, so off there; needs two ranks and the F8 engram_kv (`STRIX_DS41_F16_AS_F8`), else the full projection on each rank |
| `STRIX_DS41_PREFILL_TILES` | off (`=1`: unsplit tiles for prefill runs of >= `STRIX_DS41_PREFILL_TILES_MIN_ROWS` rows, default 2048) | re-measured on the current kernels, tiles on vs off: fresh-process TTFT 2K/8K/32K +1.0/+3.4/+1.9 %, same-process A/B 4K/8K/16K +0.89/+0.66/+0.82 % (slower); the earlier -0.37/-1.26/-1.13 % predates the current split-K kernels. Same logits either way |
| `STRIX_DIST_STAGED_READY_BYTES` | 262144 (`=0`: no readiness views) | needed by the two below |
| `STRIX_DS41_READY_LEAN`, `STRIX_DS41_MOE_READY` | on | Lean ready releases, MoE half sum through ready views |
| `STRIX_DS41_OB_READY_EPILOGUE`, `STRIX_DS41_READY_READS_ONLY` | follow `READY_LEAN` | without the lean form they are the slower previous epilogue |
| `STRIX_DS41_BATCH_PREFETCH` + `STRIX_DS41_PREFETCH_SIDE` | 24 MB, side stream on (`PREFETCH_ROW_MB` 3) | MALL read-ahead on a lowest-priority stream during the exchange waits; N2/N4/N8 -2.1/-4.9/-3.3 % on the Q8 sites |
| `STRIX_DS41_DECODE_PREFETCH` | 2 MB per row, o_b window | R6 -1.8 %, R1 0 on the Q8 sites |
| `STRIX_DS41_TRELLIS_OB_READY` | on | TP trellis o_b through the ready views (one copy+publish kernel into TX, add from RX; no SDMA D2H/H2D or events); with CRC_DEFER R1 -1.6 %, R6 -1.9 %, N8 -0.7 %, same bits |
| `STRIX_DIST_CRC_DEFER` | on | ready-view exchanges compute the TX CRC during the wire and the RX CRC after delivery, each checked through the next trailer (a step still fails on corruption; mixed modes fail at once) |
| `STRIX_DS41_IDX_DIRECT`, `STRIX_DS41_IDX_CAND_FILTER`, `STRIX_DS41_ATTN_SEG_HPW` | on | indexer skipped when the count is <= 512, candidate-only scoring for pool layers above 16K entries, shared heads-per-wave for N8 segments; short R1/R6/N8 -2.0/-2.7/-1.0 %, 128K -0.8/-1.9/-3.3 %, same bits |
| `STRIX_DS41_BATCH_SHARED` | `overlap` (`off`/`0`: before the routed experts; `split` unchanged) | BatchForward shared expert inside the half-sum exchange, N1/N2/N4/N8 -2.8/-3.2/-5.1/-5.2 % in qualification, same bits |

Attention-core qualification (in-process ABA x3 per cell, every arm bitwise on both
ranks: tokens, logits slices, attention states; prefill 32K / 400 rows logits + state digest equal):

| setting | default | effect |
|---|---|---|
| `STRIX_DS41_IDX_DIRECT` | on (`=0` off) | count <= 512: no indexer projections / scores / select. Short context R1 -2.0 %, R6 -2.7 %, N8 -1.0 % |
| `STRIX_DS41_IDX_CAND_FILTER` | on (`=0` off) | candidate users read and select only candidate blocks (> 16K entries). 128K R1 -0.8 %, R6 -1.9 % |
| `STRIX_DS41_ATTN_SEG_HPW` | on (`=0` off) | N8 segments share one heads-per-wave form (Hpw 4 instead of 1). With the above: N8 8K -1.1 %, 32K -1.4 %, 128K -3.3 % |

Still off (measured slower or neutral): `STRIX_DS41_PREFILL_SLABS` (8K +0.76 %), `STRIX_DS41_R1FUSE_QNORM_A8`,
`STRIX_DS41_R1FUSE_ROUTES` (now on the fast builder, exact, in the noise), `STRIX_DS41_NORM_ILP` (exact, in the noise), `STRIX_DS41_GRAPHS` (only 0.21 ms/step of GPU idle waits for the host), `STRIX_DS41_ATTN_LDS_SWIZZLE`, CMake `STRIX_EXPERIMENTAL_MALL_Q2_DOWN_NT`.
With the trellis o_b site (below) the o_b product is the trellis plan, so the Q8 o_b ready epilogue does not run there;
the MoE half-sum ready path does.
The read-ahead skips trellis sites (their own pack) and runs only in the Q8 o_b exchange windows, so with the trellis
attention set it has little left to read in decode; it is kept on because it is exact and never slower.

## Current two-node defaults (2026-10-26)

The qualified two-node revision `481aeb0f` keeps these additional defaults. Each was gated on the nodes;
this publication merge does not establish a new device qualification.

| setting | default | override or constraint |
|---|---|---|
| `STRIX_DS41_DENSE_SELECT` | on: exact per-site dense prefill selector | `=0` restores the configured dense forms |
| `STRIX_DS41_ATTN_WINDOW_FOLD`, `STRIX_DS41_ATTN_LIVE_CHUNKS`, `STRIX_DS41_ATTN_TILE_F16` | on: exact attention window folding, live chunk scheduling and half-precision tiles | `=0` disables each independently |
| `STRIX_DS41_MOE_PREFILL` | split MoE prefill form 7 | `=0` disables it |
| `STRIX_DS41_SERVE_WARM` | warm the serving paths before READY | `=0` disables it; the default logs and skips decode warm-up if the KV pool cannot hold the dummy slots; explicit `=1` still fails when they do not fit; split decode skips it |
| `STRIX_DS41_DSPARK_CALIB` | on with the warm-up setting outside split decode | explicit `=0` / `=1` overrides it |
| `STRIX_SERVE_OBSERVE_EXPIRY_S` | 60 seconds after an arming request | `=0` keeps observation armed until process exit; `/metrics` and `/api/status?detail=1` renew it |

Approximate BF16 attention and FP16 dense prefill remain opt-in. Their projection, layer range, row threshold,
A8, fast and split-workspace controls are part of the prefix-cache namespace, so incompatible states do not share
cache entries. The optional FP16 dense backend requires CMake `STRIX_DS41_PREFILL_FP16_LT=ON`.

## Default invocation = the measured production command (2026-10-26)

The rank, batch, prefill and serve apps resolve their production environment and flags themselves
(`include/strix/deepseek41_pair_defaults.hpp`), so the plain command

```
strix-deepseek41-serve MODEL.gguf --rank R --transport usb4auto      # rank / batch the same; prefill: --transport usb4-auto
```

runs what the 2026-10-26 qualification measured, when the binaries were built with the trellis library:

```
cmake ... -DSTRIX_DS41_TRELLIS=ON -DSTRIX_DS41_TRELLIS_LIBRARY=<trellis archive> \
  -DSTRIX_DS41_TRELLIS_INCLUDE_DIR=<dir with strix_trellis.h>    # STRIX_DS41_TRELLIS_API defaults to dense_r1
```

Without the library the build still works; the trellis packs are then skipped with one notice. Measured (TTFT 1K 2.098 -> 1.954 s, 8K 11.090 -> 9.757 s; R1 -1.9 %, R6 -2.4 %, N8 -2.2 %;
serve C1 43.58 -> 48.00 tok/s; every run exact):

| item | default | off |
|---|---|---|
| production identity: `STRIX_DS41_OVERLAY=<packs>/c-u6.gguf STRIX_DS41_CB4=1` | on when the pack is found | `STRIX_DS41_OVERLAY=0` |
| trellis mix-B at every dense site: `STRIX_DS41_TRELLIS=<packs>/ours-K4 STRIX_DS41_TRELLIS_MIX=head=<packs>/ours-K5 STRIX_DS41_TRELLIS_SITES=all STRIX_DS41_TRELLIS_DENSE_FULL=1 STRIX_DS41_TRELLIS_HEADOUT=1` (not the EP prefill app) | on when both packs are found and the build links the dense_r1 trellis library | `STRIX_DS41_TRELLIS=0` |
| rank / batch `--tp-attention --engram-direct` | on (TP attention with two ranks) | `--no-tp-attention`, `--engram-mapped` |
| prefill `STRIX_DS41_EXPERT_TP=0` (EP prefill) | on | an explicit value |
| serve `--peer-timeout 120 --prefill-path device --slots 8 --prefill-rows 2048` | on | the explicit flag |
| serve `--dspark <model dir>/*-DSpark-MXFP4.gguf --dspark-vocab auto:<packs>/vocab` | on when the draft (and the vocabulary's `core48k.bin` / `core32k.bin`) are found; off with `STRIX_DS41_BATCH_ROWS16=1` or `--prefix-session-snapshots` | `--no-dspark`, `--dspark-vocab full` |

`<packs>` is `STRIX_DS41_PACKS` when set, else `<model dir>/packs`, else the model's directory (`STRIX_DS41_PACKS=0`:
no packs). An argument on the command line wins over the default, a variable already in the environment is never changed,
and a missing precondition drops its item with one `deepseek41 defaults:` line on stderr. `--print-effective-config`
prints the resolved environment and arguments and exits before any device work.

Master switch `STRIX_DS41_BASELINE=1`: nothing above is added, and the code defaults adopted on 2026-10-25/26 are set
to `=0` (`STRIX_DS41_SHARED_GROUPED`, `SHARED_DOWN_FUSED`, `MOE_PREFILL`, `DENSE_SELECT`, `ATTN_WINDOW_FOLD`,
`ATTN_LIVE_CHUNKS`, `ATTN_TILE_F16`, `SERVE_WARM`, `DSPARK_CALIB`) unless set explicitly: the plain command as it ran
before, without those items. The CPU test `strix-deepseek41-pair-defaults-tests` checks the resolved command against the receipt
and every off switch.

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
STRIX_DS41_TRELLIS_SITES=all STRIX_DS41_TRELLIS_DENSE_FULL=1 STRIX_DS41_TRELLIS_HEADOUT=1
```

Attention sites on trellis in the qualified two-node configuration: every mix-B dense class incl. q_a / kv / q_b / o_a / o_b
reads its K4 pack. Needs the build `STRIX_DS41_TRELLIS_API=dense_r1` (the CMake default) linked against a trellis library with the dense-r1 and grouped GEMV APIs
(grouped wo_a GEMV, dense-r1 q_a + kv pair, cold schedules). `STRIX_DS41_TRELLIS_OA_GROUP` (grouped o_a) and
`STRIX_DS41_TRELLIS_QAKV_PAIR` (q_a + kv in one launch) are on by default and bit-identical to their off forms (=0).
Measured (pair, ABA, same binary, only the env differs): R1 34.7 -> 31.15 ms/token (-10.2 %), R6 window 63.17 -> 60.59 ms
(-4.1 %), N8 85.38 -> 83.56 ms/step (-2.1 %), every run exact against its reference. The previous production set
`STRIX_DS41_TRELLIS_SITES=shexp,head` keeps its bits on this build. Quality: mix-B with all dense classes on
K4 is the configuration of the trellis quality report (full set KL 0.0446, paired vs c-u6 -0.0001).

The earlier trellis mix-B (shexp + head sites, K4 dense / K5 head) passed its quality gate
(full-set KL equal to the 6.5-bit format). Either trellis library (CMake `STRIX_DS41_TRELLIS_API` v12 or splitk04)
gives the same bits. Serve: `--prefill-path device --dspark <DSpark.gguf> --dspark-vocab auto:<vocab>`.
