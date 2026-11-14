# Pair transport routing decision

Status: byte-movement implementation; new Qwen/MiniMax combinations await native
build, exactness and complete-workload timing gates. The figures below are
existing project measurements, not timings of this change.

## Evidence and message regimes

The [communicator probe](../engine/transport/communicator.md#measurements-gpu-direct-on-2-rings-data1-of-both-cables-vs-the-host-path-on-2-streams-data0)
checked every received word. Pair p50 host / GPU-direct was 26 / 33 us at 64 B,
36 / 53 us at 20 KB, 146 / 95 us at 160 KB, and 15,074 / 7,491 us at 21 MB.
After 2 ms idle, GPU-direct small-message latency increased substantially.
The production decode check measured 20.15 tok/s host, 13.7–15.8 all-direct,
and 20.18 with the size router. Thus opening GPU-direct must keep host control
and the existing router (`65536:host,1048576:gpu:2,inf:mix:2:35` on two rings).
This is not a claim that the mixed route halves application TTFT.

Qwen seam rows are 25,616 bytes. A 4,096-row chunk is 104,923,136 bytes, sent
in 16 MiB pieces and a tail. These are the large-message regime. Default pair
transport is `usb4gpu`; `usb4auto` and `usb4-auto` explicitly restore host-only.
Only seam payloads use the communicator; headers, piece/whole CRCs, ACKs and
final-state handoff retain their ordered host protocol. Direct payloads read
the immutable device snapshot and install the receiver's snapshot. System-release
readback publishes received bytes for the existing CPU CRC before consumption.
The direct receive stays posted across framing records and is retired after the
pipe joins, before handoff/exit. Both ranks need the same transport and binary.

MiniMax's rank app used `!gpu_direct` on both `xfer_load` and `pfx_load`:
the direct communicator took ownership of the host transport, while XFER/PFX
borrowed that same transport with their own progress/framing. Simply deleting
the guards would leave dangling references and outstanding communicator work at
ownership changes. The communicator now borrows a host owner that outlives it;
module entry retires communicator preposts, deferred checks and device readers.
The existing module drain/release boundaries precede communicator reuse.

| MiniMax class | Choice | Evidence / limit |
|---|---|---|
| Decode (1–8 rows, 24–192 KiB partials) | Keep XFER and its fused producer/consumer boundaries, even with `usb4gpu` | [Boundary set](../measured-gains.md): 48.56 → 44.40 ms/token (a set, not isolated link timing); [deferred CRC](../optimizations.md#mm-xfer-crc-deferred): 0.04–0.12 ms/token below neighboring inline arms, 7 processes, identical tokens. There is no matched XFER-vs-direct workload receipt; the 160 KB raw probe is not that comparison. `STRIX_MM_XFER=off` selects communicator dispatch. |
| Rank prefill | Keep PFX on USB4, including the host side of `usb4gpu` | [PFX](../optimizations.md#mm-rank-pfx): TTFT 1K −3.5%, 8K −1.5%, exact in-process A/B/A. Do not replace this measured overlap with an unmeasured direct path. `STRIX_MM_PFX=load` leaves it off; `off` omits it. |
| Rank PFX chunks | Keep 4 MiB | [Fresh-process measurements](../optimizations.md#mm-pfx-chunk): 1K 2.469 → 2.423 s; 4K 9.168 → 9.025 s; 8K unchanged around 590 tok/s because transfer is hidden. 16 MiB was no better. |
| Serve prefill / other large communicator payloads | Default `usb4gpu` with the production size router | Large-buffer probe favors direct. Serve cannot enable PFX without losing nested serving; PFX stays absent. Serve-specific 8K TTFT and exactness remain pending. |
| Control and prompt-token blocks (up to 64 KiB) | Host | Small-message probe and production decode result above. No all-direct default. |

Serve `--transport usb4-auto` restores host-only, including the previously
qualified session-cache route. Session reuse on `usb4gpu` stays declined until
its separate native qualification; this change does not widen that allowlist.
An idle paired heartbeat retires GPU receives before the DMA deadline, including
when XFER is off. Unavailable GPU streams negotiate host-only and report the
reason. Explicit `STRIX_DIST_ROUTE` continues to override communicator routing.

## Shared setup contract

[The opener](../../include/strix/dist/open_transport.hpp) owns TCP/USB4 setup,
topology receipts, direct-stream agreement and fallback. Its
[host-only parser](../../include/strix/dist/transport_spec.hpp) takes an explicit
spelling policy. DeepSeek rank/DSpark retain `usb4auto`, `usb4gpu` and raw
`usb4:DEVICES`; batch/serve omit raw devices; DeepSeek prefill retains
`usb4-auto`/`usb4-gpu`; GLM and MiniMax retain `usb4-auto`/`usb4gpu`.
Existing per-app connection timeouts, capabilities and defaults remain, except
the two default changes described above. Diagnostic/evaluation apps with narrower
host-only parsers are outside this production-parser consolidation.

Native acceptance requires strict HIP builds; Qwen 8K/32K complete seam/logit/state
bitwise checks and host/direct TTFT ABAB; MiniMax serve smoke, idle/restart,
nested serving and 8K TTFT ABAB; MiniMax rank XFER/PFX coexistence with both OFF
arms; DeepSeek/GLM pair smokes. Preserve numerical identity, capacity, model,
kernel and IOMMU settings; verify bytes/tokens/digests before comparing timings.
