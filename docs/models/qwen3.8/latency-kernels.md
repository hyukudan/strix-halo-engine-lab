# Optional single-session latency kernels

These eager decode experiments are disabled by default. They preserve the
existing projection formats, recurrent-state layout and expression order.
They are independent of the HC, injection and routed-commit fusion switches.

| Switch | Eligible path | Change |
| --- | --- | --- |
| `STRIX_QKERN_GDN_SCATTER=1` | Deferred GDN windows, 4 or 8 value columns per wave | Reduce-scatter several columns with the same ordered tree, then broadcast each root. |
| `STRIX_QKERN_ROUTER8=1` | 2560-wide staged F32 router, 2/4/8 rows | Eight experts and four loader waves per block; each output keeps its ascending FMA chain. |
| `STRIX_QKERN_HEAD_DIRECT=1` | Q8 head, 5..8 rows, width divisible by 256 and the existing lean admission bounds | Read vector BF16 inputs directly instead of staging them in LDS; retain the lean product and reduction trees. |
| `STRIX_QKERN_QSA_PERSISTENT=1` | Single-session known-window owner with `STRIX_QSA_WINDOW_POOLED=1` | Keep complete index-key pools between windows. |

The GDN switch leaves the tiled recurrence and two-column mode unchanged.
The router switch requires the existing staged-router path. The head continues
to materialize every logit; it does not prune the vocabulary.

The persistent QSA pool belongs to one eager window owner and one device/stream.
Its configurations, weights and committed session prefix remain immutable during
that owner's use. Before a window beginning at position `p`, only blocks below
`floor(p / compress_ratio)` may be reused. This invalidates complete speculative
blocks and partial tails after a rejection, regardless of the preceding row count.
After enqueue, retain validity only through that same pre-window frontier.
Blocks formed by the current window are rebuilt on reuse, so an idle owner
cannot retain rejected blocks overwritten by serial steps or another owner.
Each owner has its own pool, including owners of different window sizes.
Replacing session state requires `discard_deferred()` on its window owners;
that operation also invalidates their pools. The MTP session restart already
discards every owner's deferred state. Standalone launcher clients may explicitly
create and invalidate a pool through the typed API, retaining it through drain.

Pool storage is separate scratch and does not change exported persistent state.
It costs `complete_block_capacity * indexer_head_dim * sizeof(float)` per QSA
layer per window owner, allocated on first use. It is released after the owner's
stream drains. These switches alone do not establish workload qualification or
a throughput improvement; use native exactness gates before enabling them.
