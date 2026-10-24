# Trellis library adapters

`STRIX_DS41_TRELLIS_API` is a CMake cache string. Its default is `v12`, the
existing adapter. Every value requires an explicitly supplied library archive
and matching header directory (there is no default location); the opt-in
values `tp03` and `splitk04` need an archive that provides their entry points:

```sh
cmake -S . -B build \
  -DSTRIX_ENABLE_HIP_RUNTIME=ON -DSTRIX_DS41_TRELLIS=ON \
  -DSTRIX_DS41_TRELLIS_API=splitk04 \
  -DSTRIX_DS41_TRELLIS_LIBRARY=/absolute/path/libstrix_trellis.a \
  -DSTRIX_DS41_TRELLIS_INCLUDE_DIR=/absolute/path/include
```

The opt-in adapters remain qualification candidates. Selecting one does not
establish model exactness or workload performance. Pack transports, weight
identity, rank slices, padding and the order of inter-rank sums are unchanged.

`tp03` uses prepared linear plans, with dense TP plans for the supported K4/K5
local geometries: q_b 1280→16384, o_b 4096→5120, shared gate/up 5120→1152,
and shared down 1152→5120. Other sites use prepared linear GEMV plans.
The q_a/kv dense API geometries are supported by the adapter, but the current
forward paths do not hook those model sites to trellis packs.

`splitk04` selects the explicit split-K creator on those TP geometries;
q_b K5 at eight submitted rows keeps the prepared linear plan. Shared down
retains the zero-workspace TP kernel. This policy applies to submitted rows
after adapter padding. The original F32 SwiGLU and rank reduction remain in
the engine. Routed experts keep their existing weight format and kernels.
Split-K counter storage persists between submits. Its workspaces are separate
from generic GEMV and keyed by stream, input/output geometry, bits and submitted
rows. Reserve allocates each required layout once, shared serially across
same-layout sites. Different layouts never overwrite each other's counters.
The library workspace byte query includes these allocations before model execution.

Shared-expert TP is separately enabled with `STRIX_DS41_SHEXP_TP=1`; the
current forward implementations require `STRIX_DS41_EXPERT_TP=0` with it.
The ordinary full shared expert keeps its existing 2304-channel geometry.

Plans freeze weights, input/output addresses, row count, workspace and stream.
The context caches a plan for each such tuple. Creation happens on first use
outside stream capture. Warm every required tuple before capture; an unseen
tuple during capture throws before plan creation. Submit reuses the plan.
Buffers, streams and weights must remain live through plan destruction and
all captured graph replays. The forward owners destroy group plans and the
trellis context before their activation buffers and stream, including on
construction failure. A stream workspace must be used serially.

For qualification, the rank app's `--state-out PREFIX` saves the full
serialized attention state to `PREFIX-rR.state`. Its existing `--logits FILE`
writes full vocabulary rows during ordinary decode. With `--window-check`,
it writes each rank's vocabulary slice for every physical row, including
rejected rows; concatenate rank slices in vocabulary order for full logits.
These diagnostics are excluded from timing runs.
The batch app also accepts `--state-out PREFIX`: it writes each request's
`export_state` after its alone and batch runs, labelled by N, request and rank.
This is the batch state API's representation (without last-row top-k or the
caller-owned token list); compare the same schedule across library adapters.
