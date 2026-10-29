# MiniMax grouped host prefill

The grouped communicator backend uses the existing chunked host pipeline. Both
scheduling changes below default ON through the MiniMax resolver, gated per
grouped call: every member must have at least `STRIX_MM_PF_ASYNC_MIN_ROWS` rows
(default 2048). A group with any smaller member takes the ordinary `comm.start`
path, with byte-identical scheduling to `STRIX_MM_PF_ASYNC_START=0`. The decision
uses only the ordered member row counts and is identical on both ranks.
`STRIX_MM_PF_ASYNC_MIN_ROWS=0` removes the row gate (unconditional grouped async
start); the global off switch still wins. Invalid or overflowing thresholds
are rejected before device work. Inspect the threshold with `--print-effective-config`.

- `STRIX_MM_PF_ASYNC_START=0` disables early receive-destination publication and
  worker-side pinned receive-staging reuse. When enabled, MiniMax opts in on its
  own communicator, without exporting `STRIX_DIST_ASYNC_START` to other engines.
- `STRIX_MM_PF_LATE_SRC_FENCE=0` retains the immediate main-stream source-free
  wait after a successful async start (early destination only). With both switches
  enabled, the source-free wait precedes the next producer of that member's
  partial. The late switch has no effect when async start is disabled or falls back.

`STRIX_MM_BASELINE=1` switches both off unless explicitly overridden. Inspect them
with `--print-effective-config`; restart both ranks when changing startup settings.
The module backend, small/unchunked payloads and direct routes retain their existing
start/finish semantics. A false `start_async` result uses neither caller event.

## Exact scheduling contract and lifetime

Producer/post kernels and launch shapes, ordered group members (including
4096+443), payload bytes, rank-0/rank-1 summation, exchange sequence/tags, chunk
framing and CRC remain unchanged. All compute, including shared scratch, stays on
the main stream. Only TX/RX copies and host transport work can overlap that compute.

Timing-disabled `last_r_use` and `p_free` events are allocated once at startup for
each member of each activation bank. The destination event is initially recorded
before first use, then recorded after every admitted async-group post that reads
that member's peer partial. Ordinary groups record no destination events. If a
bank returns from an ordinary group to async, the admitted call records any
untracked ordinary readers before publishing destinations. RX waits on that
generation before overwriting the peer partial. TX records `p_free` after its last D2H read. A successful async start alone arms the
source wait; the next producer waits before overwriting the local partial. Local
post and TX may read the same partial concurrently.

Every resumable unit finishes its exchanges, enqueues every final post, and closes
all remaining source reads on main before returning. Thus intervening decode,
ungrouped tails and later groups cannot overwrite an outstanding source. Events
and grouped partial buffers persist across resumes. Park/unpark, cancellation,
retirement and bank reuse drain device work before changing ownership. Nested
children have a separate activation bank and separate event generations; current
admission limits them to one chunk. Shared partial/scratch buffers contain no live
parent data after retirement, so children may borrow them serially. Failure aborts
and joins submission workers before device synchronization. Destruction drains
outstanding work before destroying caller events. The communicator owns pinned
staging and protects its reuse on the worker before the next transport write.

## Measured regime

GPU measurements used the ungated build on two nodes,
TP2; complete token IDs were identical in every arm. Both changes were enabled
together versus `STRIX_MM_PF_ASYNC_START=0`:

| Workload | Both changes (s) | Off (s) | Result |
|---|---|---|---|
| Lone 7940-token prefill, members 4096+3844, scheduler off, ABBA | 17.44 / 17.49 | 18.00 / 17.82 | -2.5 % |
| Same, BAAB | 17.25 / 17.42 | 17.70 / 17.79 | -2.3 % |
| Lone 4539-token prefill, members 4096+443, 8+8 runs | mean 11.64 | mean 11.55 | Neutral |
| Mixed serving: 3 decoders + 4539-token long + 2 shorts, default policy, ABBA | long TTFT 16.00 | long TTFT 15.50 | Unresolved |
| Same, BAAB | long TTFT 15.29 | long TTFT 15.47 | Unresolved |

An earlier mixed-serving AB/BA pair showed long TTFT deltas +0.26 / +0.84 s.
Thus 4096+443 is neutral to slightly slower, with no reproducible mixed-serving
gain, and is gated off by default. The balanced 4096+3844 regime retains the
reproducible -2.3 / -2.5 % gain. These measurements qualify token agreement for
the measured arms; full prompt-output/state/logit and complete-workload lifetime
gates below remain pending. They do not independently attribute gains to either
switch, or establish a universal crossover at 2048 rows.

## Diagnostics and qualification

`STRIX_MM_PF_TIMELINE=1` reports pipeline receive-staging reuse wait separately
from ordinary host receive-staging reuse for grouped units. Pipeline time includes
the caller wait on the ordinary route or the worker wait on the async route; worker
time can overlap compute and must not be added to wall time. Existing GPU busy/gap
labels measure event intervals, not hardware utilization.

Host tests cover resolver defaults, independent off switches, baseline overrides
the per-call all-member threshold decision (including boundaries, zero and
global off), and source/route inventory. `strix-hip-dist-hostpath-tests` exercises fallback,
instance opt-in with the global switch off, immediate/late source fences,
member/bank event reuse, retirement, cancellation and corruption. These API tests
require a HIP GPU and do not qualify complete MiniMax execution.

Before timing, compare both ranks against the unchanged binary with the same model,
profile, capacity and prompt/chunk plan: complete IDs, all normalized prompt outputs,
KV/IK state and first/continuation logits. Include resumed units, parked children,
cancellation and bank reuse, then the complete workload. Isolate early destination
with the late fence off before testing both changes. Keep private fixtures and
receipts outside Git. See [communicator](engine/transport/communicator.md#minimax-grouped-async-start).
