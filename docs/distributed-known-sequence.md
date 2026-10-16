# Native known-input sequence component

`HipDistributedKnownSequence` is an explicit, serialized two-peer consumer of
the real token participant, native transport backend and concrete GPU boundary
staging. It accepts two through sixteen literal known input tokens. It does not
perform distributed greedy generation, sampling, memory sharding or recovery.
Every peer retains its complete local model mapping/residency and seven-state
snapshot owners. No default decoder route changes.

The first consumer admits one contiguous prefix on node0, one suffix on node1,
one nonempty boundary from the last prefix operation to the first suffix
operation, and one independently qualified physical link at index0. For the
current full model DAG, the final-hyper-mix to LM-head cut is one such bounded
placement. Cuts with other crossing dependencies are rejected. This restriction
does not constitute a balanced pipeline or a general placement scheduler.

The caller supplies an already constructed `HipDistributedTokenParticipant`,
immutable compiled model/resources, its actual retained model owner and stream,
and `HipDistributedSequenceTransport`. The latter retains the actual native
link session, GPU and receive providers, imported pools, lease registry,
physical driver, dedicated kind4 progress channel and scheduler. Positive link
qualification is never created by this component. An absent/currently invalid
capability fails admission. The step channel and dedicated kind5 transaction
channel are distinct bounded protected-control byte streams; they carry no
FP32 logit array, state or weights. The transport provider remains responsible
for the actual device payload and visibility proofs.

The canonical wire plan retains node0 and its original identity at both peers.
An explicit endpoint parameter controls staging direction. On node1 the
consumer first validates the exact sender flow and request descriptor, then
converts only the local flow direction/identity into the endpoint1 receive
flow. It does not edit a cached placement's node while retaining a stale ID.
The old flow and staging overloads retain their original node selection.

Both peers capture local state and exchange prepared receipts before target
mutation. The prefix finishes locally before staging may request completed
producer views. The suffix completes the real boundary receive, consumer
release, credit and repost before serving any remote target command. Each
remote command invokes the actual participant executor; its success ACK
follows the existing native completion checks.

The terminal peer downloads every FP32 logit, checks all values finite and
uses first occurrence of the maximum for ties before emitting local readiness.
This is a known-input qualification observation, not the next input token.
Rows remain provisional until both committed receipts and an actual minted
native retirement proof exist. Rank-ordered prepared/ready/ACK exchanges avoid
requiring both bounded control streams to send simultaneously. Only node0
returns the coordinator confirmation. The final proof retains actual owners;
each subsequent input consumes the previous proof and derives its binding
from the participant's typed frontier. The original ordinary serial session
stays quarantined.

Input/context/generation bounds and result payload capacity are admitted
before preparation. `maximum_logit_payload_bytes` bounds retained terminal
FP32 rows, not allocator overhead, snapshots, weights, driver pools or process
RSS. All those independent owners require their existing budgets. Result and
local-executor storage is allocated before the first target mutation. Cheap
request-bound transport adapters are constructed per token; weights, native
sessions and lease sequence history persist. No new GPU kernel is introduced.

There is no automatic retry, unilateral rollback or output publication after
an uncertain decision/ACK/transport outcome. Failure quarantines the participant
and drains local work. Before unwinding external DMA owners it closes the
actual native lease. If that close cannot prove release, the component stops
the process instead of freeing possibly live DMA buffers. A GPU fence alone
is never interpreted as external transport retirement.

The optional static fixture in
`tests/hip_distributed_sequence_model_qualification.hpp` is callable by an
embedded commissioning owner with actual setup. It reuses the caller's real
mapped/resident model and independent full serial reference state, runs the
literal sequence through real serial and distributed consumers, compares every
terminal FP32 bit and firstmax on peer1, checks locally owned state slices and
global write generations, and verifies the ordinary anchor remains quarantined.
It also contains pre-mutation admission negatives for cuts, canonical identity,
endpoints, channels, owners and result budget. There is deliberately no fake
standalone executable, skip-success path or automatic CTest model run. Binding
this callable case to a separately commissioned two-host physical bootstrap
and executing all compiler, native, model and physical gates remains pending.

## Retained one-input mode

An explicit `HipDistributedSequenceSetup::streaming=true` admits a positive
`max_target_inputs` before allocating the reusable row or preparing any target.
The complete limit must fit both peer contexts, compiled positions, generation
and committed-token counters. Its payload budget must fit one complete FP32
vocabulary row. The default known-input mode keeps its single-use two-through-
sixteen input contract; neither entry silently switches to the other mode.

`execute_input(token)` performs the same actual one-input transaction and returns
the committed frontier after both ACKs and a minted native retirement proof.
On peer1 its completion also carries the actual finite firstmax token and the
selected FP32 bits. Peer0 has no local terminal selection. The first input must
match the negotiated binding. Later inputs consume the previous proof and derive
the next binding through the retained participant's typed frontier. Only then
may the same executor and host row be reset and reused. The final proof retains
its actual transport, executor and row owners through the next continuation or
destruction. A transaction failure invalidates row publication and quarantines
the participant; it does not enable retry.

`last_terminal_logits()` borrows the peer1 row after successful completion until
the next admitted input or destruction, under serialized caller access. No row
is published before completion, and no full-vocabulary history is retained in
streaming mode. `completed_inputs()` counts only completed local transactions.
There is no EOS, output-budget, pending-token or emission policy here. The
autoregressive consumer must use the existing serial token cursor and a separate
control protocol. A zero-target request can negotiate finish without calling
`execute_input`; the component never fabricates prompt consumption.

The callable `qualify_hip_distributed_streaming_inputs` consumes literal oracle
inputs through this entry, retaining one independent serial reference row and
one terminal row. It compares every FP32 bit, selected bits, typed commit and
locally owned state, checks row storage reuse, and includes mode, budget, limit
and first-input admission negatives. It is a source fixture, not an executed
autoregressive, model or physical qualification result.
