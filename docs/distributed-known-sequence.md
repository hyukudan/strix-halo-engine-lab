# Native known-input sequence component

`HipDistributedKnownSequence` is an explicit, serialized two-peer consumer of
the real token participant, native transport backend and concrete GPU boundary
staging. It accepts two through sixteen literal known input tokens. It does not
perform distributed greedy generation, sampling, memory sharding or recovery.
The default full-resource route retains complete local model/state and snapshot
owners. An explicit stage owner can retain selected weights and typed packed
state as described below. No default decoder route changes.

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
`execute_input`; the component never fabricates prompt consumption. This does
not change the serial cursor's prompt-consumption policy at any output budget.

The callable `qualify_hip_distributed_streaming_inputs` consumes literal oracle
inputs through this entry, retaining one independent serial reference row and
one terminal row. It compares every FP32 bit, selected bits, typed commit and
locally owned state, checks row storage reuse, and includes mode, budget, limit
and first-input admission negatives. It is a source fixture, not an executed
autoregressive, model or physical qualification result.

## Explicit complete-layer cut

`layer_boundary_position_dependencies=true` selects a separate bounded bridge.
Its one data edge must connect `mlp_inject` in layer L to the first
`attn_hyper_mix` in layer L+1. A cut before PLE, inside a layer, or with any
additional live data crossing is rejected before preparation. The only other
admitted edges are at most64 canonical zero-payload dependencies from prefix
`position_prepare` to suffix indexed attention. Zero such dependencies is also
valid. The complete canonical plan, edge list and original identity are retained
and compared to a reconstruction; no edge is pruned or retagged.

Concrete staging receives the selected data-boundary identity and the explicit
position policy. It keeps that same complete plan but exposes only the selected
nonzero payload to real native send/receive. Neither a zero-byte DMA nor a fake
control completion is issued for logical position dependencies. The prefix
finishes before staging resolves source views; the suffix's actual participant
materializes the bound input position before executing its first operation.

After real stage completion and both committed ACKs, the participant can mint a
single-use position-completion proof for the exact token and full dependency
list. The backend consumes this logical proof before issuing retirement. Its
independent actual data handles still must retire; a host position word or token
ACK cannot replace native transport completion. A later input consumes the same
combined retirement gate before any scratch or source owner can be reused.
The default strict route retains its one-boundary admission and policy.

The callable `qualify_hip_distributed_layer_boundary_inputs` requires a separately
constructed actual participant and canonical model placement at this cut. It
compares full terminal rows, locally owned recurrent/attention/PLE state and
global generations to the independent serial reference, counts every canonical
data and logical dependency, and includes source fixtures rejecting interior,
pre-PLE, malformed logical and implicit legacy-mode admissions. It accepts the
explicit literal or retained streaming input mode. This cut enables a larger
suffix structurally; it makes no measured balance, speed or physical capacity
claim. Native, model, physical and complete-request qualification is pending.

## Actual stage-owned resources

`HipDistributedStageResources::create` can retain only the selected stage's
weight backing. Its default state policy remains `full`; the explicit
`local_owned_ranges` policy owns packed state pools and a typed snapshot of all
and only its logical state ranges. Scratch remains the complete serial arena.
The logical seven-component layout is retained as an addressing contract, not
as a claim that absent remote state is allocated locally. The reservation
reports full logical and actually retained state bytes separately.

`make_hip_distributed_stage_sequence_setup` takes this actual shared owner, the
jointly agreed initial binding, actual native transport and explicit sequence
options. It constructs the real stage participant using the owner's genuine
initialized position-zero frontier. The owner is claimed once; a failed setup
after claiming requires fresh setup rather than retrying the same owner. No
ordinary `SerialSessionContext` is created or marked healthy for a stage.

The resulting `stage_owner` pin is explicit. Admission checks its actual shared
ownership group, the participant's retained owner, every runtime-resource
pointer, all compiled/model fields and the complete launch scope. Identical IDs
on changed metadata, copied binding objects or a no-op deleter do not establish
ownership. Each input revalidates the retained backing before continuation.
The same owner is pinned by source staging and the actual native retirement
ledger. Clearing a caller's setup cannot release work or storage still retained
by the participant or proof. This path also feeds the existing greedy consumer
through `HipDistributedSequenceSetup`; it changes no emission policy.

The participant uses the distinct typed stage snapshot path when a packed map
is present. It snapshots and restores actual local pools without republishing a
full serial session. `DistributedStageStateMapping` maps each owned range's
component-relative logical offset to its packed local offset; no contiguous
component-prefix assumption is made. Prefix and suffix retain their own causal
state and the replicated complete input-position word.

The callable `qualify_hip_distributed_stage_inputs` and
`qualify_hip_distributed_stage_greedy_request` share the existing execution and
comparison machinery with the full-resource cases. They require the actual
typed owner and independent complete serial reference, with no fabricated
ordinary anchor. Every owned logical range compares against the corresponding
interior range of that reference. Known inputs additionally compare every
terminal FP32 bit; greedy compares the existing cursor's selected sequence,
consumed/emitted counts and typed distributed generations. Absent remote ranges
are not compared as though they were allocated. Source fixtures include forged
same-ID arena/step/model metadata, mapping removal, foreign resource objects,
D1 generation changes and non-owning alias pins. These remain unexecuted source
fixtures until the separate qualification gates run.

## Remaining two-host bootstrap boundary

The model consumer, stage allocation, typed snapshots and callable cases are
implemented source. A standalone two-host bootstrap still must assemble actual
live providers and owners; there is no synthetic ready path or placeholder CLI.
In particular:

- Both hosts must load the pinned model, agree the same compiled/canonical cut
  and exchange their genuine initial peer metadata before creating the bound
  participants. Each host must admit its actual weight, scratch, local state,
  snapshot, pool and headroom budgets.
- The embedding owner must supply current native sessions, imported TX/RX pools,
  GPU visibility/consumer providers, lease registry and a qualified link profile.
  `LinuxDirectTransportDriverIo` still requires independent
  `DirectGpuStreamQualification` for the exact observed device/driver/pool path;
  its stock factory supplies no positive declaration.
- The owner must establish bounded protected step, token and greedy channels,
  separately from the driver's kind4 progress channel, and retain all sessions,
  streams, driver and physical allocations through actual retirement or proven
  lease closure. No implicit multiplexing or reconnect protocol is added.

`LinuxDirectGpuStreamCommissioning` and its reports are experimental observations,
not production IO providers, readiness declarations or model transport proofs.
They cannot be substituted into this factory to bypass absent qualification.
Executing these model cases, independently qualifying the physical path and
binding a real standalone bootstrap remain outstanding; source assembly alone
does not establish hardware readiness or memory fit.
