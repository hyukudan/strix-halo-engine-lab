# Known input windows

The experimental known-prefix window compiles one layer-major traversal for 2, 4 or 8
already known inputs of one request. It does not predict the next greedy token.
The existing serial and speculative-verify APIs retain their behavior. The explicit `--known-prefix-window` command-line route now connects the native
owner and known-prompt controller. Default-profile selection remains unchanged.

`--known-prefix-rows 2|4|8` selects the logical width, defaulting to two. It
requires `--known-prefix-window` and is rejected on the prompt-lookup route.
The compiler derives disjoint row views from the actual serial allocations,
including the serial expert workspace, and preserves each serial kernel ID.
Window memory and arithmetic have separate identities. The default request uses
explicit row loops for every step. Optional closed variants reuse Q8 shared-expert
weights across native 2/4/8-row projection calls. Mixed GDN uses native
2/4/8-row calls for its three large Q8 projections while the two F32 gates and
recurrence remain ordered per-row. The all-Q8 GDN variant keeps contiguous
two-row tiles. Dispatch receipts identify each actual closed variant separately. Other operators and
unsupported serial variants keep their original one-row launches. GDN convolution
and recurrence run for row zero before row one; indexed attention and PLE remain
ordered per-row calls. P0 admits the synchronous file-mapped PLE backend without
an event-upload pool; other lookup/upload paths remain outside this window.
Both terminal heads run.

Admission validates both views, launch plans, resources, capacity, generations
and lineage before work begins. Every row-loop launcher must provide its pure metadata preflight callback;
missing callbacks fail admission. The owner must supply an owned eager stream with graph position arguments absent, completion and status
checks, a closed list of QSA reader implementation identities, and a quiescence
hook that terminates instead of unwinding if a drain cannot be proved. Merely
registering launchers does not satisfy deep admission.

Position history is a single host word. Before every admitted row reader it is
resolved again and set to that row's position. This does not add semantic state
generations. A failure after host or device mutation cancels PLE, drains, and
poisons the session without publishing positions, generations or new lineage.
Successful completion publishes a distinct prefill-chunk interval covering the complete
known input window. It cannot claim a serial-only arithmetic history.

This implementation requires source, compiler, native component, full-model
logit/state and prompt-driver qualification. Component fixtures and structural
host checks alone do not qualify a whole-model window. Reusing weights and
reducing launches do not by themselves establish a speed improvement. Greedy
speculation and snapshot-lineage restoration remain separate work.

The production opt-in is eager, full-head generation only. It rejects prepared
submission, shared overlap, islands, graph/integer heads, GPU scalar output,
RAW/reference replay, prefix-cache roundtrip and candidate recurrent/GQA/selector
layouts. It requires the resolved D1 profile and explicitly requested paired
wave8 GDN input/output; it never enables or disables incompatible flags silently.
The typed compiler resolves actual formats with mixed F32 gates admitted, rather
than assuming all five GDN projections are Q8. Compiled variant counts and actual
per-variant dispatch counters are emitted separately. No fixed whole-network R2
count is inferred from the model name.

Known windows use a separately retained window arena, owned stream and zeroed
completion record. The ordinary serial backend keeps its original resources and
D1 shadow transaction for an incomplete window suffix and autoregressive continuation.
Windows do not call that single-token shadow's begin/complete methods. Their native
owner publishes its configured causal rows before the prompt adapter reads the terminal
FP32 row, and all owners survive original-device quiescence. All LM heads remain
computed; no speculative second input or head elision is introduced.

The request controller's generation count is converted to total consumed rows
for the existing output/final-state contract by adding consumed prompt rows.
The common request timer includes the complete known-window prompt path.
The resumable cursor submits only complete selected-width windows; a shorter
suffix uses original serial R1 calls without padding or fake row commits. Per-step serial timing
is explicitly labelled as serial-tail-and-generation timing because it omits
known-window work. Actual successful-window telemetry uses saturating counters and
reports saturation without affecting computation. Source receipts remain
`not_assessed`; compiler/native, full-model, cache/prompt and timing qualification
for the complete source batch are pending. No speedup is claimed here.


## Explicit prompt-lookup request route

`--prompt-lookup --prompt-lookup-depth 0|1|2` is a separate experimental greedy
route, default off. It requires the same paired-wave8 GDN and D1 configuration
as the known-prefix owner. It rejects graph, integer heads, prepared/overlap,
islands, prefix roundtrip, RAW/reference output, alternate recurrent layouts and
candidate QSA modes rather than changing any requested flags. It is distinct
from `--known-prefix-window` and `--gpu-greedy-token`.

The prompt is consumed through the existing complete serial target. The terminal
head is selected on device; each cycle then uses the existing prompt-lookup
proposal and speculation transaction, with real R2 verification at depth two,
explicit serial verification at one, and frontier-only selection at zero.
Three persistent firstmax/finite selectors avoid full-vocabulary D2H. The serial
and R2 completion records have different execution identities on one retained
stream, initialized once; neither is retagged or cleared to hide an error.
Original state owners are reused; serial shadow completion is not applied to a
window transaction.

Accepted candidate inputs advance target state. Each cycle retains one emitted
bonus that has not been consumed. Only continuation consumes that bonus, without
appending it to draft history again. EOS, output budget and context exhaustion
leave the final emitted token pending, matching the ordinary greedy frontier.
Snapshots restore both bytes and producer lineage; native failure remains sticky.
Receipts report observed proposal/acceptance/replay/snapshot counts and explicitly
unassessed qualification. Timing uses the common complete request timer, not an
empty legacy backend vector. Final-state validation remains outside timing and
compares final logical history, not transient discarded verification work.

Source only: no compilation, tests, native execution or performance measurements
have been run for this integration. It does not implement sampling, learned
draft models, automatic depth selection or a default promotion.
