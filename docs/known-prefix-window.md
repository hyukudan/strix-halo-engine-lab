# Two known input tokens

The experimental known-prefix window compiles one layer-major traversal for two
already known inputs of one request. It does not predict the next greedy token.
The existing serial and speculative-verify APIs retain their behavior. There is
no public command-line or default-profile selection for this path.

The compiler derives two disjoint row views from the actual serial allocations,
including the serial expert workspace, and preserves each serial kernel ID.
Window memory and arithmetic have separate identities. The default request uses
explicit row loops for every step. Optional closed variants reuse Q8 shared-expert
weights and wave8 GDN projection weights across both rows. Other operators and
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
Successful completion publishes a distinct prefill-chunk interval covering both
inputs. It cannot claim a serial-only arithmetic history.

This implementation requires source, compiler, native component, full-model
logit/state and prompt-driver qualification. Component fixtures and structural
host checks alone do not qualify a whole-model window. Reusing weights and
reducing launches do not by themselves establish a speed improvement. Greedy
speculation and snapshot-lineage restoration remain separate work.
