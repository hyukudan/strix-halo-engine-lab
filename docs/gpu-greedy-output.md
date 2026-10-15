# Experimental scalar greedy output

`--gpu-greedy-token` selects an explicit eager generation path which retains the
complete LM-head output in the execution arena for each selected position,
checks every logit of that position for finiteness,
and selects the maximum on the GPU. Numeric ties, including signed zeros, select
the lowest token ID. Integer IEEE keys preserve ordering for finite subnormals
without depending on device floating-point denormal arithmetic.

The two reduction stages use persistent partials and a 32-byte reply. One owned
nonblocking stream orders reply poisoning, reduction and pinned result readback.
The reply contains execution identity, input position, epoch, token and status.
The consumer rejects stale metadata or nonfinite values before returning a token.
The owner retains the source arena, checks its allocation generation and view,
and drains its original device before freeing any pool resources, including on
partial submission failure. The source producer must already be completed.

The application uses the existing deferred completion where selected, or waits
for the eager producer. Output selection occurs after that producer's state
commit. If selection fails, the consumed position is not rolled back: the session
and arithmetic lineage are poisoned and no generated result is returned. The
selection owner drains its own stream; aborting a completed deferred producer
alone would not drain this subsequent work.

The scalar request controller preserves prefix state execution, EOS, capacity and
generated-versus-consumed token boundaries. The backend's default scalar method
retains the existing full-logits CPU selection for other implementations. The
previous full-logits request and continuation APIs remain available.

The option requires eager generation and rejects RAW recording, reference replay,
prefix roundtrip and integer-head modes. Those paths keep their complete-logits
behavior. Nonfinal prefix positions use a state-only intent: every model-state
step still runs, and only the terminal, state-free LM head is omitted. The final
prefix position and generated-token steps compute the complete head and select
from its full vocabulary. The receipt counts omitted heads. Final hidden values
remain available; this slice does not implement MTP, sampling or logprobs, or
change model arithmetic. Automatic kernel-profile selection remains intact;
the explicit output route has its own receipt identity and is not qualified by
the historical automatic profile's evidence.

For this mode, `generation_step_elapsed_ms_total` and `_per_step` include producer
completion and selection/readback for full-output steps only. Nonfinal prefix
timings are separate `prefix_state_only_elapsed_ms_*` fields and are not averaged
with full-output steps. Historical producer-only
`decode_elapsed_ms_*` fields are omitted in this mode. The receipt labels the
scope and reports vocabulary, calls, failures, kernel submissions, transfer bytes,
stream completions, omitted heads and requested pool allocation bytes. Host-dispatch telemetry
does not count the two selection kernels; their submissions have dedicated fields.
No target performance
gain is established by the reduced logical D2H payload.

Status: implemented source, not qualified. New compilation, portable/native tests,
model equivalence and performance campaigns are deferred until the implementation
stage is complete. Source fixtures cover controller boundaries and full-vocabulary
GPU selection, including nonfinite rejection, tail sizes and owner generations.

Test-only host fault boundaries simulate reported failures before each pool
allocation and after real poison, partial reduction, final reduction, readback
and completion operations. They exercise partial setup cleanup and draining of
actually queued work. Reply metadata, status and token corruption must also
reject publication and permanently fail the owner. These seams are compiled
only with `BUILD_TESTING`; builds without it contain no fault checks.
Additional source fixtures cover same-arena rebinding, all-NaN logits and retained
source lifetime. All fixtures still await execution.

Real driver allocation/launch/copy failures, failed-drain fail-stop behavior,
device changes and the application session failure paths remain separate
qualification gaps. Host boundary simulation does not establish recovery from
hardware or driver faults.
