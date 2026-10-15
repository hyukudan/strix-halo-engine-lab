# Execution observability

The offline inventory and optional host dispatcher telemetry are the first
development increments for reconstructing effective execution and attributing
cost. They do not change the automatic optimization profile or numerical kernels.

## Offline inventory

`tools/qualification/runtime_inventory.py` reads existing CMake caches and JSON
outputs from the decoder, tensor audit, model manifest and platform probe. It can
hash the actual model and executable. It launches no decoder, probe or GPU job.

```bash
python tools/qualification/runtime_inventory.py --repo . \
  --cmake-cache build-host/CMakeCache.txt --receipt /private/decoder.json \
  --context-capacity-tokens 8192 --occupied-prefix-tokens 43 \
  --output /private/runtime-inventory.json
```

The optional receipt is an existing decoder stdout JSON object. Capacity and
occupied prefix are distinct; explicit declarations must agree with receipt
fields when present. Model file SHA-256 and the versioned model-content identity
are distinct and are both reported when the model is supplied.

A CMake cache attests configuration only. Tensor requirements attest neither
executable launcher support nor row/concurrency support. Missing effective
catalog, binary bindings and native qualification stay null. Profile resolution
comes from the actual decoder receipt, without duplicating the application
resolver. Conflicting evidence produces a conflict list and exit code 3;
malformed inputs use exit code 2.
An inventory with unresolved fields is not a qualified baseline.

The decoder's optional `--execution-inventory` adds the actual selected compiled
steps and host launcher identities to each stdout receipt path. The exporter
checks execution, kernel-plan and launch-plan identities, dense steps and bound
functions without invoking any launcher. Variant precision, layouts, phase and
qualification are explicitly catalog declarations; native qualification remains
null. The selected decode geometry does not admit verify or prefill geometries.
Without this option the larger inventory is neither constructed nor emitted.

## Optional eager host dispatcher telemetry

The decoder accepts `--host-dispatch-telemetry`. One JSON summary is written to
stderr when the dispatcher owner leaves scope, including failure unwinding.
Stdout keeps its existing decoder receipt. The option preserves automatic
profile selection and is rejected with graph execution: graph capture host calls
do not represent replay execution.

The observer measures steady-clock time around actual compiled host launcher
lookup, validation and invocation. Counts distinguish successful returns,
rejections before invocation, and exceptions after invocation. Events expose
execution identity, position, step, operator, layer and logical shape to a
caller-supplied observer; the built-in summary aggregates by operator. Observers
are non-owning, noexcept and serialized with their dispatcher, and must outlive
every dispatch call. The built-in counters use fixed storage and report overflow
by saturation. With no observer there are no timestamp reads or observer calls.

A successful return is not proof of a GPU submit, completion, CPU visibility or
token commit. A failed launcher may have partially enqueued work. Those GPU and
publication counts, graph counts, HIP copy bytes, allocations and synchronization
counts remain null. The hook does not change error handling or memory ownership.
It is not a GPU duration, complete token wall time or profiler replacement.
Keep the option off during speed measurements; instrumentation overhead has not
yet been measured on the target hardware.

`summarize_gpu_trace` separately aggregates supplied profiler intervals in one
clock domain. It reports their sum and busy union across all queues, checking
invalid ranges, mixed clocks and overflow. It never subtracts GPU sums from a
host-clock duration. It does not infer dependencies or a DAG critical path.

## Remaining integration

This increment exports the selected decode plan, not a complete cross-phase
capability catalog. It does not instrument every HIP allocation/copy/synchronization, trace request/session
epochs and graph replays, or measure prepare, selection, commit and delivery.
Those are remaining inventory/telemetry work, not implied by these interfaces.
Target HIP compilation, native execution and complete-workload measurements
remain separate gates from portable tests.

## Requested HIP process settings

With `--execution-inventory`, the decoder also emits the fixed allowlist
`hip_runtime_environment`: requested hardware queues, SDMA/architecture override,
kernel/copy serialization, launch blocking and device visibility. Values are
read from the actual decoder process; unset is null and an empty value remains
an empty string. The collector preserves this receipt instead of substituting
its own process environment. Effective hardware queues and observed concurrency
remain null: an environment request or two streams does not establish overlap.
No setting is modified. These source changes have not been executed yet.
