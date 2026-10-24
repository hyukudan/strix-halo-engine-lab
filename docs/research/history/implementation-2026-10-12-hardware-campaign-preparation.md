# Hardware campaign preparation

This slice prepares target-side experiments and host-side evidence checks. It
does not install or boot a kernel, configure a connection, execute HIP/DMA, or
establish model or transport qualification. Build products and all personal
deployment inputs/results remain outside Git.

## Reproducible kernel and modules

The [private kernel builder](../../../lab/direct_gpu_stream/kernel/README.md) verifies
the pinned upstream archive and exact project patches, a hash-bound reviewed
configuration, compiler/dependency identities and a fresh private workspace.
It builds the complete kernel image, vmlinux and modules, runs actual modpost,
stages only beneath `INSTALL_MOD_PATH`, and checks module dependencies, vermagic
and GNU build IDs before writing a receipt. The development proof completed
this full workflow, including the real stream/controller/dependency modules.

That development configuration has not been booted and is not a substitute for
a reviewed target configuration. The runbook separately addresses the exact
boot kernel/configuration/symbol set, signatures, target installation, loaded
module observation, removal and retained-resource quarantine. No installation,
signing/enrollment, network configuration or physical reset is automatic.

## Independent experimental transport entry point

[The commissioning executable](../../../lab/direct_gpu_stream/COMMISSIONING.md)
connects the actual uncached HIP allocation/export, Linux UAPI, GPU
packing/acquire/unpacking, peer control and cleanup paths. The receiver compares
every acquired/unpacked byte before publishing consumer credit or reposting RX.
Controller completion, GPU acquire, consumer release, peer ACK and peer credit
remain separate observations.

The commissioning facade has no production backend/qualification interface.
Its privately owned provider remains unavailable for production even while its
isolated experimental dispatch operates. Loopback-only TCP control requires an
operator-established authenticated tunnel; the executable opens no LAN listener
and provisions no network. Both `--help` and `--check-config` run without GPU,
driver or socket work. One run carries one opaque payload in one direction;
reverse roles and use a new increasing peer-agreed epoch for another run.

The host-only C++ config check was linked against the real optional HIP library
and independently checked against Python for the full byte-pattern SHA-256 and
descriptor fingerprint. Such config parity is not an observation of hardware.

## Private plans and evidence

TensorMeld's separate `transport_commissioning` module prepares closed private
plans from pinned expectations/artifacts and checks paired native receipts. A
template covers both directions, seeds 0 and UINT64_MAX, and payloads 1, 4095,
4096, 4097, 65536, 1048576 and 16773120 bytes. Hardware, route, epoch and build
fields remain unresolved until actual target observations exist. The maximum
size requires matching ring geometry; unsupported mappings are not skipped into
success.

The paired checker recomputes the pattern and fingerprint independently, binds
current endpoint/module/GPU/runtime and descriptor identities, requires every
role-specific completion and released cleanup, and rejects missing or duplicated
campaign cases. A matching report requires independent native review; it never
emits production qualification, reservations or inference authorization.

[Numerical preparation](implementation-2026-10-12-native-numerical-campaign.md)
audits a selected actual GGUF through the existing host tools, derives actual
projection shapes and storage eligibility, and pins predetermined mathematical,
semantic and state/history criteria. Unknown references remain unknown. The
floating controls bind the actual BF16 or FP32 activation storage and remain
distinct from requested W4A16 diagnostics,
and the integer C1 eager LM head is distinct from the floating complete graph.
No phase, precision or source-storage identity is silently substituted.

The raw baseline recording and import instructions live with the numerical
laboratory. Full logits/state recordings are correctness diagnostics; recording
copies and file IO are not end-to-end performance evidence. Tokenization, chat
rendering and text decoding must be independently bound to the exact GGUF and
workload; raw token traces cannot attest those semantic transformations.

The [complete-model decode routes](implementation-2026-10-12-decode-execution-modes.md)
now connect eager floating, floating graph and eager alternate integer LM-head
execution to the RAW recorder. Each route can record an actual same-plan prefix
snapshot, intervening continuation, restore and continued generation. The
independent importer checks the full state, arithmetic lineage and exact
execution/profile/runtime tuple; a no-op restoration is insufficient. Graph
and integer execution require explicit native memory ceilings. This implements
the recording adapters, not their physical quality or speed acceptance.

## Target order and remaining execution gates

1. Observe each target runtime/GPU and confirm the reviewed boot kernel/module.
2. Run isolated native arithmetic/state fixtures, then eager/graph comparisons.
3. Collect the selected local model's reference logits, generated tokens, state
   and memory evidence before choosing another numerical plan or split.
4. Measure the copied transport baseline on the reconciled physical route.
5. Commission the GPU path in both directions, retaining failure and cleanup
   results as well as successful transfers.
6. Independently qualify exact external-DMA coherence and lifecycle bindings,
   then test model continuity and preparation-inclusive committed-token cost.

Local workstation, each companion alone, copied split and qualified direct split
are separate rows on the same supported workload. CUDA device/adapter evidence
is independent; this HIP experiment supplies no mixed-vendor direct-memory path.
One cable, two GPUs or arithmetic instruction throughput cannot select a winner.
Repeated imported-pool reuse, concurrent/dual-link behavior, graph internal peak,
quality and sustained workload/recovery remain physical execution gates.
