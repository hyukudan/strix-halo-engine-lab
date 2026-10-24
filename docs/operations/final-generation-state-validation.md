# Validation-only final generation state

The explicit --validation-final-state DIR option writes all seven actual session
state allocations after a completed eager generation request. It works with the
ordinary output path and GPU scalar selection, including prepared and shared
overlap composition. It does not enable RAW recording in GPU output mode or read
logits. Reference, RAW, prefix roundtrip and integer-head modes remain excluded.
The optional --validation-final-state-max-bytes N sets an explicit byte ceiling.

This is a validation artifact, not a speed-run option. Model/executable hashes
are collected only when recording or this export is requested. State output is
after the request and per-step timer collection. The stdout receipt marks the
entire invocation invalid for throughput interpretation; clean speed runs omit
the flag and perform no export allocations, copies, hashes or filesystem work.

Admission requires a fresh destination and an existing parent. Before model
residency allocation, the complete seven-allocation byte count plus a bounded
manifest allowance must fit the export budget. Export uses a 64 KiB reusable
copy buffer. It copies entire logical allocations, including inactive capacity,
rather than a component-wide active prefix that could omit later layers.

The manifest records actual next position, committed count, state generations,
health, arithmetic lineage, physical allocation generations and sizes, model and
executable hashes, numerical/kernel/launch identities, device/runtime identity,
and per-component sizes and hashes. The consumed history is the complete prompt
followed by all emitted tokens except the last. That last emitted token remains
unconsumed under the existing request controller, including EOS and capacity
stops. For a 43-token prompt and 512 generated tokens, the consumed frontier is
554, not 555. State equivalence compares logical component bytes at equal
consumed histories; scratch addresses and relocated execution IDs may differ.

The caller retains the real memory owner and original-device guard through all
reads. The exporter drains conservatively outside timers; every chunk uses the
existing guarded state reader. A copy, hash or binding failure drains or
fail-stops before the owner and copy destination can unwind. Uncertain device
work cannot be released as though it completed.

Files are written inside an exclusive owner-only DIR.partial directory.
Publication is an atomic no-replace directory move after all component files,
manifest and final model/executable/device checks finish. An existing final or
partial path is never reused. A failed export preserves its partial artifact and
publishes no final directory. Linux requires renameat2 RENAME_NOREPLACE; Windows
uses a same-parent MoveFile without replacement. Unsupported publication fails
closed. This is atomic visibility, not a claim of crash-durable storage.

Status: implemented source, not qualified. Compilation, tests, native model state
equivalence and source review are pending. Portable fixture source exercises the
actual request controller's bonus convention, complete chunked component bytes,
hashes, EOS/capacity stops, no-overwrite races, changed bindings and partial read/
publication failures. It is synthetic CPU evidence only when executed; no native
qualification or performance improvement is implied.
