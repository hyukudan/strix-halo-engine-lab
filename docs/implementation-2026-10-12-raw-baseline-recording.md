# Eager RAW baseline recording

The existing `strix-hip-decode-token` executable has an opt-in actual recording
path for its current native eager computation. The ordinary CLI behavior is
unchanged. This supplies data for a later independently bound model-correctness
campaign; recording, hashes and offline compilation do not authorize a numerical
mode or qualify hardware.

```sh
strix-hip-decode-token MODEL.gguf TOKEN_ID ... \
  --context 128 --record-raw "$fresh_private_directory" \
  --record-max-bytes 8589934592 --record-state-policy final
```

Adding the existing `--generate N` records the actual greedy run from the exact
supplied prefix IDs. Omitting it records the supplied IDs as teacher-forced raw
inputs. The adapter uses the existing BF16 activation storage and eager kernel
catalog, with complete FP32 logits; FP32 output storage is not a declaration of
FP32 activation arithmetic. It does not tokenize, decode text, optimize prefill,
capture graphs, select integer variants or perform prefix-cache roundtrips.
Recording with `--prefix-cache-roundtrip` is rejected.

## Artifacts and identity

The output must be a fresh absolute directory with an existing parent outside
Git. Existing files/directories and Git ancestors are rejected. Artifacts remain
private. The writer has an explicit total-byte budget (8 GiB by default), a
bounded maximum step count and chunks of 64 KiB. It checks worst-case state,
logits and metadata storage before creating the directory; every actual write
shares the same budget. Full-state snapshots use a reusable bounded copy buffer,
not a second allocation of the model's complete retained state.

`--record-state-policy final` records the actual initialized state before the
first token and the actual state after the last executed input. `each` records
every post-token state instead; use it for small correctness fixtures. Each full
snapshot includes all seven layout components covering six persistent classes:
attention KV, recurrent matrix, GDN convolution, PLE convolution, sparse index,
host n-gram history and position history. Both convolution components remain
separate files. Allocation offsets, storage, byte lengths, state flags,
generations and position are recorded. Raw state bytes are opaque native storage,
not a portable semantic restore format; their native byte order is explicit.

Each executed token produces the full vocabulary of finite FP32 logits in
little-endian order. `steps.jsonl` records its input ID, role, position, resulting
position, complete-logits count/artifact and first-index-on-tie argmax. An explicit
teacher-forced target is the next supplied ID, or null for the final supplied
input. The single `history.u32le` artifact contains all actually fed IDs in order,
including prefix and consumed greedy IDs. It excludes the final emitted greedy
ID when that ID was not fed into another execution step. The receipt's separate
`generated_output_tokens` list preserves those actual emitted IDs and exact
EOS/context/maximum-token stop precedence. Teacher-forced recordings emit no
generated-token list.

`states.jsonl` describes initial/final or initial/post-token snapshots and their
component artifacts. `receipt.json` uses schema `strix-raw-eager-baseline-v1` and
hash-binds that file, `steps.jsonl`, the complete history, and each referenced
binary artifact through its manifest. It records exact GGUF file SHA256 and
executable SHA256, the loader's separate model-content identity, execution/kernel
plan/layout/runtime compatibility IDs, GPU UUID/architecture and HIP runtime/
driver versions. Native UUID/version metadata must be present and nonzero.
The receipt's `activation_storage` is an explicit validated `bf16` or `fp32`
value; this native CLI binds `bf16` to its actual BF16 catalog and two-byte
activation configuration. An importer must match that observed value to its
declared numerical input storage, independently of the FP32 logits dtype.
Native device/runtime/allocation owners are rechecked around copies and before
publication; model and executable bytes are rehashed before publication.

The native app completes each eager token before reading logits and state.
Recording IO and hashing run outside the existing token timers, but they can
alter cache behavior and the execution environment. The recording explicitly
sets `performance_evidence=false`; neither its timers nor recorded CLI timing
output is a workload performance qualification.

## Failure and publication

An `INCOMPLETE` marker is created before capture. Copy/write/flush/hash/binding,
budget or token-order failures prevent final receipt publication. A successful
publisher closes and hashes artifacts, rechecks every artifact and native
binding, removes the marker, then renames its pending receipt to `receipt.json`.
Consumers must require that final filename and the absence of `INCOMPLETE`;
pending files or a provided `complete=true` value cannot replace those gates.

A thrown state reader can have exposed the destination to unfinished work. The
portable helper retains the bounded copy buffer and strong captured reader
owners in an explicit reachable process-lifetime registry, with no retry or
invented release proof. Readers must retain their source owners. In the native
app, errors from token execution/synchronization or synchronous D2H copies drain
the exact selected HIP device before normal exception cleanup. If that drain
cannot retain the original UUID/architecture/runtime/driver/allocation-owner
binding before and after synchronization, it also fails closed. If quiescence
cannot be proved, the diagnostic process flushes a bounded error message
and fail-stops without normal resource destructors or a success receipt. This is
not automatic recovery or a claim that failed GPU work was released normally.

The identity boundary assumes trusted, quiescent local artifacts, toolchain,
reader callbacks and serialized session ownership. Lasting changes are checked;
the recorder does not assert immutable execute-by-handle or adversarial runtime
dependency provenance.

## Evidence and remaining adapters

The portable writer tests use explicitly injected fixture readers. They cover
complete little-endian logits/history, all state components, bounds/overflow,
enum/native identity guards, stale histories/targets, nonfinite logits, exact
greedy EOS/context boundaries, unknown-copy retention, artifact drift and failed
publication. Strict portable compilation and ASan/UBSan with leak detection
passed. The modified actual CLI compiled offline with ROCm 7.2.4 for `gfx1151`.
No GPU/model execution or recording from physical hardware was performed.

An external campaign adapter still needs the exact source/compile receipt and
numerical/kernel/prefill/history contracts, explicit prompt-case and tokenizer
binding, teacher-forced evaluation span, and independently supplied decoded text
where semantic evaluation requires it. The receipt identifies these as external
bindings. It preserves `evidence_kind` as native-eager-raw or injected-fixture and
always keeps `qualified=false`. A fixture, artifact hash, raw argmax match or
successful recorder invocation cannot promote itself to native qualification.
