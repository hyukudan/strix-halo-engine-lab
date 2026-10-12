# Native numerical campaign preparation

`campaign.py` is a host-only audit, planning and recorded-evidence evaluator.
It never executes GPU code, downloads a model, promotes a kernel or supplies a
model baseline. The opt-in token decoder RAW recorder now emits complete eager
floating logits, actual greedy decisions and all seven state allocations.
`raw_import.py` converts paired recordings into this evaluator's strict record
format. Actual prompt tokenization and decoded output text remain explicit
external inputs; their pinned operator declaration is not tokenizer attestation.
The current recorder supports fresh eager state only. Restored-state, graph and
integer full-model capture producers remain unavailable. Operator fixtures and
the independent native commissioning campaign are separate gates.

Copy the neutral `workloads.json` template to a private directory before adding
actual prompts. The long-context case is a document-construction template, not
an already tokenized 16384-token workload. Replace its private text with the
complete deterministic document. Both reference and candidate must record the
same actual GGUF-tokenized input IDs and meet the declared minimum context.
Personal prompts, GGUF files, baselines and compiler/result artifacts stay
outside every Git checkout. Output files/directories must be new; resolved
symlink ancestors are checked. Nothing writes a supplied model or reference.

On Linux/WSL, audit a local GGUF v3 using the existing host-only project tools:

```sh
python3 lab/native_campaign/numerical/campaign.py audit-model \
  --model /private/model.gguf --model-sha256 ACTUAL_GGUF_SHA256 \
  --manifest-tool /private/build/strix-model-manifest \
  --manifest-tool-sha256 ACTUAL_MANIFEST_BINARY_SHA256 \
  --tensor-audit-tool /private/build/strix-audit-execution-tensors \
  --tensor-audit-tool-sha256 ACTUAL_AUDITOR_BINARY_SHA256 \
  --out /private/new-host-audit
python3 lab/native_campaign/numerical/campaign.py prepare \
  --audit /private/new-host-audit/audit_receipt.json \
  --prompts /private/prompts.json --out /private/new-plan.json
python3 lab/native_campaign/numerical/campaign.py validate \
  --plan /private/new-plan.json
```

Use the actual full-file hashes; placeholders intentionally fail validation.
Auditing hashes the full GGUF before and after the two fixed host invocations,
pins each tool before/after, bounds outputs/time, and requires the native tensor
audit to compile with zero missing/ambiguous requirements and no unclassified
tensors. It verifies tensor offsets/extents and layout/access identities. It
does not establish all operator type support or model fidelity. These auditors
are trusted local, single-process project executables; this utility is not a
general untrusted-command sandbox. A persistent rewrite is detected; a malicious
rewrite restored between observations is outside the trusted quiescent input
contract. Content hashes identify bytes, not their authenticity.

Prepared references and runtime bindings start explicitly `unknown`. The CLI
cannot invent them. The Python API exposes `prepare_plan(audit_path,prompts_path)`,
`validate_plan(plan)`, `identity(plan)` and `evaluate(plan,observations)`. A trusted
external capture adapter may bind runtime/reference artifacts before collecting
candidate outcomes; changed thresholds or bindings change the deterministic
plan SHA256. Binding fields are closed and exact. `runtime` then contains:

```json
{
  "status":"recorded",
  "executable":{"path":"/private/runtime","sha256":"ACTUAL_SHA256","bytes":123},
  "source_manifest":{"path":"/private/sources.json","sha256":"ACTUAL_SHA256","bytes":123},
  "mode":"eager_integer_lm_head",
  "candidate_id":"t0:decode:M1:W4A8",
  "input_storage":"fp32",
  "numerical_identity":"EXACT_APPROVED_RUNTIME_NUMERICAL_IDENTITY",
  "kernel_plan_identity":"EXACT_BOUND_KERNEL_PLAN",
  "prefill_identity":"EXACT_PREFILL_ARITHMETIC_AND_CHUNK_CONTRACT",
  "state_history_identity":"EXACT_ARITHMETIC_LINEAGE_CONTRACT"
}
```

The source manifest is `{"schema":"strix-source-manifest-v1","files":[PIN,...]}`.
Source file paths may be inside the source checkout, while this manifest remains
private. Every listed file is checked before and after evaluation. The caller
must provide the complete transitive source/build declaration; the evaluator
does not derive dependency completeness, prove loaded code or authenticate an
executable from this list. Root campaign execution/provenance gates remain
independent. Integer profile strings are checked against the exact current
Q4_0/IQ4_XS, A8/A4, IU8/IU4, BF16/FP32 source, binary64 group and binary32 RNE
contract. A generic precision label cannot substitute the profile.

The matrix derives M/N/K and checkpoint bytes from the audited tensor inventory:
decode M1, verify M2/4/8/16 and prefill M16/64/512 are separate experiments.
Actual expert rows remain unknown until an observed per-expert histogram exists.
Q4_0 K640/K2560 are representable; IQ4_XS K640 is rejected and its nonlinear
indices always require INT8 codebook operands. Storage block256, scale group32
and WMMA K16 are distinct. A4 preparation currently stores one byte per code.
The existing C1 LM head uses column1024 tiles, twelve internal allocations and
repeats activation preparation per tile. At N153600/K2560 this is 150 preparations
of 2560 elements. Requested scratch, padding and staging estimates are reported;
whole native allocation capacities, model/arena owner tails and runtime overhead
remain actual admission/profiler gates. Multi-row column tiling is a planned
diagnostic cost model, not an integrated prefill/verify model dispatcher.

Current integer model integration is only C1 eager LM head. It cannot bind to
prefill, verify or graph replay. W4A16 remains an unbound diagnostic request.
The floating LM head has separate `BF16_activation_control` and
`FP32_activation_control` candidates bound to its actual input storage and exact
profile. Default token decoding uses BF16 input, rather than inferring arithmetic
from FP32 output logits. Its implemented weight formats are GGML0 F32,1 F16,
8 Q8_0,20 IQ4_NL,23 IQ4_XS and30 BF16 with exact row blocks and byte extents.
GGML2 Q4_0 is accepted by the experimental integer adapter but rejected by this
default floating launcher. Controls are not labelled W4 when weights have a
different format. An integer graph combination is rejected. Floating tile
geometry/conversion costs remain unknown until measured for the exact profile.

Each reference is `{"status":"recorded","artifact":PIN}` or an explicit
unknown reason. Observations contain exact `plan_sha256` and a `cases` object
with **every** declared case ID. Each value is `{"result":PIN,"replay":PIN_OR_NULL}`.
State/context cases require separate fresh/resumed within-plan records. A `null`
replay keeps that gate unknown. Duplicating two fresh recordings is rejected.
Record fields follow `record()` in `campaign.py`:

- exact model/prompt hashes, executable/source pins, numerical/kernel/prefill/
  lineage contracts and actual canonical producer `history_sha256`;
- `capture_contract` with `origin`, `history_contract`, `restoration_record`.
  Fresh RAW imports preserve `single_row_eager_from_fresh_initialized_state` and
  a null restoration record. A restored record requires a different history
  contract and a pinned `strix-declared-state-restoration-v1` JSON declaration
  binding model/executable/source/history identities and a pinned `restore_log`.
  This trusted operator declaration is not native restoration proof; the current
  RAW producer cannot create it. See `record()` for its exact closed fields;
- actual `input_tokens`, generated `output_tokens` and decoded `output_text`;
- `teacher_forced_target_tokens`, full-vocabulary `logits` JSONL and separate
  autoregressive `generation_logits` JSONL, each pinned;
- all six state classes (`kv`, `recurrent`, `convolution`, `host_ngram`,
  `sparse_index`, `position`) as complete nonempty binary artifacts;
- elapsed `timings_ms` for input widening, activation preparation, weight repack,
  projection, scale/correction, conversion, status/synchronization, logits commit
  and independent `complete_wall`. Components may overlap; they are reported
  separately and never summed to invent end-to-end speed. Unmeasured categories
  are explicitly `null`, including diagnostic RAW captures; zeros must not be
  invented. Every evaluator report sets `performance_evidence:false`.

Each JSONL row is `{"position":0,"target_token":7,"logits":[...]}`. Positions
must be complete and ordered, every row has the exact audited vocabulary width,
and all values are finite within binary32 range. The evaluator streams bounded
8MiB rows (at most4096 positions, vocabulary262144, file8GiB), rather than loading
an entire vocabulary-by-context matrix. Generation rows must select the recorded
token by deterministic first-index argmax; teacher-forced targets are independently
bound and cannot masquerade as generated decisions. Sampling is not supported.

Predetermined defaults are absolute logit error≤0.05, mean KL≤0.001 nats,
mean NLL increase≤0.02 nats, teacher-forced argmax mismatch≤0.01, generated
token Levenshtein/max-length rate≤0.01, zero semantic failures and zero state
byte mismatches. These are explicit exploratory rejection thresholds, not
validated quality limits for a real checkpoint. Change them before inspecting
candidate outcomes and regenerate the exact plan identity. KL/NLL use independent
stable logsumexp calculations. JSON/tool/context outputs are compared with the
actual expected semantic object, including keys, argument values and boolean
types, rather than merely accepting parseable syntax. Resume requires exact
within-plan tokens, binary32 logit bits (including signed zero), state bytes, all precision/history identities and
canonical producer history. It does not infer equivalence to another A4/A8 target.

```sh
python3 lab/native_campaign/numerical/campaign.py evaluate \
  --plan /private/plan-with-real-bindings.json \
  --observations /private/observations.json --out /private/new-report.json
python3 lab/native_campaign/numerical/test_campaign.py
python3 lab/native_campaign/numerical/test_raw_import.py
```

Exit codes:0 numerical gates pass;2 invalid/tampered artifacts;3 unknown reference
or runtime;4 a measured numerical/task/state gate fails. `validate` returning0
means structural validity only. Every report sets `qualification:false` and
`native_execution_claim:false`, including synthetic test fixtures that pass all
arithmetic checks. Native execution, independent model fidelity, full-workload
cost, sampled speculation and alternative/restored model-record integration
remain separate.

## Eager RAW capture and typed import

These capture commands describe future target-hardware execution; none was run
while preparing this campaign. Use the same pinned GGUF/executable/config/context
for both runs and fresh private output directories. Supply actual numeric token
IDs, not the placeholders below. The teacher-forced run supplies the complete
prompt prefix P followed by explicit evaluation targets T. The greedy run supplies
only P and records its actual generated tokens G:

```sh
/private/build/strix-hip-decode-token /private/model.gguf PREFIX_IDS TARGET_IDS \
  --context ACTUAL_CONTEXT --record-raw /private/new-teacher \
  --record-state-policy final --record-max-bytes ACTUAL_ADMITTED_BYTES
/private/build/strix-hip-decode-token /private/model.gguf PREFIX_IDS \
  --generate CASE_MAX_NEW_TOKENS --eos-token ACTUAL_EOS_ID \
  --context ACTUAL_CONTEXT --record-raw /private/new-greedy \
  --record-state-policy final --record-max-bytes ACTUAL_ADMITTED_BYTES
python3 lab/native_campaign/numerical/raw_import.py \
  --plan /private/exact-bound-eager-plan.json --case CASE_ID \
  --teacher-forced /private/new-teacher --greedy /private/new-greedy \
  --binding /private/import-binding.json --out /private/new-typed-record
```

The closed binding object has schema `strix-raw-import-binding-v1`, exact
`plan_sha256`, `case_id`, `teacher_forced_targets`, and pins `compile_receipt`,
`tokenization`, `output_text`. `compile_binding` contains the exact runtime
`executable_sha256` and `source_manifest_sha256`. `observed` contains every exact
RAW `model_content_identity`, GPU UUID/full architecture including features,
HIP runtime/driver versions, execution/kernel/runtime compatibility/layout IDs,
context capacity, vocabulary size and `activation_storage` (`bf16` or `fp32`).
RAW activation storage must exactly match the selected runtime control and input
storage; FP32 output logits do not establish FP32 input arithmetic.
The bound runtime kernel-plan string is
the decimal RAW `kernel_plan_identity`. Tensor-manifest and persistent-state
layout identities are separate, not compared as though they were interchangeable.
The compile receipt is pinned bounded JSON with an operator-declared association;
the importer does not independently authenticate a compiled executable from it.

`tokenization` pins a `strix-external-tokenization-v1` object containing exact
`model_sha256`, `prompt_sha256` (the canonical case hash), `input_tokens`,
`output_tokens`, `output_text_sha256`, `tokenizer_identity` and
`asserted_by:"external_operator"`. The output text file must be UTF8≤1MiB. The
importer checks these associations but does not decode GGUF tokens itself.
Both recordings must share the complete observed tuple, evidence kind, allocation
layout and initial state bytes. Native-shaped input declarations retain their
evidence kind without becoming native execution or qualification claims.

Teacher-forced rows P−1 through P+T−2 are selected with their actual next-fed
target tokens. Greedy rows P−1 through P+G−2 must agree with first-index argmax.
The final generated token, including EOS, is emitted without being fed; only the
greedy final snapshot supplies output state and actual fed-token history. Both
GDN and PLE convolution allocations are included in the convolution artifact,
along with canonical component/layout metadata. All RAW rows and every initial,
final or per-step state component are checked, including unselected prefix rows.
The importer rejects `INCOMPLETE`, missing receipts, altered hashes, escaping
paths, missing components, changed layout/generations, nonfinite/partial logits,
false stops and model/source/runtime substitutions before publishing a record.

Import bounds are 32768 prompt tokens plus4096 targets, vocabulary262144,
64MiB metadata, 8MiB metadata/converted-logit rows, at most4097 state snapshots,
and64GiB aggregate RAW artifacts per capture. Metadata and logit rows are bounded;
state copying and hashing stream1MiB/4MiB chunks. Converted logit and individual
evaluator state files remain bounded to8GiB. A context of16384 at vocabulary153600
already requires about9.4GiB of RAW FP32 logits before state; the decoder's default
8GiB budget therefore requires explicit increased admission, never a silent cap
override. `final` state policy avoids quadratic full-state snapshots.
Unknown timing categories stay null and `performance_evidence` remains false.
Fresh output retains an `INCOMPLETE` marker on any failed import. Only a finished
`record.json` plus `import_receipt.json` without that marker is usable.

The importer creates an individual baseline/result artifact, not a fabricated
full-coverage observation set or resumed replay. A full numerical campaign still
needs all declared cases, an independently trusted reference, an actual restored
producer for state/context gates, and separate integer/graph capture adapters.
No real-model quality, throughput, native execution or hardware qualification
has been measured here.
