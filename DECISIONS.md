# Decision register

## Accepted

| ID | Decision | Reason | Evidence | Review |
| --- | --- | --- | --- | --- |
| D-001 | Build a dedicated inference engine for Strix Halo and Qwen3.8-Flash-Next. | Specializing for one platform and one model allows kernels, memory layout and scheduling that a general-purpose runtime cannot assume. | Scope confirmed by the owner. | Reopen only by explicit decision. |
| D-002 | Retired. | Superseded by the publication policy in `AGENTS.md`. | — | — |
| D-003 | Start with native Linux, ROCm and `gfx1151`. | It is the real target environment and allows hardware-specific optimization. | Target hardware (Bosgame M5, Ryzen AI Max+ 395). | Revisit after a first stable version. |
| D-004 | Use GGUF as the initial input format. | Public format, weights available, no dependency on private formats. | `GGUF_READER.md`. | Measure a lossless internal cache later. |
| D-005 | Quality and reproducibility are gates, not later tasks. | MTP, recurrence and quantization can produce misleading gains. | `docs/qualification-ledger.md`. | Permanent. |
| D-006 | The priority UI is a remote administration plane. | Operation, observability and benchmarks add more value than another chat client. | Scope confirmed by the owner. | Chat remains an optional integrated playground. |
| D-007 | Build and compile offline before the hardware is available. | Validates toolchain, objects and contracts without pretending to have dynamic results. | E-015. | Revalidate everything on the real machine. |
| D-008 | The first slice is AR + MTP, C1/C2/C4 and a qualified context below 262K. | Reduces numerical, memory and scheduling risk at the same time. | E-019, O-11. | Extend only after passing correctness, memory and stability. |
| D-009 | Separate engine, gateway/chat and administration as privilege domains. | A convenient UI must not gain direct control of the GPU process or a shell. | O-14, O-15. | Permanent unless the threat model is revised. |
| D-010 | Treat the official Qwen repository, paper and checkpoint as the primary architecture source. | Contracts are specified from the model's own definition. | E-020, E-021. | Reopen if Qwen publishes an incompatible revision; always pin the SHA. |
| D-011 | Expose nearly the maximum GTT, but admit workloads with an independent effective reserve of 16–20 GiB. | GTT widens the mapping; it does not create RAM or protect the host from OOM. | E-022, `MEMORY_BUDGET.md`. | Adjust the reserve only with telemetry from the real hardware. |
| D-012 | Initially keep the 128 GiB single-node profile within an envelope of about 100 GiB of resident weights, until A0.6 justifies another limit. | Preserves the effective reserve and leaves room for KV, recurrence, scratch, runtime and recovery; avoids turning the maximum GTT into a load budget. | E-028, E-032, E-033. | Reopen after real UMA measurements and quality of the candidate quantizations. |
| D-013 | Break QSA ties by the lowest block index when scores are identical. | The canonical forward uses `torch.topk`, whose tied indices are not guaranteed stable; the engine needs reproducible replay and the same rule in serial and MTP. | E-041, E-043 and fixture `qsa_tie`. | Reopen only if Qwen publishes an explicit, incompatible canonical tie-break. |
| D-014 | Model each physical USB4 link as an autonomous transport with its own epoch, lease, TX/RX pools and accounting; dual-link mode is a policy above two qualified links. | Avoids assuming summed bandwidth, isolates failures and allows flow distribution, directional split or striping only with evidence of independence. | E-044 to E-047; docs 12 and 16. | Reopen only if the hardware shows an aggregate resource with common ownership/synchronization and an end-to-end benefit. |
| D-015 | Keep low-level wave router selection OFF in manual mode; the content-bound automatic profile may select wave as part of the adopted v3 bundle after admission, with unchanged softmax/slot-order FP expressions, immutable preflight manifest admission and launcher-only K32; retain default/ranked alternatives. | Pointer identity to a validated owned deep copy avoids hot-path tensor scans and unsafe caches on mutable declared IDs. Raw GGUF rejects groups/bias before shape reduction; registration cannot inspect metadata absent from its API. | E-050; `docs/implementation-2026-10-15-wave-router-topk.md`; host oracle/admission checks. | Reopen for expressible grouped routing/bias, ISA changes, full-array mismatch or failure to improve measured latency. Canonical v3 target/model gates passed as recorded in E-056; other workloads remain separate. |
| D-016 | Keep an opt-in C1 eager mode with a persistent first MoE error and completion before the token is published. | The domain and the complete bits keep the first error; a temporary copy of the host history avoids publishing position or n-grams on failure. KV/GDN stay in place, and a failure after dispatch poisons the session. | E-051; `docs/deferred-moe-status.md`. | Review, HIP, fixtures and fidelity of the canonical v3 slice closed in E-056; other cases need qualification. Reopen on numerical differences or no measured benefit. |
| D-020 | Prototype exact whole HC projection chains with explicit 32-thread launch bounds and P1/P2 prefetch; do not integrate before measurement. | Alternative: retain rows8; reduction splitting is excluded. Bound registers using actual compiler evidence, then compare hot and rotating real-weight copies. | E-055; implementation document and diagnostic fixtures. | Risk: dependency latency, cache bias or register pressure can dominate. Reopen after fresh ISA, device exactness and real event measurements; reject if no benefit. |

D-017, D-018, D-019 and the follow-ups to D-015/D-016 are recorded as sections below. D-021 onward cover the
speculative decode path.

## Pending

| ID | Pending decision | Data needed |
| --- | --- | --- |
| P-001 | Project name and license. | Owner preferences and contribution policy. |
| P-002 | Canonical development checkpoint/quantization. | Size, quality, available MTP and weight license. Current working choice: UD-Q4_K_XL plus the Q8_0 MTP sidecar. |
| P-003 | Registered mmap versus anonymous copy/huge pages. | Load, decode, memory and page-pressure measurements. Current working choice: registered mapping (see D-028). |
| P-004 | Exact continuous batching policy. | C1/C2/C4/C6/C8 results and admission cost. |
| P-005 | Wave32/wave64 per kernel family. | Paired per-shape and end-to-end experiments. |
| P-006 | Internal weight cache. | Lossless prototype, size and startup time. |
| P-007 | Administration plane stack. | Prototype, threat model and deployment preferences. |
| P-008 (P0) | Canonical checkpoint and complete budget within 128 GB UMA. | Real per-tensor size plus OS, KV, recurrence, rollback, scratch, libraries and margin. |
| P-009 (P0) | Memory residency/registration model on `gfx1151`. | Documentation and tests with the exact Linux kernel/ROCm; confirm XNACK/HMM. |
| P-010 (P0) | Ledger of canonical architecture sources. | Repository, commit, license and contract for QSA, DeltaNet, MoE, MTP, tokenizer and template. |
| P-011 (P0) | `gfx1151` coverage of the chosen ROCm libraries. | Offline inventory of rocBLAS/hipBLASLt code objects and fallbacks. |
| P-012 (P0) | Policy compatible with the Qwen Community License 1.0. | Use case, distribution, commercial service and legal review if applicable. |
| P-013 | Final GTT, reserve and recovery values after a dirty kill. | Physical RAM, BIOS, final kernel/ROCm and pressure/restart tests. |
| P-014 (P0) | Canonical QSA semantics: top-k selection, tie-break, indexer and relation to dense attention. | Official paper/code, CPU oracle and KL/NIAH tests; do not assume dense is equivalent. |
| P-015 | Direct GPU-owned DMA-BUF backend and preferred dual-link policy. | Exact kernel/ROCm on the M5, DMA-BUF export/import support, consumer-credit/repost, reliable re-arm, 1×USB4/2×USB4 measurements and real effect on decode/prefill. |
| H-001 | Tiled/DPP GDN as the first optimization candidate after profiling. | GDN contract/oracle and own traffic/time profile. |
| H-002 | Direct PLE reader before demand-paged mmap. | n-gram/PLE contract, p99 latency, page cache and A/B on hardware. See D-030 for the current page-cache approach. |

## Rejected for now

| ID | Alternative | Reason |
| --- | --- | --- |
| X-001 | Extending a general-purpose runtime as the final product. | Does not meet the goal of a runtime specialized for one platform and model. |
| X-002 | Retired. | Superseded by the publication policy in `AGENTS.md`. |
| X-003 | Designing for multiple GPUs/models from the start. | Would add abstraction before the `gfx1151` fast path is validated. |

Rejections from the speculative decode work are in the "Measured and rejected" section below.

## Review format

Every new decision states: alternatives, evidence, risk, consequence for the spec and the condition for
reopening. Acceptance actions are linked here and in `DESIGN_ACCEPTANCE.md`.


## D-017 (2026-10-15): explicit indexed-attention Q8 input prefetch experiment

Keep the prefetch schedule off by default and in the original HIP translation
unit. Manifest eligibility is per matrix; operator bindings explicitly retain
legacy IDs for wholly ineligible layers. Receipts distinguish operator-step and
projection counts. Do not infer target qualification from an arithmetic argument
or a host test. Reopen adoption after linked legacy/candidate ISA, device fixtures,
full-array and token fidelity, and paired complete-workload measurements. The first
increment rejects speculative verify because the actual indexed launcher is C1.

Alternatives: retain only the original T256 schedule, or choose a distinct opt-in
loading schedule with the same leaves and reduction. The latter permits isolated
qualification without replacing the existing path. Risk: source equivalence does
not establish compiler behavior or complete-workload benefit. Spec consequence:
require explicit per-tensor eligibility, topology/dispatch receipts and all native
gates before adoption. Evidence: E-052. Reopening conditions are the gates above.

## D-018

Preserve each original HC gate chain, coefficient rounding, sigmoid and stream order while assigning independent chains to lanes. Keep normalization/projection/injection kernels untouched. Stage original Q8 bytes without repacking, use explicit metadata-selected base fallback, and bound dynamic LDS to 64 KiB. Rank320 bank permutation is documented narrowly; other admitted ranks are not presumed conflict free. Reject speculative selection for this serial decode slice. Alternatives that split the reduction or alter rounding are outside this exact optimization. Reopen after actual ISA/resource checks and bitwise complete-model evidence, or reject the candidate if any arithmetic mismatch appears.

D-016 follow-up: alternatives are the synchronous per-layer status boundary or
an opt-in token completion boundary with transactional host histories. The latter
keeps the complete first error through one shared host/device merge helper.
Risk: a host exception after a recorded device error remains the reported error;
this precedence is intentional and must not publish token metadata. The current
per-token reset may queue before serial preflight and is retained conservatively.
A zero-on-construction/dirty-state optimization needs owner recovery injection
coverage before changing lifecycle semantics. Evidence: E-051. Reopen after
native compiler/ISA, bounded async fixtures, model fidelity and paired workload
measurements; see `DESIGN_ACCEPTANCE.md` and `docs/deferred-moe-status.md`.

## D-019 (2026-10-15): explicit Q8 shared-expert loading experiment

Choose a default-off gate/up/down loading schedule with unchanged T256 leaves,
reduction and activation/store semantics. Alternatives: retain only the existing
schedule, or change the arithmetic/activation fusion. The isolated loading change
keeps exactness independently testable and preserves the scalar gate and SiLU.
Evidence: E-054 and the project-owned implementation note. Risk: eight rows per
workgroup and reduced waves can have shape-dependent cost; source equivalence
cannot establish compiler behavior or useful workload speed. Spec consequence:
explicit per-tensor eligibility, operator IDs, actual-dispatch receipts and all
native fidelity/latency gates precede adoption. Reopen after linked ISA, actual
fixtures, full-model equality and paired preparation-inclusive measurements.

D-015 decoder policy follow-up: use a content-bound automatic profile for the
admitted eager registered checkpoint, retaining manual selection and explicit
ranked routing. Low-level flag defaults remain OFF. The adopted source profile
`canonical-a8-{wave,ranked}-qsa-hc-shared-deferred-v3` also selects QSA input
prefetch, shared-expert prefetch and deferred MoE status after metadata/content
admission. Its own native, full-model fidelity and automatic workload gates passed
at source `265fd78762884d18eb32aa60294408f080088d52` (E-056); manual flags and
all existing fallback contracts are retained. This does not adopt unrelated
projection prototypes or other checkpoints/contexts. See `docs/decode-optimization-profile.md`.

D-015 qualified adoption: fast-forward primary source to the exact qualified v3
revision above, without another code change. Alternatives were retaining the
preceding wave+HC automatic bundle or keeping all five modes manual. The direct
same-node clean-auto comparison favored v3 by a descriptive median +15.55% across
three retained pairs after all model/native gates; evidence E-056. Risk: limited
prompt/context and early stopping do not establish population performance or
quality. Reopen on fidelity failure, adverse workload behavior, changed compiler
semantics or a newly measured alternative. Source adoption does not assert remote
binary deployment.


# Speculative decode decisions (2026-10-16 to 2026-10-17)

All measurements below are from one Bosgame M5 node (Ryzen AI Max+ 395, 128 GB), UD-Q4_K_XL, a 43-token
prompt and 256 generated tokens, compared against the same binary's serial output. They are local workload
numbers, not qualification evidence. Every adopted change kept the output bit-identical.

## D-021: exact multi-row verification windows

Verify drafts in one target forward over T = 2, 4 or 8 consecutive rows. Every multi-row kernel performs each
row's arithmetic, reduction tree and rounding exactly as the single-row kernel does, so greedy acceptance is
exact and output equals serial decode.

Why: a single-token forward is bandwidth bound at ~41 ms; a window reads each weight once for all rows, so
verifying T tokens costs far less than T forwards. Before this work a fixed R2 window ran at 15 tok/s, slower
than plain decode (19.4), because HC, the MoE union and the LM head ran per row. Alternative: tolerance-based
verification. Rejected because it gives up output identity, which is this engine's correctness gate (D-005).
Each kernel package has a bitwise test against the 1-row kernel (see `docs/VERIFY_WINDOW_PLAN.md`). Reopen
only for sampled (non-greedy) decoding.

## D-022: adaptive window size {2,4,8} with an online cost model

Each cycle, T = 1 + the number of leading drafts with probability >= `--mtp-spec-min-p`, rounded up to an
available window owner. The verifier then picks the owner (or a serial step at T = 1) with the best expected
tokens per ms, using per-bin acceptance and an EMA of measured cycle wall time per owner
(`--mtp-target-window-rows-set 2,4,8`).

Why: a fixed window wastes rows when the draft is weak and leaves tokens on the table when it is strong.
Moving from `{2,4}` to `{2,4,8}` was the largest single gain. `--mtp-spec-min-p 0.85` was best; 0.7, 0.75, 0.8,
0.88 and 0.9 were all worse.

Robust cost model (`65ca700`): the first window of each owner is not used as a sample, and samples are capped
at 2× the current estimate. The first R8 window pays one-time costs (hundreds of ms), which used to inflate the
R8 estimate enough to veto R8 for the rest of the run. Reopen if window kernels for other row counts (for
example R6) become available.

## D-023: longest-prefix acceptance with deferred GDN commit

Commit the longest accepted prefix of a window without a state checkpoint (`--mtp-window-prefix`). The GDN
window never commits its own rows: the next window first applies the previous window's accepted rows (count
and slot from a device control word) to the recurrent and conv state, writing it once, and a standalone commit
kernel applies pending rows before a serial step. Windows of different sizes share one deferred record. PLE
keeps per-row snapshots and the commit selects the last accepted row. QSA appends all rows and restores only the
indexer tail.

Why: the checkpoint/restore alternative copies the recurrent state every cycle and restores it on every partial
acceptance. On a 65-token R4 run, prefix acceptance produced identical output with zero restores and was 46%
faster (`1212d3f`). Risk: owner switches must hand over pending rows; this is covered by a GDN test that mixes
window sizes.

## D-024: A8 numerical profile in the windows

Run the windows under the A8 profile `hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup-v1` (int8 activations per
group of 32, `dot4`, for MoE gate/up and down). The routed-expert union reproduces the serial A8 kernels per row,
so all 48 MoE layers use the union (43 did before) (`d1cffab`).

Why: the profile saves ~12 ms per token on the serial path (54 → ~41 ms). The windows must use the same
arithmetic as the serial A8 path; otherwise the serial-equivalence gate cannot hold. Consequence: the
`value-key` GDN recurrent storage, which is incompatible with this profile, is not used with windows.

## D-025: routed-expert union per window

Build the per-window plan on the device (parallel CSR, rows ascending per expert) and run grouped gate/up/down
that decode each distinct expert once for all its rows (`--mtp-window-expert-union`).

Why: the union replaced a single-thread CSR kernel and per-row expert reads that cost ~17 ms per R2 window.
The union is still the dominant cost of an R8 cycle; that is inherent to verifying eight exact rows, since
eight rows touch more distinct experts. The down projection is already at its byte floor.

## D-026: draft head over a vocabulary subset, with Q4_0 copies

The MTP draft head computes logits only over a static vocabulary subset (`--mtp-draft-vocab FILE[,FILE...]`,
little-endian `uint32` IDs, built with `tools/build_draft_vocab.py`) plus up to N tokens seen in the prompt and
output (`--mtp-draft-context-vocab N`). The draft chain runs on the device without host syncs. The subset rows
are requantized from Q8_0 to Q4_0 (`--mtp-draft-q4`), and the other draft projections use Q4_0 copies by
default (`STRIX_MTP_DRAFT_Q4_ALL=0` disables them).

Why: the draft only steers acceptance; the target verifies every token, so the draft's precision and vocabulary
cannot change the output. A full-vocabulary Q8 head re-reads 0.67 GB per draft step. The Q4_0 projection copies
were neutral with `{2,4}` windows (fewer bytes per step, one more cycle) and +2.4% with `{2,4,8}`, so they became
the default with `{2,4,8}`. A smaller draft vocabulary gave 49.5 vs 48.4 tok/s on the code prompt. Risk:
vocabulary coverage is workload dependent (prose accepts fewer drafts).

## D-027: any-order launches for independent kernels

Launch kernels that do not depend on the kernel queued just before them with `hipExtAnyOrderLaunch`: shared
expert gate/up after the router top-k, GDN F32 gates after the Q8 input projection, later QSA input groups, the
q/k norm-RoPE and the MTP draft k/v projections (`47ce5fa`, `b0b6d9e`). On by default; `STRIX_ANYORDER=0`
disables it.

Why: per-kernel launch overhead is about 2 µs on this platform, and in-order queues serialize kernels that could
overlap. Measured 45.3 → 46.0 tok/s, output identical. Dependencies are explicit; only kernels whose inputs are
complete before the preceding kernel are reordered.

## D-028: fine-grained registered weights

Keep the model registered with the default (fine-grained) memory attributes. Coarse-grained registration is
available as opt-in (`STRIX_WEIGHTS_COARSE_GRAINED=1`) but measured slower: 44.3 vs 45.0 tok/s.

## D-029: no hipGraph for the decode loop

Do not capture the window/draft cycle in hipGraphs. A graph lowered the measured per-kernel cost only from
1.98 to 1.82 µs, which does not pay for capture and input plumbing on a path whose window size changes every
cycle. No ROCm environment setting lowered the ~2 µs floor. Any-order launches (D-027) reduce the cost of
serialization instead. Reopen if a lower-overhead dispatch mechanism becomes available. The existing opt-in C1
graph mode (`--execution-mode graph`) is unaffected.

## D-030: populate the file-backed PLE table after registration

The PLE n-gram table (`per_layer_token_embd`, ~29 GB IQ4_NL) is file-backed and not registered. After model
registration the engine advises `MADV_RANDOM` and populates it with `MADV_POPULATE_READ` (`dd141a7`;
`STRIX_PLE_WARM=0` skips).

Why: when another process evicted the page cache, every PLE lookup fault triggered readahead and decode read the
disk at ~1.4 GB/s, which made runs slow and noisy. After population there are no decode-time disk reads.

## D-031: other adopted bit-exact kernel changes

Each of these kept the per-row arithmetic and was measured on the complete workload:

- PLE window key/value projections in one pass over the weights; R8 windows as two R4 passes (423 → ~260 µs).
- Argmax of every window row on the window stream into host-mapped memory, no per-row selector round trips
  (`STRIX_MTP_GREEDY_ROWS=0` keeps the selectors).
- Router rows with 128 coefficients in flight per lane step.
- HC rows split-K with 2 outputs per block by default; lane-chain HC output compiled for 8 rows.
- QSA attention output with eight slots' token indices and KV values loaded ahead of the unchanged
  score × value chain (+0.6 tok/s).
- GDN rows sites with fused FMA via `__builtin_fmaf` (same single rounding; neutral, kept).

## D-032: dense A8 in the recommended profile

Recommend `hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1`: the profile of D-024 plus int8 activations per
group of 32 for every dense Q8_0 projection (`dot4` in decode, iu8 WMMA in prefill, bit-identical between the two).
The FP32-dense profile stays available.

Measured on one Strix Halo node (Qwen3.8-Flash-Next UD-Q4_K_XL):

| | FP32 dense | dense A8 |
| --- | --- | --- |
| Prefill, 1032-token prompt (one block) | 410.5 tok/s | 707.7 tok/s |
| Decode, MTP windows 2/4/8, 256 tokens (3 interleaved rounds, replay error 0) | 51.9 tok/s | 53.9 tok/s |
| Top-1 agreement with a Q8_0 reference (49,001 positions) | 95.52 % | 95.55 % |
| KL to the Q8_0 reference | 0.01940 | 0.01978 (+1.9 ± 2.0 %) |

Why: the quality cost is within the difference between two exact implementations of the same weights, and far
below quantizing activations to Q8_1 (+30 % KL). The two profiles pick different tokens at 2.8 % of positions,
almost all near-ties (median reference top-2 margin 0.22 nats, against 3.10 overall). The one per-category signal
is Spanish prose top-1 (−0.55 ± 0.23 pt) with unchanged KL.

## D-033: separate prefill numerical profile

`--prefill-numerical-profile NAME` selects the arithmetic of the prompt prefill (tiled known-prefix prefill
blocks, or every MTP prompt row); `--numerical-profile` keeps selecting decode, MTP windows and checkpoints.
Without the flag the prefill uses `--numerical-profile`, and tokens and every existing receipt field are
unchanged. Admitted today: the dense-A8 sibling of the decode profile, in both directions (see
[docs/prefill-numerical-profile.md](docs/prefill-numerical-profile.md)).

The prefill runs a second serial execution compiled from the same requests with only dense A8 changed; the two
are checked to be siblings (same steps, state access, IO and arena layout). The session is handed to the prefill
execution for each prefill block and back to decode afterwards; only the executor identities change and the
arithmetic lineage keeps the producer of every row. The prefill execution uses the decode serial arena under its
own identity, so the split adds no device memory (host side: a second compiled plan; on the MTP route a second
launcher catalog). MTP goldens (version 2) record the prefill and decode profile identities and are rejected
under another pair; version-1 goldens are admitted only for one profile.

Why: a fast prefill profile (FP16-dequant WMMA for MoE, HC and dense projections) will change prefill arithmetic
but not decode. MTP exactness can be kept from the state the prefill produced (replay against a golden of the same
pair), not from a single-profile run from the start, so both identities must travel with goldens and receipts.

Measured on one Strix Halo node (Qwen3.8-Flash-Next UD-Q4_K_XL), FP32-dense (F) and dense-A8 (A) profiles:

| Check | Result |
| --- | --- |
| Flag absent, tiled prefill T=256 (1032-token prompt, 32 tokens) and MTP 2/4/8 (43-token prompt, 256 tokens), F and A | Tokens identical to `a1c871f`; receipts differ only by added fields |
| MTP replay, split pairs F/A and A/F (decode/prefill) against goldens dumped with the same pair | Error 0 (hidden and logits, 42 rows) |
| Prompt rows of a split-pair golden | Byte-identical to the golden of the pure prefill profile |
| Golden of another pair (v2) or a v1 golden under a split pair, and the reverse | Rejected before target work |
| Split MTP generation, decode F after prefill A | All 256 tokens equal to pure A (pure F diverges at 212) |
| Split MTP generation, decode A after prefill F | Diverges from pure F and pure A at token 57 |
| Split MTP generation with R2-only windows vs adaptive 2/4/8 windows, both pairs | 256 tokens identical (replay error 0 in both) |
| Split tiled prefill, 1024-token prompt (4 blocks of 256, terminal head in the last block), 32 tokens | Decode F after prefill A equals pure A; decode A after prefill F equals pure F |
| Split tiled prefill, 1032-token prompt (8 serial suffix rows run decode) | Decode A after prefill F equals pure F; decode F after prefill A differs from both at token 0 |
| Deferred-normalization pair (decode FP32-dense, prefill dense A8), tiled 1024 | Runs; 32 tokens equal to both pure deferred profiles (which agree here) |
| Peak RSS, split vs pure prefill profile | +3.8 MB tiled, +0.1 MB MTP; no added device allocation |

## D-034: teacher-forced scoring over the tiled prefill

`--teacher-forced-score` (and `--teacher-logits-file`) is admitted on the tiled known-prefix prefill
(`--known-prefix-window --known-prefix-full-rows --known-prefix-prefill-rows T`, all-rows output policy) as
well as on the serial eager route. No block runs headless; each block's T heads run as the multi-row prefetch
Q8_0 head (packs of eight, the R1 head's per-row leaf/reduction tree) when the serial head is that head. After
a block commits, its T rows are downloaded with one strided copy into a retained pinned staging buffer of T x
vocabulary floats and scored with the serial consumer's arithmetic (shared `compute_requested_logprobs`),
rows in parallel host threads, then appended to the ledger in position order; the block's spool append runs on
a background thread overlapping the next block and is drained before the staging is reused. The unpadded suffix
runs serially through the same consumer. File format, row order, footer and score rows are unchanged. Rows
of complete blocks carry the prefill profile, suffix rows the decode profile; the receipt records the route,
the row counts per route and both profiles (see
[docs/teacher-forced-tiled-prefill.md](docs/teacher-forced-tiled-prefill.md)).

Why: the serial route is the only way to score long documents, at about 13 minutes per 16K-token document and
profile; the tiled prefill reproduces serial rows bit for bit within a profile, so the same files can come from
it in a fraction of the time. The staging buffer bounds host memory by T rows; no device memory is added (the
arena already holds every row's logits stripe).

Measured on one Strix Halo node (Qwen3.8-Flash-Next UD-Q4_K_XL), wall time including model load, serial route
with the automatic organization, tiled route with the explicit v5 flag set; F = FP32-dense profile, A = dense A8:

| Check | Result |
| --- | --- |
| 1536 tokens (wiki), T=256 (6 blocks, final input inside the last block), F and A | Logits file byte-identical to serial (SHA-256), 1535 score rows equal, same generated token; 75 s serial vs 10 s tiled |
| 2000 tokens (code), T=512 (3 blocks + 464 serial suffix rows), F and A | Byte-identical, score rows equal; 98 s vs 32-34 s |
| Split pair, decode F / prefill A, 1536 tokens, T=256 | Byte-identical to the serial A file (every row in a block) |
| Serial route, new binary vs base | Byte-identical file and score rows |
| 4096 tokens, T=2048, F | Byte-identical, 4095 score rows equal; 205 s vs 25 s |
| 16384 tokens (wiki), T=2048, F | Byte-identical (16.3 GB file), 16383 score rows equal; 1001 s serial vs 66 s tiled |
| 16384 tokens, where the tiled time goes (per 2048-row block) | 7-8 s GPU against 5.4 s for the same prefill with only the terminal head (the eight-row heads add about 2.6 s), 31 ms download, 150 ms scoring; the 0.3-2.8 s spool overlaps the next block |

## D-035: block-boundary prefix state cache

`--prefix-cache-host-bytes N` (optional `--prefix-cache-dir DIR`, `--prefix-cache-disk-bytes N`,
`--prefix-cache-every N`) keeps the complete session state at block boundaries P = k�T of the tiled
known-prefix prefill and restores the longest cached boundary below the prompt length into a fresh session;
only the rest of the prompt is prefilled. Per-position rows (KV and raw indexer keys of every attention layer)
are stored once per block and shared by the checkpoints of a prompt; the in-place state (GDN matrices, GDN/PLE
convolution histories, n-gram and position history: 111 MiB here) plus counters and arithmetic lineage is
stored per checkpoint. Keys chain SHA-256 over T-token blocks from an identity that covers model content, state
layout and context, runtime compatibility, decode, prefill-sibling and tiled-plan execution and kernel-plan
identities, both numerical profiles and their pair, T, the tail sizes and (disk tier) the executable. Host
tier: pinned memory with LRU in a byte budget; disk tier: checksummed segment/checkpoint files written behind
the request. See [docs/prefix-state-cache.md](docs/prefix-state-cache.md).

Two pieces were needed around it. `--known-prefix-tail-rows T2[,T3...]` runs the remainder after the last
T-row block as complete blocks of the largest tail size that fits, before the serial rows: serial prompt rows
cost about 50-70 ms each, so a warm request with an arbitrary suffix would otherwise spend seconds on up to T-1
serial rows. Within one numerical profile tiled rows are serial rows bit for bit (D-034), so tail blocks do not
change outputs; with a split pair they move rows from the decode to the prefill profile and are part of the
key. And `--mtp` now composes with the tiled prefill: the golden covers only `--mtp-calibration-rows R`
(default 8) prompt pairs, replayed serially to qualify the predictor; then target and predictor are reset,
complete blocks run tiled while a step observer copies every row's wide hidden state out of the block arena to
build the predictor's cache rows, and the remaining rows (at least the last, whose ticket starts the first
draft cycle) run serially through the MTP target producer. On this route a checkpoint also holds the
predictor's cache rows of pairs i < P-1 and h_{P-1}; the key adds the sidecar, MTP plan/profile and
draft-kernel switches. Split pairs are rejected there.

Why: agents resend a long, mostly identical prompt; a 16K-token prefill costs about 25 s (dense A8) to 45 s
(FP32 dense) on one node. The state at a block boundary is exact and the suffix of a restored request keeps the
full prompt's block alignment, so outputs cannot change; checkpoints at block boundaries only.

Measured on one Strix Halo node (Qwen3.8-Flash-Next UD-Q4_K_XL), F = FP32-dense and A = dense-A8 profile, FA =
decode F / prefill A. Identity checks compare against the same configuration without lookup (every request
cold) or an uncached process; timings are single smoke runs on shared nodes, not a benchmark campaign:

| Check | Result |
| --- | --- |
| 4K prompt, T=512, F, A and FA: full prompt, then in one process prompts of 3585, 3684, 2747 tokens (restored 3584, 3584, 2560), 2948 tokens diverging after 2048 (restored 2048) and the full prompt again (restored 3584) | 32 generated tokens identical to the uncached runs in all 18 requests |
| Same with `--known-prefix-tail-rows 64` and `64,16` (A) | Identical to the runs without tail blocks; warm requests 3.4-4.7 s (64) and 2.2-3.8 s (64,16) instead of 6.3-20.9 s, including 31 generated tokens |
| 16K prompt, T=1024, A and F: 15360-token request (cold, captures), then the 16384-token prompt (restored 15360, 1024-token suffix) and a 9692-token prompt diverging after 8192 (restored 8192) | Tokens identical; A 26.1 s cold vs 3.2 s warm, F 45.2 s vs 4.4 s (request time including 31 serial generated tokens, about 1.5 s) |
| Disk tier, other process (A, T=512) | 3684-token prompt restored 3584 from disk (0.22 GB read in 42 ms), tokens identical |
| Identity changes, disk tier | F instead of A, or T=256 instead of 512: miss (restored 0), no read failure |
| Tiled MTP, 4K, T=512, tail 64, A and F, one process per request through the disk tier | Full prompt, 3684- and 2948-token variants: 128 tokens identical, calibration replay error 0 (hidden and logits), SHA-256 of the whole prompt state (target rows and in-place state, predictor rows, h_{n-1}) identical between cold and warm; T=256: miss |
| Tiled MTP, 16K, T=1024, A and F (restored 15360) | 256 tokens identical, replay error 0, prompt-state digest identical cold vs warm. Prompt phase (restore, prefill, predictor rows): A 35.3 s cold vs 7.6 s warm with tail 64, 31.0 s vs 4.8 s with tails 64,16; F 54.4 s vs 8.3 s (tail 64); plus 0.5 s calibration replay |
| Checkpoint size and cost | 27,648 B per position (29,952 with MTP) + 116.4 MB per checkpoint: 0.54 GB at P=15360; capture 25-35 ms per checkpoint; restore 3.3 ms (3584 positions) to 8-11 ms (15360 positions) from host memory, about 65 GB/s; disk read 0.12-0.16 s for 0.58 GB |

MTP cycle counts can differ between identical runs (the window cost model is timing-driven); tokens, replay and
state digests do not.

## D-038: survival window controller and active rows (opt-in, pending the timing campaign)

`STRIX_MTP_CONTROLLER=survival` replaces the D-022 cut (the chain stops at the first draft below
`--mtp-spec-min-p`, and that cut also caps the window) by two separate decisions:

- Acceptance: for draft k, given that drafts 0..k-1 matched, P(draft k equals the target's greedy token) is a
  logistic in the draft's probability, its log margin over the runner-up (the draft head now also reduces the
  runner-up) and depth (1, >= 2). Prior weights were fitted offline; online only the bias adapts, pulled back
  to the prior.
- Window and chain length: E[N_R] = 1 + sum_j prod_{i<j} P_i over the option's active rows against a cost
  C = option cost + n draft steps, with separated EMAs (cycle time minus draft chain per option, per draft
  step, serial rows). The chosen option maximises E[N] - rho*C, rho the long-run tokens per ms. Drafting
  continues while some longer chain, with further drafts at a lookahead acceptance (from depth and the
  previous probability), beats the current best option. One extra draft after a weak first draft is
  available (`STRIX_MTP_EXPLORE`) but off: neutral to slightly negative in simulation.

`STRIX_MTP_ACTIVE_ROWS=1` adds options inside the physical owners: the R4 owner verifies 3 active rows and
the R8 owner 6. Padding rows run the dense R4/R8 kernels but stay outside the routed expert union (the CSR is
built over the active rows and the merge skips the rest, so their routed output is zero), and are never
accepted (acceptance stops before the active count; `commit_prefix` keeps at most that many rows, and
capture, GDN deferred commit, PLE commit and QSA rows already publish only the accepted prefix). Admission is
unchanged: the active count is a launch argument of the eager window. The union rows fixture checks, in both
profiles, that active rows are bit-identical to the full union and padding rows are zero.

Sizing with the full-chain diagnostic (`STRIX_MTP_DIAG_FULL_CHAIN=1`: a 7-deep chain from every generated
position) on 6 prompts x 512 tokens (EN, ES, FR prose, 2 code, agent; dense A8 profile, Q4 draft, auto
vocabulary):

- Calibration. Raw draft probability in [0.9, 0.99) predicts 0.957 against 0.801 observed; the calibrated
  estimate predicts 0.958 against 0.956.
- Simulation against the min-p 0.8 controller, leave-one-prompt-out (calibrator fitted without the simulated
  prompt), geometric mean:
  - controller alone: +2.1 % (+1.0 to +3.5 % per prompt);
  - with active rows: +4.3 %.
- Active rows under the min-p controller: about 0 %.
- Measured costs in single runs (cycle minus drafts): R2 44.2, R3-in-R4 49.3, R4 52.4, R6-in-R8 64.6,
  R8 72.1 ms.

Tokens are unchanged by construction: every emitted token is the target's greedy token. They were identical,
with MTP replay error 0, on every run in both profiles. The cost model also re-probes an option whose
estimate is implausible once it has gone 64 cycles unused (`STRIX_MTP_REPROBE`). In one min-p baseline run, a
single 111 ms first sample of R4 kept 262 of 267 cycles in R2.

Status: opt-in until an interleaved timing campaign against D-022 qualifies it.

## D-039: robust window cost estimates with bounded re-probes (D-022 controller, default)

The D-022 controller only samples an owner's cost when it chooses it. Under the EMA, the first sample set
the estimate and later samples were capped at 2x, so one slow window (page faults, a stalled host) could
price an owner out for the rest of the run. Once a larger owner looked too slow, `worth_drafting` stopped
every chain early, the larger owners were never chosen again, and nothing corrected the estimate.

On hardware (code prompt, dense A8 profile, min-p 0.8) this happened as follows. The first R4 sample after
its warm-up was 111 ms, against about 55 ms in other runs. R8 got only its warm-up window. 262 of 267 cycles
then ran R2, at 41.8 tok/s; a repeat of the same run gave 61.2 tok/s.

Now (`MtpMinPWindowController`, host code shared by the verifier and a unit test):

- Cost estimate per owner and for serial rows. It keeps the last 8 samples; the first window of an owner is
  still not a sample.
  - With fewer than 3 samples the estimate is provisional and equals their minimum, so a single slow sample
    cannot inflate it.
  - From 3 samples on, it is their median, so one outlier never moves it.
- Bounded re-probe. An owner whose estimate is provisional, or implausible (above the settled estimate of a
  larger owner), is chosen once more, if it fits, after it has gone 64 choices unused. Each probe doubles that
  owner's interval, so a run of N cycles pays at most about log2(N/64) probes per owner, and none once every
  estimate has settled.
- Unchanged: the chain cut, calibration and ranking (expected tokens per estimated ms), the policy flags,
  and every emitted token. Only the choice of window differs.

`STRIX_MTP_WINDOW_COST=legacy` restores the previous EMA without re-probes. Receipts record
`mtp_window_cost_model` and per-owner `probes`.

`tests/mtp_window_controller_tests.cpp` replays the stall with a synthetic cost sequence (a 111 ms first R4
sample, then a confident workload):

| Cost model | Outcome |
| --- | --- |
| Legacy | Stays in R2 |
| Robust | Re-probes R8 and R4 a few times and runs the workload in R8 |

Without the spike, both models settle on the same windows.

## D-040: draft vocabulary policy 2 (opt-in, pending the timing campaign)

The draft head proposes only from its rows: the core, the language pack chosen by the prompt, and every
prompt or emitted token (the context rows). A full-chain diagnostic run per head variant (Q4 subset as
deployed, Q8 subset, Q8 full head; 10 prompts: en, es, fr, de, ja x2, zh, two code, agent) measured what this
costs.

Findings:

- Tokens outside the rows cost only their first appearance. Later appearances are already context rows.
  - Generated positions whose token is outside the rows: 0.2-0.8 % with a Latin pack, 0.6-1.2 % in Japanese,
    1-4.3 % in English and code (core only), and 6.1 % in Chinese (core32k + pack-zh16k).
  - Share of draft misses with an out-of-rows target: 15 % (JS reasoning), 12 % (Chinese), 6.5 % (English
    chat, agent), and at most 2.6 % elsewhere.
- Normalising the draft probability over the subset barely moves it. On the first draft, where all head
  variants see the same input, p(subset) / p(full head) has a median of 1.0000-1.0047 and a p90 of
  1.003-1.085. A renormalisation or per-pack calibration has nothing to correct.
- Pack selection by prompt picked the best pack on every prompt. Prompts without a pack (English, code, agent)
  gain nothing from any pack.
- Ceiling: the Q8 full head at today's draft cost would gain only +1.1 % (min-p) / +1.2 % (survival
  controller), geometric mean. The largest gains are Chinese (+3.2 / +4.5 %), JS reasoning (+2.9 / +3.6 %) and
  English chat (+1.3 / +1.6 %).
- Simulation method. A subset head's argmax equals the full head's whenever the full argmax is one of its rows,
  so any subset's chains follow from the full-head chains. They matched the real Q8 subset runs on every one
  of 31,933 drafts.

Policy 2 (`STRIX_MTP_DRAFT_VOCAB=2` with `auto:DIR`):

- A chosen pack comes from `pack2-<lang>.bin`. This is the same ranking with core32k excluded: the shipped
  packs were ranked against a 16k core, so 0.2-4.5k of their rows repeat the core. The Chinese pack doubles to
  32k rows, which cuts its out-of-rows positions from 23 to 2 of 376.
- A prompt without a pack gets `core64k.bin`, a longer prefix of the core32k ranking.

Simulated gain against today's vocabulary, with the extra head rows priced at 0.1 ms per draft step per 16k
rows:

| | Min-p | Survival controller |
| --- | --- | --- |
| Geometric mean | +0.4 % | +0.6 % |
| Chinese | +2.3 % | +3.0 % |
| JS reasoning | +1.5 % | +1.6 % |
| English chat | +0.4 % | +0.9 % |
| Spanish | +0.7 % | +0.6 % |
| Others | -0.4 to +0.4 % | |

`STRIX_MTP_DRAFT_CORE=<N>k` forces the core. Tokens are unchanged: the draft only steers verification.

Status: opt-in until the timing campaign. Both rules are data only, and the language is still chosen with the
`pack-` files.

## Measured and rejected

| ID | Alternative | Result | Reason |
| --- | --- | --- | --- |
| X-004 | hipGraph capture of the decode cycle | 1.98 → 1.82 µs per kernel | See D-029. |
| X-005 | Coarse-grained weight registration | 44.3 vs 45.0 tok/s | Slower; kept opt-in (`STRIX_WEIGHTS_COARSE_GRAINED=1`). |
| X-006 | `value-key` GDN recurrent storage with windows | — | Incompatible with the A8 profile (D-024). |
| X-007 | Fused HC normalization in split-K | 42.6 tok/s | Bit-exact but slower; kept opt-in (`STRIX_HC_ROWS_FUSED_NORM=1`). |
| X-008 | Issuing MoE gate+up loads early | 35.7 tok/s | 256 VGPRs and 190 spills. |
| X-009 | MoE gate/up with half the registers | neutral | Kept opt-in (`STRIX_MOE_GATEUP_CHUNK=5`). |
| X-010 | Wide LM head rows variants (w16, direct) | neutral | Kept opt-in (`STRIX_LM_HEAD_ROWS_WIDE`). |
| X-011 | HC low part in LDS; q8_rows prefetch depth 4; GDN 4 or 16 columns per wave | slower or neutral | Not adopted. |
| X-012 | `--model-copy-tier experts`, `--staged-moe-router-logits` on the serial path | slower | Not used in the measured configuration. |
| X-013 | `--mtp-spec-min-p` 0.7, 0.75, 0.8, 0.88, 0.9 | lower tok/s | 0.85 is the measured optimum (D-022). |
| X-015 | Q8 instead of the Q4_0 copy of the MTP draft head subset (always, from depth 1, 2 or 3, only with a zh/ja pack, only without a pack) | simulated -1.6 % to -0.1 % | The Q8 head costs about 2x per draft step (+0.22/+0.29/+0.39 ms at 37k/49k/65k rows, `strix-hip-mtp-draft-head-tests --bench-precision`) and, even at no cost, gains nothing beyond trajectory noise on 10 prompts (full-chain logs). Keep `--mtp-draft-q4`; Q5/Q6 are bounded by Q8. |
| X-016 | Richer survival-controller inputs and stopping rules for the draft chain and verify window: acceptance in the request, cumulative chain probability, previous draft probability, entropy from top-2, token type of the draft and previous token (logistic, leave one prompt out), gradient-boosted trees, a richer or online depth-wise lookahead, drafting 2-7 steps before choosing, R5/R7 active rows | simulated -2.6 % to +0.1 % against the D-038 controller (R3/R6), base and distilled draft | Full-chain logs (16 prompts base draft, 12 distilled; prose en/es/fr/de/zh/ja, summaries, rewrites, QA, agent, code), measured R2/R4/R8 costs. Log loss improves 1.3 % at most and even trees fitted on the evaluation data reach +0.3 % over D-038. An oracle that knows the accepted length gains +18 to +21 % (base) or +25 to +28 % (distilled) over min-p 0.85, but that is information the draft does not carry when the window is chosen; D-038 (+4 %, +9.6 % with the distilled draft) is within about 1 % of the best draft-time controller found. |
| X-017 | Algorithmic cuts of verify-window bytes: choosing the window rows from their real expert-union bytes, keeping or prefetching the routed experts of the previous window or of the next layer in L2/MALL, ordering experts for cache reuse | simulated -0.1 % (byte-aware rows); cache reuse not available | Route logs (`STRIX_WINDOW_ROUTE_LOG`, 517 R8 windows, 6 prompts): a window reads 10 / 15.8 / 25.5 / 40.9 distinct experts per layer for 1 / 2 / 4 / 8 rows (1.50 / 2.38 / 3.85 / 6.16 GB, 3.1 MB per expert); a row adds 4-6 new experts per layer whether or not it is accepted, so bytes carry no acceptance information and the survival controller already prices R3/R6 at the fitted cost (23.7 ms + 8.06 ms/GB). The MALL serves a re-read of up to 32 MB at about 850 GB/s against 235 GB/s from DRAM, but keeps nothing past 16 MB of other traffic (LRU), while one layer of one row is 31 MB and an R8 window streams 6 GB; 59 % of a layer's union recurs in the next window but is long evicted. A side-stream prefetch against a saturating read costs its full bytes. |
| X-018 | Routed-expert window kernels (A8 gate/up `project_window`, Q5_1/Q8_0 `down_window`): aligned 16/8-byte Q5_1 block loads instead of byte loads, one gate/up pass for experts of 5-8 rows (G=8), half the Q4_K blocks in registers (`STRIX_MOE_GATEUP_CHUNK=5`) in windows, other expert orders and 64/128-row tiles | exact; union fixture 0 % (wide down), +4..+17 % slower (G=8); in-situ R8 trace 0 % (chunk 5) | The kernels already run near the bandwidth of their access pattern: a pure read of the same scattered expert slabs (no arithmetic) reaches 205-226 GB/s for 10-80 experts (236 GB/s for one contiguous stream), the window down kernel runs at that floor and gate/up at about 85-90 % of it (about 190 GB/s in situ). The remaining gate/up gap is worth at most about 3-4 % of an R8 window. |
