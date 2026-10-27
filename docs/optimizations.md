# Optimizations

The fast path is the default in all five public engines. Use the plain recommended command and inspect its
resolution before adding overrides. `ON` means enabled; `auto` means enabled when admission guards pass;
`opt-in` requires an explicit selection. An unlisted isolated gain means unmeasured, not zero.
The [JSON catalog](optimizations.json) provides the same controls for tools; the [README](../README.md#optimizations)
is the compact overview. Run commands and agent instructions are in [AGENTS.md](../AGENTS.md).

Exactness requires fixed model content, prompt, state, profile pair and capacity. Qwen fast FP16 prefill,
dense A8, GSQ, DeepSeek trellis and GLM R1b are distinct numerical identities. Speculation verifies committed
tokens against its target from the prefill handoff state; draft changes can affect acceptance without changing
that sequence. Gains from separate campaigns must not be added. Measurements below are prior device
qualification; a host-only publication build cannot establish new device exactness or speed.

## Baselines, inspection and exactness

| Engine | Master baseline | Inspect before device work | Scope / precedence |
|---|---|---|---|
| Qwen | `--baseline` / `STRIX_BASELINE=1` | `--print-effective-config` | Historical organization/arithmetic; controlled defaults suppressed, implicit context 8192; explicit route options retain historical validation |
| DeepSeek pair | `STRIX_DS41_BASELINE=1` | `--print-effective-config` | Stop production pack/draft/argv injection and switch the nine recent code defaults off unless explicitly set; older kernel defaults remain |
| DeepSeek RTX | `--baseline` / `STRIX_DS41_RTX_BASELINE=1` | `--print-config` | CLI baseline applies at its argument position; later flags win; env baseline applies before argv |
| MiniMax | `STRIX_MM_BASELINE=1` | `--print-effective-config` | All 22 resolver items use their off values; explicit environment wins; older built-in routes remain |
| GLM | `--baseline` / `STRIX_GLM_BASELINE=1` | `--print-effective-config` | R1, reference prefill, MTP loaded, comm decode exchange and 15 env off values; explicit overrides win; fast MoE retained for capacity |

Use a fresh process for startup-cached settings, and the same settings on both ranks where the protocol requires it.
For A/B, change one off switch at a time, first compare complete token IDs and logits/state digests, then compare
medians in interleaved runs. A baseline that changes arithmetic is a separate identity/quality comparison.
Do not add legacy opt-in flags to the default command: on Qwen they can own the explicit route and decline auto selection.

Use each engine's built-in gates now: Qwen profile-pair qualification plus operator byte checks; DeepSeek
both-rank prefill logits, greedy/verify/batch rows and transport integrity checks; RTX replica pressure,
canonical-state and plain/speculative chat gates; MiniMax golden logits, decode/verify/exchange checks;
GLM G3/G4, identity, prefill-logit and speculative-oracle gates. References must match the selected identity.
Once the bench kit lands, run `bash scripts/bench/bench.sh list`, then
`bash scripts/bench/bench.sh check --engine E --only exactness --suite quick --json` and
`bash scripts/bench/bench.sh run --engine E --only exactness --suite quick --json`.
Verdicts are `EXACT`, `MISMATCH` and `NO-REFERENCE`; the last is not a pass. The kit is not assumed present today.

## Qwen

```sh
strix-hip-decode-token MODEL.gguf $(cat prompt.ids) --generate 256
```

Add `--print-effective-config` to inspect effective argv, environment/provenance, budgets and profile pair,
then exit before HIP initialization. See [command-line admission](getting-started/engine-reference.md#command-line).
The historical baseline resolves ordinary generation with exact-v5, prepared execution, shared overlap,
GDN islands and implicit context 8192; it does not automatically select dense A8 or enlarge the context.

<a id="qwen-tiled"></a>

### Automatic tiled prefill, expert union and router rows

**Engine:** `qwen`. **Control:** `--known-prefix-window; --known-prefix-full-rows; --known-prefix-prefill-rows; --known-prefix-expert-union; --known-window-router-rows; --prefill-numerical-profile`. **Default:** auto.

The plain generation command resolves to the measured production organization with dense-A8 decode and fast-prefill-f16wmma-v1 prefill.

- **Preconditions:** Pinned verified checkpoint, qualified gfx1151 target, A8 runtime, >=16 prompt tokens, positive generation, eligible eager registered mapping, unambiguous UMA/GTT observations and sufficient backing capacity.
- **Automatic selection:** Largest prompt-fitting T in 8192/4096/2048/1024/256/64/16. Context rounds prompt + max(generation,512) up to 512, minimum 8192, clamped to the model maximum (262,144); T falls back 8192 -> 4096 -> 2048 before declining (insufficient-memory when no tile fits the budget; prompt-exceeds-model-context above the maximum); --unqualified-context is automatic above 8192. Explicit context is preserved. Full budgets: 8 GiB workspace / 1 GiB metadata / 120 GiB UMA / 4 GiB headroom, scaled down with available memory (at least 1 GiB headroom). Routed expert union and router rows are on. Tail owners follow the admitted ladder.
- **Fallback notice:** One admitted line reports T/context. Refusal prints `Qwen auto-prefill declined: <reason>; using serial prefill`. Explicit MTP/cohorts, HTTP serve, serving prefix files, lookup, known-prefix/low-level flags and target-input export retain their explicit route without this notice. The diagnostic/execution-control variables STRIX_PREFILL_LOGITS_DIGEST, STRIX_RTX_EXPORT_STATE, STRIX_QRTX_STEP_DUMP and STRIX_PROMPT_BATCH_ROWS (non-empty, not 0) decline with the line above and fall back to the historical resolution, including its 8192-token default context: a longer prompt is then refused before any work (pass an explicit --context for a slow serial run). GSQ declines the new auto route. MTP decode and the HTTP server do not get fast prefill automatically yet. FP16 operand/plan refusal can retain dense-A8 tiling with an arithmetic-fallback reason.
- **Exactness evidence:** Tokens and terminal FP32 logits digests match the reference for the selected profile pair, including 32K/64K/128K/262K. Fast FP16 prefill is exact relative to its own reference, not bit-identical to dense-A8 serial prefill; longer-context fidelity remains outside the 8192 qualification.
- **Off switch:** --baseline / STRIX_BASELINE=1 restores historical resolution; STRIX_PREFILL_F16=0 or --known-prefix-window=0 declines automatic tiling. --prefill-numerical-profile dense-a8 retains tiling with dense-A8 arithmetic. Controlled feature =0 or STRIX_QPF2K_DEFAULTS=0 also declines auto selection; exact operator A/B needs an explicit route with the same profile pair.
- **Measured gain:** One Strix Halo node: 32K 22.03 s = 1,487 tok/s; 131,072 tokens 100.3 s; 262,143 tokens 202.8 s = 1,293 tok/s. Previous documented default externally reported about 119 tok/s versus about 822 with union + router rows; separate from local measurements.
- **Implementation / evidence:** [Resolver](../apps/qwen_effective_config.hpp), [profiles](engine/prefill/prefill-numerical-profile.md), [results](performance.md#qwen-prefill-and-defaults)

<a id="qwen-residency"></a>

### UMA registered residency

**Engine:** `qwen`. **Control:** `--residency register`. **Default:** ON.

Register the mmap-backed GGUF without a second full model allocation.

- **Preconditions:** UMA GPU, mappable GGUF and capacity for model/state/workspaces.
- **Automatic selection:** CLI default on UMA.
- **Fallback notice:** Registration/setup failure is an error; there is no silent full-copy conversion.
- **Exactness evidence:** Weight bytes are unchanged; compare same-profile tokens and state/logit digests.
- **Off switch:** --residency copy (requires capacity for the copy)
- **Measured gain:** No isolated measurement
- **Implementation / evidence:** [Residency](operations/model-weight-residency.md)

<a id="qwen-l1"></a>

### PLE page reclaim and prefetch admission

**Engine:** `qwen`. **Control:** `STRIX_PLE_RECLAIMABLE=auto`. **Default:** auto.

Reclaim file-backed L1 PLE pages and prefetch the required pages, allowing the measured T8192 route at 128K/262K.

- **Preconditions:** Linux, nonempty plain CLI prompt, tiled known prefill, registered file mapping, no model copies or graph promotion, no MTP/cohort/serving/planned PLE IO. Exactly one file-backed IQ4_NL/BF16 n-gram table, valid extents and unregistered interior pages, valid prompt hashes, and a verified page drop within the existing residual bound.
- **Automatic selection:** Auto tries the existing page-drop/prefetch admission. Admitted auto uses the same credit, checks and receipt text as explicit =1. Defaulted auto is suppressed for GSQ owners and by STRIX_QPF2K_DEFAULTS=0; explicit settings retain their checks. Prefetch scheduling is unchanged.
- **Fallback notice:** Unsupported routes or failed auto setup silently keep the original warm path, without credit or PLE reclaim receipt fields. Explicit =1 still refuses failed admission.
- **Exactness evidence:** Page contents are unchanged; long-context token/logit digest evidence applies to the admitted non-GSQ identity. GSQ page-drop behavior is not qualified by these results.
- **Off switch:** STRIX_PLE_RECLAIMABLE=0
- **Measured gain:** Capacity admission for T8192 at 128K/262K; no isolated speed attribution
- **Implementation / evidence:** [Admission](../include/strix/ple_reclaim.hpp), [implementation](../src/ple_reclaim.cpp)

<a id="qwen-defaults"></a>

### Combined QPF2K and wrapper defaults

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_DEFAULTS`. **Default:** ON.

Supply unset per-item values; retain explicit overrides. The 13 QPF2K items and nine admitted wrapper settings are listed individually below.

- **Preconditions:** Eligible tiled operators and selected numerical identity; the decoder resolver admits wrappers only for the single-session automatic route.
- **Automatic selection:** Historical tiled items and route-gated PLE auto may initialize before main. Nine wrappers are deferred until decoder admission. QSA/HC masks are supplied with a newly defaulted parent enable. Injected items are tracked and suppressed per GSQ owner after lazy caching, preserving explicit environment values.
- **Fallback notice:** No general initializer notice. Operators retain guards; GSQ no longer requires disabling defaults just to run. Explicit unqualified GSQ controls still fail validation.
- **Exactness evidence:** Tokens and terminal FP32 logits digests match the reference for the selected profile pair, including 32K/64K/128K/262K. Fast FP16 prefill is exact relative to its own reference, not bit-identical to dense-A8 serial prefill; longer-context fidelity remains outside the 8192 qualification.
- **Off switch:** STRIX_QPF2K_DEFAULTS=0 in a fresh process with individual overrides unset; --baseline / STRIX_BASELINE=1 additionally suppresses controlled explicit environment items and restores historical policy.
- **Measured gain:** Combined 32K 25.95 → 23.49 s (−9.5%); later 22.03 s is a separate measurement
- **Implementation / evidence:** [Default table](../include/strix/qpf2k_defaults.hpp)

#### The 13 QPF2K defaults

Each item inherits its linked operator's preconditions, automatic selection, fallback and same-identity gates.
An explicit `=0` disables its feature and also declines the new automatic CLI route. QSA mask `0x57`
disables only stage batching; HC mask values select folds. Use explicit tiling to hold the profile pair fixed for A/B.

| Environment | Default | Off | Operator detail |
|---|---|---|---|
| `STRIX_QPF2K_SCHED` | `12` | `STRIX_QPF2K_SCHED=0` | [Merged tails and lazy tail owners](#qwen-schedule) |
| `STRIX_QPF2K_DENSE` | `1` | `STRIX_QPF2K_DENSE=0` | [Register-blocked dense projections](#qwen-dense) |
| `STRIX_QPF2K_GDN` | `1` | `STRIX_QPF2K_GDN=0` | [GDN recurrence, gates batching and core16](#qwen-gdn) |
| `STRIX_QPF2K_GDN_GATES_BATCH` | `1` | `STRIX_QPF2K_GDN_GATES_BATCH=0` | [GDN recurrence, gates batching and core16](#qwen-gdn) |
| `STRIX_QPF2K_GDN_CORE16` | `1` | `STRIX_QPF2K_GDN_CORE16=0` | [GDN recurrence, gates batching and core16](#qwen-gdn) |
| `STRIX_QPF2K_QSA` | `1` | `STRIX_QPF2K_QSA=0` | [QSA stages and stage batching](#qwen-qsa) |
| `STRIX_QPF2K_QSA_MASK` | `0x257` | `STRIX_QPF2K_QSA_MASK=0` | [QSA stages and stage batching](#qwen-qsa) |
| `STRIX_QPF2K_HC` | `1` | `STRIX_QPF2K_HC=0` | [HC folds, output v3 and band 16](#qwen-hc) |
| `STRIX_QPF2K_HC_MASK` | `14` | `STRIX_QPF2K_HC_MASK=0` | [HC folds, output v3 and band 16](#qwen-hc) |
| `STRIX_QPF2K_HC_OUTPUT3` | `1` | `STRIX_QPF2K_HC_OUTPUT3=0` | [HC folds, output v3 and band 16](#qwen-hc) |
| `STRIX_QPF2K_HC_OUTPUT3_BAND` | `16` | `STRIX_QPF2K_HC_OUTPUT3_BAND=0` | [HC folds, output v3 and band 16](#qwen-hc) |
| `STRIX_QPF2K_MOE_DA16` | `1` | `STRIX_QPF2K_MOE_DA16=0` | [Routed MoE operand image and route setup](#qwen-moe) |
| `STRIX_QPF2K_MOE_ROUTE` | `14` | `STRIX_QPF2K_MOE_ROUTE=0` | [Routed MoE operand image and route setup](#qwen-moe) |


<a id="qwen-schedule"></a>

### Merged tails and lazy tail owners

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_SCHED=12`. **Default:** auto.

Bit 4 merges eligible tail work into the main arena; bit 8 allocates only the tail owners the prompt ladder uses.

- **Preconditions:** Tiled full rows; merged tails require terminal-row output and exclude the dense8 fast component. Lazy owner pruning requires a single known CLI prompt without serving, import, prefix caches, cohorts, lookup or wide-copy work.
- **Automatic selection:** The defaults bundle supplies 12. Ineligible pruning retains the full tail-owner ladder; ineligible merging retains the configured chunks.
- **Fallback notice:** No ordinary decline line. Requesting merged/balanced scheduling with dense8 throws STRIX_QPF2K_SCHED bits 4/16 (merged / balanced prefill blocks) exclude the dense8 prefill component.
- **Exactness evidence:** The qualification campaign compared 1K code and 8K prose receipts, terminal FP32 logits FNV and generated tokens against the unchanged reference. Repeat with the same prefill/decode identity; changing arithmetic profiles is a separate quality comparison.
- **Off switch:** STRIX_QPF2K_SCHED=0
- **Measured gain:** -
- **Public evidence:** [Qwen results](performance.md#qwen-prefill-and-defaults), [chunk cursor](../src/known_prompt_chunks.cpp)

<a id="qwen-dense"></a>

### Register-blocked dense projections

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_DENSE=1`. **Default:** auto.

Reuses dense projection tiles across prompt rows while preserving the operand image and K order.

- **Preconditions:** Fast FP16 prefill dense sites, supported Q8_0 widths/segments/alignment and uniform strides; default minimum 1024 rows at GDN input and shared gate/up, 512 at other sites. The site mask defaults to all six sites; STRIX_QPF2K_DENSE_MIN_ROWS can override the threshold.
- **Automatic selection:** Selects supported sites at or above their row threshold; otherwise keeps the preceding projection.
- **Fallback notice:** Silent eligibility fallback. STRIX_QPF2K_DENSE_TRACE=1 prints qpf2k dense tiled: first launch ... when selected, not a decline line.
- **Exactness evidence:** The tiled dense fixture suite checked 120/120 bitwise cases; also use the whole-engine receipt/logit/token comparison. The source target is strix-hip-dense-f16-tiled-tests.
- **Off switch:** STRIX_QPF2K_DENSE=0
- **Measured gain:** -
- **Public evidence:** [Qwen results](performance.md#qwen-prefill-and-defaults), [dense selector](../src/hip_dense_f16_tiled.hpp)

<a id="qwen-gdn"></a>

### GDN recurrence, gates batching and core16

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_GDN=1; STRIX_QPF2K_GDN_PARTS=55; STRIX_QPF2K_GDN_GATES_BATCH=1; STRIX_QPF2K_GDN_CORE16=1`. **Default:** auto.

Uses the exact recurrence forms, batches gates and feeds eligible dense output directly from FP16 core rows.

- **Preconditions:** Eligible GDN prefill rows and supported lane/column shapes. Core16 additionally needs dense FP16 (not dense G64), tiled rows in multiples of 32, output site enabled, at least 2048 rows by default, value width 6144, head dimension 128, uniform core/output strides and a supported dense output site; graph capture declines it. The FP16 core consumer must be admitted before the producer drops FP32 rows.
- **Automatic selection:** The initializer enables GDN, gates batching and core16; the GDN reader defaults PARTS to 55. Shape, check-mode and consumer guards retain the older forms.
- **Fallback notice:** Ineligible forms fall back without a general decline line. Check modes retain FP32 core rows. Diagnostic output is QPF2K GDN check: ... differ ... or GDN prefill check: ... differ ....
- **Exactness evidence:** STRIX_QPF2K_GDN_CHECK=1 and STRIX_GDN_PREFILL_CHECK=1 compare written bytes; require zero differences. Whole-engine receipts and operand-identity checks cover the qualified special-value/NaN order.
- **Off switch:** STRIX_QPF2K_GDN=0; independently STRIX_QPF2K_GDN_GATES_BATCH=0 or STRIX_QPF2K_GDN_CORE16=0 (one at a time)
- **Measured gain:** -
- **Public evidence:** [Qwen results](performance.md#qwen-prefill-and-defaults)

<a id="qwen-hc"></a>

### HC folds, output v3 and band 16

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_HC=1; STRIX_QPF2K_HC_MASK=14; STRIX_QPF2K_HC_OUTPUT3=1; STRIX_QPF2K_HC_OUTPUT3_BAND=16; STRIX_QPF2K_HC_UPCOPY`. **Default:** auto.

Folds admitted inject/norm work and changes the lane-chain output schedule; band 16 permutes CTAs without changing sums.

- **Preconditions:** Eligible full-row HC shapes, supported norm/weight types and non-overlapping buffers; v3 requires HC enabled. Optional up-weight copy requires scratch.
- **Automatic selection:** The bundle supplies HC mask 14, output v3 and band 16; the up-weight copy is on with v3 unless disabled.
- **Fallback notice:** Unsupported folds keep original launches silently. If optional copy allocation fails or a site exceeds scratch, read the original weights; do not fail the call merely for that copy.
- **Exactness evidence:** STRIX_QPF2K_HC_CHECK=1 compares every replaced launch byte for byte and prints qpf2k HC check ... differ .... Require zero differences; band fixtures passed 519/519 and engine logits/tokens matched.
- **Off switch:** STRIX_QPF2K_HC=0; independently STRIX_QPF2K_HC_OUTPUT3=0, STRIX_QPF2K_HC_OUTPUT3_BAND=0 or STRIX_QPF2K_HC_UPCOPY=0
- **Measured gain:** HC v3/band: 32K -0.52 s (-2.2%); 4K -54 ms (-1.8%)
- **Public evidence:** [Qwen results](performance.md#qwen-prefill-and-defaults)

<a id="qwen-qsa"></a>

### QSA stages and stage batching

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_QSA=1; STRIX_QPF2K_QSA_MASK=0x257`. **Default:** auto.

Uses admitted exact stage implementations and batches independent stages with bit 0x200.

- **Preconditions:** Stage batching requires mask bits 0x3 as well as 0x200; normal eager prefill, 1..128 complete 64-row groups with no partial tiles, grouped norm enabled, BF16 KV and pooled-key storage. Index geometry is four heads of dimension 128, 16-byte aligned queries and prefixes bounded by pool capacity and 65536 blocks. The norm tensors share a weight format, have power-of-two dimensions 2..256, and admit the rotary geometry (half rotary dimension <=64). Rows need contiguous positions and common raw/KV owners, geometry, norm owners, journal, compression ratio and top-k. Private device tables/maxima and pinned descriptors must allocate before any launch/state write. Windows, cohorts, graphs, int8 KV, explicit tiled diagnostics and self-check modes keep their existing paths.
- **Automatic selection:** The bundle supplies QSA=1 and mask 0x257; ineligible stage shapes keep their prior implementation.
- **Fallback notice:** Without tracing, stage-batch declines are silent and keep the per-group path. With STRIX_QPF2K_QSA_TRACE=1 and bit 0x200 requested, a decline prints qpf2k qsa stage_batch groups 0 pool 0 range 0 score 0 rope 0 norm 0 store 0 fallback 1. HIP execution errors propagate without retry after partial execution.
- **Exactness evidence:** The qualification campaign compared 1K code and 8K prose receipts, terminal FP32 logits FNV and generated tokens against the unchanged reference. Repeat with the same prefill/decode identity; changing arithmetic profiles is a separate quality comparison. Stage-batching qualification additionally compared terminal logits and 1K code/8K prose receipts.
- **Off switch:** STRIX_QPF2K_QSA=0 for the stages; STRIX_QPF2K_QSA_MASK=0x57 for stage batching alone
- **Measured gain:** Stage batching: 32K mean -0.39 s (-1.7%); published-build ABBA -0.17 s; 64K -1.6%
- **Public evidence:** [Qwen results](performance.md#qwen-prefill-and-defaults), [stage implementation](../src/hip_indexed_attention_launcher.hip)

<a id="qwen-moe"></a>

### Routed MoE operand image and route setup

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_MOE_DA16=1; STRIX_QPF2K_MOE_ROUTE=14`. **Default:** auto.

Produces the FP16 down operand image once; reduces CSR/prologue work and moves the clear into the ordered merge.

- **Preconditions:** Production Q4_K/Q5_K gate/up, supported down type and fast FP16 prefill tunes 80/82; DA16 uses the admitted gate/up output image.
- **Automatic selection:** Injected defaults are filtered per GSQ owner for all controlled readers, including DA16/ROUTE; explicit environment values are preserved. Explicitly forcing those combinations is rejected until separately qualified.
- **Fallback notice:** Defaulted GSQ decline has no separate warning. Explicit GSQ overrides throw STRIX_QPF2K_MOE_ROUTE / STRIX_QPF2K_MOE_DA16 are not qualified with the GSQ prefill profile.
- **Exactness evidence:** The qualification campaign compared 1K code and 8K prose receipts, terminal FP32 logits FNV and generated tokens against the unchanged reference. Repeat with the same prefill/decode identity; changing arithmetic profiles is a separate quality comparison. Routed-MoE known-row fixtures check operator equality; do not use diagnostic skip-work tune bits for exactness.
- **Off switch:** STRIX_QPF2K_MOE_DA16=0 or STRIX_QPF2K_MOE_ROUTE=0, separately
- **Measured gain:** -
- **Public evidence:** [Qwen results](performance.md#qwen-prefill-and-defaults)

<a id="qwen-router"></a>

### Router selection and activation staging

**Engine:** `qwen`. **Control:** `STRIX_PREFILL_ROUTER_SELECT; STRIX_ROUTER_STAGE_ACT`. **Default:** ON.

Runs one wave per row for selection and stages reusable activations.

- **Preconditions:** Admitted router-row layouts; staged activation form must fit the supported shape/LDS limits.
- **Automatic selection:** Selection is on unless =0. Activation staging defaults on (mode 2 at the scalar/window reader); unsupported cases use the older forms.
- **Fallback notice:** No ordinary decline line. STRIX_PREFILL_ROUTER_CHECK=1 is a synchronizing diagnostic for router logits, not an automatic fallback notice.
- **Exactness evidence:** The qualification campaign compared 1K code and 8K prose receipts, terminal FP32 logits FNV and generated tokens against the unchanged reference. Repeat with the same prefill/decode identity; changing arithmetic profiles is a separate quality comparison. The router diagnostic compares against logits_kernel<2>; selection still needs route IDs/weights and engine token checks.
- **Off switch:** STRIX_PREFILL_ROUTER_SELECT=0 or STRIX_ROUTER_STAGE_ACT=0, separately; these are outside the QPF2K master initializer
- **Measured gain:** -
- **Public evidence:** [Qwen results](performance.md#qwen-prefill-and-defaults), [router readers](../src/hip_moe_route_launcher.hip)

### Admitted decoder wrapper settings

Each setting below is injected only after the plain CLI route is admitted. Ordinary unsupported shapes keep the prior launcher without a general decline line; execution failures propagate. Each uses whole-engine token/logit digest gates within the selected identity. No isolated gain is attributed to these nine settings.

<a id="qwen-intprefill-dense-rows"></a>

### Dense row batching

**Engine:** `qwen`. **Control:** `STRIX_INTPREFILL_DENSE_ROWS`. **Default:** 512.

Batch supported Q8-weight dense projections into 512-row fast-FP16 prefill work.

- **Preconditions:** Fast-FP16 prefill; supported Q8 projection widths, uniform input/output strides and a complete 512-row block; otherwise keep the previous projection.
- **Automatic selection:** Unspecified value becomes 512 only after decoder admission; explicit values win.
- **Fallback notice:** Unsupported operator shapes retain the prior launcher without a general decline notice. Declined routes do not receive this wrapper default.
- **Exactness evidence:** Qualified profile-pair token IDs and terminal FP32 logits digests match; hold T, capacity and arithmetic fixed for operator comparisons.
- **Off switch:** STRIX_INTPREFILL_DENSE_ROWS=0
- **Measured gain:** No isolated measurement

<a id="qwen-prefill-pipe"></a>

### Prefill pipeline

**Engine:** `qwen`. **Control:** `STRIX_PREFILL_PIPE`. **Default:** 255.

Apply the measured 255 bitmask for overlap, row loops, headless work and persistent row views.

- **Preconditions:** Admitted automatic CLI prefill only; MTP priming bits are not injected into declined MTP routes.
- **Automatic selection:** Unspecified value becomes 255 only after decoder admission; explicit values win.
- **Fallback notice:** Unsupported operator shapes retain the prior launcher without a general decline notice. Declined routes do not receive this wrapper default.
- **Exactness evidence:** Qualified profile-pair token IDs and terminal FP32 logits digests match; hold T, capacity and arithmetic fixed for operator comparisons.
- **Off switch:** STRIX_PREFILL_PIPE=0
- **Measured gain:** No isolated measurement

<a id="qwen-qsa-attend-batch"></a>

### QSA attend batching

**Engine:** `qwen`. **Control:** `STRIX_QSA_ATTEND_BATCH`. **Default:** 1.

Batch admitted group union/attend work.

- **Preconditions:** Complete 64-row pooled-key/BF16 QSA groups, compatible KV owners/geometry, default four-row tiles and supported scratch; pinned view slots are reused only after completion events.
- **Automatic selection:** Unspecified value becomes 1 only after decoder admission; explicit values win.
- **Fallback notice:** Unsupported operator shapes retain the prior launcher without a general decline notice. Declined routes do not receive this wrapper default.
- **Exactness evidence:** Qualified profile-pair token IDs and terminal FP32 logits digests match; hold T, capacity and arithmetic fixed for operator comparisons.
- **Off switch:** STRIX_QSA_ATTEND_BATCH=0
- **Measured gain:** No isolated measurement

<a id="qwen-qsa-score-park"></a>

### QSA score parking

**Engine:** `qwen`. **Control:** `STRIX_QSA_SCORE_PARK`. **Default:** 1.

Retain reusable score work for the admitted schedule.

- **Preconditions:** Prefill groups (at least 16 rows), supported pooled score buffers/group ownership and 16-byte aligned index query rows; windows/cohorts retain their route.
- **Automatic selection:** Unspecified value becomes 1 only after decoder admission; explicit values win.
- **Fallback notice:** Unsupported operator shapes retain the prior launcher without a general decline notice. Declined routes do not receive this wrapper default.
- **Exactness evidence:** Qualified profile-pair token IDs and terminal FP32 logits digests match; hold T, capacity and arithmetic fixed for operator comparisons.
- **Off switch:** STRIX_QSA_SCORE_PARK=0
- **Measured gain:** No isolated measurement

<a id="qwen-qsa-norm-group"></a>

### QSA grouped norm

**Engine:** `qwen`. **Control:** `STRIX_QSA_NORM_GROUP`. **Default:** 1.

Group compatible QSA normalization rows.

- **Preconditions:** Compatible grouped norm/RoPE geometry, weight types and owners; use a rotary table plus grouped normalization instead of three per-group launches.
- **Automatic selection:** Unspecified value becomes 1 only after decoder admission; explicit values win.
- **Fallback notice:** Unsupported operator shapes retain the prior launcher without a general decline notice. Declined routes do not receive this wrapper default.
- **Exactness evidence:** Qualified profile-pair token IDs and terminal FP32 logits digests match; hold T, capacity and arithmetic fixed for operator comparisons.
- **Off switch:** STRIX_QSA_NORM_GROUP=0
- **Measured gain:** No isolated measurement

<a id="qwen-qsa-index-rows"></a>

### QSA index rows

**Engine:** `qwen`. **Control:** `STRIX_QSA_INDEX_ROWS`. **Default:** 1.

Batch compatible index projection rows.

- **Preconditions:** BF16 input matrices in supported QSA prefill groups; batched index q/k projections retain the same chains and reduction trees.
- **Automatic selection:** Unspecified value becomes 1 only after decoder admission; explicit values win.
- **Fallback notice:** Unsupported operator shapes retain the prior launcher without a general decline notice. Declined routes do not receive this wrapper default.
- **Exactness evidence:** Qualified profile-pair token IDs and terminal FP32 logits digests match; hold T, capacity and arithmetic fixed for operator comparisons.
- **Off switch:** STRIX_QSA_INDEX_ROWS=0
- **Measured gain:** No isolated measurement

<a id="qwen-lm-head-cohort-lean"></a>

### Lean cohort head

**Engine:** `qwen`. **Control:** `STRIX_LM_HEAD_COHORT_LEAN`. **Default:** 1.

For an explicitly active multi-session head, split compatible 9–32-row cohorts into eight-row pieces using the lean head. The admitted plain CLI receipt includes this setting but has no active cohort.

- **Preconditions:** Admitted decoder environment; this injection does not enable active multi-session serving heads.
- **Automatic selection:** Unspecified value becomes 1 only after decoder admission; explicit values win.
- **Fallback notice:** Unsupported operator shapes retain the prior launcher without a general decline notice. Declined routes do not receive this wrapper default.
- **Exactness evidence:** Qualified profile-pair token IDs and terminal FP32 logits digests match; hold T, capacity and arithmetic fixed for operator comparisons.
- **Off switch:** STRIX_LM_HEAD_COHORT_LEAN=0
- **Measured gain:** No isolated measurement

<a id="qwen-lm-head-rows-lean"></a>

### Lean row head

**Engine:** `qwen`. **Control:** `STRIX_LM_HEAD_ROWS_LEAN`. **Default:** 2.

Use two logits per wave with fused greedy partials.

- **Preconditions:** Supported row-head shapes and unchanged reduction operand pairs.
- **Automatic selection:** Unspecified value becomes 2 only after decoder admission; explicit values win.
- **Fallback notice:** Unsupported operator shapes retain the prior launcher without a general decline notice. Declined routes do not receive this wrapper default.
- **Exactness evidence:** Qualified profile-pair token IDs and terminal FP32 logits digests match; hold T, capacity and arithmetic fixed for operator comparisons.
- **Off switch:** STRIX_LM_HEAD_ROWS_LEAN=0
- **Measured gain:** No isolated measurement

<a id="qwen-dispatch-fuse"></a>

### Dispatch fusion

**Engine:** `qwen`. **Control:** `STRIX_DISPATCH_FUSE`. **Default:** 1.

Combine eligible verification/row dispatch work.

- **Preconditions:** Admitted single-session tiled decoder, supported fused steps; active MTP/HTTP/cohort routes are excluded from injection.
- **Automatic selection:** Unspecified value becomes 1 only after decoder admission; explicit values win.
- **Fallback notice:** Unsupported operator shapes retain the prior launcher without a general decline notice. Declined routes do not receive this wrapper default.
- **Exactness evidence:** Qualified profile-pair token IDs and terminal FP32 logits digests match; hold T, capacity and arithmetic fixed for operator comparisons.
- **Off switch:** STRIX_DISPATCH_FUSE=0
- **Measured gain:** No isolated measurement

<a id="qwen-gsq"></a>

### GSQ compatibility and typed dense-A8 rounding

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_DEFAULTS`. **Default:** ON.

Allow GSQ models to run while prefill defaults are enabled, filtering injected settings per model owner; typed dense-A8 sites retain their declared activation rounding and per-format scale/minimum accumulation order across R1, window and prefill sites.

- **Preconditions:** GSQ manifest and its separately selected numerical identity. Non-GSQ FP16 and PLE qualification does not qualify GSQ.
- **Automatic selection:** Detect GSQ gate tensors and suppress only injected defaults after each reader cache; no global environment mutation when owners alternate.
- **Fallback notice:** GSQ declines the new content-qualified auto route; normal serial/explicit GSQ execution continues. Forced incompatible controls retain validation errors.
- **Exactness evidence:** Host GSQ-first/ordinary-first reader tests check isolation; typed-site device gates check rounding. Host tests alone do not establish device equality.
- **Off switch:** STRIX_QPF2K_DEFAULTS=0 suppresses injection globally; explicit feature overrides retain GSQ admission checks.
- **Measured gain:** No isolated measurement
- **Implementation / evidence:** [Typed dense sites](../include/strix/hip_gsq_a8_groups.hpp), [GSQ tests](../tests/hip_gsq_typed_sites_tests.hip)

<a id="qwen-gdn-witness"></a>

### Opt-in GDN witness core

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_GDN_PARTS`. **Default:** opt-in (bit 128 off; normal mask 55).

Use scratch state and a NaN witness to guard a faster core; replay the pinned exact kernel if the witness requires it.

- **Preconditions:** Eligible lean GDN core, bit 128 set, bit 64 excluded, supported cooperative residency and scratch capacity.
- **Automatic selection:** Never selected by the defaults mask 55. Explicit PARTS bit 128 requests witness admission.
- **Fallback notice:** Failed scratch/shape admission retains the existing core; a set witness runs the pinned fallback from untouched state, otherwise scratch state is committed.
- **Exactness evidence:** Requires GDN byte/state checks and same-profile engine digests, including NaN cases; no defaults-on performance claim.
- **Off switch:** Clear bit 128; STRIX_QPF2K_GDN_PARTS=55 restores the normal parts mask
- **Measured gain:** No isolated measurement
- **Implementation / evidence:** [Witness launcher](../src/hip_gdn_launcher.hip)

<a id="qwen-qsa-score-rows"></a>

### QSA scorer row batching

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_SCORE_ROWS`. **Default:** auto (32 rows above 4096 blocks, 16 otherwise).

Stage-batch the QSA scorer over more rows, reusing the loaded keys.

- **Preconditions:** Automatic decoder route (CLI), admitted tiled prefill; MTP/HTTP serving and cohorts keep it opt-in; not applied under GSQ.
- **Automatic selection:** The decoder injects the value only when unset; explicit values win; `STRIX_QPF2K_DEFAULTS=0` or `--baseline` disables all five.
- **Fallback notice:** `=0` disables only this item and keeps the tiled route; shapes outside the admission keep the previous kernel.
- **Exactness evidence:** Bitwise: 32K terminal FP32 logits and 4K digests identical with the item on and off; golden-code and prose gates equal the archived base.
- **Off switch:** STRIX_QPF2K_SCORE_ROWS=0 (16 rows)
- **Measured gain:** ≈ −1.6 s at 262K together with PLE row batching (~−10 ms at 32K); combined five-item set −0.71 s (−3.2 %) at 32K vs all five off (interleaved, same build)
- **Implementation / evidence:** [defaults](../include/strix/qpf2k_defaults.hpp), [engine reference](getting-started/engine-reference.md)

<a id="qwen-ple-batch"></a>

### PLE row batching

**Engine:** `qwen`. **Control:** `STRIX_PREFILL_PLE_BATCH`. **Default:** 1.

Run the per-layer-embedding projection in 64-row groups; the wide lane recipe, layout and scratch admission still apply.

- **Preconditions:** Automatic decoder route (CLI), admitted tiled prefill; MTP/HTTP serving and cohorts keep it opt-in; not applied under GSQ.
- **Automatic selection:** The decoder injects the value only when unset; explicit values win; `STRIX_QPF2K_DEFAULTS=0` or `--baseline` disables all five.
- **Fallback notice:** `=0` disables only this item and keeps the tiled route; shapes outside the admission keep the previous kernel.
- **Exactness evidence:** Bitwise: 32K terminal FP32 logits and 4K digests identical with the item on and off; golden-code and prose gates equal the archived base.
- **Off switch:** STRIX_PREFILL_PLE_BATCH=0
- **Measured gain:** Included in the scorer figure above (−0.16 s raw kernel time at 262K); combined five-item set −0.71 s (−3.2 %) at 32K vs all five off (interleaved, same build)
- **Implementation / evidence:** [defaults](../include/strix/qpf2k_defaults.hpp), [engine reference](getting-started/engine-reference.md)

<a id="qwen-shared-pre"></a>

### Shared-expert operand preimage

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_SHARED_PRE`. **Default:** 1.

Build the shared expert's FP16 operand once for gate/up and down; the existing tiled down admission and fallback apply.

- **Preconditions:** Automatic decoder route (CLI), admitted tiled prefill; MTP/HTTP serving and cohorts keep it opt-in; not applied under GSQ.
- **Automatic selection:** The decoder injects the value only when unset; explicit values win; `STRIX_QPF2K_DEFAULTS=0` or `--baseline` disables all five.
- **Fallback notice:** `=0` disables only this item and keeps the tiled route; shapes outside the admission keep the previous kernel.
- **Exactness evidence:** Bitwise: 32K terminal FP32 logits and 4K digests identical with the item on and off; golden-code and prose gates equal the archived base.
- **Off switch:** STRIX_QPF2K_SHARED_PRE=0
- **Measured gain:** ~−1 % TTFT at 32K; combined five-item set −0.71 s (−3.2 %) at 32K vs all five off (interleaved, same build)
- **Implementation / evidence:** [defaults](../include/strix/qpf2k_defaults.hpp), [engine reference](getting-started/engine-reference.md)

<a id="qwen-moe-merge-commit"></a>

### MoE ordered merge + BF16 commit

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_MOE_MC`. **Default:** 1 (mode 1; mode 2 needs an explicit value).

Merge expert outputs in a fixed order and commit BF16 rows in the same pass, behind the existing DA16/ROUTE=14 certificate.

- **Preconditions:** Automatic decoder route (CLI), admitted tiled prefill; MTP/HTTP serving and cohorts keep it opt-in; not applied under GSQ.
- **Automatic selection:** The decoder injects the value only when unset; explicit values win; `STRIX_QPF2K_DEFAULTS=0` or `--baseline` disables all five.
- **Fallback notice:** `=0` disables only this item and keeps the tiled route; shapes outside the admission keep the previous kernel.
- **Exactness evidence:** Bitwise: 32K terminal FP32 logits and 4K digests identical with the item on and off; golden-code and prose gates equal the archived base.
- **Off switch:** STRIX_QPF2K_MOE_MC=0
- **Measured gain:** ~−0.10 s at 32K; combined five-item set −0.71 s (−3.2 %) at 32K vs all five off (interleaved, same build)
- **Implementation / evidence:** [defaults](../include/strix/qpf2k_defaults.hpp), [engine reference](getting-started/engine-reference.md)

<a id="qwen-moe-header-reuse"></a>

### MoE gate/up header reuse

**Engine:** `qwen`. **Control:** `STRIX_QPF2K_MOE_HDR`. **Default:** 1.

Reuse loaded superblock headers between gate and up within the DA16 tune 80/82 path.

- **Preconditions:** Automatic decoder route (CLI), admitted tiled prefill; MTP/HTTP serving and cohorts keep it opt-in; not applied under GSQ.
- **Automatic selection:** The decoder injects the value only when unset; explicit values win; `STRIX_QPF2K_DEFAULTS=0` or `--baseline` disables all five.
- **Fallback notice:** `=0` disables only this item and keeps the tiled route; shapes outside the admission keep the previous kernel.
- **Exactness evidence:** Bitwise: 32K terminal FP32 logits and 4K digests identical with the item on and off; golden-code and prose gates equal the archived base.
- **Off switch:** STRIX_QPF2K_MOE_HDR=0
- **Measured gain:** ~−0.07 s at 32K; combined five-item set −0.71 s (−3.2 %) at 32K vs all five off (interleaved, same build)
- **Implementation / evidence:** [defaults](../include/strix/qpf2k_defaults.hpp), [engine reference](getting-started/engine-reference.md)

### Exact options that remain opt-in

MoE PK/MERGE, QSA_OUT16, attend pairing and `STRIX_QPF2K_HC_LOWBAND` are exact experiments with neutral or unmeasured complete-workload
gains. They are not part of the default set. Do not infer an environment spelling from these experiment names;
use only controls exposed by the current source and require the corresponding device gates before promotion.

<a id="qwen-serve-api"></a>

### HTTP API surface: Responses route, capabilities, stream receipts, keepalive

**Engine:** `qwen` (also `strix-deepseek41-serve`, same shared server layer). **Control:** `--http-responses`,
`--http-capabilities`, `--http-stream-receipts`, `--http-sse-keepalive`. **Default:** on, on, on, 10 seconds.

Serving features that change no generated token: a stateless `POST /v1/responses` route translated onto the chat
prompt path, a versioned `GET /v1/capabilities` document, the applied policy (requested vs effective sampling and
output limit, each sampling filter with its source, the effective seed of a sampled request) at the start and end of
every stream, and SSE keepalive comments while a stream waits. Sampling filters resolve request -> server default
(`--http-default-temperature`, `--http-default-top-k`, `--http-default-top-p`, `--http-default-min-p`; a positive
default temperature is a token-changing operator choice and needs `--http-sampling sample`) -> built-in greedy; an
unseeded sampled request gets a seed, echoed in the receipt (stream receipts on), so the same request with that seed
reproduces it.

- **Preconditions:** `--http-serve` (Qwen) or the DeepSeek serve app.
- **Automatic selection:** Always on in the server; a request disables its keepalive with `"strix":{"keepalive":false}`.
- **Fallback notice:** Unsupported Responses fields are refused with a 400 naming the field; nothing is ignored silently.
- **Exactness evidence:** Responses prompt IDs equal the equivalent chat request (CPU tests); greedy token IDs equal with
  the features on and off, and the stream bytes with every switch off equal the previous build after normalizing ids
  and timings (GPU gate).
- **Off switch:** `--http-responses off`, `--http-capabilities off`, `--http-stream-receipts off`, `--http-sse-keepalive 0`

<a id="qwen-serve-constraints"></a>

### Constrained JSON output

**Engine:** `qwen`. **Control:** `--http-constraints`. **Default:** on (requests opt in with `response_format` /
`text.format`).

Token selection restricted to a declared JSON language (`json-object` or a `strix-json-schema-v1` schema), greedy, on a
serial host route for the constrained request only.

- **Preconditions:** `--http-serve`; a request with `response_format` / `text.format` `json_object` or `json_schema`,
  greedy, no stop strings, no tools.
- **Automatic selection:** Only requests that ask for it; other requests and other batch slots are unchanged.
- **Fallback notice:** Unsupported schema keywords, sampling, stop strings or tools are refused with a 400; output is
  never produced unconstrained in their place.
- **Exactness evidence:** Constrained output is exact relative to its declared constrained target (masked greedy), not
  to unconstrained greedy. Unconstrained requests, alone and sharing the batch with constrained ones, keep their token
  IDs and draft counters (GPU gate).
- **Off switch:** `--http-constraints off`

## DeepSeek two-node

```sh
strix-deepseek41-serve MODEL.gguf --rank R --transport usb4auto
```

Launch one rank on each node using the same assets and compatible transport configuration. Plain rank,
batch, prefill and serve invocations resolve the production configuration. The EP prefill transport spelling
is `usb4-auto`. Inspect with `--print-effective-config`. The resolver discovers packs under `STRIX_DS41_PACKS`,
then the model's `packs` directory, then the model directory. Missing prerequisites print one
`deepseek41 defaults:` notice per dropped item; explicit arguments/environment win. See
[production defaults](models/deepseek-v4.1/production-defaults.md).

`STRIX_DS41_BASELINE=1` stops pack/draft/argv injection and sets the nine recent defaults off unless explicitly
overridden: grouped shared work, shared down, MoE prefill, dense selector, three attention settings, serving
warm-up and calibration. Older kernel/transport defaults remain; disable them individually for an older reference.
The combined qualified build measured 1K/8K TTFT 2.098 → 1.954 s / 11.090 → 9.757 s, R1 −1.9%,
R6 −2.4%, N8 −2.2%, and warmed C1 43.58 → 48.00 tok/s. These are build gains, not item attribution.

<a id="ds-packs"></a>

### Production dense overlay

**Engine:** `deepseek`. **Control:** `STRIX_DS41_OVERLAY; STRIX_DS41_CB4`. **Default:** auto.

Discover c-u6.gguf and enable CB4.

- **Preconditions:** Complete production packs with valid model content.
- **Automatic selection:** Set overlay and CB4=1 only when packs are found; existing environment wins.
- **Fallback notice:** Missing resolver prerequisites print a `deepseek41 defaults:` notice; ordinary operator shape dispatch has no universal decline notice. Explicit incompatible configurations retain errors.
- **Exactness evidence:** Both-rank reference gates compare prefill logits/state digests, greedy token IDs, verify rows and batch rows with fixed Q2 experts, dense overlay and trellis packs; pack identity changes require their separate quality reference.
- **Off switch:** STRIX_DS41_OVERLAY=0
- **Measured gain:** No isolated measurement
- **Implementation / evidence:** [Resolver](../src/deepseek41_pair_defaults.cpp), [contract](models/deepseek-v4.1/production-defaults.md)

<a id="ds-packs-trellis"></a>

### Production trellis dense identity

**Engine:** `deepseek`. **Control:** `STRIX_DS41_TRELLIS; STRIX_DS41_TRELLIS_MIX; STRIX_DS41_TRELLIS_SITES; STRIX_DS41_TRELLIS_DENSE_FULL; STRIX_DS41_TRELLIS_HEADOUT`. **Default:** auto.

Select ours-K4 dense, ours-K5 head, all sites, full dense and head output.

- **Preconditions:** Both packs, dense_r1 trellis library and API; excluded from the EP prefill app.
- **Automatic selection:** All-or-nothing discovered identity; no partial automatic pack set.
- **Fallback notice:** Missing resolver prerequisites print a `deepseek41 defaults:` notice; ordinary operator shape dispatch has no universal decline notice. Explicit incompatible configurations retain errors.
- **Exactness evidence:** Both-rank reference gates compare prefill logits/state digests, greedy token IDs, verify rows and batch rows with fixed Q2 experts, dense overlay and trellis packs; pack identity changes require their separate quality reference.
- **Off switch:** STRIX_DS41_TRELLIS=0; STRIX_DS41_PACKS=0 disables discovery
- **Measured gain:** No isolated measurement
- **Implementation / evidence:** [Resolver](../src/deepseek41_pair_defaults.cpp), [contract](models/deepseek-v4.1/production-defaults.md)

<a id="ds-topology"></a>

### TP attention and direct Engram

**Engine:** `deepseek`. **Control:** `--tp-attention; --engram-direct`. **Default:** auto.

Use two-rank TP attention and direct Engram reads in rank/batch apps.

- **Preconditions:** Rank/batch invocation; two-rank transport for TP attention.
- **Automatic selection:** Enable unless the corresponding explicit opt-out is present.
- **Fallback notice:** Missing resolver prerequisites print a `deepseek41 defaults:` notice; ordinary operator shape dispatch has no universal decline notice. Explicit incompatible configurations retain errors.
- **Exactness evidence:** Both-rank reference gates compare prefill logits/state digests, greedy token IDs, verify rows and batch rows with fixed Q2 experts, dense overlay and trellis packs; pack identity changes require their separate quality reference.
- **Off switch:** --no-tp-attention; --engram-mapped
- **Measured gain:** No isolated measurement
- **Implementation / evidence:** [Resolver](../src/deepseek41_pair_defaults.cpp), [contract](models/deepseek-v4.1/production-defaults.md)

<a id="ds-serve-shape"></a>

### Serving production shape

**Engine:** `deepseek`. **Control:** `--peer-timeout; --prefill-path; --slots; --prefill-rows`. **Default:** auto.

Resolve timeout 120, device prefill, 8 slots and 2048 prefill rows.

- **Preconditions:** Serve app and capacity for the requested slots/rows.
- **Automatic selection:** Prepend defaults before caller arguments so explicit flags win.
- **Fallback notice:** Missing resolver prerequisites print a `deepseek41 defaults:` notice; ordinary operator shape dispatch has no universal decline notice. Explicit incompatible configurations retain errors.
- **Exactness evidence:** Both-rank reference gates compare prefill logits/state digests, greedy token IDs, verify rows and batch rows with fixed Q2 experts, dense overlay and trellis packs; pack identity changes require their separate quality reference.
- **Off switch:** Explicit values override each flag; STRIX_DS41_BASELINE=1 suppresses injection
- **Measured gain:** No isolated measurement
- **Implementation / evidence:** [Resolver](../src/deepseek41_pair_defaults.cpp), [contract](models/deepseek-v4.1/production-defaults.md)

<a id="ds-draft"></a>

### Automatic DSpark and vocabulary

**Engine:** `deepseek`. **Control:** `--dspark; --dspark-vocab`. **Default:** auto.

Discover a DSpark-MXFP4 GGUF and auto vocabulary beside the model/packs.

- **Preconditions:** Draft asset; core48k.bin and core32k.bin for vocabulary; not BATCH_ROWS16 or prefix-session snapshots.
- **Automatic selection:** Missing draft disables drafting; missing vocabulary keeps full vocabulary.
- **Fallback notice:** Missing resolver prerequisites print a `deepseek41 defaults:` notice; ordinary operator shape dispatch has no universal decline notice. Explicit incompatible configurations retain errors.
- **Exactness evidence:** Both-rank reference gates compare prefill logits/state digests, greedy token IDs, verify rows and batch rows with fixed Q2 experts, dense overlay and trellis packs; pack identity changes require their separate quality reference.
- **Off switch:** --no-dspark; --dspark-vocab full
- **Measured gain:** No isolated measurement
- **Implementation / evidence:** [Resolver](../src/deepseek41_pair_defaults.cpp), [contract](models/deepseek-v4.1/production-defaults.md)

<a id="ds-shared-grouped"></a>

### Grouped shared expert

**Engine:** `deepseek`. **Control:** `STRIX_DS41_SHARED_GROUPED`. **Default:** ON.

Group shared-expert SwiGLU pairs.

- **Preconditions:** Supported shared-expert shapes and production dense identity.
- **Automatic selection:** Default on; shape dispatch keeps compatible kernels.
- **Fallback notice:** Missing resolver prerequisites print a `deepseek41 defaults:` notice; ordinary operator shape dispatch has no universal decline notice. Explicit incompatible configurations retain errors.
- **Exactness evidence:** Both-rank reference gates compare prefill logits/state digests, greedy token IDs, verify rows and batch rows with fixed Q2 experts, dense overlay and trellis packs; pack identity changes require their separate quality reference.
- **Off switch:** STRIX_DS41_SHARED_GROUPED=0
- **Measured gain:** No isolated measurement
- **Implementation / evidence:** [Resolver](../src/deepseek41_pair_defaults.cpp), [contract](models/deepseek-v4.1/production-defaults.md)

<a id="ds-core"></a>

### Expert TP and batched operators

**Engine:** `deepseek`. **Control:** `STRIX_DS41_EXPERT_TP; STRIX_DS41_EXPERT_SLOTS; STRIX_DS41_ROUTES_FAST; STRIX_DS41_F16_PAIR; STRIX_DS41_TP_DOWN; STRIX_DS41_BATCH_SEGMENTS`. **Default:** ON.

Splits expert work across the pair, retains expert slots, builds integer routes faster, pairs eligible F16 projections and batches attention segments.

- **Preconditions:** Two ranks and supported expert geometry; TP down width 1152; segment batching does not combine with BATCH_ROWS16=1.
- **Automatic selection:** Each is on unless disabled; BATCH_SEGMENTS declines itself when STRIX_DS41_BATCH_ROWS16=1. Shape guards keep supported older paths.
- **Fallback notice:** Routine operator eligibility has no general decline log. Invalid expert-TP geometry fails validation; do not treat a transport change as the same numerical identity.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms.
- **Off switch:** Set one of STRIX_DS41_EXPERT_TP, STRIX_DS41_EXPERT_SLOTS, STRIX_DS41_ROUTES_FAST, STRIX_DS41_F16_PAIR, STRIX_DS41_TP_DOWN or STRIX_DS41_BATCH_SEGMENTS to 0 per comparison.
- **Measured gain:** Batch segments N8 +9%; TP down R6 -3%, N8 +1.6%
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-scheduling"></a>

### Serving interleave and draft admission

**Engine:** `deepseek`. **Control:** `STRIX_DS41_SERVE_SCHED; --dspark-ranks; --dspark-max-inflight; --draft-ranks`. **Default:** auto.

Interleaves prompt work with decode; uses both drafters and switches to plain target steps above the in-flight threshold.

- **Preconditions:** Device prefill serving; two draft ranks require the drafter assets on both ranks.
- **Automatic selection:** Device serving defaults to interleave, --dspark-ranks 2 and --dspark-max-inflight 4; batch defaults --draft-ranks 2. Row prefill keeps the compatible scheduler.
- **Fallback notice:** Scheduler selection follows the prefill path; no universal decline line. Draft proposal changes are visible in acceptance/timing, while committed tokens remain target verified.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms.
- **Off switch:** STRIX_DS41_SERVE_SCHED=off; independently --dspark-ranks 1, --dspark-max-inflight 0 (always draft), or batch --draft-ranks 1
- **Measured gain:** 8K under load: decode stall <=1 s instead of ~13 s; prompt TTFT about doubles
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-transport"></a>

### USB4 eager sends and fused staging

**Engine:** `deepseek`. **Control:** `STRIX_DIST_USB4_EAGER=65536; STRIX_DIST_STAGING=fused; --transport usb4gpu`. **Default:** auto.

Sends small payloads eagerly and lets TP products write/read pinned transport buffers directly.

- **Preconditions:** Two-node USB4 routes; fused staging applies to host-path routes. GPU-direct requires streams exposing that capability.
- **Automatic selection:** Eager threshold is 65536 bytes; fused staging is the default. usb4gpu chooses the device exchange only with GPU-direct streams.
- **Fallback notice:** Without GPU-direct streams the apps print gpu-direct unavailable ... host path only and use the host path with the same bits.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms.
- **Off switch:** STRIX_DIST_USB4_EAGER=0 or STRIX_DIST_STAGING=copy, separately; transport CLI spelling for the prefill app is usb4-gpu
- **Measured gain:** Eager R1 +2.8%; fused staging R1 -0.2..-0.4 ms/token
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-ready"></a>

### Ready views, CRC overlap and deferred checks

**Engine:** `deepseek`. **Control:** `STRIX_DIST_CRC_PARALLEL; STRIX_DIST_HOST_CRC_OVERLAP; STRIX_DIST_HOST_CHUNK_BYTES=1048576; STRIX_DIST_STAGED_READY_BYTES=262144; STRIX_DS41_READY_LEAN; STRIX_DS41_MOE_READY; STRIX_DS41_OB_READY_EPILOGUE; STRIX_DS41_READY_READS_ONLY; STRIX_DS41_TRELLIS_OB_READY; STRIX_DIST_CRC_DEFER`. **Default:** auto.

Publishes readiness from producers, reduces staging copies and overlaps CRC work with transfer without removing corruption checks.

- **Preconditions:** Compatible ready-view routes and identical transport modes on both ranks; trellis ready applies at trellis o_b sites.
- **Automatic selection:** Ready lean/MoE and trellis ready are on; epilogue/reads-only follow READY_LEAN. Chunks default to 1 MiB, readiness views to 256 KiB; CRC defer checks TX/RX via the next trailer.
- **Fallback notice:** Routes without ready views retain the previous staging/checks; Q8 o_b epilogues do not run at trellis o_b sites. There is no universal decline notice. Corruption fails the step and mixed deferred modes fail immediately.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms. Transport integrity gates remain required; token equality alone does not exercise corruption handling.
- **Off switch:** Set one boolean above to 0 per comparison; STRIX_DIST_HOST_CHUNK_BYTES=0 selects one piece, STRIX_DIST_STAGED_READY_BYTES=0 removes readiness views.
- **Measured gain:** CRC overlap bundle R6 -1.37%, N8 -1.77%; trellis ready + deferred CRC R1/R6/N8 -1.6/-1.9/-0.7%
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-overlap"></a>

### Shared expert overlap and late HC

**Engine:** `deepseek`. **Control:** `STRIX_DS41_BATCH_SHARED=overlap; STRIX_DS41_LATE_HC_LAYERS=all`. **Default:** auto.

Hides shared-expert work and attention-site HC mixes inside exchange waits.

- **Preconditions:** Eligible batch half-sum / attention o_b exchanges; late HC does not combine with BATCH_PREFETCH.
- **Automatic selection:** Batch shared defaults to overlap; late HC defaults to all eligible layers. With batch prefetch the late-HC path stays out.
- **Fallback notice:** Ineligible overlap retains the preceding ordering without a general decline line; explicit split shared mode remains a separate choice.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms.
- **Off switch:** STRIX_DS41_BATCH_SHARED=off or STRIX_DS41_LATE_HC_LAYERS=none, separately
- **Measured gain:** Shared overlap N1/N2/N4/N8 -2.8/-3.2/-5.1/-5.2%; late HC R1 -0.2 ms, R6 -0.35 ms
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-prefill"></a>

### Final KV ownership, prefill tiles and Engram cache

**Engine:** `deepseek`. **Control:** `STRIX_DS41_PREFILL_FINAL_KV=all; STRIX_DS41_PREFILL_TILES=1 (opt-in); STRIX_DS41_PREFILL_TILES_MIN_ROWS=2048; STRIX_DS41_ENGRAM_PAYLOAD_CACHE`. **Default:** auto.

Transfers final state instead of intermediate FP32 K/V and caches repeated Engram payloads; unsplit prefill tiles are available as an opt-in.

- **Preconditions:** Row-split prefill with final ring/compressed-state owners; tile runs at least 2048 rows by default; a cache budget that fits memory.
- **Automatic selection:** Final KV defaults to all; prefill tiles are off (split-K kernels), and with STRIX_DS41_PREFILL_TILES=1 they apply above the row threshold; payload caching is on with STRIX_DS41_ENGRAM_PAYLOAD_CACHE_MIB controlling its cap.
- **Fallback notice:** Subthreshold tiling silently keeps split-K. Cache misses use normal reads. No general decline line is printed for these ordinary selections.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms. Final KV qualification used prefill --reference and identical decode continuation.
- **Off switch:** STRIX_DS41_PREFILL_FINAL_KV=none or STRIX_DS41_ENGRAM_PAYLOAD_CACHE=0, separately (tiles are opt-in: STRIX_DS41_PREFILL_TILES=1)
- **Measured gain:** Final KV 8K 714.9 -> 729.4 tok/s; cache 32K TTFT -1.63%; tiles on are slower on the current kernels (TTFT 2K/8K/32K +1.0/+3.4/+1.9 %), hence off
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-prefetch"></a>

### Q8 exchange read-ahead

**Engine:** `deepseek`. **Control:** `STRIX_DS41_BATCH_PREFETCH; STRIX_DS41_PREFETCH_SIDE; STRIX_DS41_PREFETCH_ROW_MB; STRIX_DS41_DECODE_PREFETCH`. **Default:** auto.

Reads future dense weights into MALL while the ranks wait for exchange.

- **Preconditions:** Q8 o_b exchange waits; trellis sites use their own packs and are skipped.
- **Automatic selection:** Batch defaults to 24 MB with side stream on and 3 MB per row; decode defaults to 2 MB per row in the o_b window. Only Q8 sites read ahead.
- **Fallback notice:** Trellis sites skip read-ahead silently. PREFETCH_SIDE=0 chooses the main-stream form rather than disabling all prefetch.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms.
- **Off switch:** STRIX_DS41_BATCH_PREFETCH=0 or STRIX_DS41_DECODE_PREFETCH=0, separately
- **Measured gain:** Batch N2/N4/N8 -2.1/-4.9/-3.3%; decode R6 -1.8% on Q8 sites
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-index"></a>

### Indexer pruning and shared segment geometry

**Engine:** `deepseek`. **Control:** `STRIX_DS41_IDX_DIRECT; STRIX_DS41_IDX_CAND_FILTER; STRIX_DS41_ATTN_SEG_HPW`. **Default:** auto.

Skips redundant indexing, scores only candidates and shares eligible segment launch geometry.

- **Preconditions:** Direct indexing count <=512; candidate filtering at pool layers above 16K entries; shared heads-per-wave for N8 segments.
- **Automatic selection:** Each switch is on; its count/context/segment predicate decides whether the optimized form applies.
- **Fallback notice:** Outside the predicate the previous indexer/segment form runs without a general decline line.
- **Exactness evidence:** In-process ABA x3 per cell compared tokens, logits slices and attention states on both ranks; prefill 32K/400 rows compared logits and state digest.
- **Off switch:** STRIX_DS41_IDX_DIRECT=0, STRIX_DS41_IDX_CAND_FILTER=0 or STRIX_DS41_ATTN_SEG_HPW=0, separately
- **Measured gain:** Short R1/R6/N8 -2.0/-2.7/-1.0%; 128K -0.8/-1.9/-3.3% (combined)
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-trellis"></a>

### Grouped trellis o_a and q_a/KV pair

**Engine:** `deepseek`. **Control:** `STRIX_DS41_TRELLIS_OA_GROUP; STRIX_DS41_TRELLIS_QAKV_PAIR`. **Default:** auto.

Groups o_a projections and submits q_a with KV in one launch.

- **Preconditions:** Production K4 dense / K5 head packs and a dense_r1 build linked to a library exposing dense-r1 and grouped GEMV APIs.
- **Automatic selection:** On for the admitted trellis sites; the production environment selects the packs independently of these scheduling switches.
- **Fallback notice:** Unsupported library/shape admission keeps supported staged forms or rejects an explicit incompatible configuration; there is no single generic decline notice for both adapters.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms. Exactness is against unchanged packs, not the original Q8_0 weights; pack quality has a separate gate.
- **Off switch:** STRIX_DS41_TRELLIS_OA_GROUP=0 or STRIX_DS41_TRELLIS_QAKV_PAIR=0, separately
- **Measured gain:** Combined R1 34.7 -> 31.15 ms (-10.2%); R6 -4.1%; N8 -2.1%
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-shared-down"></a>

### Single-launch shared down

**Engine:** `deepseek`. **Control:** `STRIX_DS41_SHARED_DOWN_FUSED`. **Default:** auto.

Replaces staged shared-down submission with one declared fused launch.

- **Preconditions:** All shared-down sites must be K4 [5120,2304] in one context, no shared-expert TP, and the linked library must pass declared-workspace probes for the supported rows.
- **Automatic selection:** Unset means try once at startup; an implicit refusal is retained for the process. Explicit =1 demands admission and throws if it fails.
- **Fallback notice:** Prints ds41 trellis: shared down fused launch not admitted by default (...): old path on implicit refusal; prints shared down as one fused launch ... when admitted. Once used, a later incompatible admission is an error.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms.
- **Off switch:** STRIX_DS41_SHARED_DOWN_FUSED=0
- **Measured gain:** -
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md), [startup probe](../src/hip_ds41_trellis.hip)

<a id="ds-moe7"></a>

### Split gate/up prefill form 7

**Engine:** `deepseek`. **Control:** `STRIX_DS41_MOE_PREFILL`. **Default:** ON.

Uses the qualified split gate/up MoE prefill schedule.

- **Preconditions:** Supported device-prefill expert shapes and quantized formats.
- **Automatic selection:** Default form 7; supported row/format dispatch determines its kernel, with the previous form selected by =0.
- **Fallback notice:** No general decline line for normal kernel selection; invalid geometry still fails the operator validation.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms.
- **Off switch:** STRIX_DS41_MOE_PREFILL=0
- **Measured gain:** -
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-dense"></a>

### Exact dense prefill selector

**Engine:** `deepseek`. **Control:** `STRIX_DS41_DENSE_SELECT`. **Default:** auto.

Enables q_b/o_a dense sites and selects an exact form per call without enabling approximate dense arithmetic.

- **Preconditions:** Exact dense prefill, supported site geometry; approximate FP16 dense and BF16 attention remain separate opt-ins.
- **Automatic selection:** On unless the value is exactly 0. At exact calls with v.tall=0 it requests the auto form; explicit nonzero form settings and approximate FP16 calls keep their configured paths.
- **Fallback notice:** No general decline log; unsupported exact shapes keep the compatible form. Optional FP16 requests without the build backend fail with FP16 dense prefill requires STRIX_DS41_PREFILL_FP16_LT.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms.
- **Off switch:** STRIX_DS41_DENSE_SELECT=0 restores loaded STRIX_DS41_DENSE_TALL / STRIX_DS41_PREFILL_DENSE_SITES configuration
- **Measured gain:** -
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md), [selector](../src/hip_deepseek41_prefill.hip)

<a id="ds-attention"></a>

### Attention fold, live chunks and F16 tiles

**Engine:** `deepseek`. **Control:** `STRIX_DS41_ATTN_WINDOW_FOLD; STRIX_DS41_ATTN_LIVE_CHUNKS; STRIX_DS41_ATTN_TILE_F16`. **Default:** auto.

Folds window state work, schedules live chunks and stores eligible attention tiles in half precision while preserving the qualified arithmetic.

- **Preconditions:** Valid attention ring/KV views and supported window rows; half tiles apply only at one or two heads per wave.
- **Automatic selection:** All three default on. Four-heads-per-wave kernels keep the non-half tile because the half form exceeds the register file.
- **Fallback notice:** Unsupported half geometry uses the non-half kernel silently. Invalid folded KV/ring geometry throws ds41 attention: window fold needs kv, kv_norm and ring_out == ring together, at most kMaxRows rows.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms.
- **Off switch:** STRIX_DS41_ATTN_WINDOW_FOLD=0, STRIX_DS41_ATTN_LIVE_CHUNKS=0 or STRIX_DS41_ATTN_TILE_F16=0, separately
- **Measured gain:** -
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-warm"></a>

### Server warm-up and DSpark calibration

**Engine:** `deepseek`. **Control:** `STRIX_DS41_SERVE_WARM; STRIX_DS41_DSPARK_CALIB`. **Default:** auto.

Warms supported serving paths before READY and prices the speculative controller on actual device costs.

- **Preconditions:** Non-split serving with dummy KV slots available; DSpark calibration requires the drafter.
- **Automatic selection:** Warm-up is on; calibration follows warm-up outside split decode, with explicit =0/=1 overrides. Split decode skips local decode warm-up.
- **Fallback notice:** If implicit warm-up cannot reserve dummy slots it prints rank ...: warm-up skipped: the KV pool has no room for ... . Explicit SERVE_WARM=1 instead fails when they do not fit.
- **Exactness evidence:** The published two-node gates compare both ranks with the reference: prefill logits, decode token IDs, verify rows and batch rows. Keep the same Q2 experts, dense overlay and trellis packs in both arms. The warmed-server campaign compared all 64 request token sequences in each run.
- **Off switch:** STRIX_DS41_SERVE_WARM=0 or STRIX_DS41_DSPARK_CALIB=0, separately
- **Measured gain:** Combined two-node build: warmed C1 43.58 -> 48.00 tok/s (+9.9%); not isolated attribution
- **Public evidence:** [Two-node defaults](models/deepseek-v4.1/production-defaults.md)

<a id="ds-observation"></a>

### Bounded serving observation

**Engine:** `deepseek`. **Control:** `STRIX_SERVE_OBSERVE_EXPIRY_S`. **Default:** 60 seconds.

Expire an armed detailed serving observation after a bounded interval.

- **Preconditions:** Serve app with an observation arming/renewal request.
- **Automatic selection:** An arming request starts the 60-second lease; metrics and detailed status requests renew it.
- **Fallback notice:** Expiry stops the observation; it does not change inference dispatch or model state.
- **Exactness evidence:** Observation lifetime changes no arithmetic or committed tokens; inference reference gates remain required.
- **Off switch:** STRIX_SERVE_OBSERVE_EXPIRY_S=0 keeps observation armed until exit
- **Measured gain:** No isolated measurement

## DeepSeek RTX PRO standalone

Use the [standalone run command and required fixtures](models/deepseek-v4.1/rtx-standalone.md#run-and-persistent-worker).
`strix-cuda-ds41-replica MODEL.gguf --print-config` inspects resolved JSON before CUDA, model or fixture reads;
small prior/pruning/residency files may still be read. Defaults target the standalone Q2 GGUF/Q8_0 identity.
`--baseline` restores reference forms at that argument; later explicit flags win. `STRIX_DS41_RTX_BASELINE=1`
applies before all argv. Baseline also restores 96 host spares, 4 pressure spares, inferred row capacity,
run controller scope, full draft vocabulary, serial load unless explicitly set, and the old arena rule.
Normal defaults use 48 host spares, 8 pressure spares, 6 rows; arena is 74.8 GiB with keep-64, 76.5 with keep-32,
71.5 with full drafter, 80 without. Explicit arena size wins; host allocation is capped at 68 GiB with a 6 GiB margin.

Every RTX item uses the replica's exact0/pressure/exact8/wforce0/spec0/exact0 and plain/speculative/plain chat gates:
at least 64 compared steps, matched == compared, gate pass, canonical position/state digest equal, complete chat
vectors equal; pressure must exercise bank/resurrection/mid-step collection. Pruning/controller/vocabulary can
change acceptance, not committed target-greedy tokens. Compare within one load where supported (load spread ~10%).

<a id="rtx-kernels"></a>

### Decode kernel bundle

**Engine:** `deepseek-rtx`. **Control:** `--dsrtxk`. **Default:** ON.

Router, projection/fork, MoE, attention and norm decode kernels.

- **Preconditions:** Admitted Q2/Q8 standalone model and supported CUDA shapes.
- **Automatic selection:** Default bundle on; explicit member flags can override.
- **Fallback notice:** Supported shapes use optimized forms; incompatible explicit requests retain validation.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --dsrtxk-off
- **Measured gain:** All-hit 12.62 → 10.50 ms/token; full loop 5.8–10.8%
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-fusions"></a>

### HC and shared-expert fusions

**Engine:** `deepseek-rtx`. **Control:** `--fuse`. **Default:** ON.

Fuse compatible HC/shared-expert work.

- **Preconditions:** Supported standalone sites and shapes.
- **Automatic selection:** On independently of decode bundle.
- **Fallback notice:** Unsupported shapes retain supported exact dispatch; invalid explicit combinations fail.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --fuse-off
- **Measured gain:** Included in decode qualification; not isolated
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-records"></a>

### Deferred step records

**Engine:** `deepseek-rtx`. **Control:** `--records-deferred`. **Default:** ON.

Move step records off the critical path.

- **Preconditions:** Replica step/qualification recording.
- **Automatic selection:** Deferred unless explicitly inline.
- **Fallback notice:** No refusal notice; inline is an explicit order choice.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --records-inline
- **Measured gain:** Qualified; no isolated measurement
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-router"></a>

### Router form 2

**Engine:** `deepseek-rtx`. **Control:** `--router-pf`. **Default:** ON.

Use form 2 router.

- **Preconditions:** Supported router rows 1–8.
- **Automatic selection:** Default form 2 in decode configuration.
- **Fallback notice:** Shape dispatch keeps a supported exact form; no general decline notice.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --router-pf0
- **Measured gain:** R1 17.3 → 15.4 us; eight-layer device time −0.5%
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-qakv"></a>

### QA/KV split 1

**Engine:** `deepseek-rtx`. **Control:** `--qakv-s1`. **Default:** ON.

Use split-1 query/KV scheduling.

- **Preconditions:** Supported query/KV geometry.
- **Automatic selection:** Default split 1.
- **Fallback notice:** Supported exact dispatch retained outside specialized geometry.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --qakv-auto
- **Measured gain:** Eight-layer device time −1.54 to −1.94%
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-q8"></a>

### Compact Q8 R1

**Engine:** `deepseek-rtx`. **Control:** `--q8-r1`. **Default:** ON.

Use compact single-row kernel without padding.

- **Preconditions:** Supported Q8 single-row shapes.
- **Automatic selection:** Use compact form when admitted.
- **Fallback notice:** One-time notice for shapes requiring the generic exact Q8 kernel.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --q8-r1-off; --q8-r1-pad selects padded form
- **Measured gain:** Eight-layer device time −0.80 / −0.53%
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-pruning"></a>

### Keep-64 drafter pruning

**Engine:** `deepseek-rtx`. **Control:** `--draft-keep 64`. **Default:** ON.

Keep most-used 64 experts per stage.

- **Preconditions:** DSpark drafter with built-in expert ranking.
- **Automatic selection:** Default keep count 64.
- **Fallback notice:** Absent drafter makes pruning irrelevant; explicit unsupported configurations fail validation.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --draft-keep 0
- **Measured gain:** Code +16% with a four-row window
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-controller"></a>

### Warm controller prior, costs and scope

**Engine:** `deepseek-rtx`. **Control:** `--controller-prior; --controller-costs; --controller-scope`. **Default:** warm, chat scope.

Seed built-in survival prior/costs; disable exploration and cost learning.

- **Preconditions:** Speculative phase with compatible drafter.
- **Automatic selection:** Built-in keep-64 prior and seeded costs, reset at chat scope.
- **Fallback notice:** Explicit none removes prior/costs; no universal decline line.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --controller-prior none removes prior and implicit costs; --controller-costs none removes costs alone; --controller-scope run restores run scope
- **Measured gain:** Qualified; no isolated measurement
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-writeback"></a>

### Deferred write-backs

**Engine:** `deepseek-rtx`. **Control:** `--wb-defer`. **Default:** ON.

Schedule bounded write-backs away from demand work.

- **Preconditions:** Exclusive VRAM/pinned-host tierstream cache and spare capacity.
- **Automatic selection:** On in normal replica defaults.
- **Fallback notice:** Cache pressure follows demand/collection protocol; setup errors fail.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --no-wb-defer
- **Measured gain:** Qualified; no isolated measurement
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-engram"></a>

### Host Engram record cache

**Engine:** `deepseek-rtx`. **Control:** `--engram-cache-mb 512`. **Default:** ON.

Retain identical record bytes in bounded host RAM.

- **Preconditions:** 512 MiB host cache capacity.
- **Automatic selection:** Cache hits reuse bytes; misses read backing storage.
- **Fallback notice:** Misses read normally without a decline notice.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --engram-cache-mb 0
- **Measured gain:** Avoids repeated storage stalls; no isolated throughput measurement
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-load"></a>

### Pipelined cold load

**Engine:** `deepseek-rtx`. **Control:** `DG_LOAD_SERIAL`. **Default:** ON (unset).

Use tierstream cold-load pipeline.

- **Preconditions:** Supported tierstream loader and model storage.
- **Automatic selection:** Pipeline unless serial environment override.
- **Fallback notice:** Loader errors propagate; no generic retry promise.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** DG_LOAD_SERIAL=1
- **Measured gain:** Full loading 108.7 → 61.8 s in the published workload
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-warmup"></a>

### Graph warm-up

**Engine:** `deepseek-rtx`. **Control:** `--warmup-graphs`. **Default:** auto (0,8 in serve).

Capture common graphs before readiness.

- **Preconditions:** Persistent serve worker and supported graph sizes.
- **Automatic selection:** Serve warms 0,8; non-serve does not unless explicit.
- **Fallback notice:** Unsupported graph/config requests retain validation; no general refusal notice.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --warmup-graphs none
- **Measured gain:** First 16 steps 56.05 → 33.99 ms/step
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-residency"></a>

### Saved expert residency

**Engine:** `deepseek-rtx`. **Control:** `--residency-seed; --residency-save`. **Default:** auto.

Reuse valid placement and save on worker exit.

- **Preconditions:** Serve mode, compatible seed identity and writable destination for saves.
- **Automatic selection:** MODEL.gguf.residency unless STRIX_DS41_RESIDENCY overrides it; non-serve needs explicit save/warm-up requests.
- **Fallback notice:** Missing auto seed quietly uses route order; invalid auto seed gives one notice and falls back. Explicit seed errors are strict. Save uses temporary file + rename.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --residency-seed none; --residency-save none
- **Measured gain:** Step 0 332.6 → 156.0 ms; first 16 steps −20%
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

<a id="rtx-vocab"></a>

### Automatic draft vocabulary

**Engine:** `deepseek-rtx`. **Control:** `--draft-vocab`. **Default:** auto.

Select vocabulary directory for the drafter.

- **Preconditions:** STRIX_DS41_DRAFT_VOCAB directory or ds41-draft-vocab beside drafter.
- **Automatic selection:** auto:DIR if a directory is available; explicit flag wins.
- **Fallback notice:** Absent directory uses full vocabulary with one line.
- **Exactness evidence:** Replica gate pass, complete target-greedy token vectors and canonical state/position digests equal within the standalone Q2/Q8 identity; pressure and forced-window gates cover committed-prefix state.
- **Off switch:** --draft-vocab full
- **Measured gain:** No isolated measurement
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

### Lean decode state

**Engine:** `deepseek-rtx`. **Control:** `--lean-state`. **Default:** ON.

Keep only the decode-time attention state: an allocation ledger and a split at the maximum decode position replace the full prefill-sized buffers, so the freed VRAM holds more experts.

- **Preconditions:** Standalone RTX replica or CUDA decode tool.
- **Automatic selection:** Always on unless switched off; `--baseline` turns it off.
- **Fallback notice:** None; the state layout is fixed at load.
- **Exactness evidence:** Bitwise against the full state at 2K, 8K, 32K and 128K: prefill states and stream hashes, 64-step greedy logits and tokens, DSpark commit windows, replica gates, chat tokens and persistent-worker requests.
- **Off switch:** --no-lean-state
- **Measured gain:** 8K state 673 to 25 MiB; 32K 2,486 to 59 MiB (+264 expert slots, plain decode +3.0 %); 128K loads with a 195 MiB state (without it the replica does not fit in host memory).
- **Implementation / evidence:** [Standalone defaults and gates](models/deepseek-v4.1/rtx-standalone.md#optimizations-default-on-and-off-switches)

## MiniMax

```sh
strix-minimax-rank MODEL_DIR R usb4-auto OUTPUT_DIR
```

Launch both ranks with the deployment's transport/assets and requested workload, or use
`strix-minimax-serve MODEL_DIR --rank R --transport usb4-auto`. Inspect either command with
`--print-effective-config` before model/GPU work. The plain resolver supplies 8 slots, capacity 8256 and
max rows 4096 unless explicitly overridden. `STRIX_MM_BASELINE=1` puts all 27 resolver items below at their off values;
an explicit per-item environment value wins. It does not undo older built-in attention/index/MoE routes.
The resolver applies once before cached readers. Build-time scalar gu4 addressing is also on by default;
`STRIX_MINIMAX_GU4_SCALAR=OFF` selects its previous reader at build time.

The default exchange routes are controlled separately by `STRIX_MM_XFER`, `STRIX_MM_PFX` and
`STRIX_MM_PF_INTERLEAVE`; the resolver prints their effective selection. Preserve topology and memory budget
when comparing items. Built-in `--gate`/`--gold`, `--decode-check`, `--verify-trace` and `--xb-check` compare
golden logits, sequential/verify rows and exchange state. Speculative sessions require identical committed
token IDs to plain sessions. Use transport corruption checks as well as model gates for CRC changes.

<a id="mm-moe-wide"></a>

### Wide expert decode

**Engine:** `minimax`. **Control:** `STRIX_MMX_MOE_WIDE`. **Default:** 2.

Use wider expert kernels for N2–4.

- **Preconditions:** Supported MoE widths, quantization and N2–4 rows.
- **Automatic selection:** Resolver supplies 2 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Measurement: rows == alone, wide expert kernels at N2-4, N2 -3.0..-3.5 %, N4 -4..-6 %; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MMX_MOE_WIDE=0
- **Measured gain:** rows == alone, wide expert kernels at N2-4, N2 -3.0..-3.5 %, N4 -4..-6 %
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-index-conc"></a>

### Concurrent indexer

**Engine:** `minimax`. **Control:** `STRIX_MM_INDEX_CONC`. **Default:** 1.

Overlap independent index work.

- **Preconditions:** Decode indexer with valid stream/buffer ownership.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Measurement: ids identical, 17/18 in-process block pairs faster, ~-0.2 ms/token; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_INDEX_CONC=0
- **Measured gain:** ids identical, 17/18 in-process block pairs faster, ~-0.2 ms/token
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-xfer-crc-deferred"></a>

### Deferred exchange CRC

**Engine:** `minimax`. **Control:** `STRIX_MMX_CRC`. **Default:** deferred.

Check transport CRC through deferred exchange completion.

- **Preconditions:** Both ranks on compatible deferred protocol; corruption checks remain enabled.
- **Automatic selection:** Resolver supplies deferred only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Protocol errors/corruption still fail; the off value 1 means inline CRC, not disabled CRC.
- **Exactness evidence:** Published same-identity qualification: Measurement: tokens identical in 7 processes, each deferred arm below its inline neighbours by 0.04-0.12 ms/token; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MMX_CRC=1
- **Measured gain:** tokens identical in 7 processes, each deferred arm below its inline neighbours by 0.04-0.12 ms/token
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-xb-wait"></a>

### Exchange boundary wait

**Engine:** `minimax`. **Control:** `STRIX_MM_XB_WAIT`. **Default:** 1.

Use fused boundary wait scheduling.

- **Preconditions:** Compatible decode Xfer exchange boundary.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_XB_WAIT=0
- **Measured gain:** Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-xb-inplace"></a>

### In-place exchange boundary

**Engine:** `minimax`. **Control:** `STRIX_MM_XB_INPLACE`. **Default:** 1.

Consume exchange results in place.

- **Preconditions:** Compatible decode Xfer buffers and ownership.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_XB_INPLACE=0
- **Measured gain:** Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-xb-publish"></a>

### Producer exchange publish

**Engine:** `minimax`. **Control:** `STRIX_MM_XB_PUBLISH`. **Default:** 1.

Publish the exchange message from the producer.

- **Preconditions:** In-place boundary plus v4 linear core and decode MoE overlap form.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Defaulted publish logs that the v4 linear/overlap core is required and switches off; explicit =1 fails when unavailable.
- **Exactness evidence:** Published same-identity qualification: Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token (falls back without the v4/overlap cores); repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_XB_PUBLISH=0
- **Measured gain:** Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token (falls back without the v4/overlap cores)
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-xb-postilp"></a>

### Boundary post ILP

**Engine:** `minimax`. **Control:** `STRIX_MM_XB_POSTILP`. **Default:** 2.

Use LDS-staged mode-2 post-exchange partials.

- **Preconditions:** Supported decode boundary shapes/LDS capacity.
- **Automatic selection:** Resolver supplies 2 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_XB_POSTILP=0
- **Measured gain:** Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-xb-warm"></a>

### Boundary warm reads

**Engine:** `minimax`. **Control:** `STRIX_MM_XB_WARM`. **Default:** 8.

Warm 8 MiB of boundary data.

- **Preconditions:** Decode exchange boundary and supported memory region.
- **Automatic selection:** Resolver supplies 8 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_XB_WARM=0
- **Measured gain:** Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-dspark-attn"></a>

### DSpark attention

**Engine:** `minimax`. **Control:** `STRIX_MM_DSPARK_ATTN`. **Default:** 2.

Use mode-2 drafter attention.

- **Preconditions:** Attached compatible DSpark drafter and attention geometry.
- **Automatic selection:** Resolver supplies 2 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Drafter measurement: bitwise, drafter attention B/A 0.964-0.975 / 0.918-0.924 / 0.815-0.819 at ctx 300/1000/4095; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_DSPARK_ATTN=0
- **Measured gain:** Drafter measurement: bitwise, drafter attention B/A 0.964-0.975 / 0.918-0.924 / 0.815-0.819 at ctx 300/1000/4095
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-draft-lin"></a>

### Grouped draft linears

**Engine:** `minimax`. **Control:** `STRIX_MM_DRAFT_LIN`. **Default:** opt.

Group q/k/v and gate/up draft projections.

- **Preconditions:** Attached EXL3 draft linears with supported shapes.
- **Automatic selection:** Resolver supplies opt only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Optimized draft linears (measurement evidence); repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_DRAFT_LIN=lib
- **Measured gain:** Optimized draft linears (measurement evidence)
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-linear-v11"></a>

### Draft-block linear core v11

**Engine:** `minimax`. **Control:** `STRIX_MMX_LINEAR_V11`. **Default:** 1.

A linear core scheduled for the 7-row DSpark draft block (q+k+v, o and gate+up groups).

- **Preconditions:** DSpark draft block at 7 rows through the scheduled linear path; other callers keep their forms.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win.
- **Fallback notice:** Shapes outside the schedule keep the previous core.
- **Exactness evidence:** Same output bits as v10 (bitwise over every scheduled form and layer); speculative ids == greedy 36/36.
- **Off switch:** STRIX_MMX_LINEAR_V11=0
- **Measured gain:** DSpark draft block −11.9 %
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-markov-cand"></a>

### DSpark Markov candidate form

**Engine:** `minimax`. **Control:** `STRIX_MM_MARKOV; STRIX_MM_MARKOV_C; STRIX_MM_MARKOV_H; STRIX_MM_MARKOV_TOPC`. **Default:** cand, C 4096, H 8192, top-C v2.

The drafter's Markov chain runs over a candidate set (exact top-C per row) with one device-to-device payload exchange
per draft instead of a full-vocabulary scan and host exchange per step.

- **Preconditions:** Attached DSpark drafter on the pair.
- **Automatic selection:** Resolver supplies the four values only when unset; explicit values win.
- **Fallback notice:** A draft whose candidate set cannot cover the chain falls back to the full chain for that draft.
- **Exactness evidence:** Target verify is unchanged, so committed tokens equal plain greedy (ids == greedy on every gated session); top-C v2 payloads == v1 == CPU, including ties, ±0 and NaN.
- **Off switch:** STRIX_MM_MARKOV=full (C/H/TOPC: 1024/2048/v1)
- **Measured gain:** ~1.7 ms saved per draft; DSpark median tok/s +1.1 to +1.6 % across json/code/English/Spanish prompts, acceptance unchanged
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-pf-core-v4"></a>

### v4 MoE prefill core

**Engine:** `minimax`. **Control:** `STRIX_MMX_PREFILL_CORE`. **Default:** v4.

Use v4 register/layout scheduling.

- **Preconditions:** Supported prefill expert shapes and quantized weights.
- **Automatic selection:** Resolver supplies v4 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Prefill core measurement: bitwise everywhere, whole call 0.34-0.58x lib; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MMX_PREFILL_CORE=lib
- **Measured gain:** Prefill core measurement: bitwise everywhere, whole call 0.34-0.58x lib
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-pf-msa-batch"></a>

### MSA prefill batching

**Engine:** `minimax`. **Control:** `STRIX_MM_PF_MSA_BATCH`. **Default:** 1.

Batch compatible MSA work across prompt rows.

- **Preconditions:** Prefill rows with supported MSA geometry.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Prefill measurement: exact, combined prefill TTFT 1K -17 %, 8K -20 %; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_PF_MSA_BATCH=0
- **Measured gain:** Prefill measurement: exact, combined prefill TTFT 1K -17 %, 8K -20 %
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-pf-attend4"></a>

### Four-way prefill attend

**Engine:** `minimax`. **Control:** `STRIX_MM_PF_ATTEND4`. **Default:** 1.

Use the admitted four-way attend schedule.

- **Preconditions:** Supported prefill attention groups and row layout.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Prefill measurement: exact, combined prefill TTFT 1K -17 %, 8K -20 %; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_PF_ATTEND4=0
- **Measured gain:** Prefill measurement: exact, combined prefill TTFT 1K -17 %, 8K -20 %
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-pf-router-batch"></a>

### Batched prefill router

**Engine:** `minimax`. **Control:** `STRIX_MM_PF_ROUTER_BATCH`. **Default:** 1.

Compute compatible router rows together.

- **Preconditions:** Supported router row blocks.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Prefill measurement: exact, combined prefill TTFT 1K -17 %, 8K -20 %; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_PF_ROUTER_BATCH=0
- **Measured gain:** Prefill measurement: exact, combined prefill TTFT 1K -17 %, 8K -20 %
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-pf-mtfit"></a>

### MoE tile fit

**Engine:** `minimax`. **Control:** `STRIX_MMX_PREFILL_MTFIT`. **Default:** 1.

Choose fitting prefill MoE tiles.

- **Preconditions:** Supported prefill core and expert row geometry.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Prefill measurement: exact, combined prefill TTFT 1K -17 %, 8K -20 %; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MMX_PREFILL_MTFIT=0
- **Measured gain:** Prefill measurement: exact, combined prefill TTFT 1K -17 %, 8K -20 %
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-pf-msa-rework"></a>

### MSA prefill mode 3

**Engine:** `minimax`. **Control:** `STRIX_MM_PF_MSA_REWORK`. **Default:** 3.

Rework attention scheduling without changing the reference result.

- **Preconditions:** Supported MSA prefill shapes and row buffers.
- **Automatic selection:** Resolver supplies 3 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Prefill attention measurement: exact, 8K B/A 0.963 / 0.957 / 0.972; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_PF_MSA_REWORK=0
- **Measured gain:** Prefill attention measurement: exact, 8K B/A 0.963 / 0.957 / 0.972
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-pf-moe-pipe"></a>

### MoE prefill pipeline

**Engine:** `minimax`. **Control:** `STRIX_MMX_PREFILL_PIPE`. **Default:** 1.

Pipeline compatible expert prefill work.

- **Preconditions:** Supported prefill expert core and ordered workspace ownership.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Prefill measurement: exact, 8K B/A 0.973 / 0.935 / 0.924; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MMX_PREFILL_PIPE=0
- **Measured gain:** Prefill measurement: exact, 8K B/A 0.973 / 0.935 / 0.924
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-pf-kv-tile"></a>

### Automatic KV tile

**Engine:** `minimax`. **Control:** `STRIX_MM_PF_KV_TILE`. **Default:** auto.

Choose the supported KV tile implementation.

- **Preconditions:** Linked trellis tile override hook and supported KV shape.
- **Automatic selection:** Resolver supplies auto only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Absent trellis lab tile hook prints `STRIX_MM_PF_KV_TILE: trellis lab tile override absent; switch is a no-op`; keep the normal KV path.
- **Exactness evidence:** Published same-identity qualification: Prefill measurement: exact, 8K B/A 0.960 / 0.949 / 1.017; needs the Trellis tile hook, else a logged no-op; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_PF_KV_TILE=off
- **Measured gain:** Prefill measurement: exact, 8K B/A 0.960 / 0.949 / 1.017; needs the Trellis tile hook, else a logged no-op
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-pf-tx-uncached"></a>

### Uncached prefill TX

**Engine:** `minimax`. **Control:** `STRIX_MM_PF_TX_UNCACHED`. **Default:** 1.

Avoid cache pollution in prefill exchange sends.

- **Preconditions:** Prefill module route and uncached TX allocation support.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unavailable uncached TX allocation prints `mmpfx: uncached TX slots unavailable ...; TX stays host-coherent`; retain the coherent buffer.
- **Exactness evidence:** Published same-identity qualification: Prefill exchange measurement: exact, exchange-send 0.23 -> 0.002 s GPU, ~1.3 % (prefill module route); repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_MM_PF_TX_UNCACHED=0
- **Measured gain:** Prefill exchange measurement: exact, exchange-send 0.23 -> 0.002 s GPU, ~1.3 % (prefill module route)
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-dist-crc-parallel"></a>

### Parallel host CRC

**Engine:** `minimax`. **Control:** `STRIX_DIST_CRC_PARALLEL`. **Default:** 1.

Compute host transport CRC on workers.

- **Preconditions:** Host exchange route; preserve compatible settings across ranks.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Host transport measurement: ABA x3 TTFT 1K -12.8 %, decode -1 %, bits unchanged; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_DIST_CRC_PARALLEL=0
- **Measured gain:** Host transport measurement: ABA x3 TTFT 1K -12.8 %, decode -1 %, bits unchanged
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-dist-host-crc-overlap"></a>

### Host CRC overlap

**Engine:** `minimax`. **Control:** `STRIX_DIST_HOST_CRC_OVERLAP`. **Default:** 1.

Overlap host CRC and transport work.

- **Preconditions:** Host exchange route; integrity checks enabled.
- **Automatic selection:** Resolver supplies 1 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Host transport measurement: ABA x3 TTFT 1K -12.8 %, decode -1 %, bits unchanged; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_DIST_HOST_CRC_OVERLAP=0
- **Measured gain:** Host transport measurement: ABA x3 TTFT 1K -12.8 %, decode -1 %, bits unchanged
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-dist-host-chunk"></a>

### Host transfer chunks

**Engine:** `minimax`. **Control:** `STRIX_DIST_HOST_CHUNK_BYTES`. **Default:** 1048576.

Send host transport in 1 MiB pieces.

- **Preconditions:** Host exchange route with chunk support.
- **Automatic selection:** Resolver supplies 1048576 only when unset; explicit values win. Shape/core dispatch determines actual use.
- **Fallback notice:** Unsupported operator geometry keeps its supported dispatch without a universal decline notice; invalid explicit settings fail validation.
- **Exactness evidence:** Published same-identity qualification: Host transport measurement: ABA x3 TTFT 1K -12.8 %, decode -1 %, bits unchanged; repeat golden-logit/token/state and relevant exchange gates for the chosen workload.
- **Off switch:** STRIX_DIST_HOST_CHUNK_BYTES=0
- **Measured gain:** Host transport measurement: ABA x3 TTFT 1K -12.8 %, decode -1 %, bits unchanged
- **Implementation / evidence:** [27-item resolver](../src/minimax/engine/mm_defaults.cpp), [rank gates](../apps/minimax_rank_main.cpp)

<a id="mm-routes"></a>

### Decode and prefill exchange routes

**Engine:** `minimax`. **Control:** `STRIX_MM_XFER; STRIX_MM_PFX; STRIX_MM_PF_INTERLEAVE`. **Default:** XFER on, PFX loaded, interleave on.

Use the host-path decode exchange module, load the prefill exchange module, and interleave prefill chunks over the selected exchange.

- **Preconditions:** Two compatible ranks on a host exchange route; module loads are skipped with GPU-direct transport.
- **Automatic selection:** Unset XFER loads/uses decode exchange. Unset PFX loads the module; explicit PFX=on installs it. Interleave uses the installed module or communicator; =comm forces the communicator.
- **Fallback notice:** GPU-direct keeps its communicator route; startup prints the module/route selection. Invalid route values fail validation.
- **Exactness evidence:** Same token IDs, K/V/index state and terminal rows in prefill qualification; retain protocol integrity gates.
- **Off switch:** STRIX_MM_XFER=off; STRIX_MM_PFX=off; STRIX_MM_PF_INTERLEAVE=off (one at a time)
- **Measured gain:** Interleaving: 8K TTFT 40.0 → 35.4 s; 1K inert

<a id="mm-gu4"></a>

### Scalar gu4 expert addressing

**Engine:** `minimax`. **Control:** `STRIX_MINIMAX_GU4_SCALAR`. **Default:** ON (build time).

Use the scalar gu4 MoE expert-address reader.

- **Preconditions:** MiniMax build and supported expert layout.
- **Automatic selection:** CMake defaults the scalar reader on; the effective config reports its build state.
- **Fallback notice:** Build-time selection has no runtime fallback notice; use the previous reader by rebuilding.
- **Exactness evidence:** Same-layout expert reader qualification; verify model tokens/digests after changing the build.
- **Off switch:** STRIX_MINIMAX_GU4_SCALAR=OFF at CMake configure time
- **Measured gain:** No isolated measurement

## GLM

```sh
strix-glm5-rank MODEL_DIR --world 2 --rank R --transport usb4-auto --prompt PROMPT.u32 --decode 64
```

Launch both ranks with the same model, prompt and compatible transport. `--print-effective-config` reports
the resolution before GPU/peer work. Defaults: `--identity r1b`, `--mtp 0`, `--prefill auto`,
`--decode-xch xfer`. R1b uses R1 prompt arithmetic and R1b generation; `--identity r1` restores R1 throughout.
`STRIX_GLM_IDENTITY` overrides the CLI identity; CLI prefill overrides `STRIX_GLM_PREFILL`;
`STRIX_GLM_DECODE_XCH` overrides CLI exchange. Explicit values win over baseline defaults.

R1b changes FP32 summation order and is **not bit-identical to R1**. Own-identity runs are stable, but the
32-prompt/8192-position comparison found 99.19% top-1 agreement and mean KL about 1.8e-3 nats; free runs
diverged on 28/32 prompts within 19–227 tokens. Measured R1b 41.93–41.96 versus R1 47.33–47.48 ms/token
is an identity comparison, not an exact R1 optimization. The faster R1b kernels separately measured R1/R8
−7.8/−5.4% with the same tokens within R1b. See [identity implementation](../src/glm/engine/glm_r1b.hip).

`--baseline` / `STRIX_GLM_BASELINE=1` restores R1, reference prefill, MTP loaded, comm decode exchange and
the listed off values unless explicitly overridden. `STRIX_GLM_MOE=fast` remains on: prepared reference
experts do not fit the real TP2 rank's GTT. Its per-item `ref` switch is not a runnable full-model baseline.
Use G3/G4, `--idcheck`, prefill logits/digests and the speculative oracle within the selected identity;
R1b-vs-R1 agreement is reported separately from those exactness gates.

<a id="glm-prefill"></a>

### Exact chunked prefill

**Engine:** `glm`. **Control:** `--prefill auto`. **Default:** auto.

Use chunked gemv prefill with the R1 arithmetic identity.

- **Preconditions:** Ordinary prefill, MTP/spec off, not r1b-all, G3-only/smoke/memory-plan/identity-check/reference-gate modes, with capacity for chunked path/workspace.
- **Automatic selection:** auto chooses gemv for normal generation; excluded diagnostic/spec modes choose ref. R1b default still prefills in R1.
- **Fallback notice:** At world 1, failed auto admission prints `prefill auto: exact chunked path not admitted (...) -> ref (same bits, slower)`. Explicit requests and world 2 failures propagate so peers cannot choose incompatible exchange patterns.
- **Exactness evidence:** Real-model logits are bitwise equal on both ranks to previous exact R1 prefill; 1K/8K/32K gates. This does not assert R1b decode equality to R1.
- **Off switch:** --prefill ref; STRIX_GLM_PREFILL=ref if no CLI prefill override
- **Measured gain:** 1K/8K/32K TTFT 4.268/34.70/144.75 s = 240/236/226 tok/s
- **Implementation / evidence:** [Resolver](../apps/glm_rank_defaults.hpp), [admission](../apps/glm_rank_main.cpp)

<a id="glm-identity"></a>

### R1b decode identity

**Engine:** `glm`. **Control:** `--identity r1b`. **Default:** ON.

Use R1 prompt arithmetic and the R1b generation identity.

- **Preconditions:** Admitted GLM model, R1b kernels and identity-specific reference.
- **Automatic selection:** Default identity is r1b unless CLI/environment chooses another; baseline uses r1.
- **Fallback notice:** No automatic substitution between identities; explicit unsupported settings retain validation.
- **Exactness evidence:** Stable within R1b; not bit-identical to R1. Teacher-forced top-1 agreement 8126/8192; mean KL ~1.8e-3 nats; free runs diverge on 28/32 prompts.
- **Off switch:** --identity r1 (unset STRIX_GLM_IDENTITY, which otherwise wins)
- **Measured gain:** 41.93–41.96 vs R1 47.33–47.48 ms/token (−11.5%)

<a id="glm-xfer"></a>

### Direct decode exchange

**Engine:** `glm`. **Control:** `--decode-xch xfer`. **Default:** ON.

Use the production Xfer decode exchange.

- **Preconditions:** Two-rank transport, compatible exchange buffers and topology.
- **Automatic selection:** Default xfer; baseline comm; STRIX_GLM_DECODE_XCH overrides CLI.
- **Fallback notice:** Invalid/incompatible transport fails; no independent per-rank protocol fallback.
- **Exactness evidence:** Same-identity rank token/state/logit gates and exchange integrity checks.
- **Off switch:** --decode-xch comm (unset overriding environment)
- **Measured gain:** Boundary bundle reduces exchange overhead; no isolated measurement

<a id="glm-hp-crc-parallel"></a>

### Parallel host CRC

**Engine:** `glm`. **Control:** `STRIX_DIST_CRC_PARALLEL`. **Default:** 1.

Compute host transport CRC using parallel workers while retaining integrity checks.

- **Preconditions:** Host-path exchange; compatible rank protocol.
- **Automatic selection:** Unset resolves to 1; the app exports it before runtime readers. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Measurement: hp host path, exact TTFT 1K -8.5 %, 8K -3.1 %, logits fnv matches reference 16/16; compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_DIST_CRC_PARALLEL=0
- **Measured gain:** hp host path, exact TTFT 1K -8.5 %, 8K -3.1 %, logits fnv matches reference 16/16
- **Implementation / evidence:** [16-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-hp-crc-overlap"></a>

### Host CRC overlap

**Engine:** `glm`. **Control:** `STRIX_DIST_HOST_CRC_OVERLAP`. **Default:** 1.

Overlap host CRC work with exchange transfers.

- **Preconditions:** Host-path exchange with integrity checks enabled.
- **Automatic selection:** Unset resolves to 1; the app exports it before runtime readers. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Measurement: hp host path, exact TTFT 1K -8.5 %, 8K -3.1 %, logits fnv matches reference 16/16; compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_DIST_HOST_CRC_OVERLAP=0
- **Measured gain:** hp host path, exact TTFT 1K -8.5 %, 8K -3.1 %, logits fnv matches reference 16/16
- **Implementation / evidence:** [16-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-hp-chunk-bytes"></a>

### Host transfer chunks

**Engine:** `glm`. **Control:** `STRIX_DIST_HOST_CHUNK_BYTES`. **Default:** 1048576.

Split host exchange payloads into 1 MiB chunks.

- **Preconditions:** Host-path exchange and chunk support.
- **Automatic selection:** Unset resolves to 1048576; the app exports it before runtime readers. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Measurement: hp host path, exact TTFT 1K -8.5 %, 8K -3.1 %, logits fnv matches reference 16/16; compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_DIST_HOST_CHUNK_BYTES=0
- **Measured gain:** hp host path, exact TTFT 1K -8.5 %, 8K -3.1 %, logits fnv matches reference 16/16
- **Implementation / evidence:** [16-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-hsa-no-fragment-allocator"></a>

### Allocator granules

**Engine:** `glm`. **Control:** `HSA_DISABLE_FRAGMENT_ALLOCATOR`. **Default:** 1.

Disable the HSA fragment allocator so production allocation granules fit GTT.

- **Preconditions:** Runtime allocator and GTT capacity for the production model.
- **Automatic selection:** Unset resolves to 1; the app exports it before runtime readers. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Measurement: production decode and prefill configuration; allocator granules fit GTT; compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** HSA_DISABLE_FRAGMENT_ALLOCATOR=0
- **Measured gain:** production decode and prefill configuration; allocator granules fit GTT
- **Implementation / evidence:** [16-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-r1b-dense-auto"></a>

### R1b dense scheduling

**Engine:** `glm`. **Control:** `STRIX_GLM_R1B_DENSE`. **Default:** 7.

Select dense variant 7 within the R1b generation identity.

- **Preconditions:** R1b generation identity and supported dense sites.
- **Automatic selection:** Unset resolves to 7; the engine default provides it; the resolver reports it. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Measurement: R1b-fast kernels -7.8 % R1 / -5.4 % R8, same tokens; compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_GLM_R1B_DENSE=2
- **Measured gain:** R1b-fast kernels -7.8 % R1 / -5.4 % R8, same tokens
- **Implementation / evidence:** [16-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-r1b-hc-v4"></a>

### R1b HC v4

**Engine:** `glm`. **Control:** `STRIX_GLM_R1B_HC`. **Default:** 4.

Use HC variant 4 within the R1b generation identity.

- **Preconditions:** R1b generation identity and compatible HC sites.
- **Automatic selection:** Unset resolves to 4; the engine default provides it; the resolver reports it. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Measurement: with r1b-dense-auto, R1b-fast kernels -7.8 % R1 / -5.4 % R8, same tokens; compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_GLM_R1B_HC=1
- **Measured gain:** with r1b-dense-auto, R1b-fast kernels -7.8 % R1 / -5.4 % R8, same tokens
- **Implementation / evidence:** [16-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-hc-pre-v7"></a>

### HC pre v7

**Engine:** `glm`. **Control:** `STRIX_GLM_HC_PRE_V`. **Default:** 7.

Use the qualified v7 pre-HC reduction/normalization schedule.

- **Preconditions:** Supported HC pre shapes and norm/weight geometry.
- **Automatic selection:** Unset resolves to 7; the engine default provides it; the resolver reports it. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Measurement: hc_pre 77.4-77.9 vs 102.1-102.8 us, gates 0-7 bitwise; tokens match reference; compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_GLM_HC_PRE_V=4
- **Measured gain:** hc_pre 77.4-77.9 vs 102.1-102.8 us, gates 0-7 bitwise; tokens match reference
- **Implementation / evidence:** [16-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-post-exchange-fast"></a>

### Fast post-exchange consumers

**Engine:** `glm`. **Control:** `STRIX_GLM_POST`. **Default:** fast.

Consume post-exchange values through fused fast kernels rather than separate reference consumers.

- **Preconditions:** Compatible two-rank post-exchange buffers.
- **Automatic selection:** Unset resolves to fast; the engine default provides it; the resolver reports it. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Measurement: fast post-exchange consumers match STRIX_GLM_POST=ref and reference (real model, 6 layers); compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_GLM_POST=ref
- **Measured gain:** fast post-exchange consumers match STRIX_GLM_POST=ref and reference (real model, 6 layers)
- **Implementation / evidence:** [16-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-attn-fast"></a>

### Fast attention

**Engine:** `glm`. **Control:** `STRIX_GLM_ATTN`. **Default:** fast.

Install FastAttn at the attention seam, preserving the selected identity and state contract.

- **Preconditions:** Supported attention geometry; reference form supports plain execution only.
- **Automatic selection:** Unset resolves to fast; the engine default provides it; the resolver reports it. Explicit values win.
- **Fallback notice:** Explicit ref is for plain execution only; speculative verify on the reference seam fails validation.
- **Exactness evidence:** Measurement: FastAttn matches STRIX_GLM_ATTN=ref (G3), Tier A 12-40x; compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_GLM_ATTN=ref
- **Measured gain:** FastAttn matches STRIX_GLM_ATTN=ref (G3), Tier A 12-40x
- **Implementation / evidence:** [16-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-attn-tierb-fused"></a>

### Fused attention Tier B

**Engine:** `glm`. **Control:** `STRIX_GLM_TIERB`. **Default:** fused.

Use fused Tier-B attention projections from raw weights instead of prepared library linears.

- **Preconditions:** TP2 attention, supported projection shapes and raw weights.
- **Automatic selection:** Unset resolves to fused; the engine default provides it; the resolver reports it. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Measurement: fused Tier B -2.46 ms/token vs strix_gemv, tokens match reference; compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_GLM_TIERB=lib
- **Measured gain:** fused Tier B -2.46 ms/token vs strix_gemv, tokens match reference
- **Implementation / evidence:** [16-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-xfer-boundary-fused"></a>

### Fused Xfer boundary

**Engine:** `glm`. **Control:** `STRIX_GLM_BOUNDARY`. **Default:** 1.

Fuse exchange reduction and its dependent boundary work.

- **Preconditions:** Compatible Xfer decode boundary.
- **Automatic selection:** Unset resolves to 1; the engine default provides it; the resolver reports it. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Measurement: Xfer reduce +1.41 -> fused boundary ~+1.10 ms/token over local, bitwise; compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_GLM_BOUNDARY=0
- **Measured gain:** Xfer reduce +1.41 -> fused boundary ~+1.10 ms/token over local, bitwise
- **Implementation / evidence:** [16-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-moe-fast"></a>

### Fast raw-expert MoE

**Engine:** `glm`. **Control:** `STRIX_GLM_MOE`. **Default:** fast.

Use raw-only expert storage and the bitwise FastMoe implementation so the TP2 model fits GTT.

- **Preconditions:** Real TP2 raw expert slices; prepared reference experts exceed GTT.
- **Automatic selection:** Unset resolves to fast; the engine default provides it; the resolver reports it. Explicit values win.
- **Fallback notice:** Baseline retains fast MoE; explicit ref may fail full-model TP2 memory admission. No silent numerical-identity substitution.
- **Exactness evidence:** Measurement: FastMoe matches DefaultMoe bitwise (real TP2 slices); raw-only experts (fits GTT); compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_GLM_MOE=ref (excluded from master baseline for capacity)
- **Measured gain:** FastMoe matches DefaultMoe bitwise (real TP2 slices); raw-only experts (fits GTT)
- **Implementation / evidence:** [16-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-pf-exact-moe-v3"></a>

### Exact prefill MoE v3

**Engine:** `glm`. **Control:** `STRIX_GLM_PF_EXACTMOE_V`. **Default:** 3.

Group expert W2 work in exact chunked prefill.

- **Preconditions:** Exact chunked prefill with supported grouped W2 expert work.
- **Automatic selection:** Unset resolves to 3; the engine default provides it; the resolver reports it. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Measurement: exact MoE v3 (W2 grouped experts) 1K/8K/32K 240/236/226 tok/s, logits bitwise; compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_GLM_PF_EXACTMOE_V=1
- **Measured gain:** exact MoE v3 (W2 grouped experts) 1K/8K/32K 240/236/226 tok/s, logits bitwise
- **Implementation / evidence:** [16-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-pf-obar-v5"></a>

### Prefill MLA o-bar v5

**Engine:** `glm`. **Control:** `STRIX_GLM_PF_OBAR`. **Default:** 5.

Use the v5 exact MLA output projection schedule.

- **Preconditions:** Exact chunked prefill and supported MLA projection geometry.
- **Automatic selection:** Unset resolves to 5; the engine default provides it; the resolver reports it. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Measurement: MLA o-bar v5 exact, 2.9x; compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_GLM_PF_OBAR=2
- **Measured gain:** MLA o-bar v5 exact, 2.9x
- **Implementation / evidence:** [16-item resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-pf-scores-v3"></a>

### Prefill MLA scores v4

**Engine:** `glm`. **Control:** `STRIX_GLM_PF_SCORES`. **Default:** 4.

Use the v4 exact MLA score kernel (same bits as v3 and v2).

- **Preconditions:** Exact chunked prefill and supported score geometry.
- **Automatic selection:** Unset resolves to 4; the resolver reports it. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Bitwise equal to v3 on the kernel fixtures; complete same-identity tokens plus logits/state digests on the pair.
- **Off switch:** STRIX_GLM_PF_SCORES=3 (v3) or =2 (v2)
- **Measured gain:** v3 -22 % vs v2; v4 a further -8..-17 % vs v3
- **Implementation / evidence:** [resolver](../apps/glm_rank_defaults.hpp)

### Prefill pipeline and async exchange start

**Engine:** `glm`. **Control:** `STRIX_GLM_PF_PIPELINE`, `STRIX_DIST_ASYNC_START`. **Default:** 2 / 1.

Overlap two whole layers of the chunked prefill, and start the host exchange asynchronously so it hides behind compute
(only the pipeline uses the async start; other engines keep the ordinary start unless they set the variable).

- **Preconditions:** Exact chunked prefill on two ranks; full-host chunked exchange path.
- **Automatic selection:** The rank app sets both when unset; explicit values win.
- **Fallback notice:** Other exchange paths use the ordinary start.
- **Exactness evidence:** Prefill gate bitwise to the reference row-by-row prefill (token, logits, all state bytes); TTFT logits equal on both ranks at 1K/8K/32K.
- **Off switch:** STRIX_GLM_PF_PIPELINE=1, STRIX_DIST_ASYNC_START=0, or --baseline
- **Measured gain:** 8K about -4 %; the exchange is fully hidden (GPU idle per 2K prefill 0.97 to 0.09 s)
- **Implementation / evidence:** [resolver](../apps/glm_rank_defaults.hpp)

### Prefill chunk rows 4096

**Engine:** `glm`. **Control:** `--max-rows`. **Default:** 4096 (baseline 1024).

Process prefill in 4096-row chunks.

- **Preconditions:** Exact chunked prefill.
- **Automatic selection:** Default 4096; --max-rows wins; --baseline uses 1024.
- **Fallback notice:** None.
- **Exactness evidence:** Logits equal to the reference prefill on both ranks at 1K/8K/32K.
- **Off switch:** --max-rows 2048 / 1024, or --baseline
- **Measured gain:** with the pipeline on: 8K 35.1 s at 1024, 32.1 s at 2048, 30.1-30.8 s at 4096; 32K 136.7 to 128.4 s vs 2048
- **Implementation / evidence:** [resolver](../apps/glm_rank_defaults.hpp)

<a id="glm-pf-indexer-v2"></a>

### Prefill indexer v2

**Engine:** `glm`. **Control:** `STRIX_GLM_PF_IDX`. **Default:** 2.

Use the v2 exact prefill indexer kernel.

- **Preconditions:** Exact chunked prefill with supported index rows.
- **Automatic selection:** Unset resolves to 2; the engine default provides it; the resolver reports it. Explicit values win.
- **Fallback notice:** Outside supported geometry, keep the supported exact dispatch; invalid explicit combinations fail. No general per-item decline line.
- **Exactness evidence:** Measurement: indexer v2 -41 %; compare complete same-identity tokens plus logits/state digests. R1b items are not a claim of R1 equality.
- **Off switch:** STRIX_GLM_PF_IDX=1
- **Measured gain:** indexer v2 -41 %
- **Implementation / evidence:** [16-item resolver](../apps/glm_rank_defaults.hpp)
