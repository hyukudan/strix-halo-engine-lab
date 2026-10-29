# HC output WMMA native release gates

Run on the native HIP target after its compiler/ISA checks. CPU tests do not
qualify numerical restoration. Keep all pins, prompts and outputs outside Git.
Use a clean environment and fresh output directories. `EVIDENCE` and `PINS` below
are private directories supplied by the operator; executable paths belong only
in the private command files. The public-main executable must be built from
05fedd80. Candidate binaries must contain this change. No timing claims arise
from diagnostic runs.

Each command file is a JSON argv array. Keep model, prompt IDs, context capacity,
profile pair, T, tails, output policy, state layout and workspace budgets identical
between builds. Only argv[0] changes. Command files contain no prefill controls or
artifact destinations. For 4k and 32k use exactly 4096 and 32768 prompt tokens.
The logits command pins `--teacher-forced-score --generate 1` with all-row tiled
output; it runs at 4k only, because full-vocabulary teacher logits of every 32k row
exceed the engine's 16 GiB teacher-spool bound. At 32k the gate omits
`--logits-command` and compares the committed state, which covers every KV/index
row, the fixed components and the token history. The gate hashes each case's 4k
logits (about 4 GB) and then deletes the file unless `--keep-logits` is given; a
reference case's `signature.json` holds that digest, so at most one logits file is on
disk at a time. The state command pins eager greedy generation, terminal-row output,
full-row tiled prefill and positive generation, leaving capacity for generation.
State-session runs need a >=1024-row cache unit and sufficient
`--session-capture-host-bytes`; do not enable session suffix or fixed MTP modes.
The gate refuses artifact destinations in base commands.

Private pin layout:

- `PINS/{public-main,candidate}/{4k,32k}/{fast,gsq,baseline}/{logits,state}.json`.
  Fast/GSQ select the same explicit fast profile and qualified component set;
  baseline uses the same explicit baseline command on both builds.
- `PINS/{public-main,candidate}/{4k,32k}/{standard,gsq,baseline}/mtp.json`.
  These use `strix-hip-mtp-model-qualification`, ordinary admitted profiles,
  complete model/sidecar/golden pins, replay and 2/4/8 window owners/budgets.
- `PINS/{public-main,candidate}/{standard,gsq,baseline}/server.json`.
  These use the decoder with the same ordinary MTP assets and serving geometry,
  no HTTP flags; standard pins use dense-A8 with T8192 for nested admission.
- `PINS/{4k,32k,short2048}.ids` contain the shared token IDs for HTTP runs.

Run the complete command set below with `PINS=../private-pins EVIDENCE=../private-evidence BUILD_DIR=build bash scripts/qualification/qwen-hc-output-wmma.sh`. The private pin files must already exist.

## CLI full logits and committed state, public main / OFF / ON

`session` is the working state path shared with unmodified public main. It hashes
all committed KV/index rows and all five complete fixed components, verifies the
envelope root and each section, and compares token history, pending token and
frontier. Unused KV/index capacity is outside this scope. Identity metadata may
differ. ON references below establish repeatability, not equivalence to OFF.

```sh
set -eu
for n in 4k 32k; do
  for scope in fast gsq baseline; do
    ref="$EVIDENCE/$n-$scope-public"
    python3 tests/qwen_prefill_public_controls_gate.py --arm public-main --capture-only --state-format session \
      --logits-command "$PINS/public-main/$n/$scope/logits.json" \
      --state-command "$PINS/public-main/$n/$scope/state.json" --output "$ref"
    python3 tests/qwen_prefill_public_controls_gate.py --arm off --state-format session \
      --logits-command "$PINS/candidate/$n/$scope/logits.json" \
      --state-command "$PINS/candidate/$n/$scope/state.json" \
      --reference-logits "$ref/public-main-legacy1/signature.json" \
      --reference-state "$ref/public-main-legacy1/state/session.bin" --output "$EVIDENCE/$n-$scope-off"
    witness=""
    if [ "$scope" = fast ]; then witness=--require-hc-witness; fi
    on="$EVIDENCE/$n-$scope-on"
    python3 tests/qwen_prefill_public_controls_gate.py --arm on --capture-only --state-format session $witness \
      --logits-command "$PINS/candidate/$n/$scope/logits.json" \
      --state-command "$PINS/candidate/$n/$scope/state.json" --output "$on"
    python3 tests/qwen_prefill_public_controls_gate.py --arm on --state-format session $witness \
      --logits-command "$PINS/candidate/$n/$scope/logits.json" \
      --state-command "$PINS/candidate/$n/$scope/state.json" \
      --reference-logits "$on/on-legacy1/signature.json" \
      --reference-state "$on/on-legacy1/state/session.bin" --output "$on-repeat"
  done
done
```

OFF runs every CLI/environment/config individual/master restriction with legacy
output selectors 1 and 2. A missing reference or execution witness is not a pass.

## Split-profile complete-allocation export

Use the same candidate state pins without session flags. Unlike public main,
the candidate admits `--validation-final-state` for split profiles. This hashes
all seven full allocations, including unused capacity. Compare each arm with its
own repeated complete-allocation reference; cross-build committed-state equality
is established separately above.

```sh
for n in 4k 32k; do
  for scope in fast gsq baseline; do
    for arm in off on; do
      ref="$EVIDENCE/$n-$scope-$arm-full"
      python3 tests/qwen_prefill_public_controls_gate.py --arm "$arm" --capture-only \
        --logits-command "$PINS/candidate/$n/$scope/logits.json" \
        --state-command "$PINS/candidate/$n/$scope/state.json" --output "$ref"
      case_name=on-legacy1
      if [ "$arm" = off ]; then case_name=cli-individual-legacy1; fi
      python3 tests/qwen_prefill_public_controls_gate.py --arm "$arm" \
        --logits-command "$PINS/candidate/$n/$scope/logits.json" \
        --state-command "$PINS/candidate/$n/$scope/state.json" \
        --reference-logits "$ref/$case_name/signature.json" \
        --reference-state "$ref/$case_name/state/manifest.json" --output "$ref-repeat"
    done
  done
done
```

## MTP full logits, IDs and seven state components

Fast HC ON is explicitly excluded from MTP and nested scheduling until qualified;
ordinary MTP profiles do not execute that arithmetic. Both ON and OFF below must
match public main. The MTP gate compares every case's full physical logits and
all seven committed/physical state components, excluding compiled identity fields.

```sh
for n in 4k 32k; do
  for scope in standard gsq baseline; do
    ref="$EVIDENCE/$n-$scope-mtp-public"
    python3 tests/qwen_prefill_mtp_gate.py --arm public-main \
      --command "$PINS/public-main/$n/$scope/mtp.json" --output "$ref"
    for arm in on off; do
      python3 tests/qwen_prefill_mtp_gate.py --arm "$arm" \
        --command "$PINS/candidate/$n/$scope/mtp.json" \
        --reference "$ref/native/result.json" --output "$EVIDENCE/$n-$scope-mtp-$arm"
    done
  done
done
```

## HTTP and nested scheduling

HTTP gates compare complete emitted IDs and committed target state. Their
existing scheduling contract excludes predictor-only differences; the MTP gate
above separately covers all seven target components and physical state. HTTP
slot comparisons retain their existing cross-route prompt-digest exclusion;
the service/nested gate also compares prompt digests.

```sh
for scope in standard gsq baseline; do
  mapfile -d '' -t public_server < <(python3 -c 'import json,sys; [sys.stdout.buffer.write(x.encode()+b"\0") for x in json.load(open(sys.argv[1]))]' "$PINS/public-main/$scope/server.json")
  mapfile -d '' -t candidate_server < <(python3 -c 'import json,sys; [sys.stdout.buffer.write(x.encode()+b"\0") for x in json.load(open(sys.argv[1]))]' "$PINS/candidate/$scope/server.json")
  ref="$EVIDENCE/$scope-http-public"
  env -u STRIX_PREFILL_EXACT_FP16 -u STRIX_PREFILL_HC_OUTPUT_WMMA \
    python3 tests/http_slots_gpu_gate.py --prompt "$PINS/4k.ids" --prompt "$PINS/32k.ids" \
      --out "$ref" --generate 64 -- "${public_server[@]}"
  service_ref="$EVIDENCE/$scope-service-public"
  env -u STRIX_PREFILL_EXACT_FP16 -u STRIX_PREFILL_HC_OUTPUT_WMMA \
    python3 tests/batch_service_gpu_gate.py --short "$PINS/short2048.ids" --long "$PINS/32k.ids" \
      --out "$service_ref" --generate 64 -- "${public_server[@]}"
  for arm in on off; do
    value=1
    if [ "$arm" = off ]; then value=0; fi
    env -u STRIX_PREFILL_EXACT_FP16 STRIX_PREFILL_HC_OUTPUT_WMMA="$value" \
      python3 tests/http_slots_gpu_gate.py --prompt "$PINS/4k.ids" --prompt "$PINS/32k.ids" \
        --reference "$ref/single-reference.json" --out "$EVIDENCE/$scope-http-$arm" \
        --generate 64 -- "${candidate_server[@]}"
    env -u STRIX_PREFILL_EXACT_FP16 STRIX_PREFILL_HC_OUTPUT_WMMA="$value" \
      python3 tests/batch_service_gpu_gate.py --short "$PINS/short2048.ids" --long "$PINS/32k.ids" \
        --reference "$service_ref/alone.json" --out "$EVIDENCE/$scope-service-$arm" \
        --generate 64 -- "${candidate_server[@]}"
  done
  if [ "$scope" = standard ]; then
    for arm in public-main on off; do
      command=("${candidate_server[@]}")
      value=1
      if [ "$arm" = public-main ]; then command=("${public_server[@]}"); fi
      if [ "$arm" = off ]; then value=0; fi
      env -u STRIX_PREFILL_EXACT_FP16 STRIX_PREFILL_HC_OUTPUT_WMMA="$value" \
        python3 tests/batch_service_gpu_gate.py --scenario nest --nest-rows 2048 \
          --short "$PINS/short2048.ids" --long "$PINS/32k.ids" \
          --reference "$service_ref/alone.json" --out "$EVIDENCE/nested-$arm" \
          --generate 64 -- "${command[@]}"
    done
  fi
done
```

GSQ and baseline retain their existing nested exclusions; do not label an
unwitnessed nested arm a pass. Run the operator fixture too:

```sh
ctest --test-dir build --output-on-failure -R '^strix-hip-hc-out16-tests$'
```

`hc_output_wmma` passed a pre-declared confirmation holdout; its NLL endpoint is measured against a rounding-equivalent control whose per-case movement is capped at +-0.02 nats/token (a bounded control allowance, not an unbiased estimate of the uncapped difference); the absolute NLL change vs the exact reference also passed (-0.37 % NLL/token, upper bound +0.38 %). The off switch restores the exact output path.
These restoration and repeatability gates do not substitute for that model-fidelity holdout.

## Bundle local CI evidence

A fresh HIP build of the final tree on gfx1151 passed native local CI:

- HC low-band: 132/132 cases equal to mode 0 without stray stores, including
  eight T8224 cases exercising 512-row pool rounding.
- All 11 device/host fixtures passed: compact-stage-images (host and HIP),
  ple-launcher, ple-staged-gather, qsa-index-batch, moe-prefill-pool16k,
  startup-policy, prefill-pool-host and three prefill-call-shapes fixtures.
- Effective config confirmed that `--baseline` and `STRIX_QPF2K_DEFAULTS=0`
  each turn every controlled bundle item off. The unconditional PLE fix remains.
- At 32K real-text prompt tokens, last-row logits were bitwise equal between
  default and each OFF arm: early setup, eager loading, T16384, index batching
  and compact images. With `hc_output_wmma` OFF, low-band 3 and 0 also matched
  bitwise. These checks do not assert full-logit or seven-state-payload equality.

Interleaved 32K A/B medians over three rounds were 20.09 s default, 20.48 s early
setup OFF, 20.34 s T16384 OFF and 20.07 s eager loading OFF. Early setup and
T16384 show measured gains; eager loading is neutral within noise. Compact
images OFF was 19.95 s (n=2), neutral within noise versus default 20.09 s; no
speed gain claimed. Compact images retain only their calculated 96 MiB saving
per T8192 owner. HC low-band is inactive on the default `hc_output_wmma` route:
exact, bitwise-checked, no timing claim. Gains are workload-specific and not additive;
setup and loading move work into startup without establishing lower process wall time.

Open, nonblocking gates:

- Native comparison of all seven committed state payloads.
- T16384 at 262K and its memory-pressure fallback.
- Staged PLE versus `STRIX_PLE_STAGE_PREFETCH=0` gathered-byte comparison,
  including the real file-mapped/reclaimed staging matrix below.

Admission under the priced native pools (`--print-effective-config`): at 131,072
and 262,143 real-text tokens the default admits automatic T16384, and with
`STRIX_QPF2K_T16K=0` it admits T8192 as before, with no fallback.

The candidate indexer remains outside READY. The matrix below defines broader
qualification coverage; it does not negate the completed local-CI checks above.

## Bundle native gates

Keep `hc_output_wmma` fixed in every exact A/B arm. Preserve the model, profile,
context capacity, tails, output policy and state/reference assets. Restart between
startup-cached arms. Native compilation with warnings as errors and compiler/ISA
inspection precedes execution; CPU success does not qualify this bundle.

1. Early setup: compare default ON against `STRIX_QPF2K_EARLY_SETUP=0`, then repeat
   OFF via `--no-early-setup` and config `qpf2k_early_setup = false`. Compare complete
   token IDs, full logits and all seven committed state payloads bitwise. Confirm
   static setup is outside the request timer only on eligible fresh full blocks;
   live invalid-token/position rejection and partial/tail admission must still run.
2. Eager loading: with `HIP_ENABLE_DEFERRED_LOADING` absent, compare default ON
   against `STRIX_EAGER_CODE_LOAD=0`, `--no-eager-code-load`, and config
   `eager_code_load = false`. Compare the same bits. Confirm effective config says
   `0` for the ON arm; user-supplied HIP values must win in both arms. Capture
   startup, request TTFT and full process wall time separately. Setup/loading
   relocate work; a shorter request timer alone is not an end-to-end gain.
3. T16384: test 32,767 / 32,768 tokens and 262K, with generation fitting capacity.
   Compare auto ON to `STRIX_QPF2K_T16K=0`, using the same arithmetic identity and
   complete output/state bit comparison. Record UMA/GTT admission and native pool
   high-water bytes. Force memory pressure separately at 32K and 262K: T16384
   must decline to T8192 without an over-budget allocation or lost fallback reason.
   Run pool lifetime fixtures, including 16384/8192/16384/1320 rows and tails.
4. Exact batching: compare default against `STRIX_QPF2K_INDEX_BATCH=0` with
   `index_proj_wmma` OFF, full tokens/logits/state bitwise. Run the native
   `strix-hip-qsa-index-batch-tests` fixture and unsupported-layout fallbacks;
   baseline and GSQ must suppress the independent default.

5. HC low-band: compare decoder mode 3 against `STRIX_QPF2K_HC_LOWBAND=0`.
   Run `strix-hip-hc-wide-rows-tests` with `STRIX_HC_WIDE_LOWBAND_FIXTURES_ONLY=1`
   (modes 0/1/2/3, all storage forms and tails); test output WMMA and other
   declined routes separately. Exercise the admitted route with hc_output_wmma
   OFF in both arms, then the default identity in both arms. Compare full tokens,
   logits and all seven state payloads bitwise. Inspect identical arithmetic
   instructions and changed addressing/stores only. Optional fixture timing:
   `STRIX_HC_WIDE_BENCH_LOWBAND=1`; require complete-workload not-slower A/B too.

6. Compact QSA images: compare default ON against `STRIX_QPF2K_COMPACT_IMAGES=0`
   after restart, at T8192, T16384 and short tails. Run
   `strix-hip-compact-stage-images-tests` (T8192/40-row production producer and
   projection reader matrix, special bits, nonfinite/overflow status and guards).
   Compare expanded images, complete logits, tokens and seven state payloads
   bitwise under both default and exact-FP16 identities, fixing each identity in
   both arms. Confirm decode/MTP/capture/segmented owners retain FP32 and
   measure the 96 MiB/T8192 saving. Only physical arena execution hashes differ.

7. PLE acceptance fix: compare staged gather ON against the existing unstaged
   copy route (`STRIX_PLE_STAGE_PREFETCH=0`), including successful independent
   row-0 prefetch, no row-0 prefetch, short tails, mismatched IDs, worker failure
   and alternating staging slots. Confirm complete gathered bytes, full logits,
   tokens and seven state payloads bitwise, no extra accepted rows on rejection,
   and full staged_rows on acceptance. Run CPU `strix-ple-staged-gather-tests`
   before native execution and `strix-hip-ple-launcher-tests` for native PLE
   launcher/tail coverage. The staged acceptance matrix additionally requires
   the real file-mapped backend and reclaimed staging path; the fixture alone
   does not qualify that matrix. This is an unconditional fix, not a new policy switch.

After equality passes, measure interleaved ON/OFF complete-workload runs for each
exact item separately, then the whole bundle. Require not-slower medians; retain
startup/request/process timings and memory qualification evidence. Do not add gains
from different workloads. Record running kernel and IOMMU PerfOpt state without
changing host policy.

Candidate `index_proj_wmma` stays default OFF. Opt in with
`STRIX_PREFILL_INDEX_PROJ_WMMA=1`; test individual env/CLI/config OFF and the master
exact switch. The candidate uses `idxproj16`, independent K16 BF16 WMMA panels,
then ascending FP32 panel additions. It is bit-changing: do not require ON/OFF
bitwise equality or borrow HC holdout evidence. Run native operand/mapping and
cancellation fixtures, model-fidelity holdouts and complete-workload gates for
each proposed active set before promotion to READY. Diagnostic `STRIX_QPF2K_QSA_TRACE=1`
reports `index_wmma` launch counts; a nonzero count establishes dispatch, not fidelity. Compare OFF restoration to
the base's full logits and seven state payloads. No local native qualification
or performance claim is established by the CPU bundle checks.
