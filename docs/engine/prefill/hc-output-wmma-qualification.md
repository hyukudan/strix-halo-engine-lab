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
row, the fixed components and the token history. The state command pins eager greedy generation, terminal-row output,
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
      --reference-logits "$ref/public-main-legacy1/logits.bin" \
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
      --reference-logits "$on/on-legacy1/logits.bin" \
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
        --reference-logits "$ref/$case_name/logits.bin" \
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
