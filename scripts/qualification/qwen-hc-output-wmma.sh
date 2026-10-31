#!/usr/bin/env bash
# Native only. See docs/engine/prefill/hc-output-wmma-qualification.md.
set -euo pipefail
: "${PINS:?set PINS to the private pinned command directory}"
: "${EVIDENCE:?set EVIDENCE to a fresh private directory outside Git}"
mkdir -p "$EVIDENCE"
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

ctest --test-dir "${BUILD_DIR:-build}" --output-on-failure -R '^strix-hip-hc-out16-tests$'
