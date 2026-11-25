# Inkling 2 Mini HIP qualification

The HIP CLI, text HTTP service, CLI image input and CLI DSpark are connected. Label the `--profile exact` reference arm **BASE** pending a bitwise reference for the same device arithmetic. Record the effective route and weight identity for every arm. A FAST selection that resolves to BASE supplies a resolver and identity check. A changing FAST route needs quality and measured speed receipts in the same regime.

| Gate | Evidence in the tree | Evidence needed for completion |
| --- | --- | --- |
| Checkpoint and tokenizer | Pinned single-shard text schema, optional index validation, tokenizer tests | Hash the shard, config, index and tokenizer used by each run. Pin the independent reference source and arithmetic. |
| CPU operators and loop | Synthetic CPU forward, loop restart and prior-pass messages, relative attention, convolution, packed experts, cache wrap and retry | Official-weight prefill and teacher-forced decode against independent logits and intermediate tensors. Cover both repeat boundaries, full/sliding attention and positions 1023–1025. |
| HIP text | Resident decoder and operator, verify and cohort fixtures | Integrated device fixtures, official-weight generation IDs, prefill and teacher-forced decode parity, with per-visit and per-block checkpoints and continuation across wrap after rollback. |
| HIP vision | Shared tower and splice, CPU/media fixtures, standalone tower comparison | Compare pixels, tower hidden/projected rows, image placement, first language logits and complete continuation IDs. Run the built-in image recipe end to end. |
| HIP DSpark | Separate `mtp.safetensors` shape validation, native drafter and synthetic device fixture | Official sidecar load, committed IDs/state vs serial, accepted/rejected windows and restore, with throughput and acceptance receipts. HTTP refuses DSpark. |
| Profiles and context | BASE and FAST CLI/serve resolution, held-out gate, context and KV planning | Effective profile identity on integrated binary, quality and gain by regime, 1K–128K and long-context capacity/continuation. |
| Placement | Single worker and two full-model replicas | Optional two-node throughput run. Layer, expert and tensor sharding require a stage/collective binding and native fidelity/transport testing. |
| Small reuse | Shared kernels and execution mechanisms | Small checkpoint contract and non-looped binding are tracked separately. |

## Device build and inputs

Run on Linux with ROCm/gfx1151. Use pre-existing local assets. `MODEL_DIR` contains the [pinned Mini checkpoint](README.md#checkpoint-inspection). `OUT` is a fresh directory outside the source tree. `FIXTURES` contains plain-text token fixtures `text-{1,1023,1024,1025}.ids` and `prompt-{1024,8192,32768,131072,1048512}.ids` in whitespace-separated form, plus `image.json` with a fixed question and local image as a PNG data URL. `REF` contains independent expected artifacts for the same cases. Record the reference package commit, compiler, revision, file hashes, device, memory budgets, kernel and power settings. Keep receipts and fixtures outside Git.

```bash
set -euo pipefail
: "${MODEL_DIR:?}" "${OUT:?}" "${FIXTURES:?}" "${REF:?}"
mkdir "$OUT"
model="$MODEL_DIR/model-00001-of-00001.safetensors"
tokenizer="$MODEL_DIR/tokenizer.json"
b=build-inkling-hip
cmake -S . -B "$b" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON -DSTRIX_ENABLE_HIP_RUNTIME=ON -DSTRIX_HIP_ARCHS=gfx1151 \
  -DSTRIX_INKLING2_TOKENIZER_JSON="$tokenizer"
mapfile -t fixtures < <(ctest --test-dir "$b" -N -L '^inkling2$' -R '^strix-hip-inkling2-mini-' \
  | sed -n 's/^ *Test *#[0-9]*: \(strix-hip-inkling2-mini-.*\)$/\1/p')
test "${#fixtures[@]}" -gt 0
cmake --build "$b" --target strix-inkling2-mini-inspect strix-inkling2-mini-cpu \
  strix-inkling2-mini-hip strix-serve strix-inkling2-vision-gate \
  strix-inkling2-vision-gate-hip "${fixtures[@]}"
ctest --test-dir "$b" -L '^inkling2$' -R '^strix-hip-inkling2-mini-' --output-on-failure \
  | tee "$OUT/hip-operators.log"
git rev-parse HEAD > "$OUT/revision.txt"
sha256sum "$b/strix-inkling2-mini-hip" "$b/strix-serve" "$model" "$tokenizer" \
  "$MODEL_DIR/config.json" "$MODEL_DIR/model.safetensors.index.json" > "$OUT/sha256.txt"
"$b/strix-inkling2-mini-inspect" "$model" "$MODEL_DIR/model.safetensors.index.json" \
  > "$OUT/checkpoint.txt"
```

## Text, image and DSpark

`REF/text-$n.generated` and `REF/image.generated` hold the independently generated lines in the form `generated ID ID ...`. `REF/*.f32` contains matching little-endian FP32 rows. The native arms export full greedy sequences and terminal prompt logits. With a bitwise independent reference, compare its corresponding files with `cmp`. For a different evaluation order, declare the comparator and limits before the run and record error across all 200058 logit entries. Report numerical agreement separately from the token-ID check.

```bash
for n in 1 1023 1024 1025; do
  read -r -a ids <<< "$(xargs < "$FIXTURES/text-$n.ids")"
  test "${#ids[@]}" -eq "$n"
  for backend in cpu hip; do
    "$b/strix-inkling2-mini-$backend" "$model" "${ids[@]}" --profile exact \
      --speculation off --generate 64 --logits "$OUT/text-$n.$backend.f32" \
      > "$OUT/text-$n.$backend.out" 2> "$OUT/text-$n.$backend.log"
    grep '^generated ' "$OUT/text-$n.$backend.out" | diff -u "$REF/text-$n.generated" -
    test "$(wc -c < "$OUT/text-$n.$backend.f32")" -eq 800232
  done
done

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 tools/vision/inkling2_vision_ref.py \
  --checkpoint "$MODEL_DIR" --out "$OUT/vision-ref" synth:40x40:7 synth:160x80:8
"$b/strix-inkling2-vision-gate" --checkpoint "$MODEL_DIR" --ref "$OUT/vision-ref" \
  --backend host synth_40x40_7 synth_160x80_8 | tee "$OUT/vision-tower-host.txt"
"$b/strix-inkling2-vision-gate-hip" --checkpoint "$MODEL_DIR" --ref "$OUT/vision-ref" \
  --backend hip synth_40x40_7 synth_160x80_8 | tee "$OUT/vision-tower-hip.txt"
for backend in cpu hip; do
  "$b/strix-inkling2-mini-$backend" "$model" --tokenizer "$tokenizer" \
    --messages "$FIXTURES/image.json" --profile exact --speculation off \
    --generate 64 --logits "$OUT/image.$backend.f32" \
    > "$OUT/image.$backend.out" 2> "$OUT/image.$backend.log"
  grep '^generated ' "$OUT/image.$backend.out" | diff -u "$REF/image.generated" -
  test "$(wc -c < "$OUT/image.$backend.f32")" -eq 800232
done

sha256sum "$MODEL_DIR/mtp.safetensors" >> "$OUT/sha256.txt"
read -r -a ids <<< "$(xargs < "$FIXTURES/text-1025.ids")"
"$b/strix-inkling2-mini-hip" "$model" "${ids[@]}" --profile exact \
  --speculation off --generate 64 > "$OUT/serial.out" 2> "$OUT/serial.log"
"$b/strix-inkling2-mini-hip" "$model" "${ids[@]}" --profile exact --speculation auto \
  --dspark "$MODEL_DIR/mtp.safetensors" --generate 64 > "$OUT/dspark.out" 2> "$OUT/dspark.log"
diff -u <(grep '^generated ' "$OUT/serial.out") <(grep '^generated ' "$OUT/dspark.out")
grep '^speculation mode ' "$OUT/dspark.out"
```

The tower gate uses its axial reference recipe. The end-to-end CLI arm must record the built-in `inkling2-vision-v1` identity and use expected logits for that recipe. For DSpark require proposed rows and both acceptance and restore witnesses, including across a wrap. The present CLI emits prompt-end scores, generated IDs and drafter counters. Independent text export, a native per-visit trace and real-checkpoint rollback digests are prerequisites for the remaining stage/state comparisons.

## FAST gate and measurements

Collect the two regimes of the explicit Q8_0 candidate with disjoint held-out prose, code and structured sources and a genuine evaluation-order null. Each class requires at least 800 scored rows. Provide enough input for 13 disjoint windows per class and band. This command covers the requested bands. Qualification also requires a measured gain of the effective candidate route in each selected regime.

```bash
python3 scripts/bench/inkling2/build-fast-gate-windows.py "$b/strix-inkling2-mini-hip" "$model" "$OUT/windows" \
  --tokenizer "$tokenizer" --count 13 \
  --source "prose=$FIXTURES/held-prose.txt" --source "code=$FIXTURES/held-code.txt" \
  --source "structured=$FIXTURES/held-structured.txt" \
  --band 1K=1024 --band 8K=8192 --band 32K=32768 --band 128K=131072
for mode in prefill decode; do
  for arm in REF REF2 NULL CAND CANDF NULLF; do
    bash scripts/bench/inkling2/fast-gate-v2.sh arm "$b/strix-inkling2-mini-hip" "$model" \
      "$OUT/windows/windows.tsv" "$OUT/q8-$mode" "$arm" "$mode"
  done
  bash scripts/bench/inkling2/fast-gate-v2.sh --expect-identical "$OUT/windows/windows.tsv" "$OUT/q8-$mode" "$mode"
  bash scripts/bench/inkling2/fast-gate-v2.sh qual "$OUT/windows/windows.tsv" "$OUT/q8-$mode" "$mode"
done
```

Run on an otherwise idle device. Measure the default HTTP path first. `ttft_ms` is client request-to-first-content. Prefill and decode rates use the server receipt. `VmHWM` is peak process RSS in KiB. GTT is device-wide sampled bytes. Record minimum MemAvailable separately on UMA to avoid double-counting. The commands run three interleaved trials per tier and context, with a fresh server per run and the model already loaded at request time. At 1M the prompt leaves room for 64 generated IDs. Record any admission refusal and its cause.

```bash
source scripts/bench/common/runlib.sh
run_case() (
  set -euo pipefail
  n=$1 tier=$2 round=$3
  dir="$OUT/http-$n-$tier-$round"
  mkdir "$dir"
  "$b/strix-serve" --model inkling2-mini --profile "$tier" \
    --bind 127.0.0.1 --port 18197 --context "$((n + 64))" --slots 1 -- \
    --checkpoint "$model" --tokenizer "$tokenizer" --backend hip \
    > "$dir/server.log" 2>&1 &
  server_pid=$!
  trap 'kit_sampler_stop; kill -INT "$server_pid" 2>/dev/null || true; wait "$server_pid" || true' EXIT
  printf '%s\n' "$server_pid" > "$dir/pid"
  kit_sampler_start "$dir"
  for attempt in $(seq 1 1800); do
    if curl -fsS --max-time 1 http://127.0.0.1:18197/health > "$dir/health.json" 2>/dev/null; then break; fi
    kill -0 "$server_pid"
    sleep 1
  done
  test -s "$dir/health.json"
  python3 scripts/bench/inkling2/http-qualification.py http://127.0.0.1:18197 \
    "$FIXTURES/prompt-$n.ids" "$n" --generate 64 > "$dir/request.json"
  awk '/^VmHWM:/{print $2}' "/proc/$server_pid/status" > "$dir/peak-rss-kib.txt"
  kit_sampler_stop
  awk 'NF==9{if(!n||$6>peak) peak=$6; if(!n||$7<avail) avail=$7; n++}
       END{if(!n) exit 1; print peak, avail}' "$dir/samples.tsv" > "$dir/peak-gtt-bytes-min-available-kib.txt"
  kill -INT "$server_pid"
  wait "$server_pid"
  trap - EXIT
)
for n in 1024 8192 32768 131072; do
  round=0
  for tier in exact fast fast exact exact fast; do
    run_case "$n" "$tier" "$((round+=1))"
  done
done
run_case 1048512 exact 1
```

| Input tokens | Tier | Effective route/identity | TTFT ms | Prefill tok/s | Decode tok/s | Peak RSS KiB | Peak GTT bytes | Min available KiB | Receipt |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1024 | BASE | Pending | | | | | | | |
| 1024 | FAST | Pending | | | | | | | |
| 8192 | BASE | Pending | | | | | | | |
| 8192 | FAST | Pending | | | | | | | |
| 32768 | BASE | Pending | | | | | | | |
| 32768 | FAST | Pending | | | | | | | |
| 131072 | BASE | Pending | | | | | | | |
| 131072 | FAST | Pending | | | | | | | |
| 1048512 | BASE | Pending | | | | | | | |

For the optional replica arm, start both workers and the router as in the [two-node example](README.md#two-nodes-replicas), with the checkpoint, tokenizer, profile, context and slots of the single-worker control. Set `SINGLE` and `PAIR` to the respective base URLs. Verify complete response IDs and each worker's request count. Record one- and two-worker prefill and decode rates and ratios, batch makespan, aggregate throughput and per-node peak memory.

```bash
for arm in single pair; do
  endpoint=$SINGLE
  if [ "$arm" = pair ]; then endpoint=$PAIR; fi
  python3 scripts/bench/inkling2/http-qualification.py "$endpoint" "$FIXTURES/prompt-8192.ids" 8192 \
    --generate 64 > "$OUT/$arm-a.json" & first_pid=$!
  python3 scripts/bench/inkling2/http-qualification.py "$endpoint" "$FIXTURES/prompt-8192.ids" 8192 \
    --generate 64 > "$OUT/$arm-b.json" & second_pid=$!
  wait "$first_pid"
  wait "$second_pid"
done
```
