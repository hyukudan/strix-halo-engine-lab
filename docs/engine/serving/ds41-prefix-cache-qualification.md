# DeepSeek pair prefix policy qualification

All native execution and timing cells for the bounded agent policy are **UNVERIFIED**.
Code enables the requested host/in-run/plain-session candidate; release qualification remains
pending. CPU checks establish neither model exactness nor speed. Disk and before-tail require
separate qualification. Use one service disk root per node; separate roots/users are outside
one aggregate disk quota.

## Build and CPU checks

Pin the same source commit, byte-identical executable/libraries, model/pack hashes, numerical
environment, transport, context/pool/slots, scheduler, prefill rows and warm-up on both nodes.
Build outside timed observations with the deployment's qualified compiler and architecture:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/bin/amdclang++ \
  -DSTRIX_ENABLE_HIP_RUNTIME=ON -DBUILD_TESTING=ON
cmake --build build --target strix-deepseek41-serve strix-deepseek41-prefix \
  strix-deepseek41-prefill strix-deepseek41-prefix-cache-tests \
  strix-deepseek41-pair-defaults-tests strix-deepseek41-serve-cache-policy-tests
ctest --test-dir build --output-on-failure \
  -R 'strix-deepseek41-(prefix-cache|pair-defaults|serve-cache-policy)-tests'
sha256sum build/strix-deepseek41-{serve,prefix,prefill}
```

The native CPU run must exercise Unix host leases, namespace ownership and directory quotas.
Add subprocess stress receipts for live namespaces, a busy quota, process death/restart, queued
writes/deletions and full-disk failures. Windows CPU coverage excludes those branches. Verify
crashed writers cannot free live reservations and services cannot delete another service's files.
Record actual RSS/allocator peaks in addition to accounted cache bytes.

## Untimed exactness tickets

Every PAIR ticket is at most 480 seconds, including load, readiness, collection and cleanup;
one ticket at a time. Start no new observation after a conservative 420-second deadline.
Shorten that deadline if measured cleanup needs more time. Split these commands across tickets
as needed; they are alternatives, not one ticket's batch. Run each on both nodes simultaneously
with `R=0` and `R=1` and the same qualified profile environment. The prompt is a frozen,
nonempty, comma-separated token fixture:

```sh
build/strix-deepseek41-prefix assets/model.gguf --rank "$R" --transport usb4-auto \
  --prompt assets/prompt.ids --rows 6000 --layers 40 --block 256 --max-rows 4096 \
  --cache-mb 512 --inrun-check --byte-check --warm 5376,5632,5888 --decode 16
build/strix-deepseek41-prefix assets/model.gguf --rank "$R" --transport usb4-auto \
  --prompt assets/prompt.ids --rows 8192 --layers 40 --block 256 --max-rows 2048 \
  --cache-mb 512 --inrun-check --byte-check --decode 16
# Explicit disk only; DIR is this campaign's service-owned test directory.
build/strix-deepseek41-prefix assets/model.gguf --rank "$R" --transport usb4-auto \
  --prompt assets/prompt.ids --rows 8177 --block 256 --max-rows 2048 --cache-mb 512 \
  --cache-dir "$DIR" --disk-mb 4096 --phase all --inrun --byte-check \
  --session 512 --decode 16 --fork 1280 --rewind 5376
# Separate consecutive tickets, retaining only this test namespace between them:
build/strix-deepseek41-prefix assets/model.gguf --rank "$R" --transport usb4-auto \
  --prompt assets/prompt.ids --rows 8192 --cache-dir "$DIR" --disk-mb 4096 \
  --phase write --inrun --byte-check --decode 16
build/strix-deepseek41-prefix assets/model.gguf --rank "$R" --transport usb4-auto \
  --prompt assets/prompt.ids --rows 8192 --cache-dir "$DIR" --disk-mb 4096 \
  --phase read --inrun --byte-check --decode 16
```

Repeat the read diagnostic with `--corrupt block` and `--corrupt snapshot` on only one rank.
The prefix tool alone does not qualify serve's collective resource/restore transactions.
`--byte-check` compares canonical continuation bytes and full logits when freshly computed;
whole hits receive an additional fresh-suffix comparison. In-run unknown-next-token states
and unknown session logits report STATE-ONLY. The fresh session-turn suffix and continued decode
must pass. Unknown logits never count as equality. Byte diagnostics are excluded from timing.

Re-run the scheduler gates, one command per ticket if needed:

```sh
build/strix-deepseek41-prefill assets/model.gguf --rank "$R" --transport usb4-auto \
  --prompt assets/prompt.ids --rows 8192 --task-rows 2048,2053,8192 \
  --task-levels whole,layer,phase,finest --task-units 1 --interleave 256
# Repeat with --task-units 3 and 32, then:
build/strix-deepseek41-prefill assets/model.gguf --rank "$R" --transport usb4-auto \
  --prompt assets/prompt.ids --rows 8192 --task-levels finest --task-budget 250 --interleave 256
```

Serving requires both-rank next tokens, final logits FNV, continuation-state digests and complete
HTTP IDs versus an isolated cold reference of the same numerical identity. Cover miss,
partial/whole hit, edited tail, stable prefix, sessions, eviction, pressure bypass, cancel/reuse
and optional disk restart/corrupt-rank fallback with the current scheduler.
`STRIX_DS41_SERVE_PREFILL_LOG=1` records rank-1 prefill evidence. Add matching untimed canonical
state/full-logit exports to A and B. Serve does not yet expose a general canonical final-decode
state export endpoint: that collection arm remains a release gate requiring instrumentation.

## Bounded HTTP observations

Pre-register frozen JSON outside Git, each with `stream:true`, `return_token_ids:true`,
`stream_options:{"include_usage":true}` and fixed greedy limits. Freeze tool outputs, all later
turn prompts and tokenization counts before timing. Never build later prompts from observed
generations. Chat fixtures must include the stable system/tools boundary at token 1280.
Reuse the established replay benchmark's request construction when available.

A uses `CACHE=(--cache-profile off)`; B uses `CACHE=()` with no cache flags. Plain mode uses
`MODE=(--no-dspark)` in both arms. The separate DSpark campaign pins the same explicit draft,
vocabulary/window/rank/max-inflight flags in MODE. Never pool modes. Pre-register feasible fixed
context/pool/slots per cell family, identical in A/B. The following 64K context/270336-position
pool is an example to qualify for C8 32K, not a memory-admission result.

Inside each authorized PAIR ticket, set the same `START_EPOCH` on both nodes; set `R`, `OUT`,
CACHE and MODE locally. This local shell bounds its server and cleans only its tracked PID:

```sh
set -euo pipefail
DEADLINE=$((START_EPOCH + 420))
REMAIN=$((START_EPOCH + 470 - $(date +%s)))
test "$REMAIN" -gt 0
mkdir -p "$OUT"
export STRIX_DS41_SERVE_PREFILL_LOG=1
build/strix-deepseek41-serve assets/model.gguf --rank "$R" --transport usb4auto \
  --context 65536 --pool-positions 270336 --slots 8 --prefill-rows 2048 \
  --print-effective-config "${MODE[@]}" "${CACHE[@]}" > "$OUT/effective-$R.txt"
timeout --signal=INT --kill-after=8s "${REMAIN}s" \
  build/strix-deepseek41-serve assets/model.gguf --rank "$R" --transport usb4auto \
  --context 65536 --pool-positions 270336 --slots 8 --prefill-rows 2048 --http-port 8080 \
  "${MODE[@]}" "${CACHE[@]}" > "$OUT/rank-$R.log" 2>&1 &
SERVER_PID=$!
trap 'kill -INT "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" || true' EXIT
if [ "$R" = 0 ]; then
  until curl -fsS http://127.0.0.1:8080/v1/capabilities > "$OUT/capabilities.json"; do
    test "$(date +%s)" -lt "$DEADLINE"
    sleep 1
  done
  test "$(date +%s)" -lt "$DEADLINE"
  python3 tools/ds41_prefix_cache_probe.py --base http://127.0.0.1:8080 \
    --requests assets/frozen/cold-8192.json --out "$OUT/probe" --deadline-epoch "$DEADLINE"
  curl -fsS http://127.0.0.1:8080/api/status > "$OUT/status.json"
else
  wait "$SERVER_PID" || true
fi
```

The coordinator ends rank 1 when rank 0 finishes, collects both logs, and cleans both local
tracked process groups within 480 seconds. Do not defer receipt collection until the hard
timeout. Socket/request timeouts must fit the ticket's remaining collection/cleanup window.
For six turns pass the six files to `--requests` in order. For C4 pass four distinct files plus
`--parallel`; repeat for every required concurrency. The probe records submit-to-first-content
TTFT, completion, complete IDs and raw SSE. Role/receipt/keepalive metadata does not start TTFT.
Content-delta gaps may coalesce tokens; collect native per-token ITL separately. A failed/censored
observation is retained and invalid, never silently dropped. Startup is excluded equally from
client TTFT and reported separately. Do not clear machine-wide page cache or unrelated files.

## Cells and decision rule

Use **three ABBA blocks per cell**, A1 B1 B2 A2. Start fresh processes before each miss/burst
observation. Retain cache only within a fixed six-turn session observation. A block may span
consecutive bounded tickets; repeat an incomplete pair in full, recording its cause.

| Family | Required cells |
|---|---|
| C1 miss | 1024/8192/32768 tokens, each also +17 tail; 16 generated tokens; B resume=0. Cold chat with stable boundary 1280; shared-block/no-snapshot miss. |
| Agent hits | Six turns: conversation A four, B two sharing system/tools. Each turn TTFT and session total separately. Force plain decode across a 256-token boundary. |
| Burst miss | Distinct first blocks, fixed lengths/barrier/order; C4 at 1K/8K/32K. Each request position and burst completion separately. |
| Concurrency claim | C2,C3,C5,C6,C7,C8; long-prefill arrival beside a session-boundary decoder; ITL/completion interference separately. |
| Resources | Both-node memory refusal, host lease exhaustion, cache-full eviction; optional disk/queue/quota pressure. |

For each block compute `d=((B1-A1)+(B2-A2))/2`, positive = slower B. Retain the six individual
pair differences too. There are **n=3 independent block contrasts**, df=2. The two-sided 95%
Student-t interval is `mean(d) +/- 4.302652729911275*sd(d)/sqrt(3)`.
A lower bound >0 in **any** TTFT/ITL/completion cell fails the candidate. No pooled lengths,
turns, concurrency, modes or compensation by other wins. A zero-crossing interval meets the
operational rule but is inconclusive about a strict no-regression guarantee. No percentage
threshold, optional stopping, post-hoc cell removal or compressed sample count. Any mismatch
blocks release regardless of timing. Every passing exact gain becomes default with an OFF switch.
Retain raw receipts, memory minima, runs, row-layers, resumes, lookup/export/insert/handoff times,
disk/queue bytes, failures and effective policy. Mark every unrun cell UNVERIFIED.
