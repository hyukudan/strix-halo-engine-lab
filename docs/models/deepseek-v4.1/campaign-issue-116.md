# DeepSeek V4.1 Flash fixed-prompt campaign

This is a new baseline for issue [#116](https://github.com/hyukudan/strix-halo-engine-lab/issues/116). The historical `cab238d6` source snapshot, prompts and receipts are unavailable. The new campaign uses an accessible main commit and a public workload with its own results.

## Reproduce on a Strix Halo USB4 pair

Use the same clean, accessible main commit and fresh HIP build on both nodes. Set `PEER` to rank 1. The example uses `build` and `models` on both ranks in the same checkout location. The [asset setup and build instructions](../../reproduce.md) describe the Q2 target, dense U6 overlay, K4/K5 packs and DSpark draft and vocabulary. Record the main SHA and hardware power state before running.

```sh
bash scripts/bench/bench.sh run --engine deepseek --peer "$PEER" --contexts 1k,8k,32k --gen-tokens 256 --repeat 3 --warmup 1 --only decode,prefill,exactness --profile exact --verify-models full --write-refs --out bench-results/issue-116-exact
bash scripts/bench/bench.sh run --engine deepseek --peer "$PEER" --contexts 1k,8k,32k --gen-tokens 256 --repeat 3 --warmup 1 --only decode,prefill,exactness --profile fast --verify-models full --write-refs --out bench-results/issue-116-fast
bash scripts/bench/bench.sh run --engine deepseek --peer "$PEER" --contexts 1k,8k,32k --gen-tokens 256 --repeat 3 --warmup 1 --only decode,prefill,exactness --profile turbo --verify-models full --write-refs --out bench-results/issue-116-turbo
```

Append `--dry-run` to see both rank commands and the config probes without loading assets or connecting to rank 1. Allow enough wall time for two full model manifests, nine native jobs per tier and teardown. The kit reads the native column for each requested tier from the DeepSeek model preset on the checked-out main commit and records both words. Greedy and DSpark explicitly select the native decode arm within each column. The rank resolver supplies the effective environment used by the native decode app. Run with ordinary ROCm installation settings and without additional tuning overrides.

The kit reads the eight project-owned [fixed prompts](../../../scripts/bench/prompts/kit.json). Its raw GGUF tokenizer produces the native input arrays without a chat wrapper. The prefill fixture repeats the tokenized English chat seed to exactly 1,024, 8,192 or 32,768 rows. Each decode vector has 256 IDs including the first prefill argmax. Native decode times the subsequent 255 tokens and continues through EOS to the fixed count. TTFT measures the native prefill wall interval ending at the first token.

Each tier runs three decode rounds. Round 1 runs greedy then DSpark, round 2 reverses them, and round 3 returns to greedy then DSpark. Prompt order rotates by one position after each round. Each decode arm runs its rotated prompt list twice in one load. The first pass warms up and the second enters the measurements. Prefill runs after decode in ascending length order, one warmup followed by three measured requests in the same load at each length. The kit's `report` command reads `results.json`. Its aggregation uses the median and range of each prompt's three samples, with separate pooled medians by group and arm.

| Group | Prompt IDs |
|---|---|
| Chat | `chat`, `chat-es`, `json`, `math` |
| Code | `code`, `sql` |
| Prose | `prose`, `summary` |

## Publication files

For each tier, retain `results.json` in the [kit schema](../../../scripts/bench/results.schema.json), `results.md`, `commands.json`, `effective-*.json`, `prompts.json`, the corresponding prompt texts and ID arrays, `*.manifest.json`, `samples.jsonl` and both ranks' native logs. `results.json` contains source and binary hashes, model/overlay/pack/vocabulary paths relative to the asset root with their sizes and SHA256, build and toolchain metadata and rank hardware. The target GGUF contains the tokenizer. The raw logs include warmups and the generated vectors. The prompt ID hash covers little-endian uint32 bytes. The argv record uses placeholders for deployment paths.

After verifying the main commit and independent stable repeats, promote the measured reference candidates into [the DeepSeek reference manifest](../../../scripts/bench/refs/deepseek-pair.json) with the commit and hardware in their provenance. Review receipts before publishing. Leave unrelated [expected workload rows](../../../scripts/bench/expected.json) unchanged.

## Results template

| Metadata | Value |
|---|---|
| Main commit | |
| Hardware, USB4 and power state | |
| Toolchain, ROCm, kernel and driver | |
| Model, tokenizer, overlay, pack and vocabulary hashes | |
| Raw receipts and reference candidates | |

| Workload | Unit | EXACT median (range) | FAST median (range) | TURBO median (range) |
|---|---|---|---|---|
| Chat, greedy | tok/s | | | |
| Chat, DSpark | tok/s | | | |
| Code, greedy | tok/s | | | |
| Code, DSpark | tok/s | | | |
| Prose, greedy | tok/s | | | |
| Prose, DSpark | tok/s | | | |
| TTFT 1,024 | s | | | |
| TTFT 8,192 | s | | | |
| TTFT 32,768 | s | | | |
