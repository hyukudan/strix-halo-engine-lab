# Benchmark campaigns

Both protocols use [the benchmark kit](reproduce.md) and its [fixed prompts](../scripts/bench/prompts/kit.json). The DeepSeek pair's public protocol has its own [reproduction commands and results template](models/deepseek-v4.1/campaign-issue-116.md).

## Product defaults and private records

Print the full eight-model, three-hardware, three-tier plan on a host without devices.

```sh
bash scripts/bench/bench.sh campaign default --dry-run
```

Each arm starts `strix serve MODEL --profile exact|fast|turbo` once. The site needs its ordinary `launch/MODEL.toml` installation configuration, with the model asset, backend and any rank or listener wiring needed by the product launcher. Use the same installation settings across tiers. Run on a dedicated Linux measurement host with a clean engine environment. Set the client port to the configured loopback listener if it differs from 8080.

```sh
bash scripts/bench/bench.sh campaign default --hardware strix-one --power-state "$POWER_STATE" --ledger "$LEDGER" --out "$ONE"
bash scripts/bench/bench.sh campaign default --hardware strix-two --peer "$PEER" --peer-repo "$PEER_REPO" --peer-logs "$PEER_LOGS" --power-state "$POWER_STATE" --ledger "$LEDGER" --out "$TWO"
bash scripts/bench/bench.sh campaign default --hardware rtx --eviction-counter "$RTX_COUNTER" --power-state "$POWER_STATE" --ledger "$LEDGER" --out "$RTX"
bash scripts/bench/bench.sh campaign compare --ledger "$LEDGER" "$ONE" "$TWO" "$RTX"
```

The peer needs the same kit and its rank-1 launch configuration. Its log directory must exist and be writable. The control channel keeps the rank alive during the tier, reads its KFD counter and closes the rank after the measurements. `--strix` and `--peer-strix` select the launcher executable. `--model` can restrict a job to a model and can be repeated. Use it on the RTX workstation to select deployed models. The default startup and measurement timeouts are each 120 seconds. The nine cells share the measurement window after the resident server is ready. A capacity or time limit leaves an incomplete receipt.

- Two fresh, exact-length 8,192 and 32,768 token requests per size use deterministic rotations of the server-tokenized chat prompt. Each of the four inputs has a different first ID. Report the median of two engine prefill rates and check the response's token count and reuse fields.
- One greedy request per committed chat, code and prose prompt uses up to 64 output tokens and respects EOS. Decode rate counts tokens after the first over the engine decode interval.
- C2, C4, C8 and C16 cycle through chat, code and prose together. Aggregate actual completion tokens over elapsed wall time from cohort release to the final complete response. Natural server-side queueing is part of that interval.
- Keep raw responses, complete generated-ID hashes, native logits digests, and per-rank eviction snapshots by cell. Strix reads the resident process's KFD `evicted_ms` counters. An RTX capture needs a read-only decimal counter for the same residency eviction event from the site's telemetry. Missing counters, counter changes or missing provenance in the response leave a cell unverified.

The report includes the same-workload pair/one speed ratio when both receipts are valid. The driver tries the plain product command on each selected configuration. A model that only has a CLI route, an installation without resident pair serving, or a server that lacks the requested digests or timing can leave some or all cells unavailable. Inspect the captured server log and cell receipt. Keep output and ledger paths outside the worktree.

### Ledger format and verdicts

The UTF-8 TSV uses this header.

```text
model	hardware	tier	cell	status	value	unit	workload_sha256	source	workload
```

Use `tok/s` and one source citation with a line number for each numeric entry. A `record` has a positive number and the cell's exact workload SHA256. Establish workload equivalence from the original prompt or request bytes, token lengths, grouping, cache behavior and timing denominator before entering that hash. `historical` entries preserve sourced numbers whose workload has not been verified against the current cell. `decode-other` can hold historical B1 receipts with different content. `missing` marks a gap and carries no number. Use the campaign receipt's `workload_sha256` field after verifying equivalence.

Each requested cell prints the tier, default rate, record rate and `reached yes/no`. Exit 0 means all selected cells met their matched records and the same-workload tiers were nondecreasing. Exit 1 also covers missing evidence, distinguished from `under-record` and `slower-than-lower-tier` by the status column. The TSV is suitable for preserving the remainder of the comparison after a partial run.
