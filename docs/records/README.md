# Checking an integration against the performance record

The [ledger](best_on_record.md) tracks public throughput for language models on one Strix Halo node or a USB4 pair. Each row ID is **engine / placement / tier / route / metric**. Routes are `cli` and `served`. Metrics are prompt throughput at nominal `prefill-2k`, `prefill-8k` or `prefill-32k`, single-request `decode-b1`, and eight-request `aggregate-c8`. Rates are in tok/s. A missing row means there is no published baseline for that cell.

1. On the integration commit, run the affected campaign cells against the ledger workloads. Preserve prompt IDs, generation limits, pack, placement, route, effective tier, timing boundaries, cache state and power conditions. Run adjacent tiers and both entry points where available. Keep default and explicit lever combinations in separate runs. Use interleaved repeats and retain their receipts.
2. Record the results in the [campaign JSON format](example-campaign.json). Include failed launches, timeouts and cells with missing timing as explicit failures. Pass the expected row IDs using `--require ROW` for each cell, or use `--complete` for a full repeat of the ledger. Check each affected arm separately when their expected cells differ.
3. Run `python3 tools/ci/check_best_on_record.py campaign.json`. By default, changes within ±3% are reported as noise. Larger gains propose a new record. Larger losses, failed measurements and missing required cells exit nonzero. Review any adjacent-tier slowdown or a served result more than 2% below CLI as an integration finding. Tier checks compare equal prompt counts and timing boundaries within a run. CLI versus served permits a 10% prompt-length difference for templating and flags any difference in prefill clocks. Use a matched repeat to confirm what caused a flagged gap.
4. Resolve a failing cell before merging or obtain a written, reviewed reason to lower the floor. With public evidence in the tree, run `python3 tools/ci/check_best_on_record.py campaign.json --write` to add new cells and raise records. To lower a specific row, also pass `--accept-regression ROW "reason"`. Each accepted loss retains the old and replacement measurements and the reason. A campaign with any remaining hard failure leaves the ledger unchanged. The CI check compares against the base branch to catch a direct edit that lowers or removes a row.

## Campaign JSON v1

The top-level object has `version: 1` and a `runs` object. Each run maps unique row IDs to measurements from one commit and one arm. Labels such as `default` and `all-on` are free-form ASCII IDs. Each measurement contains:

| Field | Meaning |
|---|---|
| `value` | Positive finite tok/s. |
| `workload` | `prompt_tokens`, `generated_tokens` per request, `requests` (1 or 8), and `timing`. A token count is an integer, a two-element inclusive range or `null` if the receipt omits it. Add `notes` for relevant conditions. |
| `timing` (inside `workload`) | `prefill` for reported prompt throughput, `ttft` for prompt tokens divided by time to first token, or `estimated` for wall-derived prefill. Use `decode` for the decode interval and `wall` for completion tokens divided by request or cohort wall time. |
| `commit`, `date`, `receipt` | Hex commit ID, ISO date and evidence reference. Ledger evidence uses a repository-relative Markdown document, optionally with an anchor. A historical entry with `receipt_revision: true` identifies the public receipt revision when the document lacks a pinned measured build. The revision date is used when the measurement date is unavailable. |

A failed cell contains `commit`, `date`, `receipt` and a `failure` string in place of the value and workload. `--report-json` emits the verdicts for processing. Keep distinct clock definitions and other workload differences in the receipt. Measurements with a different clock can surface a gap, but cannot replace an existing record through `--write`.

Prefill uses actual prompt counts in the bands 1,536–3,072, 6,144–10,240 and 24,576–40,960. Compare the same prompt and profile wherever possible. Some historical receipts use different prompt lengths, explicit route selection or a different clock. Those are recorded with the measurement so the integration can repeat the corresponding workload.

Add `record_rows` to the optimization catalog entry when a default-on speed claim applies to a ledger workload. Use the relevant engine, route, tier and metric. Scope an unrelated measured workload with `record_exclusion` and give the reason. Regenerate the Markdown view with `--write` after editing the JSON ledger.
