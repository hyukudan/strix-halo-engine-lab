# Decode anomaly measurement kit

The decode layout census reads only a GGUF header and reports eligible projection bytes plus active MoE union bounds:

```
python3 scripts/bench/turbo/decode-layout.py MODEL --verify-rows 8 --emitted-per-pass E
```

Its byte bounds are logical weight visits, not a DRAM counter. The present launcher leaves K=640 shared-down copies unused; the census reports them separately. Embedding lookup, draft, KV, activation traffic, cache residency and repeated loads are excluded.

Use the census shapes for full-model traffic accounting. The projection fixture uses representative shapes; its QSA input has 13,952 outputs, while the measured pack has 13,312 (12,288 + 512 + 512). Its microsecond figures are not whole-model pass times.

For small-row projection timing, build the existing `strix-hip-q8-rows-projection-tests` target. Run `STRIX_Q8_ROWS_ONFLY_BENCH=1 strix-hip-q8-rows-projection-tests` under the deployment's quiet GPU ticket wrapper. It converts/registers each fixture once and compares native with onfly at rows 1, 2, 4 and 8, in both exact and dense-A8 modes, with cold/warm GPU-event medians and route launch counts. A zero launch count is a fallback. Report unique-weight-byte GB/s as a streaming proxy, not measured DRAM bandwidth.

For draft/verification diagnostics, `mtp-study-ticket.py PLAN.json` runs declared arms inside one existing quiet ticket (112 seconds internally). It uses the same deployment variables as the decode study. Each arm declares `tag`, `args` (simple option/value words), optional `env` assignments, `requests` (default `C1,C1`), `tokens` (default 192), and `prompts` (a module beside the runner). `mtp-study-prompts.py` supplies code, prose, JSON and editing requests; use `S4,S4` for separate cold/warm observations. The runner rejects incomplete requests and invalid cycle accounting.

`STRIX_MTP_WEIGHT_LOG=1` prints actual draft source/transform groups and target verify/commit groups. `STRIX_MTP_VERIFY_LOG=1` records every evaluated proposal row, including rows after a rejected prefix; `reached`, `eligible`, `accepted`, and `greedy_accept` distinguish prefix acceptance from teacher-forced equality. Logs carry monotonic timestamps for request attribution. These diagnostics change no arithmetic when disabled.

An arm with `logits: true` writes a full target-row dump using `STRIX_MTP_LOGITS_DUMP`. Enable `TURBODEC_KEEP_TOKEN_IDS=1` to retain the completion ledger. `mtp-study-analyze.py RUN_E RUN_T --logits E.logits T.logits --out result.json` computes per-depth acceptance, joint proposal/target agreement and KL only for identical committed and speculative token contexts. The dump's window-input hash alone does not establish a full-context match. Forced full-chain/one-commit diagnostics are excluded from ordinary accounting gaps and must never supply qualification timings.

The shared IU4 R7/R8 candidate is controlled by `STRIX_IU4_R8_SINGLE_DECODE=1` (default off pending resource/native/timing qualification). The projection fixture byte-compares it against both R4 halves before reporting either policy's timing. Inspect compiled scratch/spills and LDS as well as elapsed time. Wider tensor conversion is a distinct, bit-changing quality decision; the census does not enable it.

The source-controlled ticket runner and HTTP client are documented in [the decode-study workflow](decode-study.md). A deployment supplies only its model-specific server command adapter and assets; binary, model and prompt paths are parameters. Use the same binary, model, MTP bundle, prompts, slot budget and explicit numerical options in both arms. Count emitted tokens, retain invalid startup attempts, record both first and last prefill, use strict alternating pairs, and include the GPU busy fraction in clocks per token. Pre-cache files consistently before bounded tickets when startup requires it.

`STRIX_CYCLE_TIMING=1` reports completed host phase intervals; `STRIX_SERVE_RECEIPT=1` reports cycle draft/verification/host totals, rows and emissions. Verification includes commit and reconciliation; it is not pure target-kernel time. Kernel traces supply actual launch names and durations. Greedy draft acceptance cannot be inferred from target-to-target KL alone.
