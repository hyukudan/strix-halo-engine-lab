# Decode anomaly measurement kit

The decode layout census reads only a GGUF header and reports eligible projection bytes plus active MoE union bounds:

```
python3 scripts/bench/turbo/decode-layout.py MODEL --verify-rows 8 --emitted-per-pass E
```

Its byte bounds are logical weight visits, not a DRAM counter. The census checks conversion dimensions and the current launch widths for every projection group, and credits savings only for eligible shapes at rows 1–8. Allocated but unused copies (including K=640 shared-down) are reported separately. Embedding lookup, draft, KV, activation traffic, cache residency and repeated loads are excluded.

Use the census shapes for full-model traffic accounting. The projection fixture uses representative shapes; its QSA input has 13,952 outputs, while the measured pack has 13,312 (12,288 + 512 + 512). Its microsecond figures are not whole-model pass times.

For small-row projection timing, build the existing `strix-hip-q8-rows-projection-tests` target. Run `STRIX_Q8_ROWS_ONFLY_BENCH=1 strix-hip-q8-rows-projection-tests` under the deployment's quiet GPU ticket wrapper. It converts/registers each fixture once and compares native with onfly at rows 1, 2, 4 and 8, in both exact and dense-A8 modes, with cold/warm GPU-event medians and route launch counts. A zero launch count is a fallback. Report unique-weight-byte GB/s as a streaming proxy, not measured DRAM bandwidth.

The source-controlled ticket runner and HTTP client are documented in [the decode-study workflow](decode-study.md). A deployment supplies only its model-specific server command adapter and assets; binary, model and prompt paths are parameters. Use the same binary, model, MTP bundle, prompts, slot budget and explicit numerical options in both arms. Count emitted tokens, retain invalid startup attempts, record both first and last prefill, use strict alternating pairs, and include the GPU busy fraction in clocks per token. Pre-cache files consistently before bounded tickets when startup requires it.

`STRIX_CYCLE_TIMING=1` reports completed host phase intervals; `STRIX_SERVE_RECEIPT=1` reports cycle draft/verification/host totals, rows and emissions. Verification includes commit and reconciliation; it is not pure target-kernel time. Kernel traces supply actual launch names and durations. Greedy draft acceptance cannot be inferred from target-to-target KL alone.

E/T study arms compare bundles: T also selects the turbo prefill policy, so their difference cannot isolate onfly weights. Use the projection fixture for projection diagnostics; bundle qualification requires separate native model-fidelity and complete-workload gates.

The `.levers` manifests are independent study selections, not generated profile-table contracts. For example, MiniMax's main study uses layers 40–59 and `fast_i4`; its profile table uses layers 0–59 and does not select that format. Record the emitted arms and inspect each engine's effective configuration before measuring. The optional final manifest column supplies explicit OFF assignments for omission arms (required when another lever enables the omitted member).

Study tags are single-use, including failed attempts: keep their outputs and choose a fresh tag. Ticket wrappers retain an atomic claim and reject existing runs or completion markers. Standalone C8 uses the inventory prompts and completes only after successful requests and analyses.
