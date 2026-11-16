# Same-pack comparison

`run-fair.py` and `summarize.py` collect the paired timing matrix. All model, executable and output locations
come from arguments. Keep one model family resident and use the device declaration/start gate.
Use `--plan` to list the fixed ABAB order, then `--run-index N` to execute one entry per reserved
FIFO window into the same output directory. Config and prompt hashes must remain identical;
the completion marker is written only after every plan entry has a complete receipt.
`--resume` keeps completed receipts only when config, prompt hashes and entry identities match; failed entries
must be preserved separately before retrying. Windows server trees use a close-on-owner-exit job, and each
load waits up to `--release-wait-s` (default 30) for GPU 0 and at least 20 GiB host RAM to be released.
For kernel attribution, `qwen_decode_bench ... N G --greedy-only` closes its CUDA profiler range after
the serial C1 emissions, excluding upload, initial prefill and the later teacher-window sweeps.
`--prefill-only` instead captures the initial `N-1` input-token prefill and emits no tokens. These
intrusive profiler scopes attribute kernels; cold/warm throughput comes from the paired serving matrix.

## Teacher-forced quality

The native teacher defaults to FP16 KV. Pass `strata-first.py --kv int8` for a separately
declared diagnostic of int8 serving arithmetic; its receipt records the effective arguments.
Keep that candidate separate from the FP16 control and the formal frozen quality gate.

Build the Release target `strix-cuda-qwen-logits`. It uses the production CUDA `Prefill` API without drafts,
consumes every input token in order, and stores only selected full-vocabulary rows in the existing TURBO
TMLLOG02 format. After any initial prefix before the first selection, this diagnostic accepts teacher
windows of 1..8 consecutive forced tokens; only windows containing a selected position copy full logits
to the host. It emits selected slices in order and checks every returned full row for finite values. Its
runtime is not a prefill benchmark. It accepts device declaration flags through the usual start gate.

Prepare a whitespace-separated ID file using the exact same tokenizer for every arm. For a Strata stride
of 256, create the selected positions and run our exporter (CAP must hold all input IDs):

```sh
python scripts/bench/stratacmp/quality.py positions --ids CASE.ids --stride 256 --out CASE.positions
strix-cuda-qwen-logits MODEL TABLES ROPE CASE.ids CASE.positions CAP E.bin 1024 8
```

The last argument is `STEP_ROWS`; omitted it preserves the scalar reference (`1`). The API accepts up to
8 decode rows (`qwen_prefill_driver.cu`, `Prefill::verify_cohort`); above 8 it changes to a padded prefill
shape. Keep the same teacher window, initial prefix, context capacity and prefill block across CUDA E/R/T
arms. Eight-row export is available without a kernel change; the default remains 1 until the GSQ raw-logit
1-vs-8 identity gate is measured. This preserves the established scalar numerical reference rather than
assuming the different launch shape is exact. Once that gate passes, use 8 for the whole paired run.
Record the actual `teacher batching:` engine line with `quality.py receipt`; `compare --receipt E=...`
`--receipt R=... --receipt T:ours-gsq=...` rejects different CUDA teacher settings or payload hashes.
Metrics without these receipts are labeled unverified diagnostics and cannot qualify an E/R/T gate.
Position zero is unsupported by CUDA `verify` because it requires nonempty prefilled state; frozen TURBO
spread positions start at 64. The stride diagnostic omits row zero consistently in both engines.

The full 48-parent, two-length corpus is roughly two million forced tokens; a completed pilot or one case
does not establish that the full kit ran. Report actual cases/rows and preserve partial-run status.

The existing `tools/qualification/turbo_bar/models.json` declares E as `qwen38-udq4kxl-exact` and R as
`qwen38-udq4kxl-R-exps3-v1` (controlled expert-block reduction from E). GSQ runs belong to named T arms;
do not replace the frozen R with GSQ. Reuse the existing frozen manifest/corpus, including their selections,
for a formal gate; the stride example above is a diagnostic only. Record binary/weights/tokenizer hashes
and resolved routes before running; never relabel a binary
as exact when it reports a FAST identity. Use an E repeat to quantify reference nondeterminism.

Run Strata offline with the same `CASE.ids`, `--max-new 1 --dump-logits strata.raw --logits-stride 256`,
`--gpu 0 --kv fp16 --suffix-draft 0 --lookup-max-t 0 --prompt-cache 0`, and the pack's native/PLE arguments.
Disable tail skipping and CPU prefill sharing through its documented quality controls. The dump must
contain a header and every expected row; an empty dump is a failure, never a successful quality run.
Convert and compare:

```sh
python scripts/bench/stratacmp/quality.py strata --raw strata.raw --ids CASE.ids --stride 256 --out strata.bin
python scripts/bench/stratacmp/quality.py compare --reference E.bin --arm R=R.bin --arm T:strata=strata.bin --out metrics.json
python scripts/bench/stratacmp/quality.py receipt --manifest frozen.json --arm E --spool E.bin --binary strix-cuda-qwen-logits --engine-log E.log --out E.receipt.json
python -m unittest discover -s scripts/bench/stratacmp -p test_quality.py
```

The native-pack route can bypass the CLI dump site. For that route, `strata-first.py` runs the engine's
offline stdio serve protocol with `STRATA_DUMP_FIRST_LOGITS`, issues one `GEN 1` per frozen corpus selection
at position p with exactly IDs `[0,p]`, and records each request hash, prompt length p+1 and true label
IDs[p+1]. It uses GPU index 0, checks 20 GiB free RAM, disables prefix caches, suffix/lookup drafts, tail
skipping and CPU prefill sharing, and retains all source rows. Run it inside an authorized GPU window:

```sh
python scripts/bench/stratacmp/strata-first.py --config strata.json --manifest frozen.json --corpus CORPUS --case CASE --out NEWDIR --timeout 120
```

The runner's timeout includes loading; size the selected case/window accordingly. It stops only its own
child process. It rejects a changed corpus hash or selected-ID/position file. The output `logits.bin` is
directly comparable with CUDA export for that exact corpus case and positions file. `index.json` binds the
actual effective args and request order. The runner does not create/replace R or certify a statistical pass.
For a bounded device ticket, add `--selection-start N --selection-count K` to collect a contiguous slice
of the unchanged frozen selections. The runner writes `selected.positions` for the CUDA exporter and
records the original selection hash, interval, total count and explicit partial status in `index.json`.
Use identical slices for every arm. A completed shard does not establish a completed case or quality gate.
`quality.py strata-first --prefix PREFIX --ids CASE.ids --positions CASE.positions --out strata.bin` also
converts previously collected `PREFIX.0`, `PREFIX.1`, ... rows when their external request-order record
proves this same conditioning contract.

The converter drops row zero (unsupported by CUDA's state API) and the final prompt row because its next generated token is not a true corpus label.
All earlier selected positions retain their true next-token label. It rejects wrong dimensions, incomplete
payloads and nonfinite logits. Comparison reuses TURBO's full-vocabulary KL(E||arm), top-1 disagreement
and dNLL calculations; it rejects differing input IDs, positions, labels, or row counts. The output reports
agreement, signed dNLL(arm minus E), bitwise identity and payload hashes.
`receipt` verifies the frozen binary/weight hashes and records the declared arithmetic/identity, actual
engine log, input-token hash, selected positions and full-logit payload hash. It performs no execution and
does not claim the diagnostic adapter is the existing HIP adapter or certify a statistical gate.

These per-case metrics are diagnostics, not a statistical TURBO pass. The frozen TURBO E/R/T protocol
still requires its corpus/task inventories, identity receipts, E repeat, independent parent samples and
paired confidence intervals. Its current HIP runner/Strix HTTP identity adapter cannot certify a foreign
server directly. Preserve the raw outputs and identify this limitation when reporting a partial gate.

Home: shared (host shared: spool/metric bridge); specific (CUDA Qwen diagnostic binding through
`apps/cuda_qwen_logits_main.cu`, `qp::Config` and `qp::Prefill`, whose state API is Qwen-specific).
