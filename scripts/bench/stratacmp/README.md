# Same-pack comparison

`run-fair.py` and `summarize.py` collect the paired timing matrix. All model, executable and output locations
come from arguments. Keep one model family resident and use the device declaration/start gate.

## Teacher-forced quality

Build the Release target `strix-cuda-qwen-logits`. It uses the production CUDA `Prefill` API without drafts,
consumes every input token in order, and stores only selected full-vocabulary rows in the existing TURBO
TMLLOG02 format. After any initial prefix before the first selection, this diagnostic uses scalar decode;
its runtime is not a prefill benchmark. It accepts device declaration flags through the usual start gate.

Prepare a whitespace-separated ID file using the exact same tokenizer for every arm. For a Strata stride
of 256, create the selected positions and run our exporter (CAP must hold all input IDs):

```sh
python scripts/bench/stratacmp/quality.py positions --ids CASE.ids --stride 256 --out CASE.positions
strix-cuda-qwen-logits MODEL TABLES ROPE CASE.ids CASE.positions CAP E.bin 1024
```

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

The converter drops the final prompt row because its next generated token is not a true corpus label.
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
