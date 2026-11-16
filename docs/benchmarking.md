# Measurement kits: what each one measures and how to run it

[English](benchmarking.md) · [Español](benchmarking.es.md) · [Main README](../README.md)

The public benchmark (`scripts/bench/bench.sh`, [reproduce.md](reproduce.md)) gives one table per engine. The
kits below are the lab's day-to-day measurements: one arm against another, in strict ABAB order, with every run
checked for validity. They live in the repository so they survive any cleanup of the machines that run them.
Nothing in them names a host, a user or a private path: binaries, models and prompts are arguments, and the
second node of a pair is the ssh alias in `KIT_PEER`.

## Shared runner (`scripts/bench/common/`)

| File | Role |
|---|---|
| `ticket.sh OUT KIT [ARGS]` | The one command a GPU queue ticket runs. Budget = `GPU_TURN_MAX` (default 120 s) minus `KIT_MARGIN` (5 s); arms that no longer fit are skipped. Ends with `OUT/summary.txt` (at most 60 lines) and `OUT/.done`, so the caller waits once and reads one file. |
| `runlib.sh` | Sampler, page-cache warm, binary check across the two nodes, budget, `kit_run`, `kit_pair_arm` (rank 1 on the peer, rank 0 here). |
| `run1.sh DIR CAP -- CMD` | One measured process with its receipt, also run on the peer for rank 1. |
| `receipt.py` | Parses one run directory into `receipt.json` and one summary line (rules below). |
| `abab.py` | `order` prints the strict ABAB order; `summary OUT [--metric M]` gives per-arm medians, per-round deltas and identity checks. |
| `serve-decode.sh` | Starts a server (and its rank 1 on the peer), waits until it is ready, runs `tools/serve_bench.py` per decode class, then stops it. `--client-cmd` runs any other client instead. |
| `serve_receipts.py` | Turns served results into receipts (`agg_C<n>`, `e2e_C<n>`, `req_C<n>`, `ttft_C<n>`). |
| `forwards_ab.py` | Steady decode cycles by (lanes, target forwards) from the cycle receipts: the in-process A/B of `STRIX_SERVE_JOINT_GROUP_LANES=ab`. |
| `prompt_ids.py` | Prompt files for any length (`p<N>.ids`, `.csv` or `.u32`). Each one is the seed text tokenized once, repeated and truncated, with a sha256 per prompt. |

Every run directory holds `cmd.txt`, `stdout`, `stdout.ts`, `stderr`, `stderr.ts` (the arrival time of every line), `samples.tsv`,
`klog.txt` and `meta.json`, plus `receipt.json` once it has been summarised.

**Validity** (a run that fails any check is listed with its reasons and left out of every median):

- the exit code is not 0;
- kfd `evicted_ms` of the run's processes grew inside the timed window. Growth during model load is only a warning; `--evict-whole-run` makes it fatal, as the decode kits do;
- `MemAvailable` fell below `KIT_MIN_AVAIL_GB` (default 2);
- `--require` lines are missing, or `--reject` lines are present (defaults: `what()`, `timed out`, and `declined` outside cache-policy lines).

Kernel-log events (amdgpu, kfd, thunderbolt, oom, reset) are printed as warnings, or count as failures with `--strict-klog`.

**Warm and cold.** When a process runs two or more prefills, the first one is reported as cold and the last as
warm. The two figures are never mixed. **GPU clocks per token** = busy × clock × window / tokens, with power and
clock integrated over the timed window. On an NVIDIA host the sampler reads `nvidia-smi`, and host memory and
kfd checks are reported as not sampled.

**Arms** are written `'NAME|EXTRA CLI ARGS|ENV=V ENV2=V'`. Inherited `STRIX_*` variables are dropped, so an arm only
sees the levers it names. To get several rounds across tickets, give every ticket the same `OUT`: each summary
covers every receipt in `OUT`, and `abab.py` pairs the arms by `--round`.

**Running a kit**, here on a node through its queue (build first in its own command, never in the same shell as the
ticket):

```
<queue>/gpu-turn.sh LABEL scripts/bench/common/ticket.sh OUT <kit> [kit args]
```

## Kits

| Kit | Measures | Command (inside `ticket.sh OUT ...`) |
|---|---|---|
| `qwen/length-matrix` | Qwen prefill, length × mode × profile: `full` (one node), `pair` (two ranks), `full2` (the same arm on both nodes at once). Optional page-cache warm (`--warm none` skips it). Long context (YaRN, 262K to 1M) through an arm with `--context N`. | `qwen/length-matrix --bin strix-hip-decode-token --model Q.gguf --ids DIR --sizes 8K,32K,128K,256K --modes full,pair --rounds 3` (arms default `exact`, `fast`) |
| `qwen/serve-decode` | Qwen served decode, classes prose / edit / json at 1, 4 and 8 chats; one arm per ticket. | `qwen/serve-decode --bin B --model Q.gguf --arm 'wide\|--mtp-ngram-draft=wide\|' --round 0` |
| `ds41/prefill-pair` | DeepSeek V4.1 pair prefill, length × rows × arm, cold and warm in one process (`--ttft-timing --repeat 2`); the logits FNV of both ranks must agree. | `ds41/prefill-pair --bin strix-deepseek41-prefill --model M --ids DIR --sizes 8K,32K --arm 'exact\|\|'` |
| `ds41/serve-decode` | DeepSeek V4.1 pair served decode, classes prose / edit / json at 1, 4 and 8 chats (server cycle receipts, end-to-end, TTFT). | `ds41/serve-decode --bin-dir DIR --model M --arm 'ctl\|\|' --round 0` (then `abab.py summary OUT --metric e2e_C8`) |
| `ds41/dense-forms` | Shared dense policy `dense-q8a8-f16-v1` weight-decode forms (scalar FP32 / packed FP16) on the DS41 rank-0 chunk shapes: exhaustive stager decode gate, bit identity of every form, strict ABAB kernel times and the per-layer-chunk census. | `ds41/dense-forms --bin strix-hip-ds41-fast-dense-tests [--rows 1024,680]` |
| `glm/prefill-pair` | GLM pair prefill, exact vs FAST, several prompts in one load (`--pf-ttft`), first prompt cold and last warm; rank agreement on logits and state FNV. | `glm/prefill-pair --bin strix-glm5-rank --model DIR --prompts p8192.u32,p8192.u32` |
| `minimax/fast-gate` | MiniMax FAST quality gate using the admitted absolute dNLL rule: reference and candidate, both ranks, then `tools/strix-gate/fast_logit_gate.py` seal and compare (`--dnll-rule abs`). | GPU: `minimax/fast-gate run --bin strix-minimax-rank --model DIR --arm 'cand\|\|STRIX_X=1' --sessions s8k=p.u32`; CPU: `fast-gate.sh seal ...`, `fast-gate.sh compare --ref ref --cand cand --dnll-rule abs` |
| unit: MiniMax index projection | Exact slab vs FAST `minimax-index-proj-wmma-v1` (direct, x16 tiles, diagnostics), strict ABAB 3 pairs at 1,024 / 2,048 / 4,096 / 8,192 rows, fixed first 12 (4,096/8,192 rows) or 30 (1,024/2,048 rows) of 60 weight sets; x16-only timings require complete byte equality with direct WMMA; historical x16-only figures are withdrawn pending recollection, cold first call, kfd `evicted_ms`. | `STRIX_MMATTN_BENCH=wmma STRIX_MMATTN_WMMA_SERIES=x16 index-proj-tests` (build line in `src/minimax/attn/index_proj_tests.hip`) |
| `vision/battery` | Vision content battery: one greedy answer per synthetic image, graded from the pixels (`score.py`): colours present or absent, pattern, orientation, palette rule. | `vision/battery --bin-dir DIR --model M --vision V.gguf --images IMGDIR`, then `python3 scripts/bench/vision/score.py IMGDIR OUT/*/battery.json` |
| `rtx/ds41-rtx-pair-batch` | DeepSeek V4.1 RTX + pair batched decode C2 / C4 / C8: RTX FRONT on `[0, cut)` with `strix-cuda-ds41-multi-front`, pair TAIL on `strix-deepseek41-batch --hetero-listen`, 2 groups in flight; `--mode both` checks every request against its pair-only run. | node ticket: `rtx/ds41-rtx-pair-batch tail --bin strix-deepseek41-batch --model M --prompts a.csv,...,h.csv --token-file TOK --n 2,4,8 --generate 48`; RTX host: `scripts/bench/rtx/ds41-rtx-pair-batch.sh front --bin strix-cuda-ds41-multi-front --model M --stage NODE:7841 --token-file TOK` |
| `rtx/serve-route` | Served RTX routes (Qwen whole model on one RTX, or any OpenAI-compatible RTX server), decode classes at 1, 4 and 8 chats. | `scripts/bench/rtx/serve-route.sh --route qwen --bin strix-cuda-qwen-serve --model Q.gguf -- --tables T --rope-table R --mtp-sidecar S --draft-vocab V` |
| `kfdevict` | KFD queue evictions without root: in-process probe of the process's own eviction events (trigger, restore) and userptr allocations, node sampler of `evicted_ms` against reclaim / compaction counters and GPU busy windows, a controlled memory-pressure generator, and a micro A/B of host-memory backings. | `scripts/bench/kfdevict/README.md` |

The RTX DeepSeek prefill service (`strix-cuda-ds41-prefill-service`) is measured with
`tools/ds41_prefill_service_client.py`. The routes themselves are described in [discrete-gpu.md](discrete-gpu.md).

The decode-class prompts are in `scripts/bench/prompts/classes/`. Drafting acceptance differs by an order of
magnitude between prose, a code edit and CSV-to-JSON, so served decode is always reported per class.

TURBO quality bar: `tools/qualification/turbo_bar/` ([README](../tools/qualification/turbo_bar/README.md)) measures
E (exact route) against R (a smaller pack of the same weights on the same kernels) and T (TURBO): KL, top-1 and
dNLL on full logits, scored tasks, retrieval, paired 95 % intervals (INCONCLUSIVE when unresolved) and a
speed/quality frontier. `rpack_kquant.py` produces R; `models.json` declares E/R/T per model.

## Checks

`scripts/bench/tests/run.sh` checks the syntax of every kit, runs shellcheck when it is installed, and runs
`scripts/bench/tests/test_runner.py`, offline tests of the receipts, validity rules, ABAB summary, served receipts
and prompt formats.

Vision scores are synthetic-image word-matching heuristics, not a general image-understanding accuracy measure. ABAB comparisons require an explicit `--base ARM` within a comparable workload group.

Served runs share the ticket deadline and require both ranks, telemetry and completed requests to pass. Custom clients must write `KIT_RUN_DIR/client-result.json` with positive `completed` and zero `failed`, and exit successfully. Intentional TERM shutdown is recorded separately from server failure. Missing prefill timings remain unqualified; request latency is a separate metric.
