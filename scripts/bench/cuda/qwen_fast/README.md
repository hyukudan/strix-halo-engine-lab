# CUDA Qwen FAST gate kit

Quality gate of the FAST prefill profile (`--profile fast`) against the exact CUDA path, over teacher-forced logits.
Everything is a parameter: no host names, users or private paths.

Pieces
- `tests/cuda/qwen_tf_logits.cu` (CMake target `qwen_tf_logits`): teacher-forced logit dump. Engine and profile options go
  through the same parser and resolver as `strix-cuda-qwen-serve` (`include/strix/cuda/qwen_options.hpp`, serve role), so
  the gated configuration is the one the server runs.
- `gate.sh`: runs the exact and the fast arm, seals both with `tools/strix-gate/fast_logit_gate.py`, compares, prints the
  KL / top-1 / dNLL verdict (limits default to KL <= 0.02, top-1 >= 0.93, dNLL <= 0.01).
  Inherited numerical overrides are cleared; the reference forces every form OFF. Each executing arm writes
  `effective-config.json` only after all prompts and repetitions succeed. It records resolved forms and per-run
  forms dispatched by the driver and synchronized successfully; inactive forms remain false in each execution.
  The gate validates these forms/contract and includes the JSON in the seal. Every FAST execution must include
  target-changing arithmetic: HC, QSA, MoE MMA or BF16 MoE routing. MTP-only arms are rejected even with a sidecar.
- `prefill_abab.sh` + `prefill_client.py`: strict ABAB server prefill arms (fresh server per arm, cold and warm prefill
  per length, GPU clock / busy samples). Env: SERVE, MODEL, TABLES, ROPE, IDS, OUT, ARMS (`name|ENV=v|--flags ;; ...`),
  optional MTP, VOCAB, PORT, GPU_INDEX, LIMIT_S, CLIENT_ARGS, COMMON; summary in `$OUT/summary.txt`. Run it inside the GPU
  ownership wrapper of your lab.
  The client owns and cleans up only its launched child; occupied ports and early child exits fail the arm.
  Prefix caching is forced OFF and samples require a reported zero cached-token count. Supply at least
  `runs * sum(lens)` input IDs: slices are disjoint, never truncated or wrapped (81,920 IDs for the defaults).
- `ident.sh`: byte identity of teacher-forced logit dumps between arms (`--arm 'NAME|ENV=v|--flags'`, the first arm is the
  reference). Fresh run directories, execution receipts and complete, size-checked dumps are required; exit 0 means
  dump equality only. Kernel-variant admission must be recorded and checked separately before qualification.
  Requires Python 3; labels use letters, digits, underscores and hyphens, with case-insensitive uniqueness.
- `mkprompt.py`: builds `NAME.u32` token files from .u32 files, id lists or text (text is tokenised with `strix-tokenize`,
  the server's tokenizer, against a GGUF; nothing is downloaded).

Use

    python3 mkprompt.py --out prompts --tokenize build/strix-tokenize --model MODEL.gguf a=a.txt b=b.txt
    ./gate.sh --tf build/qwen_tf_logits --model MODEL.gguf --tables TABLES --rope-table ROPE.bin \
              --mtp-sidecar MTP.gguf --prompt a=prompts/a.u32 --prompt b=prompts/b.u32 --out gate-out \
              --rows 64 [--wrap "gpu-wrapper --minutes 5 --"] [--fast-args "--hc-mma off"]

`qwen_tf_logits` alone (two runs give cold and warm prefill timing; size/byte mismatches fail with the prompt,
zero-based run and first differing element, before publishing a successful receipt):

    qwen_tf_logits --model M --tables T --rope-table R --profile fast --prompt a=a.u32 --out dumps --rows 64 --runs 2

`--print-effective-config` prints the resolved profile (levers with status, forms, serve differences, arithmetic contract).
Each prompt needs rows + 2 <= tokens <= `--max-positions` (default 32768). Dump layout: `q-NAME-tf-r0.f32` = [rows][vocab] FP32.
Exit codes of gate.sh: 0 pass, 1 fail, 2 identity or input error.
`--arms exact,fast,compare` (the default) selects the steps. To keep one model load per time-capped GPU turn, run `--arms exact`
then `--arms fast,compare` with the same `--out`. A compare refuses (rc 2) if a sealed arm dir is missing.
CPU test: `python tests/qwen_fast_gate_arms_tests.py`.

Decode gate (FAST vs exact on the reference's OWN continuation, decode windows): the exact arm generates and dumps in one
load, `qwen_tf_logits --profile exact --generate 128 --rows 129 --window 4` (greedy 1-row `step()`s; writes
`q-NAME-gen.u32` and the teacher-forced token file `q-NAME-tf.u32` = prompt + the first 128 generated ids, then dumps 129
rows: the last prompt position and every generated id, in `step()` windows of 4 rows); every other arm runs plain
teacher-forced mode over `--prompt NAME=q-NAME-tf.u32 --rows 129 --window 4` and is compared with `fast_logit_gate.py
compare` (pooled and per prompt). `--generate G --rows 0` only generates (free-running agreement between arms).

This target-logit gate does not qualify MTP draft/cache fidelity or speculative acceptance. MTP forms remain
provisional until a separate draft/cache check and a complete speculative-workload gate pass on native hardware.

FAST bundle `qwen-cuda-fast-bundle-v1` (`--profile fast`: FAST experts `gsq-iq-bf16mma-cuda-v1`, `hc_mma`, `qsa_mma`
`qwen-cuda-qsa-bf16mma-v2`, prompt MTP rows deferred) is opt-in and provisional (`kFastBundleQualified = false`).
- Not bit-identical to the exact profile in decode: the FAST experts also run in the decode step and MTP verify windows,
  so they change the logits of generated tokens. `hc_mma` and `qsa_mma` run on prefill blocks only; they reach decode
  through the prompt's K/V and recurrent state.
- Gate coverage so far: 2K and 8K prose (pooled and per member), 32K pooled (one 32K prompt is below the per-prompt
  top-1 bound), and short code / structured prompts (pooled and per prompt). Long code and other languages are not gated.
- The arithmetic contract of a FAST run with forms carries the forms' versioned identities (`;ids=...`).
- `hc_mma` (`qwen-cuda-hc-f16mma-v2`) shifts the FP16 inject operand by one per-model power of two (undone exactly in
  the FP32 epilogue), so no non-zero inject weight rounds to FP16 zero; the load prints the range and the shift. It is
  refused at load, with a reason, only when the HC norm weights break the FP16 activation bound, a weight is non-finite,
  or the inject range exceeds what a shift can represent (about 2^40). v1 dropped 6 weights of the Qwen3.8 pack.
