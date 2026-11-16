# CUDA Qwen FAST gate kit

Quality gate of the FAST prefill profile (`--profile fast`) against the exact CUDA path, over teacher-forced logits.
Everything is a parameter: no host names, users or private paths.

Pieces
- `tests/cuda/qwen_tf_logits.cu` (CMake target `qwen_tf_logits`): teacher-forced logit dump. Engine and profile options go
  through the same parser and resolver as `strix-cuda-qwen-serve` (`include/strix/cuda/qwen_options.hpp`, serve role), so
  the gated configuration is the one the server runs.
- `gate.sh`: runs the exact and the fast arm, seals both with `tools/strix-gate/fast_logit_gate.py`, compares, prints the
  KL / top-1 / dNLL verdict (limits default to KL <= 0.02, top-1 >= 0.93, dNLL <= 0.01).
- `prefill_abab.sh` + `prefill_client.py`: strict ABAB server prefill arms (fresh server per arm, cold and warm prefill
  per length, GPU clock / busy samples). Env: SERVE, MODEL, TABLES, ROPE, IDS, OUT, ARMS (`name|ENV=v|--flags ;; ...`),
  optional MTP, VOCAB, PORT, GPU_INDEX, LIMIT_S, CLIENT_ARGS, COMMON; summary in `$OUT/summary.txt`. Run it inside the GPU
  ownership wrapper of your lab.
- `mkprompt.py`: builds `NAME.u32` token files from .u32 files, id lists or text (text is tokenised with `strix-tokenize`,
  the server's tokenizer, against a GGUF; nothing is downloaded).

Use

    python3 mkprompt.py --out prompts --tokenize build/strix-tokenize --model MODEL.gguf a=a.txt b=b.txt
    ./gate.sh --tf build/qwen_tf_logits --model MODEL.gguf --tables TABLES --rope-table ROPE.bin \
              --mtp-sidecar MTP.gguf --prompt a=prompts/a.u32 --prompt b=prompts/b.u32 --out gate-out \
              --rows 64 [--wrap "gpu-wrapper --minutes 5 --"] [--fast-args "--hc-mma off"]

`qwen_tf_logits` alone (two runs in one process give cold and warm prefill timing; the dumps of the runs are compared bit for bit):

    qwen_tf_logits --model M --tables T --rope-table R --profile fast --prompt a=a.u32 --out dumps --rows 64 --runs 2

`--print-effective-config` prints the resolved profile (levers with status, forms, serve differences, arithmetic contract).
Each prompt needs rows + 2 <= tokens <= `--max-positions` (default 32768). Dump layout: `q-NAME-tf-r0.f32` = [rows][vocab] FP32.
Exit codes of gate.sh: 0 pass, 1 fail, 2 identity or input error.
