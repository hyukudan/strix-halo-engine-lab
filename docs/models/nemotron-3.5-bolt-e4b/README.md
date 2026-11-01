# Nemotron 3.5 Bolt E4B (gemma4 family)

NVIDIA Nemotron 3.5 Bolt E4B is a post-trained Gemma 4 E4B (`model_type: gemma4_text`): 42 layers, hidden 2560,
GeGLU FFN 10240, 35 sliding-window layers (window 512, head dim 256) and 7 global layers (head dim 512, proportional
partial RoPE), per-layer embeddings (PLE), shared KV in the last 18 layers, logit softcap 30, tied head, vocabulary
262144, 64K context. The engine runs it natively as the `gemma4` family (`STRIX_FAMILY_GEMMA4=ON`).

## Profiles

One switch selects everything: `--profile default|fast|quality` (default: `default`). An explicit option or
environment variable always wins over the profile. Every receipt records the profile, each lever's value and source,
and any pending row. A pending (not yet qualified) row is never applied: the profile runs every ready row, the pending
row keeps the exact behaviour, the receipt lists it with its declared target, and one stderr line says so. A profile
refuses only for a missing required asset (`quality` without `--exl3-arm`) or an explicit request for something this
build does not contain.

| Profile | Decode | Prefill | Guarantee |
|---|---|---|---|
| `default` | `g4-v1` (FP32 activations, Q4_K_M weights) + MTP drafter at C1 | `prefill-exact` | Exact: output bytes equal the declared serial path for every batch size (C4 == C1) and with the drafter on or off |
| `fast` | `g4-a8-v1` (int8 activations, qualified) + MTP drafter | WMMA prefill (pending `g4-wmma-pf-v2`; exact prefill runs until then) | Named profile, qualified against frozen fidelity bands |
| `quality` | EXL3 pack (`--exl3-arm DIR`), identity chosen once per pack and session (`g4-exl3-v1`/`v3`) | `prefill-exact` | Exact within its EXL3 identity; higher fidelity than Q4_K_M at similar size |

The identity of the EXL3 route is fixed at startup from the pack and `--streams`, never per step, so a session's
output does not depend on load. `STRIX_G4_EXL3_IDENTITY=v1|v2|v3` forces it.

## Assets

- Model: the BF16 safetensors directory (tokenizer, chat template) and, for `default`/`fast`, the Q4_K_M GGUF.
- Rotary tables, generated once offline (see `src/gemma4/ref/README.md`):
  `python src/gemma4/ref/make_rope_tables.py rotary-tables --positions 65536 --workers 6`
- Drafter: an MTP assistant package in `<model dir>/drafter-mtp/` (config, weights, LICENSE, NOTICE, LINEAGE,
  MANIFEST) is found automatically; the published drafter is [hyukudan/Nemotron-3.5-Bolt-E4B-MTP-Drafter](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-MTP-Drafter). `--draft-mtp DIR` points elsewhere; `--draft off` or `STRIX_G4_DRAFT=0`
  unloads the drafter (prompt lookup, below, still drafts); `STRIX_G4_LOOKUP=0` turns prompt lookup off and
  `--dflash off` turns all drafting off.
- EXL3 packs (`quality`): one directory per pack arm, loaded with `--exl3-arm DIR` next to the BF16 model; the
  published pack is [hyukudan/Nemotron-3.5-Bolt-E4B-EXL3](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-EXL3) (M4).

## Serving

```
strix-gemma4-serve MODEL --rope-tables rotary-tables --tokenizer MODEL/tokenizer.json --profile default
```

With `MODEL/drafter-mtp/` present, single-request decode drafts automatically; output is identical to drafting off.
With more than one active request the drafter is admitted only where its measured cost pays off.
Single-request decode also drafts by prompt lookup (the continuation of the longest earlier match of the last tokens
in the prompt and output), verified token by token, so output is identical to drafting off. Beside the drafter a
per-request selector uses it only on a strong match (equal to the drafter alone within noise); without a drafter,
agent-style prompts that copy from their context gain most (C1 `default`: 114.9 vs 48.2 tok/s). `STRIX_G4_LOOKUP=0`
turns it off.
`--print-effective-config` (decode) prints the resolved profile and switches without loading a request.

## Measured (one Strix Halo, gfx1151, 128 GB, Q4_K_M)

Decode tok/s, `default` profile, 256-token English chat: C1 56.0 plain, 92.9 with the MTP drafter (ids identical to
drafter off); C4 169.3 aggregate; C8 238.5 aggregate. Exact prefill uses FP32 arithmetic without contraction and is
compute-bound (471 tok/s at 512 tokens, 358 at 32K); the WMMA prefill of the `fast` profile is pending qualification.
All profiles, prefill lengths and the same-machine llama.cpp reference are in the
[main README](../../../README.md#nemotron-35-bolt-e4b).
Numbers vary with node load (shared LPDDR5X bandwidth); compare runs from the same session only.

## Off switches

Every speed lever that keeps bytes identical is on by default with an `STRIX_G4_*=0` switch that restores the
previous exact path; the active switches appear under `off_switches` in each receipt. Module READMEs list them:
`src/gemma4/kernels`, `src/gemma4/prefill`, `src/gemma4/exl3`, `src/gemma4/eagle`, `src/gemma4/engine`.
