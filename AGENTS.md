# Documentation and research publication policy

Keep comparative evaluations, external benchmark records, review correspondence and
source-identified external inspection artifacts outside Git. Public documentation
describes project-owned contracts, independent mathematical semantics, implementation
status and qualification evidence without naming unrelated runtime projects.

Preserve accurate provenance, licenses and notices for actual adopted code, dependencies,
model assets and official mathematical/ISA references. Neutralization must not erase
required attribution or manufacture independent authorship.

Archive original comparative inputs privately before editorial cleanup. Do not rewrite
repository history as part of ordinary documentation editing. Numerical proposals remain
drafts until their compiler, native execution, model fidelity and complete-workload gates
pass; theoretical throughput and external results do not establish local qualification.

# Production defaults for humans and coding agents

The fast path is the default for all five public engines. Start with the plain command; let the
resolver select qualified production settings. Do not pass legacy opt-in organization flags that
are already defaults: they can force an explicit route and decline the automatic one.

| Engine | Recommended command / contract | Inspect | Baseline |
|---|---|---|---|
| Qwen | `strix-hip-decode-token MODEL.gguf $(cat prompt.ids) --generate 256`; [auto route](docs/getting-started/engine-reference.md#command-line) | `--print-effective-config` | `--baseline` / `STRIX_BASELINE=1` |
| DeepSeek pair | `strix-deepseek41-serve MODEL.gguf --rank R --transport usb4auto`, one rank per node; [production defaults](docs/models/deepseek-v4.1/production-defaults.md) | `--print-effective-config` | `STRIX_DS41_BASELINE=1` |
| DeepSeek RTX | Use the [standalone command and required receipt/state/reference assets](docs/models/deepseek-v4.1/rtx-standalone.md#run-and-persistent-worker) | `--print-config` | `--baseline` / `STRIX_DS41_RTX_BASELINE=1` |
| MiniMax | `strix-minimax-rank MODEL_DIR R usb4-auto OUTPUT_DIR`, one rank per node; [22 defaults](src/minimax/engine/mm_defaults.cpp) | `--print-effective-config` | `STRIX_MM_BASELINE=1` |
| GLM | `strix-glm5-rank MODEL_DIR --world 2 --rank R --transport usb4-auto --prompt PROMPT.u32 --decode 64`, one rank per node; [resolver](apps/glm_rank_defaults.hpp) | `--print-effective-config` | `--baseline` / `STRIX_GLM_BASELINE=1` |

Inspection exits before device work; preserve the engine's required assets and two-rank launch contract
for execution. Check [all item guards, fallback notices and off switches](docs/optimizations.md) or
the [JSON catalog](docs/optimizations.json). Qwen auto fast prefill currently excludes MTP and HTTP serving.
HTTP servers (Qwen, DeepSeek) describe what they execute at `GET /v1/capabilities`; the `strix` object of each response
states the applied policy (requested vs effective sampling). Generation fields the server does not execute are refused
with a 400 naming the field ([list](docs/engine/serving/serving.md#chat-requests)).
Fast FP16 Qwen prefill is its own identity, not dense-A8 serial bit equality. GLM defaults to R1b decode
with R1 prompt prefill; `--identity r1` restores R1. Exactness references must match that identity.

Use each engine's built-in exactness gates and compare complete token IDs and logits/state digests.
Qwen: profile-pair qualification and operator byte checks; DeepSeek pair: both-rank prefill/greedy/verify/batch
and transport gates; RTX: replica canonical/pressure/plain-speculative chat gates; MiniMax: golden logits,
decode/verify/exchange checks; GLM: G3/G4, identity, prefill and speculative-oracle gates.
Once the bench kit lands, inspect availability and run:

```sh
bash scripts/bench/bench.sh list
bash scripts/bench/bench.sh check --engine E --only exactness --suite quick --json
bash scripts/bench/bench.sh run --engine E --only exactness --suite quick --json
```

Use the engine key reported by `list`. Verdicts are `EXACT`, `MISMATCH` and `NO-REFERENCE`;
missing reference data is not an exactness pass. Do not assume the planned kit is installed today.

For A/B, change one off switch at a time, preserve model/profile/state/capacity, restart startup-cached
readers and both ranks as needed, verify tokens/digests first, then compare medians from interleaved runs.
Do not add individual gains from different workloads or call a changed numerical identity bit-identical.
Keep deployment endpoints, machine/account identifiers, private paths, prompts, receipts and unpublished
benchmark/review material out of public changes. Use relative asset placeholders in commands and public
implementation or qualification contracts for links.
