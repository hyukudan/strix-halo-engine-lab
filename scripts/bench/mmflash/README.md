# mmflash: MiniMax MSA flash layer-range qualification kit

Runs the MiniMax pair prefill gate for the FAST MSA flash attention (`STRIX_MM_PF_MSA_FLASH=1|2` on a layer range
`STRIX_MM_PF_MSA_FLASH_LAYERS`, shared core `src/attn_core/sparse_flash_keysplit.hpp`) against the exact path, with
onset-matched one-ulp nulls for the floor rule, and optional multi-site / magnitude nulls as a diagnostic.

## Files
- `arm.sh RID SECS CAP SESSIONS [ENV=VAL...]`: one arm, both ranks (rank 1 over ssh), run inside one pair GPU ticket;
  several prompts in one process (first = cold, last = warm); kfd `evicted_ms`, busy, sclk and MemAvailable recorded.
- `seal.sh RID PROMPTS [--rank1-only]`: execution witness + seal of both ranks with `tools/strix-gate/fast_logit_gate.py`.
- `ranges.sh REF ARM...`: the whole qualification in one detached run (two ticket lanes), then the floor and absolute
  comparisons and `ranges.summary`. Arm syntax: `e<tag>`, `n<L>s<seed>`, `f<A>-<B>m<11|12>`, `d<A>-<B>s<seed>g<mag>`.
- `summary.py`: timing line per run (cold / warm TTFT, busy, sclk, evicted_ms, GPU clocks per token) and the result table.
- `null_patch.py`: DIAGNOSTIC hook for a scratch copy of `src/minimax/engine/engine.hip` (never merged), needed by the
  `n` and `d` arms. Single-site `L:seed` is the gate's onset null; `A-B:seed[:mag]` logs `DIAGM` and is never a floor.

## Use
```
export MMF_PEER=<rank-1 host> MMF_BIN=<gate binary, same path on both nodes> MMF_MODEL=<model dir>
export MMF_PROMPTS=<dir of <name>.u32 prompts> MMF_RUN=<scratch>/run MMF_GATE=<repo>/tools/strix-gate/fast_logit_gate.py
export MMF_TURN="<pair GPU turn wrapper: LABEL CMD...>"   # tickets <= 120 s
setsid nohup bash scripts/bench/mmflash/ranges.sh e-a ea n0s1 n0s2 n0s3 n0s4 n0s5 f0-59m11 f0-59m12 > ranges.log 2>&1 &
```
Run the exact reference arm first (`ranges.sh` compares against a sealed REF); all arms must use the same binary and
prompts. The floor comparison needs the gate tool's MiniMax onset-floor rule (`--engine minimax --profile fast-prefill`).
Timing from gate arms is indicative only: report it when `evicted_ms` stays low, and use strict ABAB runs for claims.
