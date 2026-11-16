# Tiered expert residency (cards smaller than the model)

An option for discrete cards that cannot hold every routed expert: the card keeps the dense core and the hot
experts, the CPU computes the cold experts from host memory with the card's own arithmetic, and a planner moves
experts between the two as the conversation goes. It is not a new default: `auto` keeps today's fully resident
route whenever the model fits, and that route is unchanged (no new allocation, launch or sync).

Bound today: Qwen3.8-Flash-Next on the CUDA route (`strix-cuda-qwen-serve`, `qwen_decode_bench`).

## Flags (one resolver for the server and the CLI)

| Flag / environment | Values | Meaning |
|---|---|---|
| `--expert-residency` / `STRIX_EXPERT_RESIDENCY` | `auto` (default), `full`, `tiered` | `auto` = `full` when everything fits the budget, `tiered` otherwise |
| `--vram-budget-gib` / `STRIX_VRAM_BUDGET_GIB` | number, e.g. `8` or `7.5` | card bytes the engine may use (default: the whole card); emulates a smaller card on a bigger one |
| `STRIX_TIER_CPU_THREADS` | integer | CPU expert threads (default min(16, hardware threads)) |

`strix-cuda-qwen-serve --print-effective-config` resolves without loading the model and prints the choice and its
reason (`expert_residency`, `reason`, `vram_budget_bytes`, `expert_pool_bytes`, `slots_per_layer`, `positions`,
`host_expert_bytes`). The same parser and resolver run inside `qwen_decode_bench`
(`include/strix/residency/residency_cli.hpp`, `include/strix/residency/expert_residency.hpp`); a host test
(`tests/expert_residency_parity_tests.cpp`) checks that both entry points resolve the same request identically.

Card budget (memory owner): the budget pays, in order, the dense core (attention, GDN, router, shared expert, LM
head, MTP head), scratch, the session bytes of the requested context, one zero expert per layer and the margin; the
rest is the expert pool (the same number of slots in every layer). If the requested context does not fit next to
the core, it is reduced in steps of 1,024 positions and the reason says so. The host must hold every expert
(about 77 GB here) plus the host tables (PLE 28.8 GB): the resolver refuses tiered mode, with the numbers, when the
available host memory is smaller.

## How it works

Per MoE layer of a step, on one stream (capturable in the step's CUDA graphs):

1. The card's prologue quantizes the rows (A8) as today.
2. `tier_publish` maps each routed expert id to its slot through the layer's id -> slot table; a miss goes to the
   layer's zero slot (zero-filled weights, finite zero output) and is written, with the row's A8 codes, to an
   exchange in mapped host memory; one sequence number is published last.
3. The card runs the unchanged routed kernels on the slot ids (hits and zero-slot misses), while a host worker
   computes the misses with `strix::cpu::expert::Executor` straight from the memory-mapped GGUF.
4. `tier_gather` waits on the host's sequence number (watchdog: fail closed, never hang), copies the CPU raw dots
   into the card's per-(row, slot) buffer and ORs the CPU status; the unchanged merge and commit follow.

Exactness: every (row, slot, output) value of the routed MoE depends only on that (row, slot) and the expert's
bytes (`include/strix/cuda_qwen_moe.hpp`), the merge reads per-(row, slot) values in slot order and never the
expert ids, and the CPU executor implements the decode chains operation for operation (integer group dots, then
only single-rounding FP32 multiplies/adds in the card's order, the gfx1151 expf twin, the same h quantizer):
`src/cpu/core/routed_expert.hpp`. Tiered output is therefore bitwise the fully resident output; MTP verification
and rollback are unchanged and only emitted tokens are counted.

Between steps the planner (`strix::residency::Planner`) keeps a decayed reuse score per (layer, expert) and
promotes a candidate over the coldest resident expert only when its score beats the victim's by a margin of reuses.
Because the CPU's miss time is proportional to the expert's bytes, saved time per resident byte is the reuse rate,
so scores compare directly; the margin pays the upload and stops thrashing. Promotions are bounded per step; the
first prefill's routes seed the scores.

Why the CPU, not uploads, executes misses (measured on the lab PC, i7-13700K, dual-channel DDR5-5200, RTX PRO
6000 on PCIe 5.0 x16, `scripts/bench/residency/`): an exact CPU expert (3.07 MB) costs 52-57 us at 16 threads
(54-59 GB/s, the DDR wall is 60 GB/s), an upload of the same bytes 84 us (36-40 GB/s pinned) and it reads the
same DDR, so splitting misses between both cannot pass the DDR wall. Rows that share an expert are nearly free on
the CPU (memory-bound), so verification rows only cost through the union of their experts. The per-layer exchange
through mapped memory costs 1.15 us round trip (a stream sync + copies costs 72.8 us).

## Strix Halo

No VRAM/RAM tier: the GPU and the CPU read the same LPDDR5X, so there is nothing to promote and a CPU overflow
arm would compete with the GPU for the same bandwidth. The resolver is never asked there; the HIP routes are
unchanged.

## Other families

The planner and resolver are host-shared. DeepSeek V4.1 on CUDA already keeps a per-layer id -> slot table with
LRU dynamic slots and remote homes (`include/strix/cuda_ds41_hot_residency.hpp`); its LRU refill is the binding
target of this planner (follow-up in a DS41 window). The CPU executor does not bind DS41 (IQ2_XXS / Q2_K experts and
the split combine of `include/strix/ds41_hot_placement.hpp`). GLM and MiniMax have no CUDA route (`src/cuda`).

## Limits of this first version

Declined in tiered mode, each with a reason at start: BF16 route storage, prefill overlap, expert types other than
Q4_K/Q5_K gate-up and Q5_1/Q8_0 down. Prefill runs the misses on the CPU in blocks of 256 rows (a streaming
prefill of whole layers is the follow-up for long prompts). Promotions run on the engine thread between steps.

## Measured

See the results table below (filled from `qwen_decode_bench` receipts).
