# Prefill core8 (draft)

Status: implemented behind a default-off switch; not compiled or qualified on gfx1151.
No performance improvement is claimed until native compilation, ISA inspection,
GPU byte comparisons, model fidelity, and complete-workload checks pass.

## Selection

STRIX_MMX_PREFILL_CORE8=0 (default), 1, or 2 is captured once at Engine startup.
Engine::set_prefill_core8(int) and moe::opt::set_prefill_core8(int) accept -1 to
restore the captured environment. Invalid values throw. Selection applies only
to the v4 MT8 class. Other classes, cores, item tables, splits, transforms, finish
kernels, and decode kernels keep their existing paths.

- 1: 128 outputs by 128 rows. The CTA cooperatively loads 4096 unique A bytes per
  k tile, including explicit zero rows past n, into alternating LDS buffers.
  Next A loads precede next weight-word loads; the A values remain live across
  current decode/WMMA work and are stored to the other LDS buffer afterward.
  One priming barrier plus K-1 handoff barriers gives K barriers for K k tiles.
- 2: paired MT8 row tiles use 64 outputs by 256 rows, with two column pairs and
  two row halves. Wave w handles column pair w%2 and row half w/2. It decodes
  tile w/2 of that column pair, splitting k=0..7 and k=8..15 between lane halves.
  permlanex16 assembles the full fragment; the lower lanes publish it to LDS.
  The two row halves read the same B fragments. A remains global. Each lane
  evaluates eight codebook states per k tile, versus sixteen in the old core.
  Consecutive by values select the two 64-column halves of the same row pair.
  Items with fewer than 256 rows or an unpaired last by use variant 1.

## Exactness argument

Every accumulator retains its original 16-row tile and 16-column tile. Variant 1
copies the same 32 bytes for row c to both WMMA lane halves. Its zero-row test is
the original row<n test. Variant 2 reconstructs the same sixteen trellis states
for each B column, calls the unchanged codebook_mul1, and exchanges f16 bits
through LDS. Its two k halves are operand assembly, not an accumulation split.
Each output still starts at zero and performs the same WMMA chain over ascending
k tiles within the same split. Part addresses and stores are unchanged. The
remapped CTA coverage is a bijection within each pair of original row tiles.

Both LDS exchanges alternate buffers. The previous barrier ensures all readers
of the buffer being reused have finished; the release plus s_barrier publishes
the new buffer. No extra arithmetic, split reduction, or output conversion is
introduced.

## Resource budget and ISA gates

These are source-level planning estimates, not compiler resource reports.

| Variant | LDS per CTA | Expected VGPR planning range per thread |
| --- | --- | --- |
| 0 | 0 | Existing MT8 allocation (approximately 225) |
| 1 | 8192 bytes (two 4096-byte A buffers) | Approximately 225-249; eight next-A words, no live xr array |
| 2 | 8192 bytes reserved; 4096 used for B on paired tiles | Approximately 225-249; global A, four decoded words per lane; allocation also covers variant-1 fallback |

All variants have 128 accumulator words per thread and at most four live A
fragments (32 words). Compilation must confirm VGPRs stay below the spill limit;
reject scratch use rather than accepting a slowdown hidden by register spills.
Inspect every bit-width specialization and both branches of variant 2:

- Scratch size and scratch loads/stores must be zero.
- Expect paired ds_load_b128 operations for each 32-byte A fragment in variant 1,
  cooperative vector global A loads, and vector LDS stores/loads for exchange.
- LDS release uses __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup") followed
  by __builtin_amdgcn_s_barrier(), never __syncthreads(). Check the emitted LDS
  completion waits and s_barrier, and verify no s_waitcnt vmcnt(0) is introduced
  at the steady-state barrier to drain ln[] prefetches. An initial A fill must
  complete its global loads; there are no prefetched weight words at that barrier.
- Variant 1 issues next A before ln[] and consumes it after current WMMAs. Check
  that compiler scheduling preserves useful overlap and waits only as far as A
  requires while leaving younger weight loads outstanding. Source order alone
  does not establish this property.
- Verify sixteen WMMAs per wave per k tile and ascending k accumulation order.

## Node qualification

Run the existing benchmark executable with --test --tests test_prefill_core8.
Repeat with --model DIR --layer L --rank R for real L3 weights on both ranks.
The section compares 0 against 1 and 2 with mtfit off and on, including every
split partial, raw tiled gate/up inputs, tiled down inputs, and final outputs.
It retains the original crafted row boundaries and adds 383/384/385,
499, and 511/512/513; mixed synthetic weights cover 3/4/5/6/8 bits. Uniform random
distinct routes cover T100, T1024, and T4096. Fresh poisoned workspaces catch
missing stores. Mismatches throw and produce a nonzero process exit.
The T1024/T4096 cases also print gate_up.mt8 and down.mt8 marker timings for
0/1/2; the loaded-weight pass uses real L3 slices when --model is supplied.

--ab core8:PROMPT.u32:BLOCK forces v4 and mtfit, then runs A B A B A B A.
A selects 0; B selects STRIX_MM_AB_CORE8=1 (default) or 2. Both state and final-row
FNV checks are mandatory for this flag; mismatches use the existing failure exit.
Afterward, the core and mtfit/core8 switches restore their environment defaults.
The pfall flag is unchanged pending qualification.

## Local checks

No GPU or ROCm compiler was available. No C++ compiler was found on PATH.
CPU scalar checks passed for 2,745,728 output mappings and 655,360 exchanged B
states across bits 3/4/5/6/8, plus 82,944 A fragment address/zero checks.
They do not execute WMMA, prove compiler lowering,
or replace node qualification. Native build, resource counts, ISA barrier waits,
GPU bitwise tests, stage timings, and end-to-end rank checks remain outstanding.
