Home: shared (backend core CUDA / arch specialisation SM120).
Implemented canonical NVFP4 operands (E2M1, per-16 E4M3, FP32 tensor scale),
host reference/quantizer, byte-identical CUDA activation quantizer, dense GEMM,
and transposed split-K decode for M=1..8, all in shared CUDA core.
Native opcode: OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X.
CUDA 13 requires sm_120a; plain sm_120 rejects block-scaled FP4 MMA.
Caller owns buffers/stream/workspace; no hidden allocations or synchronization.
Arithmetic identity: nvfp4-mma-f32-v1; activation FP4 rounding is explicit.
Tests: MSVC 19.44 Release, nvcc 13.0.88, host-only and root/native CMake lanes.
ctest --output-on-failure: host 1/1 passed; native lane 2/2 passed (0.50 s).
Covered codecs/ties, quantizer byte equality, exact dyadic dots, general-error
bounds, all decode row counts, tails, padded strides, residual alias, uneven
splits, K=17408, deterministic repeats, nondefault streams and graph capture.
ISA inspected: both GEMM and decode use native FP4, no local-memory spills.
Device-0 observations: direct cold launch / median of three warm graph batches.
M=8192, (N,K): cold/warm ms, TFLOP/s, useful GB/s:
(5120,5120): 2.486 / 2.372, 181.07, 86.89.
(17408,5120): 8.758 / 8.571, 170.37, 75.15.
(5120,17408): 8.589 / 9.291, 157.17, 32.09.
Decode M=1..8 warm, ordered by M (same shape order):
GB/s: 1011,951,1022,960,938,865,883,898.
TF/s: 3.59,6.74,10.84,13.57,16.54,18.29,21.73,25.21.
GB/s: 1645,1633,1636,1535,1502,1450,1439,1375.
TF/s: 5.84,11.58,17.37,21.71,26.51,30.67,35.46,38.66.
GB/s: 1859,1288,1635,1751,1741,1636,1637,1637.
TF/s: 6.61,9.15,17.41,24.84,30.86,34.78,40.57,46.34.
All 27 requested shape/row cases were measured; no speedup claim.
Packing, uploads, allocation and activation quantization are outside timings.
Reusable kit: scripts/bench/nvfp4/run.ps1; each child capped at 110 seconds.
Final harness also supports warm graph replay and in-process NVML clocks/busy
samples; device-wide readings (SM 180..2842 MHz, memory up to 13365 MHz) include
idle transitions and cannot isolate graphics/system activity or token costs.
An earlier pressured sweep was stopped; the final full sweep began device-idle.
Remaining: TURBO+/Qwen adapter binding, model fidelity, complete-workload gates,
and broader tuning/qualification. No production numerical default is promoted.
No downloads, network, push, model-local kernels, or device-1 execution.
Docs: docs/engine/kernels/cuda-nvfp4.md and docs/architecture/backends.md.
