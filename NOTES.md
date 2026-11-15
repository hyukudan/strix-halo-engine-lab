Home: shared (host shared).

Implemented Qwen3.8-27B host declarations without a hard-coded geometry preset.
HF/GGUF config parsing covers dense geometry, vocab, RMS epsilon, biases,
tied heads, RoPE/partial RoPE/linear/YaRN and optional explicit MTP counts.
Shared mapped storage and strict safetensors parsing were extracted from Gemma;
both families bind that reader. Shared GGUF row decoding is reused unchanged.
GGUF Q4_K/Q4_0/IQ4_NL/IQ4_XS and mixed supported types retain packed storage.
NVFP4 import exposes E2M1 codes, E4M3 block scales, global multiplier and
optional input/pre-quant scales, including indexed checkpoint shards.
Family mapping checks complete tensor sets and dimensions, and aliases tied heads.
The shared dense graph declares attention, dense MLP, norms, KV and optional MTP.
Public API: include/strix/qwen27_model.hpp; target: strix_qwen27_model.
Format/declaration contract: docs/models/qwen3.8-27b-host.md.

Tests: MSVC 19.44 Release, /W4 /WX, standalone synthetic fixture project.
ctest --test-dir build-qwen27-host -C Release --output-on-failure: 2/2 passed.
The generated model fixtures passed 3,357 host checks.
Top-level CMake configuration and new host target build: passed.
CUDA 13 nvcc -c -arch=sm_120 public-header integration check: passed.
No GPU execution, downloads, network requests or kernel changes.

Remaining: real model/export validation and backend core dispatch/upload wiring.
Input smoothing/activation scale execution remains a backend obligation.
Only the documented normalized-input/shared-head MTP layout is declared.
Alternative packed names, inverse/swizzled scales and auxiliary layouts need policy.
Native execution, model fidelity and full-workload gates remain unqualified.
No production defaults, performance or numerical exactness claims were added.
