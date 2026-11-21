# Inkling 2 Mini bring-up contract (work in progress)

This document tracks the **native** Inkling 2 Mini integration. It does not imply that the model currently runs in this engine.

## Priority and non-goals

Mini is the first target; Inkling 2 Small follows only after Mini's architecture and shared operators are validated. Do not implement inference by invoking llama.cpp or Python Transformers. Use a pinned upstream implementation strictly as a numerical oracle.

## Before writing kernels

1. Pin the exact upstream repository revision and record model config, processor/tokenizer config, safetensors index and tensor names/shapes/dtypes.
2. Generate a machine-readable mapping of checkpoint tensor names to native weight objects. Reject unknown layouts and missing required tensors.
3. Validate the looped-layer schedule and residual/message semantics from upstream source, including per-pass attention state. Do **not** treat repeated physical layers as independent weights.
4. Identify quantization layouts from actual checkpoint metadata, not merely from dtype labels. Check per-block scales and alignment.
5. Write tiny deterministic CPU reference tests before introducing HIP fused kernels.

## Shared-core reuse targets

| Responsibility | Preferred ownership |
| --- | --- |
| Quantized weight formats, dequantization, GEMV/GEMM | shared weight/kernel layer |
| Routed MoE dispatch and expert cache | shared MoE/runtime |
| Activation buffers and memory planner | shared allocator/planner |
| Prefill batching, decode scheduling, metrics | shared scheduler |
| KV and auxiliary recurrent state snapshots | shared session interface with model-owned payload |
| USB4 transport and distributed topology | shared device planner |
| Loop scheduling, message gates, block attention residual | Inkling family adapter |
| Vision-specific encoder and projection | Inkling family adapter using shared primitives |

## Bring-up acceptance gates

- [ ] Pinned upstream config + checkpoint tensor inventory with source revision.
- [ ] CPU reference: loop schedule, per-pass message/residual, attention and MoE operators.
- [ ] Numeric comparisons on tiny fixture and real model intermediate tensors.
- [ ] Single-node HIP prefill and decode with official Mini weights.
- [ ] Text generation token parity and deterministic regression tests.
- [ ] Vision processor, encoder and end-to-end image+text parity.
- [ ] Memory planning and long-context KV/state correctness.
- [ ] Performance receipts: TTFT, prefill, decode and peak memory at 1K/8K/32K; longer contexts as memory permits.
- [ ] Only then consider speculative decoding and USB4 partition experiments.

## Safety and correctness

Do not label a mode EXACT until a pinned reference comparison has passed. Keep FAST opt-in and document any accuracy deviations. Fail closed when the runtime cannot load a format or when the planner cannot allocate enough memory. The future Small model may require distributed partitioning or weight streaming and must not inherit Mini's recurrent schedule by accident.

Tracking: #909 (Mini), #908 (Small).
