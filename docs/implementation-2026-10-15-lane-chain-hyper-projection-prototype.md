# Exact HC projection diagnostic prototype

This prototype adds no decoder option, catalog entry or production dispatch. Only `strix-hip-lane-chain-hyper-projection-tests` selects its kernels. All original HC device bodies remain unchanged. The new kernels share the original translation unit and compiler flags.

An active lane owns one whole down output or injection chain. Outputs per wave 8, 16 and 32 select participating lanes. Each workgroup has one 32-lane wave, declared with `__launch_bounds__(32)`, and uses no LDS or barriers. Down loads eight original Q8 blocks as 272 aligned bytes, stages one or two groups, decodes half scale times signed code into a separate FP32 coefficient, then advances the dot chain in ascending feature order from positive zero. It never splits K. Division and sigmoid tails preserve the original expressions. Injection reads the original F32 weights in ascending order.

P1 and P2 stage 68 and 136 packed dwords per active lane before other state. These are source payload sizes, not VGPR predictions: register allocation and its effective limit depend on launch bounds and the actual compiler. An earlier native compilation without explicit bounds used 43 VGPR for P1 and 145 for P2, with zero scratch or spills. The bounded version requires a fresh audit; no spill is inferred merely from payload size. P4 is excluded because 272 payload dwords already exceed 256 VGPR. Actual registers, scalar normalized-input loads and ordered FMA/FP modes remain compiler gates.

Admission requires width divisible by 256 and at most 10240, streams 2 through 8 dividing width, rank 1 through 512, rows 1 through 65535, aligned owners with sufficient extents, and disjoint mutable/read ranges. Inputs and outputs remain owned through stream completion. There is no repacking, quantization or reduction reordering.

The default diagnostic supplies 40 shape/adversary/injection cases across all six variants: ordinary finite inputs, extreme Q8 scales, isolated finite subnormal/zero inputs with moderate coefficients, explicit overflow, and full-mantissa FP32 inputs. The full-mantissa fixture first requires a host fused chain to differ from a volatile-product non-fused chain. Separate controls require the final Q8 code of the final row/group and final injection coefficient to change reference output bits. Subnormal outputs must be finite and at least one must be nonzero. Both unchanged plain and rows8 references consume the same inputs. Every variant compares full output bits, poison, guards, eager execution and three graph replays. Negative cases require the precise admission error and intact poisoned outputs for geometry, variants, missing pairs, alignment, short extents and overlap. These source fixtures do not establish native execution.

`--bench` requires named original Q8 down and F32 injection tensors from GGUF plus one externally captured normalized FP32 row. Requiring injection binds streams to an actual tensor dimension. Architecture, runtime and live library checks precede fixture allocation or kernel launch. The receipt records payload hashes/offsets, executable identity and runtime libraries. The coordinator separately binds capture provenance, source, layer and position; this executable does not validate an arbitrary capture JSON schema.

Each OPW/P choice records eager and graph samples, at least five discarded alternating warmup pairs and 200 measured alternating pairs. Each event sample batches a fixed number of calls. All timings and paired ratios are retained, and every touched copy must produce reference bits and preserve guards.

`--cold-copies N` also measures a separate rotating-copy mode when N is greater than one. N is bounded to 128; all private owners together are bounded to 512 MiB. Baseline and candidate use identical independently advanced cyclic copy sequences, and each copy has its own graph in graph mode. The receipt separates hot-single-copy and rotating-private-copies, with copy count and down working-set bytes. Rotation does not prove a specific LLC capacity or eviction rate; choose a sufficiently large working set using actual hardware evidence. Registered model residency and complete-model throughput are separate gates.

Example after source review and native compiler approval:

```sh
strix-hip-lane-chain-hyper-projection-tests --bench \
  --model /absolute/checkpoint.gguf --down-tensor blk.0.hc_attn_down.weight \
  --inject-tensor blk.0.hc_attn_inject.weight --normalized /absolute/normalized.f32 \
  --width 10240 --streams 4 --rank 320 --warmup 5 --pairs 200 --iterations 16 \
  --cold-copies 64 --output /absolute/new-projection-microbench.json
```

If the exact-chain floor or spills dominate, report that outcome or evaluate another exact producer schedule; do not change arithmetic silently. An OPW8/P2 real-input median above approximately 50 microseconds is a screening stop for integration and a reason to investigate the chain floor or an exact producer pipeline. It is not a prediction of the current kernel's latency. A later integration still requires all 554 FP32 arrays and 512 generated IDs to match. No performance gain, full-model fidelity or adoption is claimed here.
