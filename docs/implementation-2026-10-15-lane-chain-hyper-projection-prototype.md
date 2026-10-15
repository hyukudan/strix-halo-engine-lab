# Exact HC projection diagnostic prototype

This prototype adds no decoder option, catalog entry or production dispatch. Only the explicit `strix-hip-lane-chain-hyper-projection-tests` diagnostic selects its kernels. All original HC device function bodies remain unchanged, and the new kernels live in the original HC translation unit without changing compiler flags.

An active lane owns one whole down output or injection chain. Outputs-per-wave8,16,32 select how many lanes participate; each workgroup is one32-lane wave and needs no LDS or CTA barriers. The down path loads groups of eight original Q8 blocks:272 aligned bytes, represented by17 vectors of16 bytes. It stages one or two groups in registers, decodes the original half scale and signed code into a separately rounded FP32 coefficient, and advances the FP32 dot chain in ascending feature order from positive zero. It never splits K. The low-rank division and sigmoid tail are the original expression. The optional injection role reads original F32 weights, preserves its ascending chain, division and `2*sigmoid` tail.

PrefetchP1 andP2 are compiled as separate instantiations. They require68 and136 packed dwords per active lane before accumulators, addresses and temporaries; these are source-level payload sizes, not measured VGPR allocations. P4 would require272 packed dwords alone, exceeding the256-VGPR limit before live state, and is deliberately not instantiated. P2 can still spill if the compiler fails to scalarize the fully unrolled packed arrays, so actual ISA scratch/register evidence is mandatory before performance interpretation. Scalar normalized-input loads and the exact FMA/denormal sequence also require target ISA verification.

Diagnostic admission requires width divisible by256, width at most10240, streams2..8 dividing width, rank1..512, bounded rows1..65535,16-aligned down/injection owners, exact minimum extents and disjoint mutable/read owners. Inputs and outputs remain owned until explicit stream completion. There is no repacking, quantization or reduction reordering.

The default diagnostic supplies24 synthetic shape/adversary/injection cases across all six OPW/P choices. Cases include odd output tails, a one-groupP2 tail, canonical10240-wide chains, signed Q8 extremes, half subnormals/signed zero, extreme FP32 normalized inputs, different output poison, extent guards, eager execution and three captured replays. Both original plain and rows8 references consume the same inputs. The source fixture alone does not establish device execution.

The `--bench` mode reads named original Q8 down and optional F32 injection tensors directly from a GGUF, and consumes an externally captured normalized FP32 row. It validates tensor shapes/types, reports payload hashes/offsets and capture hash, and copies those original bytes into private device allocations. This isolates repeated warm projection calls; it does not reproduce complete-model registered residency or establish full-model throughput. Every measured result must still equal the original rows8 output bits, including injection and guards.

For each OPW/P choice, eager and captured modes discard at least five alternating warmup pairs, then record at least200 alternating pairs with HIP events. Each sample measures a fixed batch of calls and reports microseconds per call. All baseline/candidate samples and paired ratios are retained. No decoder integration is allowed on estimated occupancy or FLOP counts. First inspect actual ISA, then measure real captures; if the exact-chain floor or spills dominate, stop and report that outcome rather than silently changing arithmetic. Complete-model554-array and512-ID identity gates would still be required before any later integration.

Example after independent source review and native ISA approval:

```sh
strix-hip-lane-chain-hyper-projection-tests --bench \
  --model /absolute/checkpoint.gguf --down-tensor blk.0.hc_attn_down.weight \
  --inject-tensor blk.0.hc_attn_inject.weight --normalized /absolute/normalized-layer0-position0.f32 \
  --width 10240 --streams 4 --rank 320 --warmup 5 --pairs 200 --iterations 16 \
  --output /absolute/new-projection-microbench.json
```

The coordinator must bind the normalized capture to its layer/position/source and collect executable/compiler/ISA and loaded-library receipts around this command. The diagnostic records supplied bytes and observed architecture/runtime versions; it does not manufacture capture provenance. No speedup or hardware qualification is claimed at this source checkpoint.
