# Dated implementation notes

Implementation notes written while each slice landed, kept as historical records. They describe
the code and evidence as of their date; current behaviour is documented in the topic pages listed in
the [documentation index](../../README.md). Notes marked *cited* are still referenced as evidence by
current documents.

| Date | Note | Topic | Summary |
|---|---|---|---|
| 2026-10-11 | [Portable correctness and precision foundations](implementation-2026-10-11.md) *(cited)* | runtime | Distributed envelope v2, physical link epochs/sequences, RX completion proofs and poison/drain rules before hardware. |
| 2026-10-12 | [Complete C1 graph and experimental GPU transport](implementation-2026-10-12-complete-graph-and-export.md) | runtime | Explicit device-input complete C1 HIP graph capture/replay with resource fencing; experimental GPU transport export. |
| 2026-10-12 | [Complete-model experimental decode recording](implementation-2026-10-12-decode-execution-modes.md) | runtime | strix-hip-decode-token eager vs graph modes, memory ceilings, and optional eager integer LM head flags. |
| 2026-10-12 | [Device-dynamic GDN and indexed attention](implementation-2026-10-12-dynamic-state-operators.md) | gdn | Device-position capture mode for GDN and indexed attention launchers; GraphOperatorControlPlan preflight. |
| 2026-10-12 | [Reusable execution components](implementation-2026-10-12-execution-components.md) *(cited)* | runtime | GraphAttentionStatePlan post-projection norm/RoPE/KV stage with status codes and plan identity binding. |
| 2026-10-12 | [Hardware campaign preparation](implementation-2026-10-12-hardware-campaign-preparation.md) | hardware-campaign | Reproducible private kernel/module builder and commissioning executable for direct GPU transport experiments. |
| 2026-10-12 | [Experimental tiled integer LM-head consumer](implementation-2026-10-12-integer-runtime.md) *(cited)* | lm-head | RuntimePlan wiring of Q4_0/IQ4_XS integer LM-head projection with exact binary64/binary32 arithmetic identity. |
| 2026-10-12 | [Native numerical campaign preparation](implementation-2026-10-12-native-numerical-campaign.md) | hardware-campaign | campaign.py binds GGUF hash, auditors and RAW recordings to predetermined numerical/task gates. |
| 2026-10-12 | [Optimization lifetime implementation](implementation-2026-10-12-optimization-lifetimes.md) | runtime | Retained arithmetic history for prefix-state restore and current-resource graph eligibility guards. |
| 2026-10-12 | [Pre-hardware experiment implementation](implementation-2026-10-12-prehardware-experiments.md) | numerics | Integer projection lab (Wave32 IU8/IU4) and trusted DirectTransportDriverBackend adapter. |
| 2026-10-12 | [RAW token and restored-prefix recording](implementation-2026-10-12-raw-baseline-recording.md) | hardware-campaign | Opt-in --record-raw path capturing native token logits/state for later correctness campaigns. |
| 2026-10-14 | [GDN delta broadcast lifetime](implementation-2026-10-14-gdn-delta.md) | gdn | Fixes GDN recurrence race by publishing deltas in separate shared array; identities advanced. |
| 2026-10-14 | [Serialized normalization scales](implementation-2026-10-14-gguf-norms.md) | numerics | Stop re-adding one to pre-folded GGUF norm scales in hyper mixer, PLE, attention; identities v2. |
| 2026-10-14 | [Optional full-file SHA-256 provider](implementation-2026-10-14-model-hash.md) | runtime | Optional OpenSSL EVP bulk SHA-256 for model file hashing; identity format unchanged. |
| 2026-10-14 | [FP32 routed-expert accumulation and validation](implementation-2026-10-14-moe-accumulation.md) | moe | Routed-expert output kept FP32 across experts; device validation of expert IDs and route weights. |
| 2026-10-14 | [Routed experts with K-quant and affine blocks](implementation-2026-10-14-moe-kquant.md) | moe | HIP routed-expert launcher accepts Q4_K, Q5_K and Q5_1 packed weights with native fixtures. |
| 2026-10-14 | [Experimental parallel hyper-mix](implementation-2026-10-14-parallel-hyper-mix.md) | hyper-connections | Opt-in --parallel-hyper-mix splits normalization, projection, injection, output across workgroups with admitted workspace. |
| 2026-10-14 | [File-mapped IQ4_NL PLE lookup](implementation-2026-10-14-ple-iq4-nl.md) | numerics | File-mapped PLE backend decodes IQ4_NL rows to BF16 without whole-table expansion. |
| 2026-10-14 | [Router reduction broadcast ordering](implementation-2026-10-14-router-broadcast.md) | router | Barrier added so router softmax max is read before shared slot reuse; GDN resource declaration fixed. |
| 2026-10-14 | [Experimental staged hyper projections](implementation-2026-10-14-staged-hyper-projections.md) | hyper-connections | Opt-in --staged-hyper-projections using 128-thread shared FP32 tiles, preserving ascending accumulation. |
| 2026-10-15 | [Experimental fused-launch hyper injection](implementation-2026-10-15-fused-hyper-injection.md) | hyper-connections | Opt-in --fused-staged-hyper-mix appends injection workgroups to the low-rank projection grid. |
| 2026-10-15 | [Typed hyper-connection row prefetch probe](implementation-2026-10-15-hyper-row-prefetch-probe.md) | hyper-connections | Compile/ISA-only probe: --prefetch-typed-hyper-rows loads eight decoded rows into registers (rows8-v2). |
| 2026-10-15 | [Experimental exact lane-chain HC output](implementation-2026-10-15-lane-chain-hyper-output.md) *(cited)* | hyper-connections | Exact lane-chain HC output kernel; now part of adopted automatic v3 decode profile. |
| 2026-10-15 | [Exact HC projection diagnostic prototype](implementation-2026-10-15-lane-chain-hyper-projection-prototype.md) | hyper-connections | Test-only lane-owned exact HC down/injection chain kernels with microbenchmark and cold-copy mode. |
| 2026-10-15 | [Parallel HC projection diagnostic](implementation-2026-10-15-parallel-hc-projection.md) *(cited)* | hyper-connections | Diagnostic split-K16 reassociated HC down projection; distinct numerical identity, not production. |
| 2026-10-15 | [Experimental two-phase MoE router](implementation-2026-10-15-parallel-moe-router.md) | router | Opt-in --parallel-moe-router: tiled logit projection kernel then original selection tail. |
| 2026-10-15 | [Experimental Q8 hyper-connection output producer](implementation-2026-10-15-q8-hyper-output-probe.md) | hyper-connections | Source/compiler probe: --prefetch-q8-hyper-output Q8 register-prefetched HC output producer. |
| 2026-10-15 | [Opt-in Q8 indexed-attention input prefetch](implementation-2026-10-15-qsa-input-prefetch.md) *(cited)* | runtime | Q8 indexed-attention input projection prefetch schedule; adopted in automatic v3 profile. |
| 2026-10-15 | [Experimental ranked router selection](implementation-2026-10-15-ranked-router-topk.md) | router | Opt-in --ranked-moe-router-topk replaces insertion top-k with integer ranking in parallel router. |
| 2026-10-15 | [Opt-in Q8 shared-expert projection prefetch](implementation-2026-10-15-shared-expert-prefetch.md) *(cited)* | moe | Q8 shared-expert gate/up/down prefetch with original reduction tree; adopted in automatic v3 profile. |
| 2026-10-15 | [Spec v1 reversible draft integration (2026-10-15)](implementation-2026-10-15-spec-v1-draft-integration.md) *(cited)* | runtime | Integration branch combining wave router, QSA prefetch, HC output, deferred MoE status, shared-expert prefetch. |
| 2026-10-15 | [Typed fused hyper-connection projection](implementation-2026-10-15-typed-hyper-projection.md) | hyper-connections | Opt-in --typed-fused-hyper-projections requiring Q8_0 down and F32 injection; v6 identity. |
| 2026-10-15 | [Virtual256 Q8 GDN output projection (experimental)](implementation-2026-10-15-virtual-q8-gdn-output.md) | gdn | Opt-in one-wave Q8_0 GDN output projection preserving original 256-leaf arithmetic tree. |
| 2026-10-15 | [Experimental virtual-leaf Q8 LM head](implementation-2026-10-15-virtual-q8-lm-head.md) | lm-head | Opt-in --virtual-q8-lm-head 32-thread virtual-leaf Q8 LM head preserving reduction tree. |
| 2026-10-15 | [Experimental wave router selection](implementation-2026-10-15-wave-router-topk.md) *(cited)* | router | Wave-level router top-k selection (--wave-moe-router-topk); adopted in automatic v3 profile. |

Other dated or historical records in this directory:

- [Design acceptance actions](2026-09-28-design-acceptance.md): Design-review acceptance requirements recorded 2026-09-28 plus spec v1 draft integration notes (D-015..D-020). *Historical.*
- [Strix Halo optimization review — 2026-09-29](2026-09-29-optimization-review.md): Dated research review of optimization mechanisms, UMA, dual USB4 and partitioning, with proposed amendments. *Historical.*
- [Platform qualification plan](platform-review.md): Short stub: platform qualification principles; defers to docs/architecture and implementation ledger. **Superseded** by [13 — Platform Discovery and Capability Negotiation](../../architecture/13-platform-discovery.md).
- [Technical specification v0.4 — inference engine for Strix Halo](spec-draft-v0.4.md): Technical specification v0.4 research draft: objectives, principles, architecture, phases A0-E, decision gates. **Superseded** by [Strix Halo Architecture Guide](../../architecture/README.md).
