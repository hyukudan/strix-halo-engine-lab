# Decision log

These decisions retain the evidence and scope recorded when they were made.
Consult the [qualification ledger](docs/qualification-ledger.md) for current
implementation gates and [design acceptance](DESIGN_ACCEPTANCE.md) for the
review requirements.

## Accepted

| ID | Decision | Reason | Evidence | Review |
| --- | --- | --- | --- | --- |
| D-001 | Build an independent engine; external runtime A and external runtime B are references. | Avoid dependence on a closed foundation or treating external runtime B as the destination for changes. | Scope confirmed by the user. | Reopen only by explicit decision. |
| D-002 | Maintain non-derivative observation of external runtime A. | Obtain general ideas without copying, reconstructing or redistributing implementations. | External runtime A license and confirmed scope. | Audit artifacts and provenance before publishing code. |
| D-003 | Start with native Linux, ROCm and `gfx1151`. | This is the actual environment of both engines and permits target-specific optimization. | E-001, documentation of both projects. | Review after the first stable version. |
| D-004 | Use GGUF as the initial input. | Public format, available weights and less dependence on private formats. | E-009. | Later measure a lossless internal cache. |
| D-005 | Quality and reproducibility are gates, not later tasks. | MTP, recurrence and quantization can produce misleading gains. | E-008, E-010. | Permanent. |
| D-006 | Prioritize a remote administrative control plane for the UI. | Operations, observability and benchmarks offer more value than duplicating existing chat clients. | E-016 and scope confirmed by the user. | Chat remains an optional integrated playground. |
| D-007 | Advance through offline compilation before hardware arrives. | Validate the toolchain, objects and contracts without claiming dynamic results. | E-015. | Revalidate everything on the actual machine. |
| D-008 | The first slice is AR + MTP, C1/C2/C4 and a qualified context below 262K. | Reduce numerical, memory and scheduling risk together. | E-019, O-11. | Expand only after correctness, memory and stability gates pass. |
| D-009 | Separate engine, gateway/chat and administration into privilege domains. | A convenient UI must not gain direct control over the GPU process or shell. | E-016, O-14, O-15. | Permanent unless the threat model is revised. |
| D-010 | Treat official Qwen repository, paper and checkpoint as primary architecture sources. | Specify contracts without inferring them from other engines. | E-020, E-021. | Reopen if Qwen publishes an incompatible revision; always pin the SHA. |
| D-011 | Expose near-maximum GTT, but admit loads with an independent effective reserve of 16–20 GiB. | GTT expands mapping; it does not create RAM or protect the host from OOM. | E-022, E-023, `MEMORY_BUDGET.md`. | Adjust the reserve only using telemetry from actual hardware. |
| D-012 | Initially keep the single-host 128 GiB profile within an approximate 100 GiB resident-weight envelope until A0.6 justifies another limit. | Preserve the effective reserve and capacity for KV, recurrence, scratch, runtime and recovery; maximum GTT is not the load budget. | E-011, E-028, E-032, E-033. | Reopen after actual UMA measurements and quality assessment of candidate quantizations. |
| D-013 | Break equal QSA scores by lower block index. | The canonical forward uses `torch.topk`, whose tied indices are not guaranteed stable; reproducible replay requires the same rule in serial and MTP. | E-041, E-043 and the `qsa_tie` fixture. | Reopen only if Qwen publishes an incompatible explicit canonical tie rule. |
| D-014 | Model each physical USB4 link as an autonomous transport with its own epoch, lease, TX/RX pools and accounting; dual-link mode is a policy over two qualified links. | Avoid assuming additive bandwidth, isolate failures and permit flow distribution/directional split/striping only with evidence of independence. | E-044 through E-048; docs 12 and 16. | Reopen only if hardware demonstrates an aggregate resource with shared ownership/synchronization and an end-to-end benefit. |

## Pending

| ID | Pending decision | Required data |
| --- | --- | --- |
| P-001 | Project name and license. | Owner preferences and contribution policy. |
| P-002 | Canonical development checkpoint/quantization. | Size, quality, available MTP and weight license. |
| P-003 | Registered mmap versus anonymous copy/huge pages. | Load, decode, memory and page-pressure measurements. |
| P-004 | Exact continuous-batching policy. | C1/C2/C4/C6/C8 results and admission cost. |
| P-005 | Wave32/wave64 by family. | Paired experiments by shape and end-to-end. |
| P-006 | Internal weight cache. | Lossless prototype, size and startup time. |
| P-007 | Administrative control-plane stack. | Prototype, threat model and deployment preferences. |
| P-008 (P0) | Canonical checkpoint and complete budget within 128 GB UMA. | Actual per-tensor size plus OS, KV, recurrence, rollback, scratch, libraries and headroom. |
| P-009 (P0) | Memory residency/registration model on `gfx1151`. | Documentation and a test with exact Linux/ROCm versions; confirm XNACK/HMM. |
| P-010 (P0) | Canonical architecture source ledger. | Repository, commit, license and QSA, DeltaNet, MoE, MTP, tokenizer and template contracts. |
| P-011 (P0) | `gfx1151` coverage of selected ROCm libraries. | Offline rocBLAS/hipBLASLt code-object inventory and fallbacks. |
| P-012 (P0) | Policy compatible with Qwen Community License 1.0. | Use case, distribution, commercial service and legal review where applicable. |
| P-013 | Final GTT, reserve and recovery values after an unclean kill. | Physical RAM, BIOS, final kernel/ROCm and pressure/restart tests. |
| P-014 (P0) | Canonical QSA semantics: top-k selection, ties, indexer and relation to dense attention. | Official paper/code, CPU oracle and KL/NIAH tests; do not assume dense equivalence. |
| P-015 | Direct GPU-owned DMA-BUF backend and preferred dual-link policy. | Exact kernel/ROCm in M5, DMA-BUF export/import support, consumer-credit/repost, reliable re-arm, 1×USB4/2×USB4 measurements and actual decode/prefill effect. |
| H-001 | Tiled/DPP GDN as the first optimization candidate after profiling. | GDN contract/oracle and our own traffic/time profile. |
| H-002 | Direct PLE reader before demand-paged mmap. | N-gram/PLE contract, p99 latency, page cache and hardware A/B. |

## Rejected for now

| ID | Alternative | Reason |
| --- | --- | --- |
| X-001 | Modify external runtime B as the final product. | Does not match the goal of an independent solution. |
| X-002 | Copy or reconstruct external runtime A kernels. | Outside clean-room scope and contrary to the goal of independently authored code. |
| X-003 | Design for multiple GPUs/models from the start. | Would add abstraction before validating the `gfx1151` fast path. |

## Review format

Every new decision records alternatives, evidence, risk, consequences for the
specification and conditions for reopening. Link acceptance actions here and in
`DESIGN_ACCEPTANCE.md`.
