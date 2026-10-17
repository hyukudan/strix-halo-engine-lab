# Source and provenance ledger

Status: P0 partially resolved. A reference becomes a canonical source for this
engine only when it is pinned here with an immutable revision, its applicable
license and a verifiable contract.

| Component | Pinned candidate source | Verified license | Independent contract | Status |
| --- | --- | --- | --- | --- |
| Flash-Next architecture | `QwenLM/Qwen3.8-Flash-Next@69885871a64393807d988b27b1b5e380e8f28526`, paper `arXiv:2608.30320` and official checkpoint `Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540` | Qwen Community License 1.0; use case/distribution terms to be reviewed | QSA/GDN/Gated Residual/n-gram contracts still to be extracted | **Primary source pinned** |
| Flash-Next Q4 configuration and weights | `unsloth/Qwen3.8-Flash-Next-GGUF@38bb39ee97821de2c9009abb7e93950eec396e66` | Derivative; mapping of the official model's conditions pending | Hash/tensor map against the official checkpoint | Candidate; not approved |
| MTP Q8_0 predictor | same snapshot, `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | Pending | Pending | Candidate; not approved |
| Cross-check MTP implementation | `ggml-org/llama.cpp#27836@1d8de7c1b0c7d2febf8f983174d8e6a711e2b1af` | Project MIT license; open/non-canonical PR | Structural and input-combiner cross-check; its attention contains a QSA TODO, so it is NOT an oracle for the full forward | **Secondary; input fusion corroborated, full forward blocked** |
| MoE routing and experts | Official Qwen paper and checkpoint/config pinned above | Yes, subject to the usage policy | Tie-breaking, ordering and layout pending | Partial |
| MTP acceptance algorithm | `huggingface/transformers@ebd5de00b43e94cae2da6d7aeb5a76ecdda51dae`, `src/transformers/generation/utils.py` and `candidate_generator.py` | Apache-2.0 | Greedy acceptance/crop fixture; sampled mode still has RNG pending | **Greedy pinned; sampled RNG open** |
| Tokenizer | Must be extracted from the canonical snapshot and pinned by hash | Pending | Exact IDs in fixtures | Open |
| Chat template/reasoning controls | Must be extracted from the canonical snapshot and pinned by hash | Pending | Golden prompts and tokens | Open |
| N-gram/PLE | Pinned official config (`ngram_size=3`, base 20M, 128 partitions, PLE at layer 2) and Qwen paper/code | Qwen Community License 1.0 | Hashing, rows, conv and lookup timing pending | **P0 open** |
| Hyper-connections/output gate | Official config (`hc_count=4`, `hc_lowrank=320`, sigmoid) and Qwen paper/code | Qwen Community License 1.0 | Residual mixing, scratch and numerical order pending | **P0 open** |
| Partial/interleaved mRoPE | Pinned official config (`partial_rotary_factor=0.25`) and Qwen code | Qwen Community License 1.0 | Multimodal positions and text oracle pending | Open |
| Transformers `qwen4_exp` semantics | `huggingface/transformers@ebd5de00b43e94cae2da6d7aeb5a76ecdda51dae`, `src/transformers/models/qwen4_exp/modeling_qwen4_exp.py` | Apache-2.0 | Independent tiny fixtures for RMSNorm, partial RoPE, recurrent GDN and PLE n-gram hashing | **Pinned for A0.5b; greedy MTP acceptance fixed, sampled RNG/QSA ties remain open** |
| IQ4_XS storage/dequant contract | `ggml-org/llama.cpp@8019dc563b1ecbae6b161a70c3a1359f1b206c1e`, `ggml/src/ggml-common.h` and `ggml/src/ggml-quants.c` | MIT | Independent byte-layout parser, FP16 conversion and full 256-value dequant fixture | **Pinned for A1.1 compact-format experiments; not a canonical checkpoint decision** |
| GGUF v3 structure | `ggml-org/ggml@353b63b439f27ab2cc19dac97ab1681ba6d2d084`, `docs/gguf.md`; geometries in `ggml-org/llama.cpp@6c7a87f7e5e5cd75b8a641c3471f2dee84a6ed17` | MIT for both projects; public specification/catalog | Parsing, bounds, block and range fixtures | Primary source pinned for A0.2/A0.3 |
| Experimental USB4 DMA-BUF extension | [Linux USB4STREAM source](https://github.com/torvalds/linux/blob/72d3fcf802c45d00b300f25b848a93c3a2bd7c7e/drivers/thunderbolt/stream.c), pinned commit `72d3fcf802c45d00b300f25b848a93c3a2bd7c7e` | GPL-2.0; upstream Intel notices preserved in the integration patch. Project kernel code GPL-2.0-only; public UAPI GPL-2.0 WITH Linux-syscall-note. Exact upstream license/exception texts retained in `lab/direct_gpu_stream/kernel/` | Project-owned experimental ABI, attachment/SG/controller-drain contracts and shared property tests | Enabled real kernel driver object compiled with strict warnings; no module load, DMA execution or GPU/cache qualification |

## Approval rule

Each row needs a URL/repo, a SHA or immutable revision, the applicable license,
artifact hashes and a verifiable contract. Secondary references, such as
open, non-canonical pull requests, may guide tests and structural checks, but
none of them is the sole oracle for an implementation.
