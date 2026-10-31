# Gemma 4 text host reference

The `g4-v1` contract is in `../contracts/g4_identity.hpp`; ownership and integration interfaces
are in `../contracts/g4_types.hpp`. This implementation follows the Hugging Face Transformers
5.21 Gemma4 text graph and proportional rotary semantics. It is an unqualified host reference,
with FP32 activations and KV. BF16 and quantized manifests have different numerical identities.
There is no GPU execution or default production admission here.

Build only the package targets with `STRIX_FAMILY_GEMMA4=ON`, `STRIX_ENABLE_HIP_RUNTIME=OFF`
and C++20. `strix_gemma4_contracts` exports the strict floating-point flags transitively;
`strix_gemma4_model` owns importing and `strix_gemma4_ref` implements `Target`. Tests are
`strix-gemma4-{ops,manifest,rows,npz}-tests` and require no Python or checkpoint.

Generate immutable rotary tables offline (Python plus gmpy2/MPFR required only for generation):

```sh
python src/gemma4/ref/make_rope_tables.py rotary-tables --positions 65536 --workers 6
strix-gemma4-host-ref MODEL_DIR --rope-tables rotary-tables FIXTURE.npz --threads 16
strix-gemma4-host-ref MODEL_DIR --rope-tables rotary-tables FIXTURE.npz --q4km MODEL.gguf
```

The generator checks MPFR rounding bounds before exporting binary32 values, including the
intermediate power, reciprocal and position product. Both host and future device code consume
the same hashed bytes. Global inactive pairs remain exactly `(1,+0)`. A shorter generated table
is useful for local tests; execution refuses positions beyond it. Full-context qualification
requires all 65536 rows. The loader verifies sizes, hashes, finite entries and inactive pairs.

The CLI accepts the supplied NPZ fixture schema and executes both a teacher-forced stream and
an independent greedy stream. It compares 65 teacher rows for a 64-token continuation, including
the last prompt row. It emits raw greedy IDs, top-1 agreement, last-prompt full-vocabulary KL,
logit max error/RMSE, and lower/upper KL bounds from top-K plus logsumexp. Coarsening the remaining
vocabulary gives the lower bound; its total mass and minimum engine logit give a conservative
upper bound. A small lower bound alone does not establish full-vocabulary fidelity. Successful
CLI exit means observations were computed; proposed fidelity bands are not an exactness verdict.
Full-model checks are deliberately excluded from ctest.

Safetensors loading requires the text configuration in the multimodal wrapper and its index.
All shard ranges are validated, every required text tensor must match its shape, and every
ignored tensor is reported. Only the 54 K/V/K-norm tensors in shared layers 24 through 41 are
accepted as compatibility exclusions within the text graph. Other modalities are never bound
to the reference graph. The head is tied to the unscaled embedding; a materialized head must
be verified equal before being omitted.

GGUF keeps each tensor's BF16/F32/Q8_0/Q4_K/Q6_K format and original blocks. Its first dimension
is contiguous input width, so dimensions are reversed into the manifest's `[output,input]`
shape without transposing bytes. Norms are direct multipliers, without an offset or a Q/K
permutation. The admitted proportional rotary marker has 64 unit divisors and 192 large
divisors; it is validated and translated to the contract's zero frequencies. Unknown metadata
overrides, names, missing weights and unsupported formats fail loading.

`HostStream::begin()` grants one unpublished transaction per stream. `forward_rows` accepts up
to eight active rows across streams and requires each transaction's positions to form a
contiguous prefix of its frontier. Plans may be permuted; `output_row` selects result placement.
Inactive rows have no state effect. Snapshots remain immutable until publication, and all
candidate producer rows stay in an overlay through the final layer. `commit(n)` publishes only
the first `n` processed input rows, once; destruction discards the remainder. Consumers alias
their producer's normalized, rotated K and normalized V. The caller serializes stream operations
at scheduling boundaries. Concurrent host calls on a target or stream are unsupported.

Workers split output coordinates only. `--threads 1` or `STRIX_G4_HOST_PARALLEL=0` selects
the serial output loop; worker count does not change a dot's accumulators or reduction order.
No changed-precision profile is implemented.
Device byte gates, compiler/ISA inspection, full multilingual/long-context fidelity and complete
workload qualification remain required before deployment.
