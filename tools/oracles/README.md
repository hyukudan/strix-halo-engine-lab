# Semantic and Numerical Oracles

This directory contains deliberately small, dependency-free reference implementations used to qualify optimized engine paths.

The oracles are not runtime code and are not performance baselines.

## Scope of the first slice

Implemented:

- centred-weight RMSNorm, including grouped normalization;
- partial rotary application;
- deterministic candidate top-k policy;
- sequential Gated DeltaNet state update;
- hashed PLE/n-gram row IDs;
- Gated Residual/Hyper-Connection mixing with synthetic matrix fixtures;
- QSA block pooling/selection for fixtures whose cutoff has no score tie;
- fresh-versus-resumed equivalence tests for recurrent and n-gram state.

The numerical functions use Python double-precision arithmetic so that GPU-specific reduction order and lower-precision rounding are not accidentally baked into the reference.

## Canonical source pin

The model semantics used by this slice are pinned to:

```text
huggingface/transformers
revision: ebd5de00b43e94cae2da6d7aeb5a76ecdda51dae
model: qwen4_exp
```

Relevant public definitions include the model RMSNorm, partial rotary application, recurrent Gated DeltaNet rule, QSA indexer and PLE n-gram hashing.

The oracle is a separate implementation of the mathematical contract; it does not import Transformers.

## Run

From the repository root:

```bash
python3 tools/oracles/test_qwen4_exp_oracle.py
```

The fixtures are intentionally tiny and human-auditable.

## Still blocked

The following MUST NOT be treated as closed by this first slice:

- canonical sampling RNG and exact sampled replay;
- MTP acceptance algorithm;
- QSA tie semantics where the upstream primitive does not specify a stable tie order;
- full QSA mask/selection fixture coverage;
- Gated Residual/Hyper-Connection fixtures from the pinned real checkpoint;
- tokenizer and chat-template token fixtures;
- full first-layer and logit replay from the canonical development checkpoint.

Those are subsequent A0.5b gates.

## Rule

An optimized path may match another optimized path and still be wrong.

Promotion requires both:

1. model-fidelity comparison against an independent oracle/fixture;
2. within-plan equivalence checks such as serial versus batched, fresh versus resumed and speculative versus serial.
