# Shared numerical arithmetic policies

| Identity | Explicit binding | Default | Qualification |
| --- | --- | --- | --- |
| Declared exponential, fixed-point round-to-nearest-even v2 | `declared_exp<ExpForm::fixed_rne_v2>(x)` in `strix::moe::declared_math` | Engine bindings use the frozen v1 form | Host edge vectors cover overflow, subnormals, floating-point rounding modes and flush-to-zero. Device identity and workload quality are pending. |
| RMSNorm, unfused square accumulation | `SquareForm::mul_add_uncontracted` in the shared norm trait | Engine bindings keep their existing square form | Host midpoint distinguishes fused and unfused results. Device ISA, scratch usage and logits require qualification for each binding. |
| Host logprob combine, unfused FP64 | `-DSTRIX_HOST_LOGPROB_UNCONTRACTED=ON` at configure time | off | Applies strict FP only to host logprob partials. On hosts that used FMA, scores can differ at rounding boundaries. The path scores generated tokens and has its own documented tolerance. |

The optional forms can change float bits and require a separate numerical identity before
any model binding is qualified. The device qualification must include scalar
edge vectors and digests of complete engine outputs for the selected identity.
