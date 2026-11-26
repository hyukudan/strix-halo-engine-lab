# Profile receipt coverage

The host receipt check reads the profile headers and the recorded public evidence.
The gate and timing artefacts needed to reproduce these measurements are held by
the qualification owners. This check covers the fields present in the recorded
public evidence.

| Header constants | Public evidence checked |
| --- | --- |
| `qwen_dense_hip.hpp` `kGate{2,8,24}k`, `kTime{2,8,24}k`, `kWide{2,8,24}k` | Gate outcomes and rounded timing arms in the [Qwen model record](../models/qwen3.8-27b/README.md#fast-prefill-window-hip-decode-app). Receipt strings are also compared to the [band table](../band-edges.md). |
| `inkling2_mini.hpp` `kMiniWeightLoadRows` | Date, sample count and context band in the [Q8 evidence register](../decisions/evidence-register.md#inkling-2-mini-q8_0-attention-and-head-on-gfx1151). |
| `inkling2_mini.hpp` `kMiniWeightLoadDecisionRows`, `kMiniLowBitPackIdentity`, `kMiniLowBitMeasuredExtent`, `kMiniLowBitStepBand` | Sample count, scope, pack prefix, gate metrics and timing arms in the [calibration record](lowbit-calibration.md#inkling-2-mini-rows-gfx1151-hip-71). |

## Receipt fields without a checked-in source artefact

| Header | Constants or fields awaiting a source artefact |
| --- | --- |
| `hy4.hpp` | All seven `qualified` rows (prompt width, head and prompt attention), and the corresponding `incumbent_grants` rows. The detailed gate, timings and state hashes are quoted in the header. |
| `qwen_dense_hip.hpp` | `kNoise`, exact millisecond precision below the published rounding and the original lower-edge runs cited by `kBandRows`. |
| `qwen_dense_cuda.hpp` | The four `qualified` rows, matching `incumbent_grants` and `kExactLargeIdentity`. |
| `qwen_cuda.hpp` | `ladder_rows` carry provisional incumbent grants. The historical grant marker has no source artefact. |
| `inkling2_mini.hpp` | Full build, checkpoint and manifest digests, the full pack digest, the q4 receipt date, `kMiniIu4PrefillReceipt` and its two `kMiniLadder` rows, and the device ticket prose in `kMiniLevers`. |

Add the source, identity and regime when publishing another receipt field. Keep
unavailable fields in the inventory until a corresponding artefact can be checked.
