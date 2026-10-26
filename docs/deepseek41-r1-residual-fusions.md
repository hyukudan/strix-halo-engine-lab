# Residual decode boundary fusions

Two independent environment switches are off by default in DeviceForward and BatchForward:

| Switch | Behavior |
|---|---|
| `STRIX_DS41_R1FUSE_QNORM_A8=1` | The in-place query RMSNorm also emits the A8 codes, deltas and sums for the following q_b projection, removing its separate quantizer. A trellis q_b continues to consume FP32 without this A8 emission. |
| `STRIX_DS41_R1FUSE_ROUTES=1` | After the ordinary router scores, one block selects each row using the existing wave selection and builds the integer route CSR after a block barrier. The following routed-expert site reuses those routes. |

The query norm preserves its chunk partition, butterfly and FP32 output. Its A8 groups use the existing quantizer arithmetic. Route selection preserves the comparison tree, tie ordering, selected-score sum and weight normalization. Route preparation preserves expert/slot order, ownership range, base, low-half counts and untouched padding. Neither switch changes the model identity or FP32 fold order.

The route fusion supports up to eight rows and 64 selected entries. Unsupported shapes, explicitly selected fast-router forms, and host-streamed expert routing retain their existing paths. Route preparation runs before any consumer reads the selected ids. No additional inter-block synchronization or temporary device allocation is needed.

`strix-hip-ds41-r1fuse-tests` compares the fused paths with the separate GPU operators for rows 1 through 8, image routing, tied/NaN biases, empty/partial/full ownership ranges, signed zeros and subnormal norm inputs. Full-model qualification and performance are separate gates. The switches remain experimental and do not imply a production performance improvement.

For mixed-window diagnostics, `strix-deepseek41-rank --window-check ... --op-hashes` prints operation hashes for its R1 calls as well as the existing physical-row logits records.
