# Persistent HC mix for verify-window rows

`STRIX_HC_PERSISTENT=1` runs each HC step of a verify window (2, 4 or 8 rows; attention, MLP and final mix)
as one launch instead of three dependent ones (normalisation, split-K projection, lane-chain output).
`STRIX_HC_PERSISTENT_INJECT=1` (together with the first) also folds the inject step that writes the
step's input streams into the same launch. Both default off. Prompt prefill, single-row decode and the
MTP draft keep their kernels.

## Admission

The persistent form is used when the HC step has the split-K F32 projection and the rank-320 lane-chain
output selected (the recommended profiles), hidden 2560, 4 streams, Q8_0 down/up and an F32 or absent
inject matrix. Otherwise the three-kernel form runs. The launch has 40 workgroups of 512 threads that
must be resident at once: two per WGP on gfx1151 (at most 192 VGPRs per lane, no scratch). The launcher
checks this once per kernel instantiation from the compiled resources and falls back when it does not
hold. `STRIX_HC_PERSISTENT_VERBOSE=1` prints the decision.

## Arithmetic

Every value keeps the expression and operation order of the three-kernel form:

- Normalisation: 256 leaves per (row, stream), leaf `l` summing features `l, l + 256, ...` in order, the
  same reduction tree (strides 128..1), `rsqrtf(sum / hidden + epsilon)`, and `input * inverse * weight`.
- Split-K projection: per (row, output, 640-wide tile) the lane chain over the tile's five 128-wide
  segments, the xor lane reduction, the ascending sum over the 16 tiles, then `x * sigmoid(x)` for the
  low values and `2 * sigmoid(x)` for the injection gates. Rows in groups of up to four share each
  decoded coefficient; each row's chain is unchanged.
- Output: per row and up row the serial chain over its 320 coefficients, then the ascending stream mix.
- Folded inject: `hyper + hidden * gate`, stored in the activation format, exactly as the inject launch.

The fixture `strix-hip-known-window-hc-persistent-tests` (`strix-hip-known-window-tests --hc-rows`
with both switches) requires the 36 HC rows cases to equal the serial R1 launches bit for bit with the
persistent form launched, and the folded inject + HC launch to equal the inject and HC rows launches over
the whole window arena for three inject/HC pairs at 2, 4 and 8 rows.

## Synchronisation

The phases are separated by grid barriers on a per-stream counter that only grows; launches on one
stream run in order, so the host passes each launch its base. The split-K phase hands out its 160 units
(two low outputs over the 16 tiles) and 4 inject units from a second launch-relative counter, so that
workgroups whose weights arrive late take fewer units. One wave per workgroup arrives with an
atomic add and polls with a scalar load that bypasses the scalar and GL1 caches; every wave waits for its
own stores before the barrier and invalidates L0/GL1 after it. A workgroup that never arrives makes the
pollers trap after about 2^26 polls instead of hanging. The barriers order only work of the same launch
on the same stream: the launch is eager-only like the other window rows launches.

## Workspace

The window arena may give an HC step's workspace (normalised rows and low values, dead after the step)
the memory of the preceding inject's inputs. A folded launch cannot then write the workspace while other
workgroups still read those inputs, so it works in an owner-held per-stream workspace and leaves the
arena workspace unwritten. The streams, the HC output and the injection gates are the bytes of the two
separate launches.

## Status

Opt-in. Bit-exact in the component fixtures and on complete MTP decode runs (tokens identical, replay
error 0, with and without the folded inject). One-run decode timings are within run-to-run noise of the
three-kernel form: the persistent launch removes two dispatches and its normalisation and output phases
are shorter, but its split-K phase is longer than the split-K kernel. Decode throughput is to be measured
on the complete workload before it becomes a default.
