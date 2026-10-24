# CUDA prefill MoE pilot

Status: **draft**, with both kernel options off by default. Complete-workload
qualification is pending. The exact prefill path remains the reference.

With `--fast-moe`, the app and service accept two independent options:

- `--fast-moe-wide-tiles`: group up to 256 routed entries of an expert. A
  512-thread kernel shares the decoded weight tile across those entries and
  prefetches through three shared-memory stages. Groups of at most 128 entries
  use the existing kernel. The route and activation workspaces retain their
  existing sizes; no persistent decoded expert weights are added.
- `--fast-moe-signed-lut`: decode IQ2 signs through a 128-entry shared table,
  generated from the canonical sign-byte function. The table contains 1 KiB of
  masks. Weight scales and packed model bytes are unchanged.

Both options require the INT8 fast MoE input. They preserve the integer codes,
the order of scaled dot products within each output, the activation quantizer,
and the slot-order combine. They do not enable the fast path when it is off.

For each nonzero IQ2 magnitude byte `a` and sign mask byte `m` (zero or 255),
`(a XOR m) + (m AND 1)` encodes the signed byte. The format's nonzero magnitudes
prevent carry into adjacent bytes when this operation is packed into a word.
The operator tests check all 32,768 codebook/sign pairs against scalar codes.

Native operator checks exercise real weights, partial groups, tensor-range
bases, a nonidentity expert permutation, readiness events, repeated execution
and CUDA graph replay. Builds cover sm_120, sm_89 and sm_86; native checks cover
sm_120 and sm_89. Full-model state/logit checks and performance qualification
are separate gates. Neither compilation nor an operator microbenchmark
establishes end-to-end speed or complete model qualification.

The app's diagnostic `--moe-routes-out FILE` captures the actual selected routes
after each layer's router. Each batch member has its own file (later members
use a numeric suffix). Files are overwritten per run and use the route-file
format consumed by the expert microbenchmark. Capture synchronizes the compute
stream, so traced elapsed times are diagnostic rather than performance results.
