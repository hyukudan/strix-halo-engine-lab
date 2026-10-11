# Portable correctness and precision foundations

This implementation slice does not qualify native model inference or direct GPU
transport. It prepares independent correctness checks before hardware measurements.

## Logical messages and physical links

Distributed boundary metadata retains logical plan/request/generation/boundary
identity. Each native slice additionally records its own leased link epoch and
physical sequence. Physical sequences span logical requests sharing a connection
epoch and survive same-epoch lease reacquisition in the shared lease registry.
That registry must outlive the physical connection epochs it governs; replacing
it requires fresh peer-agreed epochs and the backend drain contract.
Reassembly validates each physical commit against its expected slice and retains
the complete logical message identity.

The distributed control envelope is version 2. Version 1 envelopes are rejected;
peers must agree on the new envelope before using native slice metadata. Transfer
descriptor and command protocol versions remain distinct from envelope versioning.

An inbound staging provider returns an explicit completion object. RX consumer
ownership ends only after that object reports completion. Pending, failed, missing
or throwing completion proofs cannot authorize credit or repost. Error handling
poisons the backend and quarantines affected consumer-owned sessions. Recovery
requires the backend-specific drain and resource-lifetime contract; deleting an
object is not proof that a GPU has stopped using memory.

Native progress remains serialized by one owner. These changes do not establish
thread-safe simultaneous callers or GPU event/visibility semantics. Native adapters
must provide and qualify those proofs on the actual hosts and links.

## Low precision arithmetic

`strix::low_precision` provides portable references for integer interpretation,
affine zero-point corrections, ordered scaled-group combination and deterministic
row-local activation quantization. Nonlinear four-bit indices require their lookup
interpretation; their storage width does not make them raw IU4 operands.

The signed-high/unsigned-low decomposition reconstructs already quantized INT8
activations with two four-bit products. It does not preserve original floating
activations or establish a performance advantage. Integer accumulation and
correction overflow are checked without silent saturation.

Candidate accounting checks storage-block eligibility independently of execution
tile geometry. It records theoretical work and bytes, not achieved bandwidth,
kernel selection or model-quality qualification. IQ4_XS storage rejects K=640
without an explicitly supported tail representation.

Compiler inspection, native correctness, full preparation costs, model fidelity,
speculative state equivalence and cost per committed token remain separate gates.
See the [precision proposal](proposals/2026-10-11-phase-specific-low-precision.md)
and the portable experiment instructions under `lab/low_precision`.
