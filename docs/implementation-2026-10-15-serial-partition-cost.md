# Serial partition cost model

Partition search now models the supported distributed token schedule as a
serial chain. Its compute cost is
`node0_compute_us + node1_compute_us`, and estimated token cost adds uncovered
communication and control costs to that sum. The existing
`critical_compute_us` evaluation field reports this serial compute total.
Balanced memory placement alone does not imply concurrent execution or lower
token latency.

`PartitionSearchPolicy::compute_model` defaults to
`PartitionComputeModel::serial_chain`. The reserved `stage_overlap` model and
unknown enum values are rejected. Aggregate per-node timing totals cannot
establish a concurrent critical path; supporting overlap would require a
validated per-stage schedule, dependencies and matching timing evidence.

Capacity feasibility, resident/all-manifest byte accounting, maximum node
weight, memory imbalance and synchronization/transfer accounting are
unchanged. Capacity ranking still prioritizes lower maximum node weight and
imbalance, then estimated token cost. A feasible capacity candidate can be
selected even when no candidate improves latency. The latency objective uses
the corrected serial estimate and the configured improvement threshold.

Inputs must remain finite and nonnegative. Checked additions reject serial
compute, communication or control overflow, and a nonfinite improvement
fraction is rejected. Tests cover placement-derived timing sums, balanced
half-workloads with communication overhead, capacity-only selection, no
eligible candidate, unsupported cost models and overflow.

These are planning calculations. A structurally accepted placement, supplied
timing totals or a favorable estimate does not qualify its kernels, transport,
state ownership or end-to-end distributed execution. The change introduces
no new scheduling, state replication or concurrent runtime behavior.
