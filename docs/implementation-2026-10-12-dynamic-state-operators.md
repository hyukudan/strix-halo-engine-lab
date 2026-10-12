# Device-dynamic GDN and indexed attention

The existing C1 GDN and indexed-attention launchers now have an explicitly enabled
device-input capture mode. `enable_device_graph_inputs` defaults to false. The
selected kernel ID distinguishes `.device` from `.scalar`, and the launcher
identity binds the capability and model position limit. Registry graph safety
applies to the enabled device variant; it does not establish numerical or model
qualification, or make a different execution plan capture compatible.

Both complete operators use the stream supplied by `HipLaunchScope`. GDN captures
its projections, causal convolution, parameter transforms, normalization,
recurrent update, gated normalization and output projection. Indexed attention
captures its projections, raw-key update, block scoring/selection, norm/RoPE,
KV update, attention scores, softmax, value/gate and output projection. Eager
calls continue to use scalar position and the default stream unless an owned
stream scope is supplied.

The C1 device argument layout supplies a 64-bit position. GDN validates it before
either persistent state write. Indexed attention derives visible tokens,
complete blocks, selected blocks and partial-tail length on the device. Its
capture uses fixed capacity grids and workspace strides; inactive blocks and
slots exit without changing the graph topology. Device position therefore
controls raw-index storage, RoPE, KV storage and the selection frontier on every
replay. Indexed contexts must fit the unsigned 32-bit token-count contract;
positions are checked as 64-bit values before narrowing.

`GraphOperatorControlPlan` is a portable launch preflight consumed by these
operators. It checks exact C1 argument ABI, execution identity, context, four-byte
status, extents, alignment and allocation generations, and rejects controls that
overlap operator resources. Its identity binds these resource views. It neither
retains owners nor reserves memory: the enclosing scheduler must retain all
allocations, exclude conflicting writes/rebinding, and wait for actual stream
completion before inspecting status or publishing state generations.

Status is sticky across operators: 2 means invalid position, 4 means nonfinite
source, and 8 means invalid arithmetic/intermediate values. The operator does
not clear status. The enclosing graph initializes it at the first input stage.
Blocks with barriers atomically observe status once and broadcast the decision
before any barrier-dependent work. Invalid position, upstream failure and
nonfinite input prevent persistent writes. GDN also rejects nonfinite projection
results before convolution. A later arithmetic failure can follow an earlier
state write; this is not transactional execution. The enclosing serial session
must reject the token and poison its arithmetic lineage until explicit recovery.

The indexed block-score RoPE stage now waits for every shared partner read before
overwriting the pooled vector. Softmax waits for every lane to snapshot the
reduced maximum before reusing reduction scratch. These barriers also apply to
the scalar path and preserve its intended arithmetic tree.

Portable control tests execute without HIP. Native fixtures capture the complete
operators on an explicitly owned stream and keep all resources alive through
completion. The GDN fixture compares repeated changing inputs against eager
output/state bits and an independent CPU state trajectory. The indexed fixture
replays positions 0 through 7 with compression ratio 2 and budget 2, crossing
full-visible/top-k and partial-tail boundaries, and compares every output, KV
and raw-index bit with restored eager controls. Both fixtures check oversized
64-bit positions, sticky upstream failure and nonfinite input; GDN additionally
checks overflowing finite projection inputs before persistent mutation.

The portable preflight tests passed locally, and all four changed HIP launcher
and fixture translation units compiled offline for gfx1151 using ROCm 7.2.4.
No GPU execution was available. Capture/replay correctness, native numerical
agreement, whole-model fidelity, internal graph memory consumption and latency
remain hardware qualification gates. Indexed top-k tie qualification remains
unresolved; its registry entries stay experimental.
