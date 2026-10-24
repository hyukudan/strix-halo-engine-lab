# Experimental staged hyper projections

The native decode executable exposes an optional projection implementation:

```sh
strix-hip-decode-token MODEL.gguf TOKEN_ID \
  --parallel-hyper-mix --staged-hyper-projections
```

`--staged-hyper-projections` requires `--parallel-hyper-mix`; duplicate flags
and a staged configuration without parallel mixing are rejected. Both flags
default to disabled. Omitting the staged flag retains the existing parallel
projections, and omitting both retains the original mixer.

The staged path uses 128-thread projection workgroups. Loader lanes read
contiguous input features and decode the existing serialized weight formats
into shared FP32 tiles. Eight lanes own independent output accumulators.
Coefficient rows use a 129-float shared-memory pitch for 128-feature tiles.
Injection uses the same tiling with only the configured streams active.
Output mixing processes eight hidden features per workgroup, traversing
streams in order and tiling each low-rank dot product.

Each output retains its ascending input-feature accumulation chain. Stream
scaling, SiLU, sigmoid, stream accumulation and final activation conversion
use the existing expressions. The normalization kernel is unchanged: a
256-thread per-stream RMS reduction with one thread broadcasting the inverse.
There is no split dot-product reduction, reassociation policy or projection
weight conversion. These source-level properties are not a guarantee of
bitwise equivalence across compilers or floating-point contraction modes.

All lanes participate in producer and consumer barriers, including lanes
without valid output features. The consumer barrier prevents loaders from
overwriting a tile while another wave still reads it. Tail loops consume only
valid features and outputs; they do not add padded zero terms to the dot
product.

The generation-owned `hyper_workspace` remains
`rows * (hidden_size * hc_count + hc_lowrank) * sizeof(float)`:
42,240 bytes for one canonical model row. It holds normalized input and
low-rank outputs. Staging adds no global temporary, per-step allocation or
host synchronization. The launcher rejects missing, wrongly typed, undersized
and unaligned scratch bindings.

The catalog conservatively declares 4,864 bytes of LDS for the staged variant,
covering the 4,640-byte projection arrays and the normalization kernel's
1,028-byte reduction-plus-inverse storage, rounded to a 256-byte boundary.
The existing parallel variant declares 1,280 bytes to include its shared
inverse. These are admission bounds rather than measured compiler resource
usage or occupancy. The maximum declared workgroup remains 256 threads
because normalization still uses that size; staged projections use 128.
Kernel IDs add `.parallel.staged`, the compiler identity is
`hip-gfx1151-hyper-mix-v4-staged`, and the implementation identity includes
the staged selection.

FP32 and BF16 activation fixtures compare against independent mathematical
references and paired parallel-path outputs and injection weights. Packed
IQ4_XS/Q8_0 fixtures cover attention, MLP and final mixing. A dense signed-weight
fixture uses hidden width 67, three streams and low rank 139 to exercise
feature and output tails. Tests dispatch eagerly and capture static launches
on an explicitly owned stream, then compare three graph replays. They also
check invalid selection, scratch rejection, resource declarations and distinct
identities. Static operator capture does not qualify the complete model graph
or its dynamic inputs.

The host build passes 144 tests. The integrated gfx1151 HIP build passes the
174-test suite on two devices. A private full-checkpoint shadow compares all
output bytes from 1,940 paired hyper-mix calls over 20 model steps: 9,963,520
bytes agree exactly, with maximum absolute error zero. A separate process-level
comparison checks every complete vocabulary logit array over the same 20 steps;
parallel and staged paths agree bitwise, including a staged run with kernel
profiling active. These observations apply to the tested checkpoint, compiler,
device and single-row eager workload.

A longer raw-token workload with context capacity 8,192 generates 512 new
tokens in each path. All 554 complete vocabulary logit arrays, including
conditioning, agree bitwise between parallel and staged execution. This is
one checkpoint and prompt, rather than a broad model-quality qualification.

Multiple rows, repeated complete-workload performance and distributed
execution remain qualification gates. The implementation remains experimental
and opt-in; the tiling design alone does not establish a speed improvement.
