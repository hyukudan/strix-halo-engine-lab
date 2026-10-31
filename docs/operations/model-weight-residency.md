# Selective model weight backing

`strix-hip-decode-token` can copy an explicitly selected set of stored tensor
bytes into one owned HIP allocation while retaining the immutable source mapping.
This is an opt-in experiment. The default `--model-copy-tier none` retains the
existing registered mapping. Changing backing does not select another codec,
change tensor values, or establish a speed improvement.

| Tier | Selected local, resident tensors |
| --- | --- |
| `none` | No selective copy |
| `normalization` | Normalization tensors |
| `hot_nonexpert` | Output head, normalization, attention/dense projections, router, recurrent, sparse-index and PLE parameters |
| `experts` | MoE expert tensors only |

Remote and unpromoted file-backed tensors are excluded. Selection uses typed
manifest roles. Each active tier requires all three explicit byte arguments:

```
--residency register --model-copy-tier experts --coalesce-gap 0
--model-copy-payload-bytes PAYLOAD
--model-retained-budget RETAINED
--model-setup-peak-budget PEAK
```

`PAYLOAD` must equal the sum of selected tensor payload bytes. The retained and
setup limits bound declared model backing: the page-rounded source mapping plus
the copied arena, including copied boundary padding. Expert setup peak also
includes one temporary pinned host source of at most 32 MiB (or the largest
logical copied range, if smaller). This temporary lease is retired before
consumer bindings are published and does not increase retained backing.
Registered aliases do not add another backing reservation. These limits exclude decoder workspaces and
driver allocations; they are not observations of physical RAM use or available
capacity. A model mapping and its copied file cache can coexist during setup.

The `experts` tier requires zero registration coalescing gap. It registers only
the remaining promoted tensors; shared first/last source pages can still be
registered. Admission rejects registration of an expert range's interior pages
or complete coverage of a copied range by registered boundary pages. This avoids
keeping the entire copied expert population explicitly registered as well.
A backing range can use several transfers, split at source registration boundaries.
For experts, each bounded chunk is copied from the mapping into the same pinned
source lease, then transferred to its unchanged arena offset. The original-device
completion guard drains each transfer before the lease is reused or released.
Source payload and boundary padding bytes are preserved. Normalization and hot
tiers retain their direct-copy path. Staging adds startup memory traffic and does
not establish a speed improvement or removal of file-cache pressure.
It does not assert how the HIP runtime stages or pins copy sources internally.

Copies finish before bindings become available. Tensor addresses preserve the
source offset modulo the system page size. The mapping and the single copied
arena remain owned through consumer completion. Early capture/launch admission,
payload and budget checks, range coverage, move ownership and teardown guards
apply before the backing can be used. A failed setup never falls back silently.

The CLI currently admits ordinary eager execution without serving, graph replay,
MTP, integer heads, GPU token selection, planned PLE IO, prefix reuse or alternate
controllers. All tiers accept `exact-v5`. Explicit `experts` experiments also
accept `hc-splitk-f32+q51-q80-a8-down-v1` and
`hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup-v1`; other arithmetic compositions
are rejected. An explicit `--optimization-profile auto` composes the backing
choice with automatic organization. Alternatively, supply the complete manual
organization. `--coalesce-gap` is a low-level option and otherwise prevents
implicit automatic selection.

Receipts report the effective tier, payload, copied arena, tensor/range counts,
registered alias bytes and retained/setup reservations. Backing inventory counts
the mapping and copied arena once each. `model_copy_qualification` remains
`not_assessed`; an explicit selection is not an adoption or qualification claim.

Qualify every composed recipe against its own unchanged-backing baseline using
complete logits, generated history, initial/final state payloads, dispatch
counters and selected kernels. Separate these checks and instrumented copy/API
measurements from ordinary full-request performance samples. A result for one
arithmetic recipe does not qualify another, and isolated gains are not additive.

## Registered page lock

Registered (`--residency register`) model ranges are locked in RAM by default
(`mlock2` with `MLOCK_ONFAULT`) when `RLIMIT_MEMLOCK` covers every registered
byte. A registered page that the kernel reclaims or migrates invalidates the
userptr registration, and the driver then evicts every GPU queue of the process
until the whole registration is revalidated. Under host memory pressure, for
example an HTTP server sizing its slots while other files fill the page cache,
that restore can take minutes or not complete, and the next GPU submission
waits indefinitely. Locked pages stay resident and are no longer reported as
`MemAvailable`, so automatic HTTP slot sizing sees the real headroom.

When the limit is too small, or `mlock2` fails, startup prints one notice and
the pages stay unlocked (previous behaviour). HTTP slot sizing then subtracts
the unlocked registered bytes from `MemAvailable`, because reclaiming them is
not free capacity for this process. Weight bytes, kernels and arithmetic are
unchanged.

Off switch: `STRIX_MODEL_REGISTER_MLOCK=0` (unlocked pages; slot sizing still
subtracts them).
