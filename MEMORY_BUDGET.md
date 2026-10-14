# UMA/GTT memory budget — operational draft

For current qualification, see the [ledger](docs/qualification-ledger.md).
The numerical examples below retain their original planning assumptions and
external provenance; they are not measurements on a local host.

## Correct model

Strix Halo physically shares RAM. GTT limits the address space that `amdgpu`
can map; it is not additional physical memory. Increasing it lets the GPU see
nearly all RAM, but capacity must still be reserved for the kernel, services,
page cache, recovery and temporary peaks.

## Proposed policy for 128 GB

- BIOS: use the smallest graphics carve-out supported by firmware (target
  0.5–2 GiB; never use `Auto` without checking the resulting allocation).
- GTT: calculate from actual RAM and offer a maximum profile near 124 GiB.
  The value published by external runtime A for a nominal 128 GB host is
  `amdgpu.gttsize=126976` MiB and `ttm.pages_limit=32505856` 4 KiB pages.
- Admission: preserve 16–20 GiB of effective headroom for the host, page cache,
  recovery and peaks, even when the mapping limit is higher. Maximum mappable
  capacity is not maximum allocatable capacity.
- Monitor `mem_info_gtt_used`, `MemAvailable`, locked/registered memory,
  RSS/PSS, page cache and internal engine categories.
- If an unclean process termination leaves an apparent 35–45 GiB in GTT,
  prevent automatic model restart, drain the service and recommend a host
  restart. This is a published observation from external runtime A and must
  be reproduced locally.

`amd_iommu=off` will not be a default: it may improve prefill on a dedicated
host, but reduces DMA isolation and disables the NPU. It may exist only as a
separate opt-in profile with a warning and rollback. Neither
`amdgpu.vm_update_mode=0`, `amdgpu.noretry=0` nor `amdgpu.sg_display=0` will be
adopted without measurement.

## Categories the admission controller must calculate

1. resident weights and mappings;
2. resident PLE/n-gram, row cache and direct-read buffers;
3. per-session KV;
4. GDN recurrent state;
5. MTP, snapshots and rollback;
6. scratch, graphs, input rings and BF16/dequant caches;
7. runtime/libraries and driver headroom;
8. OS, services, page cache and recovery reserve.

The pinned official `config.json` declares 48 layers, full attention every
four layers, 2 KV heads and `head_dim=256`. For conventional BF16 KV:

`12 layers × 2 heads × 256 × (K+V) × 2 bytes = 24 KiB/token/session`.

Thus 262,144 tokens consume 6 GiB per session and C4 would consume 24 GiB,
before recurrence, indexer, MTP, rollback and scratch. This is a planning lower
bound, not an implementation promise; the first context will be selected from
C1/C2/C4 measurements.

The full-attention MTP layer provisionally adds 2 KiB/token, giving
26 KiB/token: 6.5 GiB at 262,144. The drafter's next-step state already has a
fixed geometry of `hc_count × hidden_size = 10,240` elements per session.
The planner accounts for it separately: the default scenario uses
2 bytes/element (20,480 bytes/session), while `--mtp-draft-state-bytes` changes
that assumption. This element size is a **planning assumption**, not the
canonical MTP forward dtype. The indexer might add ~768 B/token if the config's
compression translates directly into a cache; confirm this against the
official forward. Estimated minimum FP32 GDN state is
`36 × 48 × 128 × 128 × 4 = 108 MiB/session`, plus conv/history. Seven full
snapshots would be ~756 MiB/session, so minimal snapshots plus recomputation
will be studied. These values are planning inferences, not finalized contracts.

The planner will publish two scenarios: resident PLE and direct/nonresident
PLE. Row cache, page cache and every BF16/dequant cache have hard limits and
separate accounting; `O_DIRECT` and/or `posix_fadvise(DONTNEED)` will be compared
to prevent `pread` from consuming the host reserve as invisible page cache.

Disk size is not the working set. As external references, external runtime A
publishes about 118 GiB of files for base+overlay but about 68 GiB resident plus
a mapped n-gram table; external platform study keeps a 27.5 GB PLE outside the
resident set using direct reads. Both results require reproduction on our
hardware and selected model.

## Exercise the portable planner

After the [portable build](README.md), run from the repository root:

```bash
./build/strix-memory-plan --tokens 32768 --sessions 1 \
  --resident-gib 68 --baseline-gib 2 --reserve-gib 18
```

These are example inputs, not a qualified host profile. Use measured physical
RAM, baseline usage and the selected model's actual resident bytes when
preparing an admission decision. See [implementation history](IMPLEMENTATION_STATUS.md)
for optional indexer, rollback and MTP planning switches.

## Administrative UI

The panel may calculate and show the boot profile, check RAM, BIOS, kernel
and current parameters, and generate/apply a versioned change. Applying changes
is limited to allowlisted GTT/TTM settings and requires a one-shot boot,
confirmation, a health check and automatic return to the previous boot entry
if the new boot is not confirmed. IOMMU and experimental parameters are changed
only through a console or SSH.
