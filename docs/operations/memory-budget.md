# UMA/GTT budget — working draft

## Correct model

On Strix Halo the RAM is physically shared. GTT limits how much address space
`amdgpu` can map; it is not additional physical memory. Raising it lets the GPU
see almost all of the RAM, but it does not remove the need to reserve capacity
for the kernel, services, page cache, recovery and transient peaks.

## Proposed policy for 128 GB

- BIOS: the minimum graphics carve-out the firmware allows (target 0.5–2 GiB,
  never `Auto` without checking the result).
- GTT: compute from the actual RAM and offer a maximum profile close to 124 GiB.
  For a nominal 128 GB system the candidate values, still to be measured on our
  hardware, are `amdgpu.gttsize=126976` MiB and `ttm.pages_limit=32505856`
  4 KiB pages.
- Admission: keep a 16–20 GiB effective margin for the host, page cache,
  recovery and peaks, even if the mapping limit is higher. The maximum mappable
  amount is not the maximum allocatable amount.
- Monitor `mem_info_gtt_used`, `MemAvailable`, locked/registered memory,
  RSS/PSS, page cache and the engine's internal categories.
- If a process dies uncleanly and 35–45 GiB apparently remain in GTT, block
  automatic model restart, drain the service and recommend a host reboot. This
  range has not been reproduced locally and must be measured on our hardware.

`amd_iommu=off` will not be a default: it can improve prefill on a dedicated
host, but it reduces DMA isolation and disables the NPU. It may only exist as a
separate opt-in profile, with a warning and rollback. Likewise,
`amdgpu.vm_update_mode=0`, `amdgpu.noretry=0` and `amdgpu.sg_display=0` will not
be adopted without measurement.

AMD's IOMMU PerfOpt (`amdgpu.iommu_perfopt`, Linux 7.4 / IOMMU tree) is the measured middle ground: the
integrated GPU skips IOMMU translation while the IOMMU stays on and the USB4 controllers stay translated.
Same kernel, on vs off, measured on one node: Qwen 32K prefill +2.9..3.1 %, identical tokens. It removes the GPU's own DMA
containment; the lab nodes boot it by default (off: `amdgpu.iommu_perfopt=0` or the previous kernel); see
[PerfOpt: what, trade-off, enable, verify, roll back](https://github.com/hyukudan/strix-halo-usb4-driver/blob/main/docs/PERFOPT.md).

## Categories the admission controller must compute

1. resident weights and mappings;
2. resident PLE/n-gram, row cache and direct-read buffers;
3. KV per session;
4. GDN recurrent state;
5. MTP, snapshots and rollback;
6. scratch, graphs, input rings and BF16/dequant cache;
7. runtime/libraries and driver margin;
8. OS, services, page cache and recovery reserve.

The pinned official `config.json` declares 48 layers, one full-attention layer
every four layers, 2 KV heads and `head_dim=256`. For a conventional BF16 KV:

`12 layers × 2 heads × 256 × (K+V) × 2 bytes = 24 KiB/token/session`.

Therefore 262,144 tokens consume 6 GiB per session and C4 would consume 24 GiB,
before recurrence, indexer, MTP, rollback and scratch. This is a planning lower
bound, not an implementation promise; the first context length will be chosen
from C1/C2/C4 measurements.

The full-attention MTP layer provisionally adds 2 KiB/token, giving
26 KiB/token: 6.5 GiB at 262,144. In addition, the drafter's next-step state
already has a pinned geometry of `hc_count × hidden_size = 10,240` elements per
session. The planner accounts for it as a separate category: the default
scenario uses 2 bytes/element (20,480 bytes/session), but
`--mtp-draft-state-bytes` allows changing the assumption. That element size is
a **planning assumption**, not the canonical dtype of the MTP forward. The
indexer could add ~768 B/token if the config's compression translates directly
into cache; this must be confirmed against the official forward. The estimated
minimum FP32 GDN state is `36 × 48 × 128 × 128 × 4 = 108 MiB/session`, plus
conv/history. Seven full snapshots would be ~756 MiB/session, so a minimal
snapshot plus recompute will be studied. These figures are planning
inferences, not closed contracts.

The planner will publish two scenarios: resident PLE and direct/non-resident
PLE. The row cache, page cache and any BF16/dequant cache have a hard cap and
their own accounting; `O_DIRECT` and/or `posix_fadvise(DONTNEED)` will be
compared to prevent `pread` from turning the host reserve into invisible page
cache.

On-disk size is not the working set. Base and overlay files, resident weights
and a mapped n-gram table must be accounted for separately, and keeping the PLE
outside the resident set through direct reads must be measured on our hardware
with the selected model.

## Administrative UI

The panel may compute and display the boot profile, check RAM, BIOS, kernel and
current parameters, and generate/apply a versioned change. Applying is limited
to allowlisted GTT/TTM parameters and requires a one-shot boot, confirmation, a
health check and automatic return to the previous entry if the new boot is not
confirmed. IOMMU and experimental parameters are changed only via console/SSH.
