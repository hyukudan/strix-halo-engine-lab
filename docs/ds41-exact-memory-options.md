# Exact memory options (experimental)

Both options below are disabled by default. They preserve stored values and the arithmetic
order. A default change requires complete-workload qualification and interleaved timings.

* `STRIX_DS41_ATTN_LDS_SWIZZLE=1`: the HIP packed-attention kernels permute 16-byte LDS
  vectors with `slot ^ ((slot >> 3) & 1)`. This map is its own inverse. Stores and reads use
  the same map; dequantization, dot products, softmax, reductions and chunk combination keep
  their original order. This also applies to segmented launches.
* `STRIX_DS41_ENGRAM_PAYLOAD_CACHE=1`: the Linux host row reader keeps 264-byte Engram
  payloads in a per-reader CLOCK cache and deduplicates absolute offsets within each batch.
  The open descriptor owns file identity; model bytes must remain immutable for the reader's
  lifetime, as for resident weights. The cache never spans readers or model files.

The payload cache budget includes its fixed entries and hash index. It is at most 1/64 of
`MemAvailable` and 256 MiB. `STRIX_DS41_ENGRAM_PAYLOAD_CACHE_MIB=0..256` can lower the
budget; it does not enable the feature. Unavailable RAM information, invalid sizing, or
allocation failure disables the optional cache. Direct reads retain their existing aligned
page buffers; the cache stores only useful row payloads. Duplicate outputs are copied only
after every unique miss has completed successfully. Invalid row requests are rejected before
cache lookup.

`EngramRowReader::stats()` exposes unique hits, unique misses, within-batch duplicates,
payload and duplicate bytes served, requested read bytes, output bytes, capacity, occupied
entries, fixed storage bytes and evictions. An enabled reader also prints these counters at
destruction. Read bytes count bytes requested from the file API, not measured physical NVMe
traffic. Hits exclude within-batch duplicates. Failed batches may increase read counters but
do not populate the cache.

The CPU cache test checks direct-page crossings, aliases, eviction, invalid requests, bounded
storage and byte identity with caching off at multiple worker counts. The trace replay tool
accepts one request per line of comma-separated token ids, resets hash history between
requests, and checks every payload against an uncached reader. HIP attention tests check
the layout against the host semantics, including segmented calls, combine modes and A8.
