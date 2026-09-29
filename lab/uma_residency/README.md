# Strix Halo UMA Residency Lab

This lab qualifies memory-placement decisions on native Linux Strix Halo hosts.

It is intentionally standalone. The main engine build does not require ROCm or HIP in order to compile the host-side core.

## Questions

The lab is designed to answer:

1. How expensive is first-touch faulting for large anonymous and file-backed mappings?
2. What does bounded prefaulting buy?
3. How does random page access behave with sequential, random and no-readahead advice?
4. What is the cost of reclaiming and refaulting file-backed data?
5. Can file-backed or anonymous mappings be registered and mapped through HIP on the target host?
6. How fast is direct GPU access to registered system pages compared with explicit staging?
7. How much physical-memory pressure and cleanup latency does each mode create?

Results from this lab are qualification data. They MUST NOT be generalized from another GPU or operating system.

## Host build

```bash
cmake -S lab/uma_residency -B build/uma-residency -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/uma-residency
```

Examples:

```bash
./build/uma-residency/strix-uma-host \
  --mode anon --bytes 8GiB --pattern scan --iterations 3

./build/uma-residency/strix-uma-host \
  --mode mmap --file /fast/model-pages.bin --bytes 8GiB \
  --pattern random-pages --advice random --drop-pages --iterations 3

./build/uma-residency/strix-uma-host \
  --mode mmap --file /fast/model-pages.bin --bytes 8GiB \
  --pattern scan --prefault --iterations 3
```

## HIP build

Build this only on a native ROCm host:

```bash
cmake -S lab/uma_residency -B build/uma-residency-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRIX_UMA_ENABLE_HIP=ON
cmake --build build/uma-residency-hip
```

Examples:

```bash
./build/uma-residency-hip/strix-uma-hip --mode anon --bytes 2GiB

./build/uma-residency-hip/strix-uma-hip \
  --mode mmap --file /fast/hip-map.bin --bytes 2GiB
```

The HIP probe compares:

- direct GPU read from a registered mapped region;
- direct GPU write to the mapped region;
- host-to-device staging copy;
- device-to-host staging copy.

The direct path is not assumed to be faster. On a unified-memory system, transport-visible or registered system pages can have different cache and write behavior from ordinary GPU allocations.

## Qualification matrix

Run at several sizes that cross working-set boundaries, for example:

- 256 MiB;
- 1 GiB;
- 4 GiB;
- 16 GiB;
- a size representative of the sparse lookup table;
- a size representative of the resident model body.

For file-backed cases run:

- warm page cache;
- cold/reclaimed pages;
- sequential advice;
- random advice;
- prefaulted;
- non-prefaulted.

Record the platform-probe JSON beside every result.

## Output

Both tools emit JSON so results can later be collected by the benchmark harness.

The host tool reports timing plus major/minor fault deltas.

The HIP tool reports direct mapped read/write bandwidth and explicit staging bandwidth.

## Safety

Start with modest sizes.

Pinned or registered memory reduces reclaimability. Do not register most of physical RAM until small and medium tests have demonstrated correct cleanup.

A failed or interrupted HIP registration test should be followed by a platform/memory health check before running a large model.
