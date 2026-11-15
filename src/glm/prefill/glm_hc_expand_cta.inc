// Included inside glm_prefill.hip's anonymous namespace. Expand CTA placement only.
// Pair addition and hoist 0-3 repeat the existing sum-on-load/hoisted expand arithmetic;
// those levers are controls, not gains attributable to the two-row organization.
// Compile with the enclosing translation unit's strict FP flags and wave32 contract.
// LDS = kRows * 256 * sizeof(float): 1/2 KiB for 256/512 threads. Per-thread live
// output/hoist arrays are unchanged. Native VGPR, spills and residency remain a gate.
__device__ __forceinline__ float hc_expand_row_tree256_lane0(float* s, float part, unsigned l) {
    s[l] = part;
    __syncthreads();  // all row groups, including an inactive odd-tail group
    float w = 0.0F;
    if (l < 32) {
        const float t0 = s[l] + s[l + 128], t1 = s[l + 64] + s[l + 192], t2 = s[l + 32] + s[l + 160], t3 = s[l + 96] + s[l + 224];
        w = (t0 + t1) + (t2 + t3);
#pragma unroll
        for (unsigned h = 16; h > 0; h >>= 1U) w = w + __shfl_down(w, h, 32);
    }
    return w;
}

template <bool kPair, unsigned kHoist, unsigned kRows>
__global__ __launch_bounds__(kRows * 256) void hc_expand_cta_kernel(
    const float* X, const float* mix, const float* y, const float* y1, float* Xo,
    float* rr_out, unsigned* flag, unsigned rows) {
    static_assert(kRows == 1 || kRows == 2);
    static_assert(kHoist <= 3);
    constexpr unsigned n = kHc * kHidden;
    static_assert(kHc == 4 && kHidden % 256 == 0);
    __shared__ float s[kRows][256];
    const unsigned group = threadIdx.x / 256, l = threadIdx.x % 256;
    const unsigned r = blockIdx.x * kRows + group;
    float a = 0.0F;
    if (kRows == 1 || r < rows) {  // one-row control has no tail branch; no inactive-row memory access
        bool bad = false;  // the one-row kernel's steps and chain (hc_expand_row_steps), same per-row order
        a = hc_expand_row_steps<kPair, kHoist>(X, mix, y, y1, Xo, r, l, bad);
        if (flag && bad) atomicOr(flag, 1U);
    }
    // Do not call lds_tree256_lane0: it indexes with the whole CTA's threadIdx.x.
    const float ss = hc_expand_row_tree256_lane0(s[group], a, l);
    if ((kRows == 1 || r < rows) && l == 0) rr_out[r] = inv_sqrt(ss / static_cast<float>(n) + kEps);
}

template <bool kPair, unsigned kRows>
void hc_expand_cta_launch(unsigned hoist, const float* X, const float* mix, const float* y, const float* y1,
                          float* Xo, float* rr, unsigned* flag, unsigned rows, hipStream_t stream) {
    const unsigned grid = rows / kRows + (rows % kRows != 0);
    if (hoist == 0) hc_expand_cta_kernel<kPair, 0, kRows><<<grid, kRows * 256, 0, stream>>>(X, mix, y, y1, Xo, rr, flag, rows);
    else if (hoist == 1) hc_expand_cta_kernel<kPair, 1, kRows><<<grid, kRows * 256, 0, stream>>>(X, mix, y, y1, Xo, rr, flag, rows);
    else if (hoist == 2) hc_expand_cta_kernel<kPair, 2, kRows><<<grid, kRows * 256, 0, stream>>>(X, mix, y, y1, Xo, rr, flag, rows);
    else hc_expand_cta_kernel<kPair, 3, kRows><<<grid, kRows * 256, 0, stream>>>(X, mix, y, y1, Xo, rr, flag, rows);
}

template <bool kPair, unsigned kHoist, unsigned kRows>
bool hc_expand_cta_resource_case() {
    hipFuncAttributes attr{};
    const auto kernel = hc_expand_cta_kernel<kPair, kHoist, kRows>;
    ok(hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel)), "hc expand CTA attributes");
    int blocks = 0;
    ok(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, kRows * 256, 0), "hc expand CTA occupancy");
    std::printf("hc-expand-resources rows=%u pair=%u hoist=%u threads=%u regs=%d lds=%zu local=%zu max_threads=%d blocks_per_cu=%d\n",
                kRows, unsigned(kPair), kHoist, kRows * 256, attr.numRegs, attr.sharedSizeBytes,
                attr.localSizeBytes, attr.maxThreadsPerBlock, blocks);
    return attr.maxThreadsPerBlock >= int(kRows * 256) && blocks > 0 && attr.localSizeBytes == 0;
}
