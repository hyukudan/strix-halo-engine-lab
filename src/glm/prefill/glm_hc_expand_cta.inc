// Included inside glm_prefill.hip's anonymous namespace. Expand CTA placement only.
// Pair addition and hoist 0/1/2 repeat the existing sum-on-load/hoisted expand arithmetic;
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
    static_assert(kHoist <= 2);
    constexpr unsigned n = kHc * kHidden, B = kHidden / 256;
    static_assert(kHc == 4 && kHidden % 256 == 0);
    __shared__ float s[kRows][256];
    const unsigned group = threadIdx.x / 256, l = threadIdx.x % 256;
    const unsigned r = blockIdx.x * kRows + group;
    float a = 0.0F;
    if (kRows == 1 || r < rows) {  // one-row control has no tail branch; no inactive-row memory access
        const float* m = mix + static_cast<std::size_t>(r) * kHcMix;
        const float* c = m + 2 * kHc;
        using CP = std::conditional_t<kHoist == 2, const float* __restrict__, const float*>;
        using OP = std::conditional_t<kHoist == 2, float* __restrict__, float*>;
        const CP Xr = X + static_cast<std::size_t>(r) * n;
        const OP Xor = Xo + static_cast<std::size_t>(r) * n;
        float v[kHc - 1][B];
        float yv[kHoist != 0 ? B : 1];
        if constexpr (kHoist != 0) {
            const float* __restrict__ yr = y + static_cast<std::size_t>(r) * kHidden;
            const float* __restrict__ y1r = kPair ? y1 + static_cast<std::size_t>(r) * kHidden : nullptr;
#pragma unroll
            for (unsigned b = 0; b < B; ++b) yv[b] = kPair ? yr[b * 256 + l] + y1r[b * 256 + l] : yr[b * 256 + l];
        }
        bool bad = false;
#pragma unroll
        for (unsigned b = 0; b < B; ++b) {
            const unsigned d = b * 256 + l;
            const std::size_t yi = static_cast<std::size_t>(r) * kHidden + d;
            const float yd = kHoist != 0 ? yv[b] : kPair ? y[yi] + y1[yi] : y[yi];
#pragma unroll
            for (unsigned i = 0; i < kHc; ++i) {
                const float mx = ((c[0 * kHc + i] * Xr[d] + c[1 * kHc + i] * Xr[kHidden + d]) + c[2 * kHc + i] * Xr[2 * kHidden + d]) +
                                 c[3 * kHc + i] * Xr[3 * kHidden + d];
                const float o = m[kHc + i] * yd + mx;
                bad = bad || !is_finite(o);
                Xor[i * kHidden + d] = o;
                if (i == 0) a = __builtin_fmaf(o, o, a);
                else v[i - 1][b] = o;
            }
        }
        if (flag && bad) atomicOr(flag, 1U);
#pragma unroll
        for (unsigned i = 0; i < kHc - 1; ++i)
#pragma unroll
            for (unsigned b = 0; b < B; ++b) a = __builtin_fmaf(v[i][b], v[i][b], a);
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
    else hc_expand_cta_kernel<kPair, 2, kRows><<<grid, kRows * 256, 0, stream>>>(X, mix, y, y1, Xo, rr, flag, rows);
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
