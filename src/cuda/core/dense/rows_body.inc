// Included in both the incumbent and epilogue dense rows kernels.
    pdl_trigger();
    const std::uint32_t g = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);  // warp group: ORO consecutive output rows
    if (g >= ma.row_end[kMaxMulti - 1]) {  // row_end counts groups and is padded with the total
        pdl_wait();
        return;
    }
    // compile-time unrolled selection: no dynamic indexing of the kernel parameter array (would force a local-memory copy)
    const void* wsel = ma.w[0];
    float* y = ma.y[0];
    std::uint32_t start = 0, out = ma.out[0];
#pragma unroll
    for (int jj = 0; jj < kMaxMulti - 1; ++jj) {
        if (g >= ma.row_end[jj]) {
            start = ma.row_end[jj];
            wsel = ma.w[jj + 1];
            y = ma.y[jj + 1];
            out = ma.out[jj + 1];
        }
    }
    const std::uint32_t o0 = (g - start) * ORO;
    if constexpr (RowGrid) {
        static_assert(!FUSED);
        const std::uint32_t first = blockIdx.y * R;
        x += static_cast<std::size_t>(first) * in;
        y += static_cast<std::size_t>(first) * out;
        rows = min(static_cast<std::uint32_t>(R), rows - first);
    }
    const int lane = threadIdx.x & 31;
    prefetch_row<F>(wsel, o0, in, lane);
    if constexpr (FUSED) prefetch_row<F>(w1, o0, in, lane);
    pdl_wait();
    float a0[ORO][R], a1[ORO][R];
#pragma unroll
    for (int oi = 0; oi < ORO; ++oi)
#pragma unroll
        for (int r = 0; r < R; ++r) a0[oi][r] = a1[oi][r] = 0.0f;
    dot_any<F, R, ORO>(wsel, o0, x, in, rows, lane, a0);
    if constexpr (FUSED) dot_any<F, R, ORO>(w1, o0, x, in, rows, lane, a1);
#pragma unroll
    for (int oi = 0; oi < ORO; ++oi) {
        const std::uint32_t o = o0 + oi;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float gsum = warp_sum_xor(a0[oi][r]);
            float v = gsum;
            if constexpr (FUSED) {
                const float u = warp_sum_xor(a1[oi][r]);
                v = __fmul_rn(cc_silu(gsum), u);
            } else if (resid != nullptr) {
                const std::size_t idx = static_cast<std::size_t>(r) * out + o;
                if (static_cast<std::uint32_t>(r) < rows) v = __fadd_rn(resid[idx], gsum);
            }
            if (lane == 0 && static_cast<std::uint32_t>(r) < rows) y[static_cast<std::size_t>(r) * out + o] = STRIX_DENSE_STORE(v);
        }
    }
