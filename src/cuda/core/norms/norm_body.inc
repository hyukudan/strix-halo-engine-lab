// Shared grouped norm schedule.
    __shared__ float red[8];
    pdl_trigger();
    const PdlWaited waited = pdl_wait();
    const std::uint32_t r = blockIdx.y, g = blockIdx.x, n = xv.n;
    const float* xp = x(waited) + static_cast<std::size_t>(r) * xv.rs() + static_cast<std::size_t>(g) * xv.gs();
    float* yp = y(waited) + static_cast<std::size_t>(r) * yv.rs() + static_cast<std::size_t>(g) * yv.gs();
    constexpr bool kGated = M == Mode::gated_swish || M == Mode::gated_sigmoid;
    [[maybe_unused]] const float* zp = nullptr;
    if constexpr (kGated) zp = z(waited) + static_cast<std::size_t>(r) * zv.rs() + static_cast<std::size_t>(g) * zv.gs();
    // gain / z for all NREG elements are loaded with x, in one batch BEFORE the first store. Inside the store loop each
    // such load had to wait for the previous store (possible aliasing): NREG dependent global round trips per block.
    // Every element is read and written by this thread only, so hoisting the reads is the same program; the per-element
    // ops and the reduction order are unchanged. Precondition: y overlaps x / z only as the identical buffer, or not at all.
    float xr[NREG];
    [[maybe_unused]] float gr[NREG];
    [[maybe_unused]] float zr[NREG];
    float p = 0.0f;
#pragma unroll
    for (int i = 0; i < NREG; ++i) {
        const std::uint32_t idx = threadIdx.x + 256u * i;
        xr[i] = idx < n ? xp[idx] : 0.0f;
        if constexpr (M == Mode::rms_bf16) {
            gr[i] = (gain != nullptr && idx < n) ? bf16_lo(gain[std::size_t{g} * n + idx]) : 1.0f;
        } else {
            gr[i] = (M != Mode::l2 && gain != nullptr && idx < n) ? gain[idx] : 1.0f;
        }
        if constexpr (kGated) zr[i] = idx < n ? zp[idx] : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < NREG; ++i) p = __fadd_rn(p, __fmul_rn(xr[i], xr[i]));
    const float ss = block_sum256(p, red);
    float inv;
    if constexpr (M == Mode::l2) {
        inv = cuda::core::row_math::l2_inverse(ss, eps);
    } else {
        inv = STRIX_CUDA_RMS_INVERSE_RN(ss, static_cast<float>(n), eps);
    }
#pragma unroll
    for (int i = 0; i < NREG; ++i) {
        const std::uint32_t idx = threadIdx.x + 256u * i;
        if (idx < n) {
            float v = __fmul_rn(xr[i], inv);
            if constexpr (M == Mode::l2) {
                v = __fmul_rn(v, scale);
            } else {
                if (gain != nullptr) v = __fmul_rn(v, gr[i]);
                if constexpr (M == Mode::gated_swish) v = __fmul_rn(v, cc_silu(zr[i]));
                if constexpr (M == Mode::gated_sigmoid) v = __fmul_rn(v, cc_sigmoid(zr[i]));
            }
            if constexpr (M == Mode::rms_bf16) v = __bfloat162float(__float2bfloat16_rn(v));
            yp[idx] = v;
        }
    }
