// Store the shared split-K output fragment or finalize an unsplit job.
    // epilogue: C fragment (c0,c1: row g, cols 2t,2t+1; c2,c3: row g+8)
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        if (!on[mt]) continue;
        const std::uint32_t m = (bx * kWarpsPerBlock + warp) * MT + mt;
#pragma unroll
        for (int rb = 0; rb < RB; ++rb)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const std::uint32_t n = static_cast<std::uint32_t>(rb * 8 + 2 * t + (q & 1));
                const std::uint32_t o = m * 16 + static_cast<std::uint32_t>(g + ((q >> 1) << 3));
                if (n >= rows) continue;
                if (ks == 1 && J.w[1] == nullptr) {
                    float v = __fmul_rn(acc[mt][rb][q], gs0);
                    if (resid != nullptr) v = __fadd_rn(resid[static_cast<std::size_t>(n) * out + o], v);
                    y[static_cast<std::size_t>(n) * out + o] = v;
                } else {
                    part[((static_cast<std::size_t>(blockIdx.z) * ks + kidx) * rows + n) * out + o] = acc[mt][rb][q];
                }
            }
    }
