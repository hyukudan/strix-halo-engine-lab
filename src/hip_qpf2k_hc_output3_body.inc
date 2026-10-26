    const auto* up_rows = static_cast<const unsigned char*>(up) + feature * kRowBytes;
    using ConstantFloat = const __attribute__((address_space(4))) float;
    using GlobalFloat = const __attribute__((address_space(1))) float;
    auto* low_t = reinterpret_cast<ConstantFloat*>(reinterpret_cast<std::uintptr_t>(rows.low_t + row0));
    auto* low_v = reinterpret_cast<GlobalFloat*>(reinterpret_cast<std::uintptr_t>(rows.low_t + row0));
    const std::uint32_t pitch = rows.low_t_pitch;
    // Q > 0: wave 0 of the block issues the bulk prefetch (the scalar cache is shared by the block's waves).
    const bool lead = __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x)) < 32;
    const std::uint64_t low_bytes = reinterpret_cast<std::uintptr_t>(rows.low_t + row0);
    unsigned sqc = 0U;
    float g[S * R];
#pragma unroll
    for (unsigned i = 0; i < S * R; ++i) g[i] = 0.0F;
    // Prefetch registers: slot block % A holds the two loads for block b issued at the start of block b - A.
    float pf[A ? A : 1U][2];
    const auto prefetch = [&](unsigned block, float (&d)[2]) {
        const unsigned b = block < kBlocks ? block : kBlocks - 1U;
#pragma unroll
        for (unsigned i = 0; i < 2U; ++i) {
            GlobalFloat* step_row = low_v + static_cast<std::uint64_t>(b * 32U + 16U * i + lane / 2U) * pitch;
            d[i] = step_row[16U * (lane % 2U)];
        }
    };
    // Block 0 is not prefetched (no lead time; its first consume must not wait); blocks 1 .. A - 1 here, then block b + A
    // at the start of block b.
#pragma unroll
    for (unsigned a = 0; a < (A ? A : 1U); ++a) pf[a][0] = pf[a][1] = 0.0F;
    if constexpr (A == 2U) prefetch(1U, pf[1]);
    unsigned words[S][9];
    const auto fetch = [&](unsigned block) {
        const bool odd = block % 2U;
#pragma unroll
        for (unsigned s = 0; s < S; ++s) {
            const auto* aligned = reinterpret_cast<const unsigned*>(up_rows + static_cast<std::uint64_t>(s) * hidden * kRowBytes +
                block * 34U - (odd ? 2U : 0U));
#pragma unroll
            for (unsigned n = 0; n < 9U; ++n) words[s][n] = aligned[n];
        }
    };
    fetch(0U);
    for (unsigned block = 0; block < kBlocks; ++block) {
        const bool odd = block % 2U;
        float scale[S];
        unsigned codes[S][8];
#pragma unroll
        for (unsigned s = 0; s < S; ++s) {
            scale[s] = __half2float(__ushort_as_half(static_cast<unsigned short>(odd ? words[s][0] >> 16U : words[s][0] & 0xffffU)));
            // FP32 scale materialized as in the base (v_cvt_f32_f16, then v_mul_f32): without this the 32-step form folds the
            // conversion into v_fma_mix_f32(scale16, x, -0), a different instruction on non-finite FP16 scales.
            asm volatile("" : "+v"(scale[s]));
#pragma unroll
            for (unsigned m = 0; m < 8U; ++m)
                codes[s][m] = (odd ? words[s][m + 1U] : __builtin_amdgcn_alignbyte(words[s][m + 1U], words[s][m], 2U)) ^ 0x80808080U;
        }
        if (block + 1U < kBlocks) fetch(block + 1U);
        if constexpr (A != 0U) {
            // The registers loaded A blocks ago (for this block) are consumed (free wait) and reloaded for block + A.
            if constexpr (A == 1U) {
                asm volatile("" ::"v"(pf[0][0]), "v"(pf[0][1]));
                prefetch(block + 1U, pf[0]);
            } else {
                if (block % 2U == 0U) { asm volatile("" ::"v"(pf[0][0]), "v"(pf[0][1])); prefetch(block + 2U, pf[0]); }
                else { asm volatile("" ::"v"(pf[1][0]), "v"(pf[1][1])); prefetch(block + 2U, pf[1]); }
            }
        }
#pragma unroll 1
        for (unsigned half = 0; half < 32U / U; ++half) {
#pragma unroll
            for (unsigned m = 0; m < U / 4U; ++m) {
#pragma unroll
                for (unsigned j = 0; j < 4U; j += Steps) {
                    const unsigned k = block * 32U + half * U + 4U * m + j;
                    float low[Steps][R];
#pragma unroll
                    for (unsigned h = 0; h < Steps; ++h)
#pragma unroll
                        for (unsigned t = 0; t < R; ++t) low[h][t] = low_t[static_cast<std::uint64_t>(k + h) * pitch + t];
#pragma unroll
                    for (unsigned h = 0; h < Steps; ++h) {
                        float c[S];
#pragma unroll
                        for (unsigned s = 0; s < S; ++s) c[s] = scale[s] * (qpf2k_cvt_ubyte(codes[s][m], j + h) - 128.0F);
                        hc_wide2_output_step(g, c[0], c[1], c[2], c[3], low[h]);
                        if constexpr (Q != 0U) {
                            const unsigned i = 4U * m + j;  // step index in the U-step iteration (h == 0 here)
                            if (h == 0U && i % Q == 0U && lead && k + 2U * Q <= 320U) {
                                // After this batch's wait: the Q steps of the next group into the scalar cache.
                                const std::uint64_t at = low_bytes + static_cast<std::uint64_t>(k + Q) * pitch * 4U;
                                const unsigned lo = static_cast<unsigned>(at), hi = static_cast<unsigned>(at >> 32U), stride = pitch * 4U;
                                if constexpr (Q == 8U)
                                    asm volatile("s_mov_b32 s100, %1\n\ts_mov_b32 s101, %2\n\t" QPF2K_SQC_STEP QPF2K_SQC_STEP QPF2K_SQC_STEP QPF2K_SQC_STEP
                                        QPF2K_SQC_STEP QPF2K_SQC_STEP QPF2K_SQC_STEP QPF2K_SQC_STEP : "=s"(sqc) : "s"(lo), "s"(hi), "s"(stride) : "s100", "s101", "scc");
                                else
                                    asm volatile("s_mov_b32 s100, %1\n\ts_mov_b32 s101, %2\n\t" QPF2K_SQC_STEP QPF2K_SQC_STEP QPF2K_SQC_STEP QPF2K_SQC_STEP
                                        QPF2K_SQC_STEP QPF2K_SQC_STEP QPF2K_SQC_STEP QPF2K_SQC_STEP QPF2K_SQC_STEP QPF2K_SQC_STEP QPF2K_SQC_STEP QPF2K_SQC_STEP
                                        QPF2K_SQC_STEP QPF2K_SQC_STEP QPF2K_SQC_STEP QPF2K_SQC_STEP : "=s"(sqc) : "s"(lo), "s"(hi), "s"(stride) : "s100", "s101", "scc");
                            }
                            // One batch later (its lgkmcnt(0) waited for the bulk too): the dummy may be reused.
                            if (h == 0U && i % Q == 2U) asm volatile("" ::"s"(sqc));
                        }
                    }
                }
            }
            if constexpr (U == 16U) {
#pragma unroll
                for (unsigned s = 0; s < S; ++s)
#pragma unroll
                    for (unsigned n = 0; n < 4U; ++n) codes[s][n] = codes[s][n + 4U];
            }
        }
    }
    if constexpr (P == 0U) {
#pragma unroll
        for (unsigned t = 0; t < R; ++t) {
            if (row0 + t >= rows.count) break;
            float mixed = 0.0F;
            const auto* normalized = wide_row(rows.normalized, rows.normalized_stride, row0 + t);
#pragma unroll
            for (unsigned s = 0; s < S; ++s)
                mixed += normalized[static_cast<std::uint64_t>(s) * hidden + feature] * sigmoid(g[4U * t + s]);
            store_activation(&wide_row(static_cast<Storage*>(rows.output), rows.output_stride, row0 + t)[feature],
                mixed / static_cast<float>(S));
        }
    } else {
        // Rows past the end load the last row (never used); the valid rows are a prefix, as with the base's break.
        float nv[P][S];
        const auto nload = [&](unsigned t, float (&d)[S]) {
            const unsigned r = row0 + t < rows.count ? row0 + t : rows.count - 1U;
            auto* normalized = reinterpret_cast<GlobalFloat*>(reinterpret_cast<std::uintptr_t>(wide_row(rows.normalized, rows.normalized_stride, r)));
#pragma unroll
            for (unsigned s = 0; s < S; ++s) d[s] = normalized[static_cast<std::uint64_t>(s) * hidden + feature];
        };
#pragma unroll
        for (unsigned t = 0; t < P; ++t) nload(t, nv[t]);
#pragma unroll
        for (unsigned t = 0; t < R; ++t) {
            float cur[S];
#pragma unroll
            for (unsigned s = 0; s < S; ++s) cur[s] = nv[t % P][s];
            if (t + P < R) nload(t + P, nv[t % P]);
            if (row0 + t < rows.count) {
                float mixed = 0.0F;
#pragma unroll
                for (unsigned s = 0; s < S; ++s) mixed += cur[s] * sigmoid(g[4U * t + s]);
                store_activation(&wide_row(static_cast<Storage*>(rows.output), rows.output_stride, row0 + t)[feature],
                    mixed / static_cast<float>(S));
            }
        }
    }
