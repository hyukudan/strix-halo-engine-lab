// Included twice by f16_tile_gemm.hpp. The compile-time packed variant
// supplies BF16 operands while the dense variant retains its FP16 instruction sequence.
#if defined(STRIX_F16_TILE_PACKED_OPERANDS)
template <Layout L, class Store, bool kA16, class Weights>
__device__ __forceinline__ void core(const Args& p, Storage& lds, const Store& store, const Weights& weights) {
    static_assert(kA16 && L == Layout::NK);
#else
template <Layout L, class Store, bool kA16 = false>
__device__ __forceinline__ void core(const Args& p, Storage& lds, const Store& store) {
#endif
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U, wm = wave & 3U, wn = wave >> 2U;
    const unsigned nt = p.N / kBN;
    const unsigned n0 = (p.order == 2U ? blockIdx.x % nt : blockIdx.x) * kBN;
    const unsigned r0 = (p.order == 1U ? blockIdx.z : blockIdx.y) * kBM;
    const unsigned b = p.order == 0U ? blockIdx.z : p.order == 1U ? blockIdx.y : blockIdx.z * p.group + blockIdx.x / nt;
#if defined(STRIX_F16_TILE_PACKED_OPERANDS)
    const unsigned wave_row = r0 + 32U * wm;
    const unsigned live_tiles = wave_row + 16U < p.rows ? 2U : (wave_row < p.rows ? 1U : 0U);
#endif
    const unsigned c = lane & 15U;
    // A stager: row tid / 2, k half (tid % 2) * 16 of the stage. Rows past `rows` read row 0 (never stored).
    const unsigned ar = tid >> 1U, ak = (tid & 1U) * 16U;
    const std::size_t aoff = b * p.a_batch + static_cast<std::size_t>(r0 + ar < p.rows ? r0 + ar : 0U) * p.a_row + ak;
    const float* ap = kA16 ? nullptr : p.a + aoff;
    const std::uint16_t* ap16 = kA16 ? p.a16 + aoff : nullptr;
    // W stager. NK: column n0 + tid / 2, k half (tid % 2) * 16. KN: k = lane, columns n0 + 16 * wave .. + 15.
#if defined(STRIX_F16_TILE_PACKED_OPERANDS)
    typename Weights::Tile16 packed_rb;
#else
    const std::uint16_t* wp = p.w + b * p.w_batch +
                              (L == Layout::NK ? static_cast<std::size_t>(n0 + (tid >> 1U)) * p.w_ld + (tid & 1U) * 16U
                                               : static_cast<std::size_t>(lane) * p.w_ld + n0 + 16U * wave);
#endif
    Float4 ra[4];
    U4 ra16[2];
#if !defined(STRIX_F16_TILE_PACKED_OPERANDS)
    U4 rb[2];
#endif
    const auto load = [&](unsigned k0) __attribute__((always_inline)) {
        if constexpr (kA16) {
            ra16[0] = *reinterpret_cast<const U4*>(ap16 + k0);
            ra16[1] = *reinterpret_cast<const U4*>(ap16 + k0 + 8U);
        } else {
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i) ra[i] = *reinterpret_cast<const Float4*>(ap + k0 + 4U * i);
        }
#if defined(STRIX_F16_TILE_PACKED_OPERANDS)
        packed_rb = weights.template load_tile16<true>(n0 + (tid >> 1U), k0 + ak);
#else
        const std::uint16_t* q = wp + (L == Layout::NK ? std::size_t{k0} : std::size_t{k0} * p.w_ld);
        rb[0] = *reinterpret_cast<const U4*>(q);
        rb[1] = *reinterpret_cast<const U4*>(q + 8);
#endif
    };
    const auto publish = [&]() __attribute__((always_inline)) {
        U4 h0, h1;
        if constexpr (kA16) {
            h0 = ra16[0];
            h1 = ra16[1];
        } else {
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i) {
                const Float4 f = ra[i];
                const std::uint32_t lo = detail::pack(f[0], f[1]), hi = detail::pack(f[2], f[3]);
                if (i < 2U) { h0[2U * i] = lo; h0[2U * i + 1U] = hi; }
                else { h1[2U * (i - 2U)] = lo; h1[2U * (i - 2U) + 1U] = hi; }
            }
        }
        *reinterpret_cast<U4*>(&lds.a[ar][ak / 2U]) = h0;
        *reinterpret_cast<U4*>(&lds.a[ar][ak / 2U + 4U]) = h1;
#if defined(STRIX_F16_TILE_PACKED_OPERANDS)
        std::uint32_t words[8];
        weights.decode_bf16(packed_rb, words);
        *reinterpret_cast<U4*>(&lds.b[tid >> 1U][ak / 2U]) = U4{words[0], words[1], words[2], words[3]};
        *reinterpret_cast<U4*>(&lds.b[tid >> 1U][ak / 2U + 4U]) = U4{words[4], words[5], words[6], words[7]};
#else
        if constexpr (L == Layout::NK) {
            *reinterpret_cast<U4*>(&lds.b[tid >> 1U][(tid & 1U) * 8U]) = rb[0];
            *reinterpret_cast<U4*>(&lds.b[tid >> 1U][(tid & 1U) * 8U + 4U]) = rb[1];
        } else {
#pragma unroll
            for (unsigned i = 0; i < 16U; ++i) {
                const std::uint32_t word = rb[i / 8U][(i % 8U) / 2U];
                reinterpret_cast<std::uint16_t*>(&lds.b[16U * wave + i][0])[lane] =
                    static_cast<std::uint16_t>(word >> (16U * (i % 2U)));
            }
        }
#endif
    };
    Float8 acc[2][4] = {};
    load(0U);
    for (unsigned k0 = 0; k0 < p.K; k0 += kBK) {
        __syncthreads();  // every wave finished reading the previous stage
        publish();
        __syncthreads();
        if (k0 + kBK < p.K) load(k0 + kBK);  // in flight across this stage's WMMAs
#if defined(STRIX_F16_TILE_PACKED_OPERANDS)
        // Keep the block-wide staging barriers for inactive waves.
        if (live_tiles == 0) continue;
#endif
#pragma unroll
        for (unsigned u = 0; u < 2U; ++u) {
            Frag fa[2], fb[4];
#pragma unroll
            for (unsigned m = 0; m < 2U; ++m)
#if defined(STRIX_F16_TILE_PACKED_OPERANDS)
                if (m < live_tiles)
#endif
                    fa[m] = detail::fragment(&lds.a[32U * wm + 16U * m + c][8U * u]);
#pragma unroll
            for (unsigned t = 0; t < 4U; ++t) fb[t] = detail::fragment(&lds.b[64U * wn + 16U * t + c][8U * u]);
#pragma unroll
            for (unsigned m = 0; m < 2U; ++m)
#pragma unroll
                for (unsigned t = 0; t < 4U; ++t)
#if defined(STRIX_F16_TILE_PACKED_OPERANDS)
                    if (m < live_tiles) {
                        namespace ops = strix::attn_core::tile;
                        acc[m][t] = ops::mma<ops::Operand::bf16>(
                            __builtin_bit_cast(ops::Frag, fa[m]), __builtin_bit_cast(ops::Frag, fb[t]), acc[m][t]);
                    }
#else
                    acc[m][t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        __builtin_bit_cast(Half16, fa[m]), __builtin_bit_cast(Half16, fb[t]), acc[m][t]);
#endif
        }
    }
    // acc[m][t][j] -> row 32 wm + 16 m + 2 j + lane / 16, column 64 wn + 16 t + lane % 16.
#pragma unroll
    for (unsigned m = 0; m < 2U; ++m)
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) {
            const unsigned r = r0 + 32U * wm + 16U * m + 2U * j + (lane >> 4U);
            if (r < p.rows) {
#pragma unroll
                for (unsigned t = 0; t < 4U; ++t) store.put(b, r, n0 + 64U * wn + 16U * t + c, acc[m][t][j]);
            }
        }
}
