// Included by f16_tile_gemm.hpp for each operand policy. Keep the FP16 and
// packed BF16 staging and instruction sequences intact.
#if defined(STRIX_F16_TILE_PLAIN_BF16) || defined(STRIX_F16_TILE_Q8)
#define STRIX_F16_TILE_PROJECTIONS
#endif
#if defined(STRIX_F16_TILE_PACKED_OPERANDS)
// A binding can reuse the packed tile inside a grid with different row and column origins.
struct PackedTileOrigin {
    unsigned row, column;
};

template <Layout L, class Store, bool kA16, class Weights, class Pair = UnpairedColumns, class Rows = DirectRows,
          class Sched = IncumbentSchedule, bool kBoundOrigin = false>
__device__ __forceinline__ void core(const Args& p, typename Sched::Storage& lds, const Store& store,
                                     const Weights& weights, const Pair& pair = {}, const Rows& source = {},
                                     PackedTileOrigin origin = {}) {
    static_assert(kA16 && L == Layout::NK && (!kBoundOrigin || !Pair::kPaired));
#elif defined(STRIX_F16_TILE_PLAIN_BF16)
// Test hook: apply(sum, row, output column of the projection) sees each completed sum before the projection epilogue;
// the identity default leaves production launches unchanged. A fixture substitutes sums to put chosen payloads through
// the real epilogue.
struct AccumulatorTap {
    __device__ static __forceinline__ float apply(float sum, unsigned, unsigned) { return sum; }
};
// kA16: A arrives as BF16 bits (Args::a16), loaded as is: the operand bits the FP32 staging (pack2<bf16>, RNE) makes of
// the values the image was encoded from. No K16 tail form.
template <bool kHalfTail, unsigned Capacity, bool kGated = false, bool kA16 = false,
          RoundStore Round = RoundStore::incumbent, class Tap = AccumulatorTap, bool Compact = false>
__device__ __forceinline__ void core(const Args& p, std::conditional_t<Compact, Bf16CompactStorage, Storage>& lds,
                                     const Bf16ProjectionGroup<Capacity>& group,
                                     const ControllerRows& gate = {}) {
    static_assert(!(kA16 && (kHalfTail || kGated)));
    static_assert(!Compact || (!kA16 && !kGated));
    constexpr Layout L = Layout::NK;
#elif defined(STRIX_F16_TILE_Q8)
// Staging: the Q8_0 block decoder, or a low-bit storage decoder of the same interface (core/lowbit_f16_staging.hpp).
template <unsigned Capacity, class Staging = q8_f16_staging::Q8F16Staging>
__device__ __forceinline__ void q8_core(const Args& p, Storage& lds, const Q8ProjectionGroup<Capacity>& group) {
    constexpr bool kA16 = false;
#else
template <Layout L, class Store, bool kA16 = false>
__device__ __forceinline__ void core(const Args& p, Storage& lds, const Store& store) {
#endif
#if defined(STRIX_F16_TILE_PACKED_OPERANDS)
    // Schedule (TileSchedule): kRT 16-row WMMA tiles per wave, kWM waves along M, stage width kSK (kU K16 units), kAC A
    // and kWC W chunks of 16 inputs staged per thread and stage, kRowStep rows (columns) between a thread's chunks.
    constexpr unsigned kRT = Sched::kRowTiles, kWM = Sched::kWavesM, kSK = Sched::kStageK, kU = Sched::kUnits;
    constexpr unsigned kAC = Sched::kAChunks, kWC = Sched::kWChunks, kRowStep = Sched::kThreads / kU;
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U, wm = wave % kWM, wn = wave / kWM;
#else
#if !defined(STRIX_F16_TILE_PLAIN_BF16)
    constexpr bool Compact = false;
#endif
    using Shape = Bf16TileShape<Compact>;
    constexpr unsigned row_tiles = Shape::row_tiles, col_tiles = Shape::col_tiles;
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U;
    const unsigned wm = Compact ? wave : wave & 3U, wn = Compact ? 0U : wave >> 2U;
#endif
#if defined(STRIX_F16_TILE_PROJECTIONS)
    constexpr unsigned b = 0U;
    const unsigned r0 = blockIdx.y * Shape::rows;
    // The incumbent (non-compact) statement order: the projection reference before col0 keeps its machine code.
    [[maybe_unused]] const unsigned panel =
        Compact ? 0U : group.count == 1U ? 2U * blockIdx.x + wn : blockIdx.x + wn * gridDim.x;
    unsigned index = Compact ? blockIdx.x : 0U;
    if constexpr (!Compact)
        while (index + 1U < group.count && panel >= group.first_panel[index + 1U]) ++index;
    const auto& projection = group.projections[index];
    const unsigned col0 = Compact ? 0U : (panel - group.first_panel[index]) * kProjectionPanel;
#if defined(STRIX_F16_TILE_Q8)
    const unsigned left = panel < group.first_panel[index + 1U] ? projection.outputs - col0 : 0U;
#else
    const unsigned left = col0 < projection.outputs ? projection.outputs - col0 : 0U;
#endif
    const RawStore store{projection.output + (left != 0U ? col0 : 0U), projection.outputs, 0U};
#else
    const unsigned nt = p.N / kBN;
#if defined(STRIX_F16_TILE_PACKED_OPERANDS)
    const unsigned n0 = kBoundOrigin ? origin.column : (p.order == 2U ? blockIdx.x % nt : blockIdx.x) * kBN;
    const unsigned r0 = kBoundOrigin ? origin.row : (p.order == 1U ? blockIdx.z : blockIdx.y) * Sched::kRows;
    const unsigned b = kBoundOrigin ? 0U : p.order == 0U ? blockIdx.z : p.order == 1U ? blockIdx.y :
                       blockIdx.z * p.group + blockIdx.x / nt;
#else
    const unsigned n0 = (p.order == 2U ? blockIdx.x % nt : blockIdx.x) * kBN;
    const unsigned r0 = (p.order == 1U ? blockIdx.z : blockIdx.y) * kBM;
    const unsigned b = p.order == 0U ? blockIdx.z : p.order == 1U ? blockIdx.y : blockIdx.z * p.group + blockIdx.x / nt;
#endif
#endif
#if defined(STRIX_F16_TILE_PACKED_OPERANDS)
    // Live 16-row WMMA tiles of this wave (rows past `rows` are never multiplied or stored).
    const unsigned wave_row = r0 + 16U * kRT * wm;
    const unsigned live_tiles = wave_row >= p.rows ? 0U : p.rows - wave_row >= 16U * kRT ? kRT : (p.rows - wave_row + 15U) / 16U;
#elif defined(STRIX_F16_TILE_PROJECTIONS)
    const unsigned wave_row = r0 + 16U * row_tiles * wm;
    const unsigned live_tiles = row_tiles == 2U && wave_row + 16U < p.rows ? 2U : (wave_row < p.rows ? 1U : 0U);
#endif
    const unsigned c = lane & 15U;
#if defined(STRIX_F16_TILE_PACKED_OPERANDS)
    // A stager: rows tid / kU + kRowStep j (j < kAC), k (tid % kU) * 16 of the stage (incumbent: row tid / 2, k half
    // (tid % 2) * 16). Rows past `rows` read row 0 (never stored). A row of the operand: the stager's row, or where the
    // binding's row map puts it (MappedRows).
    const unsigned ar = tid / kU, ak = (tid % kU) * 16U;
    const std::uint16_t* ap16[kAC];
#pragma unroll
    for (unsigned j = 0; j < kAC; ++j) {
        const unsigned row = r0 + ar + kRowStep * j;
        ap16[j] = p.a16 + b * p.a_batch + static_cast<std::size_t>(source(row < p.rows ? row : 0U)) * p.a_row + ak;
    }
#else
    // A stager: row tid / 2, k half (tid % 2) * 16 of the stage. Rows past `rows` read row 0 (never stored).
    const unsigned ar = tid >> 1U, ak = (tid & 1U) * 16U;
    const std::size_t aoff = b * p.a_batch + static_cast<std::size_t>(r0 + ar < p.rows ? r0 + ar : 0U) * p.a_row + ak;
    const float* ap = kA16 ? nullptr : p.a + aoff;
    const std::uint16_t* ap16 = kA16 ? p.a16 + aoff : nullptr;
#endif
    // W stager. NK: column n0 + tid / 2, k half (tid % 2) * 16. KN: k = lane, columns n0 + 16 * wave .. + 15.
#if defined(STRIX_F16_TILE_PACKED_OPERANDS)
    // Columns ar + kRowStep j (j < kWC), k = ak. Paired columns (placement only): local columns 0..63 are logical
    // columns 64 x .. 64 x + 63, local 64..127 the same columns + pair.half (the up half of a [gate | up] matrix),
    // x = blockIdx.x.
    typename Weights::Tile16 packed_rb[kWC];
    unsigned wcol[kWC];
#pragma unroll
    for (unsigned j = 0; j < kWC; ++j) {
        const unsigned wl = ar + kRowStep * j;
        wcol[j] = Pair::kPaired ? (wl < 64U ? 64U * blockIdx.x + wl : pair.half + 64U * blockIdx.x + (wl - 64U)) : n0 + wl;
        if constexpr (kBoundOrigin) {
            if (wcol[j] >= p.N) wcol[j] = n0;  // pad a partial output tile with a valid weight row
        }
    }
#elif defined(STRIX_F16_TILE_Q8)
    const unsigned wc = (tid >> 1U) % kProjectionPanel;
    const std::uint8_t* wp = wc < left ? projection.weights + std::size_t{col0 + wc} * p.w_ld : nullptr;
    typename Staging::Raw rb;
#elif defined(STRIX_F16_TILE_PLAIN_BF16)
    // Inactive columns stage a valid weight row. Only live columns store.
    const unsigned wc = (tid >> 1U) % Shape::panel;
    const std::uint16_t* wp = projection.weights +
        static_cast<std::size_t>(wc < left ? col0 + wc : 0U) * p.w_ld + ak;
#else
    const std::uint16_t* wp = p.w + b * p.w_batch +
                              (L == Layout::NK ? static_cast<std::size_t>(n0 + (tid >> 1U)) * p.w_ld + (tid & 1U) * 16U
                                               : static_cast<std::size_t>(lane) * p.w_ld + n0 + 16U * wave);
#endif
#if defined(STRIX_F16_TILE_PACKED_OPERANDS)
    U4 ra16[kAC][2];
    const auto load = [&](unsigned k0) __attribute__((always_inline)) {
#pragma unroll
        for (unsigned j = 0; j < kAC; ++j) {
            ra16[j][0] = *reinterpret_cast<const U4*>(ap16[j] + k0);
            ra16[j][1] = *reinterpret_cast<const U4*>(ap16[j] + k0 + 8U);
        }
#pragma unroll
        for (unsigned j = 0; j < kWC; ++j) packed_rb[j] = weights.template load_tile16<true>(wcol[j], k0 + ak);
    };
    // A copied, W decoded once per CTA and stage (shared by every wave of the CTA).
    const auto publish = [&]() __attribute__((always_inline)) {
#pragma unroll
        for (unsigned j = 0; j < kAC; ++j) {
            *reinterpret_cast<U4*>(&lds.a[ar + kRowStep * j][ak / 2U]) = ra16[j][0];
            *reinterpret_cast<U4*>(&lds.a[ar + kRowStep * j][ak / 2U + 4U]) = ra16[j][1];
        }
#pragma unroll
        for (unsigned j = 0; j < kWC; ++j) {
            std::uint32_t words[8];
            ::strix::core::decode_tile_bf16(weights, packed_rb[j], words);
            *reinterpret_cast<U4*>(&lds.b[ar + kRowStep * j][ak / 2U]) = U4{words[0], words[1], words[2], words[3]};
            *reinterpret_cast<U4*>(&lds.b[ar + kRowStep * j][ak / 2U + 4U]) = U4{words[4], words[5], words[6], words[7]};
        }
    };
    Float8 acc[kRT][4] = {};
    load(0U);
    for (unsigned k0 = 0; k0 < p.K; k0 += kSK) {
        __syncthreads();  // every wave finished reading the previous stage
        publish();
        __syncthreads();
        if (k0 + kSK < p.K) load(k0 + kSK);  // in flight across this stage's WMMAs
        // Keep the block-wide staging barriers for inactive waves.
        if (live_tiles == 0) continue;
#pragma unroll
        for (unsigned u = 0; u < kU; ++u) {
            Frag fa[kRT], fb[4];
#pragma unroll
            for (unsigned m = 0; m < kRT; ++m)
                if (m < live_tiles) fa[m] = detail::fragment(&lds.a[16U * kRT * wm + 16U * m + c][8U * u]);
#pragma unroll
            for (unsigned t = 0; t < 4U; ++t) fb[t] = detail::fragment(&lds.b[64U * wn + 16U * t + c][8U * u]);
#pragma unroll
            for (unsigned m = 0; m < kRT; ++m)
#pragma unroll
                for (unsigned t = 0; t < 4U; ++t)
                    if (m < live_tiles) {
                        namespace ops = strix::attn_core::tile;
                        acc[m][t] = ops::mma<ops::Operand::bf16>(
                            __builtin_bit_cast(ops::Frag, fa[m]), __builtin_bit_cast(ops::Frag, fb[t]), acc[m][t]);
                    }
        }
    }
#else
    Float4 ra[4];
    U4 ra16[2];
#if !defined(STRIX_F16_TILE_PACKED_OPERANDS) && !defined(STRIX_F16_TILE_Q8)
    U4 rb[2];
#endif
    const auto load = [&](unsigned k0) __attribute__((always_inline)) {
#if defined(STRIX_F16_TILE_PLAIN_BF16)
        // Both halves participate in LDS staging. On a K16 tail, the high
        // half loads the valid low half and its fragment is never used.
        if constexpr (kHalfTail) {
            if (ak != 0U && p.K - k0 == 16U) {
#pragma unroll
                for (unsigned i = 0; i < 4U; ++i) ra[i] = *reinterpret_cast<const Float4*>(ap + k0 - ak + 4U * i);
                rb[0] = *reinterpret_cast<const U4*>(wp + k0 - ak);
                rb[1] = *reinterpret_cast<const U4*>(wp + k0 - ak + 8U);
                return;
            }
        }
#endif
        if constexpr (kA16) {
            ra16[0] = *reinterpret_cast<const U4*>(ap16 + k0);
            ra16[1] = *reinterpret_cast<const U4*>(ap16 + k0 + 8U);
        } else {
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i) ra[i] = *reinterpret_cast<const Float4*>(ap + k0 + 4U * i);
        }
#if defined(STRIX_F16_TILE_PACKED_OPERANDS)
        packed_rb = weights.template load_tile16<true>(wcol, k0 + ak);
#elif defined(STRIX_F16_TILE_Q8)
        rb = Staging::load(wp, k0, tid & 1U);
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
#if defined(STRIX_F16_TILE_PLAIN_BF16)
                namespace ops = strix::attn_core::tile;
                const std::uint32_t lo = ops::pack2<ops::Operand::bf16>(f[0], f[1]);
                const std::uint32_t hi = ops::pack2<ops::Operand::bf16>(f[2], f[3]);
#else
                const std::uint32_t lo = detail::pack(f[0], f[1]), hi = detail::pack(f[2], f[3]);
#endif
                if (i < 2U) { h0[2U * i] = lo; h0[2U * i + 1U] = hi; }
                else { h1[2U * (i - 2U)] = lo; h1[2U * (i - 2U) + 1U] = hi; }
            }
        }
        *reinterpret_cast<U4*>(&lds.a[ar][ak / 2U]) = h0;
        *reinterpret_cast<U4*>(&lds.a[ar][ak / 2U + 4U]) = h1;
#if defined(STRIX_F16_TILE_PACKED_OPERANDS)
        std::uint32_t words[8];
        ::strix::core::decode_tile_bf16(weights, packed_rb, words);
        *reinterpret_cast<U4*>(&lds.b[tid >> 1U][ak / 2U]) = U4{words[0], words[1], words[2], words[3]};
        *reinterpret_cast<U4*>(&lds.b[tid >> 1U][ak / 2U + 4U]) = U4{words[4], words[5], words[6], words[7]};
#elif defined(STRIX_F16_TILE_Q8)
        Staging::publish(rb, lane, &lds.b[tid >> 1U][ak / 2U]);
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
    Float8 acc[row_tiles][col_tiles] = {};
#if defined(STRIX_F16_TILE_Q8)
    // Group-scaled staging (low-bit codes): each kGroupK group is a WMMA chain from +0 into sum, then
    // acc = fma(sum, d, acc) with the column's group scale (the small-M schedule's arithmetic).
    constexpr bool kGroupScaled = detail::group_scaled<Staging>;
    [[maybe_unused]] detail::LowBitGroup sum[2][4] = {};
#endif
    load(0U);
    for (unsigned k0 = 0; k0 < p.K; k0 += kBK) {
        __syncthreads();  // every wave finished reading the previous stage
        publish();
        __syncthreads();
        if (k0 + kBK < p.K) load(k0 + kBK);  // in flight across this stage's WMMAs
#if defined(STRIX_F16_TILE_PACKED_OPERANDS) || defined(STRIX_F16_TILE_PROJECTIONS)
        // Keep the block-wide staging barriers for inactive waves.
        if (live_tiles == 0) continue;
#endif
#if defined(STRIX_F16_TILE_Q8)
        if constexpr (kGroupScaled) {
            if (k0 % Staging::kGroupK == 0U)
#pragma unroll
                for (unsigned m = 0; m < 2U; ++m)
#pragma unroll
                    for (unsigned t = 0; t < 4U; ++t) sum[m][t] = {};
        }
#endif
#pragma unroll
        for (unsigned u = 0; u < 2U; ++u) {
#if defined(STRIX_F16_TILE_PLAIN_BF16)
            if constexpr (kHalfTail) {
                if (u != 0U && p.K - k0 == 16U) break;
            }
#endif
            Frag fa[row_tiles], fb[col_tiles];
#pragma unroll
            for (unsigned m = 0; m < row_tiles; ++m)
#if defined(STRIX_F16_TILE_PACKED_OPERANDS) || defined(STRIX_F16_TILE_PROJECTIONS)
                if (m < live_tiles)
#endif
                    fa[m] = detail::fragment(&lds.a[16U * row_tiles * wm + 16U * m + c][8U * u]);
#pragma unroll
            for (unsigned t = 0; t < col_tiles; ++t) fb[t] = detail::fragment(&lds.b[16U * col_tiles * wn + 16U * t + c][8U * u]);
#pragma unroll
            for (unsigned m = 0; m < row_tiles; ++m)
#pragma unroll
                for (unsigned t = 0; t < col_tiles; ++t)
#if defined(STRIX_F16_TILE_PACKED_OPERANDS) || defined(STRIX_F16_TILE_PROJECTIONS)
                    if (m < live_tiles)
#endif
#if defined(STRIX_F16_TILE_PACKED_OPERANDS) || defined(STRIX_F16_TILE_PLAIN_BF16)
                    {
                        namespace ops = strix::attn_core::tile;
                        acc[m][t] = ops::mma<ops::Operand::bf16>(
                            __builtin_bit_cast(ops::Frag, fa[m]), __builtin_bit_cast(ops::Frag, fb[t]), acc[m][t]);
                    }
#else
#if defined(STRIX_F16_TILE_Q8)
                    {
                        if constexpr (kGroupScaled)
                            sum[m][t].add(fa[m], fb[t]);
                        else
                            acc[m][t] = STRIX_HIP_WMMA_F16(
                                __builtin_bit_cast(Half16, fa[m]), __builtin_bit_cast(Half16, fb[t]), acc[m][t]);
                    }
#else
                    acc[m][t] = STRIX_HIP_WMMA_F16(
                        __builtin_bit_cast(Half16, fa[m]), __builtin_bit_cast(Half16, fb[t]), acc[m][t]);
#endif
#endif
        }
#if defined(STRIX_F16_TILE_Q8)
        if constexpr (kGroupScaled) {
            if ((k0 + kBK) % Staging::kGroupK == 0U) {
#pragma unroll
                for (unsigned t = 0; t < 4U; ++t) {
                    const float d = Staging::scale(&lds.b[64U * wn + 16U * t + c][0]);
#pragma unroll
                    for (unsigned m = 0; m < 2U; ++m)
                        if (m < live_tiles) sum[m][t].apply(d, acc[m][t]);
                }
            }
        }
#endif
    }
#endif
    // acc[m][t][j] -> row 32 wm + 16 m + 2 j + lane / 16, column 64 wn + 16 t + lane % 16 (packed: row 16 kRT wm + 16 m +
    // 2 j + lane / 16).
#if defined(STRIX_F16_TILE_PACKED_OPERANDS)
    if constexpr (Pair::kPaired) {
        // Waves wn = 0 hold the gate sums and waves wn = 1 the up sums of the same rows and columns (same accumulator
        // map). 2 kRT passes of 8 rows per wave: both halves go to LDS (2 x 2 kWM KB), then every thread calls
        // pair.put for 8 elements, so the epilogue math is spread over every wave.
        float* exchange = reinterpret_cast<float*>(&lds);
#pragma unroll
        for (unsigned q = 0; q < 2U * kRT; ++q) {
            const unsigned m = q / 2U, h = q % 2U;
            __syncthreads();  // the K loop's (or the previous pass's) LDS reads are done
#pragma unroll
            for (unsigned jj = 0; jj < 4U; ++jj)
#pragma unroll
                for (unsigned t = 0; t < 4U; ++t)
                    exchange[wn * (512U * kWM) + (8U * wm + 2U * jj + (lane >> 4U)) * 64U + 16U * t + c] =
                        acc[m][t][4U * h + jj];
            __syncthreads();
#pragma unroll
            for (unsigned i = 0; i < 8U; ++i) {
                const unsigned e = tid + Sched::kThreads * i, local = e / 64U, column = e % 64U;
                const unsigned r = r0 + 16U * kRT * (local / 8U) + 16U * m + 8U * h + (local % 8U);
                const unsigned col = 64U * blockIdx.x + column;
                if (r < p.rows && col < pair.half) pair.put(r, col, exchange[e], exchange[512U * kWM + e]);
            }
        }
        return;
    }
#endif
#if defined(STRIX_F16_TILE_PLAIN_BF16)
    if constexpr (kGated) {
        // Group [forget, gain] of equal width: waves wn = 1 hold the gain sums of the same rows and columns (same
        // accumulator map) as waves wn = 0 hold the forget sums. Gain goes through LDS, 16 rows of every wave per pass
        // (4 x 16 x 64 FP32 = 16 KB); forget waves store controller_mix(current, message, forget, gain).
        float* exchange = reinterpret_cast<float*>(&lds);
#pragma unroll
        for (unsigned m = 0; m < 2U; ++m) {
            __syncthreads();  // the K loop's (or the previous pass's) LDS reads are done
            if (wn == 1U) {
#pragma unroll
                for (unsigned j = 0; j < 8U; ++j)
#pragma unroll
                    for (unsigned t = 0; t < 4U; ++t)
                        exchange[(16U * wm + 2U * j + (lane >> 4U)) * kProjectionPanel + 16U * t + c] = acc[m][t][j];
            }
            __syncthreads();
            if (wn == 0U) {
#pragma unroll
                for (unsigned j = 0; j < 8U; ++j) {
                    const unsigned r = r0 + 32U * wm + 16U * m + 2U * j + (lane >> 4U);
                    if (r >= p.rows) continue;
#pragma unroll
                    for (unsigned t = 0; t < 4U; ++t) {
                        const unsigned col = 16U * t + c;
                        if (col >= left) continue;
                        const std::size_t i = std::size_t{r} * gate.width + col0 + col;
                        gate.output[i] = ::strix::core::controller_mix(
                            gate.current[i], gate.message[i], acc[m][t][j],
                            exchange[(16U * wm + 2U * j + (lane >> 4U)) * kProjectionPanel + 16U * t + c]);
                    }
                }
            }
        }
        return;
    }
#endif
#if defined(STRIX_F16_TILE_PACKED_OPERANDS)
#pragma unroll
    for (unsigned m = 0; m < kRT; ++m)
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) {
            const unsigned r = r0 + 16U * kRT * wm + 16U * m + 2U * j + (lane >> 4U);
            if (r < p.rows) {
#pragma unroll
#else
#pragma unroll
    for (unsigned m = 0; m < row_tiles; ++m)
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) {
            const unsigned r = r0 + 16U * row_tiles * wm + 16U * m + 2U * j + (lane >> 4U);
            if (r < p.rows) {
#pragma unroll
#endif
#if defined(STRIX_F16_TILE_PROJECTIONS)
                for (unsigned t = 0; t < col_tiles; ++t) {
                    const unsigned col = 16U * t + c;
#if defined(STRIX_F16_TILE_Q8)
                    if (col < left) store.put(b, r, col, acc[m][t][j]);
#else
                    // Projections with round_bf16 store the BF16-rounded FP32 sum (same rounding as fma_rows; Round selects the function).
                    if (col < left) {
                        const float sum = Tap::apply(acc[m][t][j], r, col0 + col);
                        store.put(b, r, col, projection.round_bf16 ? round_stored<Round>(sum) : sum);
                    }
#endif
                }
#else
                for (unsigned t = 0; t < 4U; ++t) store.put(b, r, n0 + 64U * wn + 16U * t + c, acc[m][t][j]);
#endif
            }
        }
}
#undef STRIX_F16_TILE_PROJECTIONS
