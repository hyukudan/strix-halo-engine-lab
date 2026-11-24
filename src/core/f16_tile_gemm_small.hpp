#pragma once
// Small-M schedule of the f16 tile GEMM family for low-bit resident rows (q4_g64 / q6_g64, strix/weight/host_lowbit.hpp):
// 16 or 32 rows per row block, FP16 WMMA for the row products (the cost per weight does not grow with the rows the way
// the VALU rows sweep's does); the row blocks of one panel re-read its bytes from the cache, not from DRAM.
//
// Class: FAST (FP16 activations). Arithmetic: the one of the 128-row projection tile with the low-bit staging trait
// (f16_tile_gemm.hpp gemm_lowbit_projections, core/lowbit_f16_staging.hpp), bit for bit:
//   y[r][n] = sum over K64 groups g in ascending order, from +0, of  fma(S_g, float(d[n][g]), y),
//   S_g = WMMA chain from +0 over the four K16 units of g in ascending k of  fp16(a[r][k]) x code[n][k],
// codes as exact FP16 integers (lowbit_f16::integer_pairs), A packed by detail::pack. There is no K split, so a row's
// bits depend only on its activation row, the weights and K: not on the row count, the row tile, the projection
// grouping, or which of the two routes (core/resident_row_plan.hpp) runs the launch.
//
// CTA: one 64-column projection panel (the panel of the projection tile), 4 waves x 16 columns, 128 threads. A stage is
// one superblock (K128): all threads stage A (FP32 -> FP16) into LDS once per stage. A lane holds ONE column (lane % 16;
// lanes 16..31 replicate it, as the WMMA operand layout requires): it loads that column's superblock (q4 64 B codes, q6
// + 32 B 2-bit plane, the scale word) kPrefetch stages ahead, so its accumulator elements share one scale per group.

#include "core/f16_tile_gemm.hpp"
#include "core/lowbit_f16_staging.hpp"

#if defined(__HIPCC__)
namespace strix::core::f16_tile_gemm::small_m {

// Row tiles of a row block: 1 or 2 (three tiles per CTA measured slower than three one-tile row blocks, whose re-reads of
// the panel hit the cache).
inline constexpr unsigned kWaveColumns = 16U, kMaxRowTiles = 2U, kGroups = 2U, kPrefetch = 2U;

template <unsigned RowTiles>
struct Schedule {
    static_assert(RowTiles >= 1U && RowTiles <= kMaxRowTiles);
    static constexpr unsigned kRows = 16U * RowTiles, kThreads = 128U, kStageK = 64U * kGroups;
    static constexpr unsigned kChunksPerRow = kStageK / 8U, kAChunks = kRows * kChunksPerRow / kThreads;
    static constexpr unsigned kPitch = kStageK / 2U + 4U;  // dwords per LDS row: data + 4 pad (b128 reads conflict free)
    static_assert(kRows * kChunksPerRow % kThreads == 0U);
    struct alignas(16) Storage {
        std::uint32_t a[kRows][kPitch];
    };
};

// Superblock `index` of one column (two groups).
template <class Storage>
struct BlockWords {
    lowbit_f16::Words4 c[2U * kGroups];  // 4-bit plane, 32 bytes per group, element j in nibble j of the byte stream
    lowbit_f16::Words4 h[Storage::high_bytes != 0U ? kGroups : 1U];  // q6 2-bit plane, 16 bytes per group
    std::uint32_t scale;  // FP16 scale of group gi in bits 16 gi .. 16 gi + 15
};

template <class Storage>
__device__ __forceinline__ BlockWords<Storage> load_block(const std::uint8_t* row, unsigned index) {
    const auto* block = row + std::size_t{index} * Storage::bytes;
    BlockWords<Storage> w{};
#pragma unroll
    for (unsigned i = 0; i < 2U * kGroups; ++i) w.c[i] = *(lowbit_f16::Words4G*)(block + 16U * i);
    if constexpr (Storage::high_bytes != 0U)
#pragma unroll
        for (unsigned i = 0; i < kGroups; ++i) w.h[i] = *(lowbit_f16::Words4G*)(block + Storage::low_bytes + 16U * i);
    w.scale = *(lowbit_f16::WordsG*)(block + Storage::scale_offset);
    return w;
}

template <class Storage, unsigned RowTiles, unsigned Capacity>
__global__ __launch_bounds__(128)  // Schedule::kThreads
void lowbit_kernel(const float* a, unsigned rows, unsigned K, std::size_t w_row_bytes, Q8ProjectionGroup<Capacity> group) {
    constexpr unsigned kDepth = kPrefetch;
    using Sched = Schedule<RowTiles>;
    using Words = BlockWords<Storage>;
    __shared__ typename Sched::Storage lds;
    const unsigned tid = threadIdx.x, lane = tid & 31U, wn = tid >> 5U;
    const unsigned c = lane & 15U;
    const unsigned panel = blockIdx.x;
    unsigned index = 0U;
    while (index + 1U < group.count && panel >= group.first_panel[index + 1U]) ++index;
    const auto& projection = group.projections[index];
    const unsigned col0 = (panel - group.first_panel[index]) * kProjectionPanel;
    const unsigned col = col0 + kWaveColumns * wn + c;
    const bool live = col < projection.outputs;
    // Columns past the projection read its first panel column (valid bytes) and never store.
    const std::uint8_t* wrow = projection.weights + std::size_t{live ? col : col0} * w_row_bytes;
    const unsigned r0 = blockIdx.y * Sched::kRows;
    const unsigned live_rows = rows - r0 >= Sched::kRows ? Sched::kRows : rows - r0;
    const unsigned live_tiles = (live_rows + 15U) / 16U;
    const unsigned stages = K / Sched::kStageK;  // block t = stage t (K % 128 == 0)

    // A stager: 8-float chunk q = tid + 128 j of stage t = row q / kChunksPerRow, k kStageK t + 8 (q % kChunksPerRow).
    // Rows past `rows` stage zeros.
    Float4 ra[Sched::kAChunks][2];
    const auto load_a = [&](unsigned t) __attribute__((always_inline)) {
#pragma unroll
        for (unsigned j = 0; j < Sched::kAChunks; ++j) {
            const unsigned q = tid + Sched::kThreads * j, row = q / Sched::kChunksPerRow;
            const float* src = a + std::size_t{r0 + row} * K + Sched::kStageK * t + 8U * (q % Sched::kChunksPerRow);
            ra[j][0] = row < live_rows ? *reinterpret_cast<const Float4*>(src) : Float4{};
            ra[j][1] = row < live_rows ? *reinterpret_cast<const Float4*>(src + 4) : Float4{};
        }
    };
    const auto publish_a = [&]() __attribute__((always_inline)) {
#pragma unroll
        for (unsigned j = 0; j < Sched::kAChunks; ++j) {
            const unsigned q = tid + Sched::kThreads * j, row = q / Sched::kChunksPerRow;
            const unsigned dword = 4U * (q % Sched::kChunksPerRow);
            *reinterpret_cast<U4*>(&lds.a[row][dword]) =
                U4{detail::pack(ra[j][0][0], ra[j][0][1]), detail::pack(ra[j][0][2], ra[j][0][3]),
                   detail::pack(ra[j][1][0], ra[j][1][1]), detail::pack(ra[j][1][2], ra[j][1][3])};
        }
    };

    Words ring[kDepth];
#pragma unroll
    for (unsigned p = 0; p < kDepth; ++p) ring[p] = p < stages ? load_block<Storage>(wrow, p) : Words{};
    load_a(0U);
    Float8 acc[RowTiles] = {};
    for (unsigned t0 = 0; t0 < stages; t0 += kDepth) {
#pragma unroll
        for (unsigned p = 0; p < kDepth; ++p) {
            const unsigned t = t0 + p;
            if (t >= stages) break;  // uniform
            __syncthreads();  // every wave finished reading the previous stage
            publish_a();
            __syncthreads();
            if (t + 1U < stages) load_a(t + 1U);  // in flight across this stage's WMMAs
            const Words w = ring[p];
            if (t + kDepth < stages) ring[p] = load_block<Storage>(wrow, t + kDepth);
#pragma unroll
            for (unsigned gi = 0; gi < kGroups; ++gi) {
                Float8 sum[RowTiles] = {};  // the group chain starts from +0
#pragma unroll
                for (unsigned u = 0; u < 4U; ++u) {
                    // Unit u of group gi: code words 8 gi + 2 u and + 1 (8 elements each), 2-bit plane word 4 gi + u.
                    const unsigned word = 8U * gi + 2U * u;
                    std::uint32_t hi = 0U;
                    if constexpr (Storage::high_bytes != 0U) hi = w.h[gi][u];
                    std::uint32_t words[8];
                    lowbit_f16::integer_pairs<Storage>(w.c[word / 4U][word % 4U], w.c[word / 4U][word % 4U + 1U], hi, words);
                    const Frag fb{words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]};
#pragma unroll
                    for (unsigned m = 0; m < RowTiles; ++m)
                        if (m < live_tiles) {
                            const Frag fa = detail::fragment(&lds.a[16U * m + c][32U * gi + 8U * u]);
                            sum[m] = STRIX_HIP_WMMA_F16(__builtin_bit_cast(Half16, fa), __builtin_bit_cast(Half16, fb), sum[m]);
                        }
                }
                const float d = lowbit_f16::scale_value(w.scale >> (16U * gi));
#pragma unroll
                for (unsigned m = 0; m < RowTiles; ++m)
                    if (m < live_tiles)
#pragma unroll
                        for (unsigned j = 0; j < 8U; ++j) acc[m][j] = __builtin_fmaf(sum[m][j], d, acc[m][j]);
            }
        }
    }
    if (!live) return;
    // acc[m][j] -> row r0 + 16 m + 2 j + lane / 16, column col.
#pragma unroll
    for (unsigned m = 0; m < RowTiles; ++m)
        if (m < live_tiles)
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) {
                const unsigned r = 16U * m + 2U * j + (lane >> 4U);
                if (r < live_rows) projection.output[std::size_t{r0 + r} * projection.outputs + col] = acc[m][j];
            }
}

template <class Storage, unsigned RowTiles, unsigned Capacity>
inline void launch(const float* a, unsigned rows, unsigned K, const Q8ProjectionGroup<Capacity>& group, hipStream_t stream) {
    const dim3 grid(group.first_panel[group.count], (rows + 16U * RowTiles - 1U) / (16U * RowTiles));
    const auto w_row_bytes = static_cast<std::size_t>(Storage::row_bytes(K));
    STRIX_HIP_LAUNCH(Rdna3Layout, lowbit_kernel<Storage, RowTiles, Capacity><<<grid, 128, 0, stream>>>(a, rows, K, w_row_bytes, group));
}

// rows >= 1, K % 128 == 0: 16-row blocks, or one 32-row block per 32 rows when `wide_block` (the caller's device plan,
// core/resident_row_plan.hpp lowbit_wide_block; speed only: both blocks give the same bits per row).
template <class Storage, unsigned Capacity>
inline hipError_t gemm_lowbit(const float* a, unsigned rows, unsigned K, const Q8ProjectionGroup<Capacity>& group,
                              hipStream_t stream, bool wide_block = false) {
    static_assert(Capacity <= 32U);
    if (wide_block) launch<Storage, 2U>(a, rows, K, group, stream);
    else launch<Storage, 1U>(a, rows, K, group, stream);
    return hipGetLastError();
}

}  // namespace strix::core::f16_tile_gemm::small_m
#endif
