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
//
// The K loop (run) is the shared code-as-operand trait of the family: an Operand trait supplies the column's superblock
// as exact FP16 WMMA operands and one FP32 multiplier per group (LowBitOperand here; NVFP4 / MXFP8 in
// core/microscaled_f16_staging.hpp), a Tile binding supplies rows, weights, A chunks and the store (ProjectionTile here;
// the grouped expert tiles in moe_core/grouped_microscaled.hpp). Every operand and binding runs this one schedule.

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

// Operand trait of the q4_g64 / q6_g64 resident rows: Words = one superblock (two groups) of one column, unit u of group
// gi = the exact FP16 integer codes of elements 64 gi + 16 u .. + 16 of the block, scale = the group's FP16 scale.
// ScalePolicy::folded (FAST / TURBO candidate, core/lowbit_f16_staging.hpp): unit = fp16(code x d) pairs and kFolded,
// so run() chains every unit into the accumulator (the tile's folded staging arithmetic, bit for bit).
template <class Storage, lowbit_f16::ScalePolicy Policy = lowbit_f16::ScalePolicy::group>
struct LowBitOperand {
    static constexpr bool kFolded = Policy == lowbit_f16::ScalePolicy::folded;
    static constexpr unsigned kGroup = Storage::group, kUnits = Storage::group / 16U, kSuperK = Storage::elements;
    static_assert(kGroup == 64U && kSuperK == 64U * kGroups);
    using Row = const std::uint8_t*;
    using Words = BlockWords<Storage>;
    __device__ __forceinline__ static Words load(Row row, unsigned index) { return load_block<Storage>(row, index); }
    __device__ __forceinline__ static void unit(const Words& w, unsigned gi, unsigned u, std::uint32_t (&words)[8]) {
        // Unit u of group gi: code words 8 gi + 2 u and + 1 (8 elements each), 2-bit plane word 4 gi + u.
        const unsigned word = 8U * gi + 2U * u;
        std::uint32_t hi = 0U;
        if constexpr (Storage::high_bytes != 0U) hi = w.h[gi][u];
        if constexpr (kFolded)
            lowbit_f16::folded_pairs<Storage>(w.c[word / 4U][word % 4U], w.c[word / 4U][word % 4U + 1U], hi,
                                              w.scale >> (16U * gi), words);
        else lowbit_f16::integer_pairs<Storage>(w.c[word / 4U][word % 4U], w.c[word / 4U][word % 4U + 1U], hi, words);
    }
    __device__ __forceinline__ static float scale(const Words& w, unsigned gi) {
        return lowbit_f16::scale_value(w.scale >> (16U * gi));
    }
};

// The K loop of the small-M schedule over one bound tile. Operand: the weight trait (LowBitOperand, or
// core/microscaled_f16_staging.hpp): kGroup elements per scale group (a multiple of 16, dividing 64), kUnits = kGroup / 16
// WMMA units per group, kSuperK = 128 elements per superblock (one stage), Words = the superblock of a column, load(row,
// index), unit() = FP16 pairs of 16 of its elements, scale() = the group's FP32 multiplier (an Operand with kFolded true
// has the scale inside unit() and every unit chains into the accumulator). Tile (a binding): K (% 128 ==
// 0), rows (live rows of the tile), columns() (output columns), row(column) (the column's weight row), AChunk / load_a(row, k, kl, in) /
// pack_a (8 A elements of row `row` from k + kl, zero when !in, as 4 FP16 pair words) and store(row, column, sum); an optional
// before_store() runs on every thread before the stores. The CTA covers rows 16 RowTiles row_block .. + 16 RowTiles.
template <class Operand> inline constexpr bool folded_operand = requires { requires Operand::kFolded; };

template <class Operand, unsigned RowTiles, class Tile>
__device__ __forceinline__ void run(const Tile& tile, unsigned row_block, unsigned col0) {
    constexpr unsigned kDepth = kPrefetch, kStageGroups = 64U * kGroups / Operand::kGroup;
    using Sched = Schedule<RowTiles>;
    using Words = typename Operand::Words;
    static_assert(Operand::kSuperK == Sched::kStageK);
    __shared__ typename Sched::Storage lds;
    const unsigned tid = threadIdx.x, lane = tid & 31U, wn = tid >> 5U;
    const unsigned c = lane & 15U;
    const unsigned K = tile.K, rows = tile.rows;
    const unsigned col = col0 + kWaveColumns * wn + c;
    const bool live = col < tile.columns();
    // Columns past the outputs read the panel's first column (valid bytes) and never store.
    const typename Operand::Row wrow = tile.row(live ? col : col0);
    const unsigned r0 = row_block * Sched::kRows;
    const unsigned live_rows = rows - r0 >= Sched::kRows ? Sched::kRows : rows - r0;
    const unsigned live_tiles = (live_rows + 15U) / 16U;
    const unsigned stages = K / Sched::kStageK;  // block t = stage t (K % 128 == 0)

    // A stager: 8-element chunk q = tid + 128 j of stage t = row q / kChunksPerRow, k kStageK t + 8 (q % kChunksPerRow).
    // Rows past `rows` stage zeros.
    typename Tile::AChunk ra[Sched::kAChunks];
    const auto load_a = [&](unsigned t) __attribute__((always_inline)) {
#pragma unroll
        for (unsigned j = 0; j < Sched::kAChunks; ++j) {
            const unsigned q = tid + Sched::kThreads * j, row = q / Sched::kChunksPerRow;
            ra[j] = tile.load_a(r0 + row, Sched::kStageK * t, 8U * (q % Sched::kChunksPerRow), row < live_rows);
        }
    };
    const auto publish_a = [&]() __attribute__((always_inline)) {
#pragma unroll
        for (unsigned j = 0; j < Sched::kAChunks; ++j) {
            const unsigned q = tid + Sched::kThreads * j, row = q / Sched::kChunksPerRow;
            const unsigned dword = 4U * (q % Sched::kChunksPerRow);
            if constexpr (requires { tile.pack_a(ra[j], row); })
                *reinterpret_cast<U4*>(&lds.a[row][dword]) = tile.pack_a(ra[j], row);
            else
                *reinterpret_cast<U4*>(&lds.a[row][dword]) = Tile::pack_a(ra[j]);
        }
    };

    Words ring[kDepth];
#pragma unroll
    for (unsigned p = 0; p < kDepth; ++p) ring[p] = p < stages ? Operand::load(wrow, p) : Words{};
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
            if (t + kDepth < stages) ring[p] = Operand::load(wrow, t + kDepth);
            if constexpr (folded_operand<Operand>) {
#pragma unroll
                for (unsigned v = 0; v < kStageGroups * Operand::kUnits; ++v) {
                    std::uint32_t words[8];
                    Operand::unit(w, v / Operand::kUnits, v % Operand::kUnits, words);
                    const Frag fb{words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]};
#pragma unroll
                    for (unsigned m = 0; m < RowTiles; ++m)
                        if (m < live_tiles) acc[m] = detail::LowBitMma{}(detail::fragment(&lds.a[16U * m + c][8U * v]), fb, acc[m]);
                }
                continue;
            }
#pragma unroll
            for (unsigned gi = 0; gi < kStageGroups; ++gi) {
                detail::LowBitGroup sum[RowTiles] = {};
#pragma unroll
                for (unsigned u = 0; u < Operand::kUnits; ++u) {
                    std::uint32_t words[8];
                    Operand::unit(w, gi, u, words);
                    const Frag fb{words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]};
#pragma unroll
                    for (unsigned m = 0; m < RowTiles; ++m)
                        if (m < live_tiles) {
                            const Frag fa = detail::fragment(&lds.a[16U * m + c][8U * (Operand::kUnits * gi + u)]);
                            sum[m].add(fa, fb);
                        }
                }
                const auto d = Operand::scale(w, gi);
#pragma unroll
                for (unsigned m = 0; m < RowTiles; ++m)
                    if (m < live_tiles) sum[m].apply(d, acc[m]);
            }
        }
    }
    if constexpr (requires { tile.before_store(); }) tile.before_store();
    if (!live) return;
    // acc[m][j] -> row r0 + 16 m + 2 j + lane / 16, column col.
#pragma unroll
    for (unsigned m = 0; m < RowTiles; ++m)
        if (m < live_tiles)
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) {
                const unsigned r = 16U * m + 2U * j + (lane >> 4U);
                if (r < live_rows) tile.store(r0 + r, col, acc[m][j]);
            }
}

// Binding of a projection group (dense rows, FP32 activations [rows][K] staged as FP16): panel blockIdx.x of the group,
// row block blockIdx.y.
struct ProjectionTile {
    struct AChunk {
        Float4 v[2];
    };
    const float* a;
    unsigned rows, K;
    std::size_t w_row_bytes;
    const Q8Projection* projection;

    __device__ __forceinline__ unsigned columns() const { return projection->outputs; }
    __device__ __forceinline__ const std::uint8_t* row(unsigned column) const {
        return projection->weights + std::size_t{column} * w_row_bytes;
    }
    __device__ __forceinline__ AChunk load_a(unsigned row, unsigned k, unsigned kl, bool in) const {
        const float* src = a + std::size_t{row} * K + k + kl;
        AChunk chunk;
        chunk.v[0] = in ? *reinterpret_cast<const Float4*>(src) : Float4{};
        chunk.v[1] = in ? *reinterpret_cast<const Float4*>(src + 4) : Float4{};
        return chunk;
    }
    __device__ __forceinline__ static U4 pack_a(const AChunk& x) {
        return U4{detail::pack(x.v[0][0], x.v[0][1]), detail::pack(x.v[0][2], x.v[0][3]),
                  detail::pack(x.v[1][0], x.v[1][1]), detail::pack(x.v[1][2], x.v[1][3])};
    }
    __device__ __forceinline__ void store(unsigned r, unsigned column, float sum) const {
        projection->output[std::size_t{r} * projection->outputs + column] = sum;
    }
};

template <class Storage, unsigned RowTiles, unsigned Capacity, lowbit_f16::ScalePolicy Policy = lowbit_f16::ScalePolicy::group>
__global__ __launch_bounds__(128)  // Schedule::kThreads
void lowbit_kernel(const float* a, unsigned rows, unsigned K, std::size_t w_row_bytes, Q8ProjectionGroup<Capacity> group) {
    const unsigned panel = blockIdx.x;
    unsigned index = 0U;
    while (index + 1U < group.count && panel >= group.first_panel[index + 1U]) ++index;
    const auto& projection = group.projections[index];
    const unsigned col0 = (panel - group.first_panel[index]) * kProjectionPanel;
    const ProjectionTile tile{a, rows, K, w_row_bytes, &projection};
    run<LowBitOperand<Storage, Policy>, RowTiles>(tile, blockIdx.y, col0);
}

template <class Storage, unsigned RowTiles, unsigned Capacity, lowbit_f16::ScalePolicy Policy = lowbit_f16::ScalePolicy::group>
inline void launch(const float* a, unsigned rows, unsigned K, const Q8ProjectionGroup<Capacity>& group, hipStream_t stream) {
    const dim3 grid(group.first_panel[group.count], (rows + 16U * RowTiles - 1U) / (16U * RowTiles));
    const auto w_row_bytes = static_cast<std::size_t>(Storage::row_bytes(K));
    STRIX_HIP_LAUNCH(Rdna3Layout, lowbit_kernel<Storage, RowTiles, Capacity, Policy><<<grid, 128, 0, stream>>>(a, rows, K, w_row_bytes, group));
}

// rows >= 1, K % 128 == 0: 16-row blocks, or one 32-row block per 32 rows when `wide_block` (the caller's device plan,
// core/resident_row_plan.hpp lowbit_wide_block; speed only: both blocks give the same bits per row).
template <class Storage, unsigned Capacity, lowbit_f16::ScalePolicy Policy = lowbit_f16::ScalePolicy::group>
inline hipError_t gemm_lowbit(const float* a, unsigned rows, unsigned K, const Q8ProjectionGroup<Capacity>& group,
                              hipStream_t stream, bool wide_block = false) {
    static_assert(Capacity <= 32U);
    if (wide_block) launch<Storage, 2U, Capacity, Policy>(a, rows, K, group, stream);
    else launch<Storage, 1U, Capacity, Policy>(a, rows, K, group, stream);
    return hipGetLastError();
}

}  // namespace strix::core::f16_tile_gemm::small_m
#endif
