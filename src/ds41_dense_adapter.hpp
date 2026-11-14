#pragma once
#include "ds41_integer_operand.hpp"
#include "strix/hip_ds41_layer.hpp"
#include "strix/ds41_layer.hpp"
namespace strix::ds41::integer_core {
// Dense is one group on tiled_core; row/column tails and strides are model glue.
template<unsigned RowTiles = 1, bool ColFirst = false> struct DenseAdapter {
    static constexpr bool kActive = true, kTiles = true;
    struct Tile {
        int rows = 0, first_col = 0, split = 0, splits = 1;
        const unsigned* packed = nullptr;
        unsigned begin = 0;
    };
    Q8A8DensePrefillView view;
    __device__ __forceinline__ const Q8A8DensePrefillView& tensor() const { return view; }
    __device__ __forceinline__ float half_exact(unsigned short h) const { return f16_to_float(h); }
    __device__ __forceinline__ float half_select(unsigned short h) const { return f16_to_float_select(h); }
    __device__ __forceinline__ bool next(Tile& t, unsigned iter, int bn) const {
        if (iter) return false;
        t.begin = (ColFirst ? blockIdx.y : blockIdx.x) * (16U * RowTiles);
        if (t.begin >= view.rows) return false;
        t.rows = int(min(16U * RowTiles, view.rows - t.begin));
        t.first_col = int(blockIdx.y) * bn;
        return true;
    }
    __device__ __forceinline__ const signed char* code_row(const Tile& t, int r) const {
        return view.codes + std::uint64_t(t.begin + unsigned(r)) * view.x_stride;
    }
    __device__ __forceinline__ float delta(const Tile& t, int r, int group) const {
        return view.deltas[std::uint64_t(t.begin + unsigned(r)) * (view.x_stride / 32U) + unsigned(group)];
    }
    __device__ __forceinline__ const unsigned long long* grid() const { return nullptr; }
    __device__ __forceinline__ const unsigned char* weight_row(const Tile&, int col) const {
        const unsigned row_bytes = (view.row_width ? view.row_width : view.width) / 32U * 34U;
        return view.weight + std::uint64_t(min(unsigned(col), view.count - 1U)) * row_bytes + view.k_offset / 32U * 34U;
    }
    __device__ __forceinline__ void store_raw(const Tile& t, int r, int col, float value) const {
        if (unsigned(col) >= view.count) return;
        auto* dst = view.out + std::uint64_t(t.begin + unsigned(r)) * view.out_stride + unsigned(col);
        *dst = view.accumulate ? *dst + value : value;
    }
    __device__ __forceinline__ void finish_tile(const Tile& t, const float* values, int bm, int bn) const {
        if (!view.a8_codes) return;
        const unsigned lane = threadIdx.x & 31U, waves = blockDim.x / 32U;
        for (unsigned r = threadIdx.x / 32U; r < unsigned(bm); r += waves) {
            if (r >= unsigned(t.rows)) continue;
            const auto a = strix::core::quantize_a8_row32(values[r * unsigned(bn) + lane]);
            const auto row = std::uint64_t(t.begin + r);
            const unsigned col = unsigned(t.first_col) + lane;
            if (col < view.count) view.a8_codes[row * view.a8_stride + col] = a.code;
            if (!lane && unsigned(t.first_col) < view.count) {
                const auto g = row * (view.a8_stride / 32U) + unsigned(t.first_col) / 32U;
                view.a8_deltas[g] = a.delta; view.a8_sums[g] = a.sum;
            }
        }
    }
};

template<unsigned RT, unsigned OT, unsigned Lean = 0, bool ColFirst = false, unsigned Parts = 1,
         bool Wide = false, bool Staged = true>
inline void launch_dense(const Q8A8DensePrefillView& v, hipStream_t stream) {
    using Plan = std::conditional_t<Wide, strix::core::WideIntegerPlan<Staged>, strix::core::DenseIntegerPlan<Lean, ColFirst, Parts>>;
    using C = strix::core::IntegerContract<strix::core::IntegerOperand<Ds41PrefillTraits, WeightFormat::q8_0, false, Plan>>;
    using Ad = DenseAdapter<RT, ColFirst>;
    using namespace strix::core::wmma_tiled;
    using G = Geometry<16 * RT, 16 * OT, 16 * RT, 16 * OT, typename C::Operand>;
    const unsigned x = (v.rows + 16 * RT - 1) / (16 * RT), y = (v.count + 16 * OT - 1) / (16 * OT);
    hipLaunchKernelGGL((tiled_core<C, 16 * RT, 16 * OT, 16 * RT, 16 * OT, false, 0, false, false, 64, false, Ad>),
        dim3(ColFirst ? y : x, ColFirst ? x : y), dim3(G::threads), 0, stream,
        nullptr, nullptr, nullptr, int(v.width), int(y * 16 * OT), int(v.rows), 1,
        nullptr, nullptr, nullptr, 1.0f, 0.0f, 1, nullptr, nullptr, nullptr, nullptr, GroupedArgs{}, Ad{v});
}
}
