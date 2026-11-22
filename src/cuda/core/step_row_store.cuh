#pragma once

// Position-indexed BF16 row store for step programs (shared, backend body = CUDA).
//
// Row r of a step block writes its FP32 input row, encoded to BF16 with the same rounding as the
// resident row stores (attn::cache_rows::encode), to output + position(r) * pitch. The position is
// read from the device step descriptor, so one captured program serves every step without a
// recapture: a decode step (R = 1) and every verify window width write the rows of the positions
// they run. Rows that are not valid in the descriptor, and positions at or past `positions`, write
// nothing. A row written past the committed frontier is simply overwritten when that position is
// committed later (position-indexed storage needs no rollback).
//
// Consumers: drafter feature taps (hidden rows a block drafter reads from committed positions).

#include "strix/attn/cache_rows.hpp"
#include "strix/exec/descriptor.hpp"

#include <cstdint>

namespace strix::cuda::core {

struct StepRowStoreBf16 {
    const exec::StepBlock* step;
    const float* input;      // [row][width] FP32, rows packed by descriptor row
    std::uint16_t* output;   // base of the position-major store (first element of the slice at position 0)
    unsigned width;          // elements stored per row
    unsigned pitch;          // elements between consecutive positions in the store (>= width)
    unsigned positions;      // positions the store holds
};

[[nodiscard]] inline bool valid_step_row_store(const StepRowStoreBf16& a) noexcept {
    return a.step && a.input && a.output && a.width != 0U && a.pitch >= a.width && a.positions != 0U;
}

[[nodiscard]] constexpr unsigned step_row_store_blocks(unsigned width) noexcept {
    const unsigned blocks = (width + 255U) / 256U;
    return blocks > 64U ? 64U : (blocks == 0U ? 1U : blocks);
}

#if defined(__CUDACC__) && !defined(__HIPCC__)
namespace step_row_store_detail {

template <unsigned Rows>
__global__ void __launch_bounds__(256) store_rows(StepRowStoreBf16 a) {
    const unsigned r = blockIdx.y;
    if (r >= Rows || !exec::step_row_valid(*a.step, r)) return;
    const auto position = a.step->row[r].position;
    if (position >= a.positions) return;
    const float* source = a.input + static_cast<std::size_t>(r) * a.width;
    std::uint16_t* target = a.output + static_cast<std::size_t>(position) * a.pitch;
    for (unsigned c = blockIdx.x * blockDim.x + threadIdx.x; c < a.width; c += gridDim.x * blockDim.x)
        target[c] = attn::cache_rows::encode(source[c]);
}

}  // namespace step_row_store_detail

template <unsigned Rows>
cudaError_t enqueue_step_row_store_bf16(const StepRowStoreBf16& a, cudaStream_t stream) {
    if (!valid_step_row_store(a)) return cudaErrorInvalidValue;
    step_row_store_detail::store_rows<Rows><<<dim3(step_row_store_blocks(a.width), Rows), 256, 0, stream>>>(a);
    return cudaGetLastError();
}
#endif

}  // namespace strix::cuda::core
