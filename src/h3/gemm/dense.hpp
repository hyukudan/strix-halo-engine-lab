#pragma once
#include <cstdint>
namespace strix::h3::gemm {
// Dense BF16 WMMA, row-major x[M,K], w[N,K], y[M,N]. FP32 accumulation.
// This path is independent of W8A8; its arithmetic requires its own native gates.
void dense_bf16(const std::uint16_t* x, const std::uint16_t* w, std::uint32_t m,
                std::uint32_t n, std::uint32_t k, std::uint16_t* y, void* stream);
}
