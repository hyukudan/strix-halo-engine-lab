#pragma once
// Block-tree scalar GEMV: the reference projection topology of the HIP launchers (GDN, PLE,
// indexed attention, shared and routed experts, LM head). One block of Threads lanes per output
// row; lane t accumulates features t, t + Threads, ... in ascending order as
// sum = sum + activation * weight in FP32, then an LDS tree with strides Threads / 2 .. 1 folds the
// Threads leaves into scratch[0]. The order is the shared body's (gemv_tree.inc); a binding supplies
// only operands and the epilogue, so every binding keeps the same leaves and the same tree.
//
// Binding trait for rows_kernel (all static, device):
//   using Input, Output; static constexpr std::uint32_t threads;
//   float activation(const Input* input, std::uint32_t feature)
//   float weight(const void* weights, std::uint32_t width, std::uint32_t row, std::uint32_t feature)
//   Output finish(float sum)
// Kernels with another prologue (route lookup, address resolution) or a read-modify-write epilogue
// include the textual body gemv_tree.inc in their own body with their operand macros.
#include <cstdint>

#include <hip/hip_runtime.h>

namespace strix::core::gemv_tree {

// output[row] = finish(sum over features of activation * weight(row)), one block per row.
template <class Binding>
__global__ void rows_kernel(const typename Binding::Input* input, const void* weights, std::uint32_t input_width,
                            std::uint32_t output_width, typename Binding::Output* output) {
    const auto out = static_cast<std::uint32_t>(blockIdx.x);
    if (out >= output_width) {
        return;
    }

    __shared__ float scratch[Binding::threads];
#define GEMV_TREE_THREADS Binding::threads
#define GEMV_TREE_SCRATCH scratch
#define GEMV_TREE_WIDTH input_width
#define GEMV_TREE_ACTIVATION(f) Binding::activation(input, f)
#define GEMV_TREE_WEIGHT(f) Binding::weight(weights, input_width, out, f)
#include "gemv_tree.inc"

    if (threadIdx.x == 0) {
        output[out] = Binding::finish(scratch[0]);
    }
}

}  // namespace strix::core::gemv_tree
