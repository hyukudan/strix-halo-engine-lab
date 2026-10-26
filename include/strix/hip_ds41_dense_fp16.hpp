#pragma once
#include "strix/hip_ds41_layer.hpp"
#include <memory>

namespace strix::ds41 {
// Model-owned cache. Weight addresses must remain immutable until destruction.
// The input is the original FP32 activation, before A8 quantization.
class DenseFp16Cache {
public:
    DenseFp16Cache();
    ~DenseFp16Cache();
    DenseFp16Cache(const DenseFp16Cache&) = delete;
    DenseFp16Cache& operator=(const DenseFp16Cache&) = delete;
    // keep_a8: FP16 operands from the view's A8 codes/deltas (the exact path's activations) instead of x
    void enqueue(const Q8A8DensePrefillView&, const float* x, void* stream, bool keep_a8 = false);
    // FP16 output form (STRIX_DS41_PREFILL_FP16_FAST): the GEMM writes FP16 (gfx1151 hipBLASLt: 377 HHS solutions
    // against 7 with FP32 output) and one finish kernel widens it into each view's out (accumulate honoured).
    // The views share the activation (same width, x_stride) and end at its last row: views[i] covers the last
    // views[i].rows of `rows` rows. They run as ONE GEMM over their concatenated weights (q_a || kv: N = 1792).
    // x16: the activation already in FP16 (row stride x16_stride, unscaled: NormView::f16); nullptr: converted here
    // from x (FP32, row stride views[0].x_stride, power-of-two row scale), once for all the views.
    // groups > 1 (one view; attn_output_a): one GEMM per group, group g = weight rows [g count, (g + 1) count)
    // x input columns [g width, (g + 1) width) -> output columns [g count, (g + 1) count); one conversion, one finish.
    void enqueue_fast(const Q8A8DensePrefillView* views, unsigned count, std::uint32_t rows, const std::uint16_t* x16,
                      std::uint32_t x16_stride, const float* x, void* stream, unsigned groups = 1, bool split = false);
    // split: the weights as two FP16 halves, W = hi + lo / 2^12 (the Q8_0 values d * code to ~22 bits instead of 11), two
    // GEMMs on the same activation summed in FP32 by the finish (diagnostic / precision variant, ~2x the GEMM time)
    // Non-finite outputs of the FP16 output form since the last call (synchronises the stream).
    unsigned take_nonfinite(void* stream);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
