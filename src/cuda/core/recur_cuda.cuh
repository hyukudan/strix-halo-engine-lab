#pragma once

#include "recur_decode_plan.hpp"
#include "recur_plan.hpp"
#include "strix/numeric/math_gfx1151.hpp"

#include <cuda_runtime_api.h>

namespace strix::cuda::recur {

struct Weights {
    const float* conv;      // [channels][taps], oldest first
    const float* ssm_a;     // [value_heads], negative exp(A_log)
    const float* dt_bias;   // [value_heads]
    const float* gain;      // [value_dim], gated RMS epilogue
    float rms_eps;
};

struct Rows {
    const float* raw;       // [steps][channels], Q then K then V
    const float* a;         // [steps][value_heads]
    const float* b;         // [steps][value_heads]
    const float* z;         // [steps][value_width]
    float* prepared;       // [steps][channels] scratch, omitted when matrix replay owns this storage
    float* out;            // [steps][value_width]
};

// Enqueue a linear window with a validated make_plan result. Input, weights,
// output, scratch and both state allocations are disjoint device addresses.
// Success queues the writes, leaving publication to the caller. The final
// (or accepted ring) image is selected by image_slot; zero accept keeps published.
cudaError_t enqueue(const Plan&, const Weights&, Rows, StateView matrix, StateView history,
                    std::uint32_t steps, cudaStream_t);

// After partial acceptance, regenerate only replay-declared states from the
// published images and retained rows. Pass the original bound views and an
// accepted prefix of the already enqueued window. Ring state needs only image
// selection. Both replay states share one dispatch. Zero accept queues no work.
cudaError_t enqueue_replay(const Plan&, const Weights&, StateView matrix, StateView history,
                           std::uint32_t accepted, cudaStream_t);

// One row using the supplied gfx1151 tables. Row buffers are disjoint except
// for the documented QKV alias. State image selection and publication belong
// to the caller. A8 is enabled by supplying both buffers.
cudaError_t enqueue_decode(Gfx1151Decode, const DecodeWeights&, DecodeRow, DecodeState,
                           const numeric::gfx1151::Tables&, DecodeA8, cudaStream_t);

}  // namespace strix::cuda::recur
