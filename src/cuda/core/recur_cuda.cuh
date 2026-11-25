#pragma once

#include "ops/recur/recur_decode_step.hpp"
#include "recur_plan.hpp"
#include "strix/numeric/math_gfx1151.hpp"

#include <cuda_runtime_api.h>

namespace strix::cuda::recur {

using ops::recur::DecodeA8;
using ops::recur::DecodeReplayStep;
using ops::recur::DecodeRow;
using ops::recur::DecodeState;
using ops::recur::DecodeStep;
using ops::recur::DecodeWeights;
using ops::recur::GdnDecodeShape;

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

// Decode launches require a compiled striped specialization for the supplied
// shape. One row uses the supplied gfx1151 tables. Row buffers are disjoint except
// for the documented QKV alias. State image selection and publication belong
// to the caller. A8 is enabled by supplying both buffers.
cudaError_t enqueue_decode(GdnDecodeShape, const DecodeWeights&, DecodeRow, DecodeState,
                           const numeric::gfx1151::Tables&, DecodeA8, cudaStream_t);

// X3 verify or cohort step. Each valid row writes its own state images and
// optional pre-convolution FP32 capture. The slot directory remains read-only.
// The only allowed row-buffer alias is prepared == qkv with equal strides.
cudaError_t enqueue_decode_step(GdnDecodeShape, unsigned width, const DecodeWeights&, const DecodeStep&,
                                const numeric::gfx1151::Tables&, cudaStream_t);

// X3 replay policy for one contiguous chain. The final row leaves its images
// in the selected spare. Rerun a shorter prefix from the same published inputs
// to replace those images after partial acceptance. Zero acceptance leaves
// the published images alone. The caller owns publication and synchronization.
// Invalid device bindings/rows mask the whole step. Row-buffer rules match
// enqueue_decode_step. Keep QKV (or restore it from raw_capture) for replay.
cudaError_t enqueue_decode_replay_step(GdnDecodeShape, unsigned width, const DecodeWeights&, const DecodeReplayStep&,
                                       const numeric::gfx1151::Tables&, cudaStream_t);

}  // namespace strix::cuda::recur
