#pragma once
// Internal: tensor-core dense path (see dense.hpp "mma arithmetic").
#include "strix/cuda_core/dense.hpp"
#include "../../../core/bf16_projection_batch.hpp"
#include "strix/core/rows.hpp"

namespace strix::cuda_core {

struct MmaGroupJob {
    const void* w;
    const float* x;
    float* y;
    const float* residual;
    std::uint32_t out, in;
    bool round_bf16;
};

struct MmaGroupTable {
    MmaGroupJob jobs[core::rows::kMaxJobs];
    decode::GroupIndex<core::rows::kMaxJobs> tiles;
};

// Own only the launch table; all operand pointers address resident device storage.
inline MmaGroupTable mma_group_args(const core::rows::Launch& launch) {
    MmaGroupTable args{};
    args.tiles = launch.tiles;
    for (unsigned j = 0; j < launch.tiles.count; ++j) {
        const auto& job = launch.jobs[j];
        const auto* residual = job.epilogue == kRowsAddResidual ? static_cast<const float*>(job.aux[0].data) : nullptr;
        args.jobs[j] = {job.w.data, static_cast<const float*>(job.x.data),
                        const_cast<float*>(static_cast<const float*>(job.y.data)), residual, job.out, job.in,
                        job.epilogue == kRowsRoundBf16};
    }
    return args;
}

inline bool valid_bf16_mma3_r1(const MmaGroupTable& a) {
    namespace rows = core::rows;
    if (a.tiles.count == 0 || a.tiles.count > rows::kMaxJobs || a.tiles.first[0] != 0) return false;
    core::Bf16ProjectionBatch batch;
    for (unsigned i = 0; i < a.tiles.count; ++i) {
        const auto& job = a.jobs[i];
        if (job.residual || job.in % rows::kBlockIn || job.out % rows::kTileOut ||
            a.tiles.first[i + 1] < a.tiles.first[i] ||
            a.tiles.first[i + 1] - a.tiles.first[i] != job.out / rows::kTileOut) return false;
        const core::Bf16RowProjection projection{static_cast<const std::uint16_t*>(job.w), job.out, job.y, job.round_bf16};
        if (!batch.append(job.x, job.in, std::span(&projection, 1))) return false;
    }
    return batch.valid_r1_operands();
}

cudaError_t dense_bf16_mma3_r1(const MmaGroupTable& args, cudaStream_t stream);

constexpr int kMmaMaxMulti = 4;
// The legacy launch keeps its original argument layout. Ready jobs with
// independent operands are bound through dense_mma3_rows_grouped instead.
struct MmaCall {
    const void* w[kMmaMaxMulti];
    float* y[kMmaMaxMulti];
    std::uint32_t out[kMmaMaxMulti];
    int n = 0;
    const void* w1 = nullptr;  // fused gate/up: the up weight (n == 1)
    WeightFormat format = WeightFormat::bf16;
    std::uint32_t in = 0, rows = 0;
    const float* x = nullptr;
    const float* residual = nullptr;
    int terms = 3;
};
// true when the mma path can serve the call (format, in % 32, out % 16); the caller falls back to the FP32 kernels otherwise.
bool dense_mma_supported(const MmaCall& c);
cudaError_t dense_mma_launch(const MmaCall& c, cudaStream_t stream);

}  // namespace strix::cuda_core
