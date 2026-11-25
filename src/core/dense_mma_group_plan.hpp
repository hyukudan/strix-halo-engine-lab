#pragma once

#include "bf16_projection_batch.hpp"
#include "strix/core/rows.hpp"

// Keep the incumbent type names used by CUDA kernel signatures.
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
inline MmaGroupTable mma_group_args(const core::rows::Launch& launch, core::rows::ChainId add_residual,
                                   core::rows::ChainId round_bf16) {
    MmaGroupTable args{};
    args.tiles = launch.tiles;
    for (unsigned j = 0; j < launch.tiles.count; ++j) {
        const auto& job = launch.jobs[j];
        const auto* residual = job.epilogue == add_residual ? static_cast<const float*>(job.aux[0].data) : nullptr;
        args.jobs[j] = {job.w.data, static_cast<const float*>(job.x.data),
                        const_cast<float*>(static_cast<const float*>(job.y.data)), residual, job.out, job.in,
                        job.epilogue == round_bf16};
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

}  // namespace strix::cuda_core
