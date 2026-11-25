#include "inkling2_mini_block_candidate_cuda.hpp"

#include "exec/cuda_decode_wr2.hpp"
#include "inkling2_mini_schedule.hpp"

#include <utility>

namespace strix::inkling2 {

void MiniCudaBlockCandidateStep::append_tap(exec::CudaProgramRecipe& recipe, const exec::StepBlock* device_step,
                                           std::span<const float> hidden, std::span<std::uint16_t> features,
                                           unsigned tap_count, unsigned tap) const {
    const auto rows = descriptor_.h.rows;
    if (tap >= tap_count || tap_count > kLogicalVisits + 1U || hidden.size() < std::size_t{rows} * kHiddenWidth)
        throw std::invalid_argument("Inkling CUDA block candidate: invalid feature tap geometry");
    const unsigned pitch = tap_count * kHiddenWidth;
    if (features.size() / pitch < positions_)
        throw std::invalid_argument("Inkling CUDA block candidate: feature store is smaller than session capacity");

    auto leaf = exec::cuda_decode::bind_step(cuda::core::StepRowStoreBf16{
        device_step, hidden.data(), features.data() + std::size_t{tap} * kHiddenWidth, kHiddenWidth, pitch, positions_,
    }, rows);
    const auto stage = recipe.stage_label();
    recipe.stage("draft_tap");
    std::move(leaf).append_to(recipe);
    recipe.stage(stage);
}

}  // namespace strix::inkling2
