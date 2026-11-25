#pragma once

#include "inkling2_mini_config.hpp"
#include "strix/exec/descriptor.hpp"
#include "strix/spec/block_draft_policy.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>

namespace strix::exec { class CudaProgramRecipe; }

namespace strix::inkling2 {

// Builds the forced-row descriptor and feature-store leaf for a prospective
// single-session block candidate. Production admission still requires a bound
// native drafter and verify target through require_block_draft_execution.
class MiniCudaBlockCandidateStep {
public:
    [[nodiscard]] static std::optional<MiniCudaBlockCandidateStep> prepare(
        const profile::Resolution& resolution, std::span<const unsigned> ids,
        unsigned first_position, unsigned positions) {
        const auto selection = spec::block_draft_selection(resolution);
        if (!selection.enabled()) return std::nullopt;
        if (selection.sampled) spec::require_block_draft_execution(selection);

        if (ids.empty() || ids.size() > selection.block || positions > kMaxContext || first_position >= positions)
            throw std::invalid_argument("Inkling CUDA block candidate: invalid verify row count or position");
        const auto rows = static_cast<unsigned>(ids.size());
        const auto width = session::padded_step_replay_width(rows);
        if (width > positions - first_position)
            throw std::invalid_argument("Inkling CUDA block candidate: padded verify window exceeds session capacity");

        exec::StepBlock descriptor{};
        descriptor.h.rows = width;
        descriptor.h.step = first_position;
        for (unsigned row = 0; row < rows; ++row) {
            if (ids[row] >= kVocabulary)
                throw std::invalid_argument("Inkling CUDA block candidate: verify token out of range");
            descriptor.h.valid_mask |= 1U << row;
            descriptor.row[row] = {ids[row], first_position + row, 0U, row == 0U ? exec::kStepNoParent : row - 1U,
                                   descriptor.h.valid_mask, exec::step_valid, 0U, 0U};
        }
        return MiniCudaBlockCandidateStep(descriptor, positions);
    }

    [[nodiscard]] const exec::StepBlock& descriptor() const noexcept { return descriptor_; }

    // The adapter supplies the tap index in checkpoint order. The device views
    // must stay stable across replays. The consumer reads only committed rows.
    void append_tap(exec::CudaProgramRecipe&, const exec::StepBlock* device_step,
                    std::span<const float> hidden, std::span<std::uint16_t> features,
                    unsigned tap_count, unsigned tap) const;

private:
    MiniCudaBlockCandidateStep(const exec::StepBlock& descriptor, unsigned positions)
        : descriptor_(descriptor), positions_(positions) {}

    exec::StepBlock descriptor_;
    unsigned positions_;
};

}  // namespace strix::inkling2
