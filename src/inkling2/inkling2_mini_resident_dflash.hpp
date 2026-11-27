#pragma once

// Inkling 2 Mini's target boundary of the shared device block-diffusion drafter (src/spec/dflash_resident.hpp): its
// stream, BF16 dense rows, embedding table + embedding norm, raw head rows and the committed tap plane (the DSpark tap
// plan, visits 5 / 23 / 41 / 48, which the drafter's config.json must name in the same order). The model adapter only.

#include "inkling2_mini_dspark_taps.hpp"
#include "inkling2_mini_resident_decoder.hpp"
#include "spec/dflash_resident.hpp"

#include <algorithm>
#include <filesystem>
#include <span>
#include <stdexcept>

namespace strix::inkling2 {

class MiniDFlashTarget {
public:
    explicit MiniDFlashTarget(MiniResidentDecoder& target) : target_(&target) {}

    [[nodiscard]] mini_gpu::Stream stream() const { return target_->stream_.get(); }
    [[nodiscard]] unsigned max_rows() const { return target_->max_rows_; }
    void dense(const float* input, unsigned width, unsigned rows, std::span<const core::Bf16RowProjection> projections) {
        target_->backend_.dense(input, width, rows, projections);
    }
    [[nodiscard]] const std::uint16_t* embed_table() const { return target_->weights_.bf16("embed.weight").data(); }
    [[nodiscard]] const std::uint16_t* embed_norm() const { return target_->weights_.bf16("embed_norm.weight").data(); }
    [[nodiscard]] static float embed_eps() { return 1e-6F; }  // the token rows' embedding norm (as the DSpark block)
    void head(const float* rows, unsigned count, float* logits) {
        target_->project_head(rows, std::span(logits, std::size_t{count} * kOutputVocabulary), false, count);
    }
    [[nodiscard]] static unsigned vocab_out() { return kOutputVocabulary; }
    [[nodiscard]] std::span<const std::uint16_t> committed_taps() const { return target_->committed_taps(); }

    // The drafter directory against this target, read before any device resource is reserved for it: empty when the
    // drafter binds (hidden width, embedding vocabulary, tap visits in the tap plane's order), else the reason.
    [[nodiscard]] static std::string problem(const std::filesystem::path& dir) {
        try {
            const auto c = spec::dflash::load_config(dir);
            if (c.hidden != kHiddenWidth) return "hidden width differs from the target";
            if (c.mask_token >= kVocabulary || c.vocab < kOutputVocabulary) return "vocabulary differs from the target";
            if (!std::ranges::equal(c.target_layers, MiniDSparkTapPlan::visits)) return "tap visits differ from the tap plan";
            return {};
        } catch (const std::exception& error) {
            return error.what();
        }
    }

private:
    MiniResidentDecoder* target_;
};

using MiniResidentDFlash = spec::dflash::Resident<MiniDFlashTarget>;

}  // namespace strix::inkling2
