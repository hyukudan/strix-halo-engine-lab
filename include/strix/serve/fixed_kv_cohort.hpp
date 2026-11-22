#pragma once

#include "strix/serve/cohort_decode.hpp"

#include <algorithm>
#include <concepts>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strix::serve {

// Bind a resident decoder with preallocated KV slots to the greedy cohort
// contract. Its decode_slots callback must execute one forward for all rows
// and publish the frontiers together, before returning the selected tokens.
// A decoder that also offers final_logits_supported() / logit_width() /
// read_slot_logits(row, out) serves sampled requests: the final scores of the
// last decode_slots() call are copied out only when a sampled lane asks.
template <class Decoder>
class FixedKvCohort final : public CohortBackend {
public:
    FixedKvCohort(std::unique_ptr<Decoder> decoder, std::uint32_t context,
                  std::uint32_t rows, std::uint32_t lanes)
        : decoder_(std::move(decoder)), context_(context), rows_(rows), lanes_(lanes) {
        if (!decoder_ || context_ == 0U || lanes_ == 0U || lanes_ > rows_)
            throw std::invalid_argument("invalid fixed KV cohort geometry");
        slots_.reserve(lanes_);
        decoder_->allocate_slots(lanes_);
    }

    std::uint32_t lanes() const override { return lanes_; }
    std::uint32_t rows_per_forward() const override { return rows_; }
    std::uint32_t max_positions() const override { return context_; }

    void prefill(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) override {
        decoder_->prefill_slot(lane, tokens, first);
    }

    [[nodiscard]] bool supports_prefill_head() const override { return kPrefillHead; }

    std::uint32_t prefill_head(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) override {
        if constexpr (kPrefillHead) {
            const auto top = decoder_->prefill_slot_head(lane, tokens, first);
            slots_.assign(1, lane);  // the single prompt head occupies compact row zero
            return top;
        } else {
            return CohortBackend::prefill_head(lane, tokens, first);
        }
    }

    void verify(std::span<const CohortGroup> groups, std::span<const std::uint32_t> tokens,
                std::span<std::uint32_t> top) override {
        slots_.clear();
        for (const auto& group : groups) {
            if (group.rows != 1U) throw std::logic_error("fixed KV cohort requires one decode row per lane");
            slots_.push_back(group.lane);
        }
        decoder_->decode_slots(slots_, tokens, top);
    }

    [[nodiscard]] bool supports_final_logits() const override {
        if constexpr (kReadsLogits) return decoder_->final_logits_supported();
        else return false;
    }

    std::span<const float> read_final_logits(std::uint32_t lane) override {
        if constexpr (kReadsLogits) {
            for (std::size_t row = 0; row < slots_.size(); ++row) {
                if (slots_[row] != lane) continue;
                logits_.resize(decoder_->logit_width());
                decoder_->read_slot_logits(static_cast<std::uint32_t>(row), logits_);
                return logits_;
            }
            throw std::logic_error("fixed KV cohort: final logits requested for a lane outside the window");
        } else {
            return CohortBackend::read_final_logits(lane);
        }
    }

    void commit(std::span<const std::uint32_t> keep) override {
        if (keep.size() != slots_.size() || !std::ranges::all_of(keep, [](auto n) { return n == 1U; }))
            throw std::logic_error("fixed KV cohort must keep each guaranteed decode row");
    }

    // A subsequent first prefill resets the reused slot's frontier and history.
    void release(std::uint32_t) override {}

private:
    static constexpr bool kPrefillHead = requires(Decoder& decoder, std::uint32_t lane,
                                                  std::span<const std::uint32_t> tokens, bool first) {
        { decoder.prefill_slot_head(lane, tokens, first) } -> std::same_as<std::uint32_t>;
    };
    static constexpr bool kReadsLogits = requires(Decoder& decoder, std::span<float> out) {
        { decoder.final_logits_supported() } -> std::convertible_to<bool>;
        { decoder.logit_width() } -> std::convertible_to<std::uint32_t>;
        decoder.read_slot_logits(std::uint32_t{}, out);
    };

    std::unique_ptr<Decoder> decoder_;
    std::uint32_t context_, rows_, lanes_;
    std::vector<std::uint32_t> slots_;
    std::vector<float> logits_;
};

} // namespace strix::serve
